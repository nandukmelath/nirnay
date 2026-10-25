"""Train the Nirnay fusion head on precomputed frozen-encoder states, calibrate, export.

Objective: log score (cross-entropy) on every question + ranked probability score on ordinal
(`score`) questions. Both are strictly proper scoring rules, the same family Laya's RLCD
rewards with REINFORCE on noisy logits; here the gradient of the rule is taken exactly.

  python train\\train_head.py                        (NIRNAY_TAG=e5s by default)
"""
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from nirnay.encoder import load_hf_encoder  # noqa: E402
from nirnay.engine import temp_bucket  # noqa: E402
from nirnay.model import QTYPES, DecisionHead, DeepFusionHead, StateAdapter  # noqa: E402

TAG = os.environ.get("NIRNAY_TAG", "e5s")
CACHE = os.path.join(HERE, "cache", TAG)
RUN = os.environ.get("NIRNAY_RUN", TAG + "-b%s" % os.environ.get("NIRNAY_BLOCKS", "2"))
RUND = os.path.join(HERE, "runs", RUN)
BLOCKS = int(os.environ.get("NIRNAY_BLOCKS", "2"))
BATCH = int(os.environ.get("NIRNAY_BATCH", "64"))
EPOCHS = float(os.environ.get("NIRNAY_EPOCHS", "2"))
LR = float(os.environ.get("NIRNAY_LR", "3e-4"))
WARMUP = int(os.environ.get("NIRNAY_WARMUP", "300"))
MAX_OPTS = int(os.environ.get("NIRNAY_MAX_OPTS", "64"))
MAX_STATE = int(os.environ.get("NIRNAY_MAX_STATE", "320"))   # training-time cap on state tokens
EVAL_EVERY = int(os.environ.get("NIRNAY_EVAL_EVERY", "600"))
SPLIT = int(os.environ.get("NIRNAY_SPLIT", "0"))          # >0: train encoder layers SPLIT+1.. on the state
ADAPTER_LR = float(os.environ.get("NIRNAY_ADAPTER_LR", "5e-5"))
FUSION = os.environ.get("NIRNAY_FUSION", "stack")          # "stack": frozen encoder + fusion blocks; "deep": top layers fused
WORLD = int(os.environ.get("WORLD_SIZE", "1"))             # >1: one process per CPU socket, gloo all-reduce
RANK = int(os.environ.get("RANK", "0"))
SEED = 1234
QNAMES = {v: k for k, v in QTYPES.items()}


class Model(torch.nn.Module):
    """Decision head, optionally preceded by a trainable state adapter (split encoder)."""

    def __init__(self, head, adapter=None, pool="cls"):
        super().__init__()
        self.head, self.adapter, self.pool_kind = head, adapter, pool

    def forward(self, S, sm, Q, qm, O, om, ok, t, sv, ov):
        if self.adapter is not None:
            S = self.adapter(S, sm)
            if self.pool_kind == "cls":
                v = S[:, 0]
            else:
                m = sm.unsqueeze(-1).to(S.dtype)
                v = (S * m).sum(1) / m.sum(1).clamp_min(1.0)
            sv = F.normalize(v, dim=-1)
        return self.head(S, sm, Q, qm, O, om, ok, t, sv, ov)


class Store:
    def __init__(self, d):
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.arr, self.off, self.vec = {}, {}, {}
        for n in ("states", "questions", "options"):
            mid = (FUSION == "deep") or (SPLIT and n == "states")
            f = n + ("_mid" if mid else "") + ".f16.npy"
            self.arr[n] = np.load(os.path.join(d, f), mmap_mode="r")
            self.off[n] = np.load(os.path.join(d, n + "_off.npy"))
            self.vec[n] = np.load(os.path.join(d, n + "_vec.npy"))
        # small tables fully in RAM
        for n in ("questions", "options"):
            self.arr[n] = np.asarray(self.arr[n])
        self.index = [json.loads(l) for l in open(os.path.join(d, "index.jsonl"), encoding="utf-8")]

    def rows(self, n, i):
        return self.arr[n][self.off[n][i]:self.off[n][i + 1]]


def make_batch(st: Store, recs, rng, train: bool):
    d = st.meta["d"]
    B = len(recs)
    opts, golds = [], []
    for r in recs:
        o, y = list(r["o"]), r["y"]
        if len(o) > MAX_OPTS:
            keep = [y] + rng.sample([i for i in range(len(o)) if i != y], MAX_OPTS - 1)
            keep.sort()
            y = keep.index(y)
            o = [o[i] for i in keep]
        opts.append(o)
        golds.append(y)
    n = min(MAX_STATE, max(st.off["states"][r["s"] + 1] - st.off["states"][r["s"]] for r in recs))
    Lq = max(st.off["questions"][r["q"] + 1] - st.off["questions"][r["q"]] for r in recs)
    K = max(len(o) for o in opts)
    Lo = max(st.off["options"][i + 1] - st.off["options"][i] for o in opts for i in o)
    S = np.zeros((B, n, d), np.float32)
    sm = np.zeros((B, n), bool)
    Q = np.zeros((B, Lq, d), np.float32)
    qm = np.zeros((B, Lq), bool)
    O = np.zeros((B, K, Lo, d), np.float32)
    om = np.zeros((B, K, Lo), bool)
    ok = np.zeros((B, K), bool)
    sv = np.zeros((B, d), np.float32)
    ov = np.zeros((B, K, d), np.float32)
    for b, (r, o) in enumerate(zip(recs, opts)):
        x = st.rows("states", r["s"])[:MAX_STATE]
        S[b, :len(x)] = x
        sm[b, :len(x)] = True
        sv[b] = st.vec["states"][r["s"]]
        x = st.rows("questions", r["q"])
        Q[b, :len(x)] = x
        qm[b, :len(x)] = True
        for j, i in enumerate(o):
            x = st.rows("options", i)
            O[b, j, :len(x)] = x
            om[b, j, :len(x)] = True
            ok[b, j] = True
            ov[b, j] = st.vec["options"][i]
    t = torch.tensor([r["t"] for r in recs])
    return (torch.from_numpy(S), torch.from_numpy(sm), torch.from_numpy(Q), torch.from_numpy(qm), torch.from_numpy(O),
            torch.from_numpy(om), torch.from_numpy(ok), t, torch.from_numpy(sv), torch.from_numpy(ov)), torch.tensor(golds)


def loss_fn(logits, y, qtype):
    ce = F.cross_entropy(logits, y, reduction="none")
    is_score = qtype == QTYPES["score"]
    if is_score.any():
        p = torch.softmax(logits[is_score], -1)
        tgt = F.one_hot(y[is_score], logits.shape[1]).float()
        k = torch.isfinite(logits[is_score]).sum(-1).clamp(min=2).float()
        rps = ((p.cumsum(-1) - tgt.cumsum(-1)) ** 2).sum(-1) / (k - 1)
        ce = ce.clone()
        ce[is_score] = ce[is_score] + rps
    return ce.mean()


def shape_key(st, r):
    """(option count, longest option, state length): batching on this keeps padding small."""
    oo = st.off["options"]
    k = min(len(r["o"]), MAX_OPTS)
    lo = max(oo[i + 1] - oo[i] for i in r["o"])
    so = st.off["states"]
    return (k, (lo + 3) // 4, so[r["s"] + 1] - so[r["s"]])


def batch_cost(st, batch):
    """Padded tokens the fused layers process for this batch (what step time scales with)."""
    so, qo, oo = st.off["states"], st.off["questions"], st.off["options"]
    n = min(MAX_STATE, max(so[r["s"] + 1] - so[r["s"]] for r in batch))
    lq = max(qo[r["q"] + 1] - qo[r["q"]] for r in batch)
    k = max(min(len(r["o"]), MAX_OPTS) for r in batch)
    lo = max(oo[i + 1] - oo[i] for r in batch for i in r["o"])
    return len(batch) * (n + lq + k * lo)


def batches_by_length(st, recs, bs, rng):
    recs = list(recs)
    rng.shuffle(recs)
    out = []
    chunk = bs * 64
    for s in range(0, len(recs), chunk):
        part = sorted(recs[s:s + chunk], key=lambda r: shape_key(st, r))
        out += [part[i:i + bs] for i in range(0, len(part), bs)]
    if WORLD > 1:
        # pair batches of similar cost so the ranks finish each step together (DDP waits for the slowest)
        out.sort(key=lambda b: batch_cost(st, b))
        groups = [out[i:i + WORLD] for i in range(0, len(out) - WORLD + 1, WORLD)]
        rng.shuffle(groups)
        return [b for g in groups for b in g]
    rng.shuffle(out)
    return out


@torch.no_grad()
def collect_logits(head, st, recs, bs=128):
    head.eval()
    rng = random.Random(0)
    res = []
    recs = sorted(recs, key=lambda r: shape_key(st, r))
    for i in range(0, len(recs), bs):
        part = recs[i:i + bs]
        if any(len(r["o"]) > MAX_OPTS for r in part):
            for r in part:
                (xb, yb) = make_batch(st, [r], rng, False) if len(r["o"]) <= MAX_OPTS else (None, None)
                if xb is None:
                    continue
                lg = head(*xb)
                res.append((r, lg[0].numpy(), int(yb[0])))
            continue
        xb, yb = make_batch(st, part, rng, False)
        lg = head(*xb).numpy()
        for r, z, y in zip(part, lg, yb.tolist()):
            res.append((r, z, y))
    head.train()
    return res


def metrics(rows, temps=None):
    by = defaultdict(lambda: [0, 0, 0.0])
    tot = [0, 0, 0.0]
    for r, z, y in rows:
        z = z[np.isfinite(z)]
        T = (temps or {}).get(temp_bucket(QNAMES[r["t"]], len(z)), 1.0)
        zz = z / T
        p = np.exp(zz - zz.max())
        p /= p.sum()
        c = int(np.argmax(p) == y)
        nll = -math.log(max(p[y], 1e-12))
        for acc in (by[r["src"]], tot):
            acc[0] += 1
            acc[1] += c
            acc[2] += nll
    out = {"acc": tot[1] / max(1, tot[0]), "nll": tot[2] / max(1, tot[0]), "n": tot[0]}
    out["by_src"] = {k: round(v[1] / v[0], 4) for k, v in sorted(by.items())}
    out["macro_acc"] = float(np.mean(list(out["by_src"].values()))) if by else 0.0
    return out


def fit_temperatures(rows):
    buckets = defaultdict(list)
    for r, z, y in rows:
        z = z[np.isfinite(z)]
        buckets[temp_bucket(QNAMES[r["t"]], len(z))].append((z, y))
    temps = {}
    grid = np.exp(np.linspace(math.log(0.2), math.log(5.0), 161))
    for b, items in sorted(buckets.items()):
        if len(items) < 30:
            continue
        best = (1e9, 1.0)
        for T in grid:
            nll = 0.0
            for z, y in items:
                zz = z / T
                m = zz.max()
                nll += -(zz[y] - m - math.log(np.exp(zz - m).sum()))
            best = min(best, (nll / len(items), float(T)))
        temps[b] = round(best[1], 4)
    return temps


def main():
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    torch.set_num_threads(int(os.environ.get("NIRNAY_THREADS", "16")))
    os.makedirs(RUND, exist_ok=True)
    st = Store(CACHE)
    tr = [r for r in st.index if r["split"] == "train"]
    dev = [r for r in st.index if r["split"] == "dev"]
    cal = [r for r in st.index if r["split"] == "cal"]
    if os.environ.get("NIRNAY_DEV_LIMIT"):
        dev = dev[:int(os.environ["NIRNAY_DEV_LIMIT"])]
    print("train %d dev %d cal %d" % (len(tr), len(dev), len(cal)), flush=True)
    meta = st.meta
    enc = load_hf_encoder(meta["backbone"], meta["pool"], meta["prefix"])
    d, heads, ff = enc.cfg.hidden, enc.cfg.heads, enc.cfg.ff
    adapter = None
    if FUSION == "deep":
        assert SPLIT and meta.get("mid_layer") == SPLIT, "deep fusion needs NIRNAY_SPLIT = cached mid layer"
        head = DeepFusionHead(list(enc.layers)[SPLIT:], d, pool=meta["pool"])
        pre = [p for n, p in head.named_parameters() if n.startswith("layers.")]
        new = [p for n, p in head.named_parameters() if not n.startswith("layers.")]
        groups = [{"params": new, "lr": LR}, {"params": pre, "lr": ADAPTER_LR}]
    else:
        head = DecisionHead(d, heads, ff, BLOCKS)
        head.init_from_encoder(enc.layers)
        if SPLIT:
            assert meta.get("mid_layer") == SPLIT, "cache holds layer %s states, not %d" % (meta.get("mid_layer"), SPLIT)
            adapter = StateAdapter(list(enc.layers)[SPLIT:], enc.cfg.ln_eps)
        groups = [{"params": list(head.parameters()), "lr": LR}]
        if adapter is not None:
            groups.append({"params": list(adapter.parameters()), "lr": ADAPTER_LR})
    model = Model(head, adapter, meta["pool"])
    if os.environ.get("NIRNAY_INIT"):   # warm start (e.g. a second epoch) from a saved model_best.pt
        model.load_state_dict(torch.load(os.environ["NIRNAY_INIT"], weights_only=True))
        print("warm start from", os.environ["NIRNAY_INIT"], flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print("trainable params %.2fM, fusion %s, blocks %d, split %d, world %d" % (n_params / 1e6, FUSION, BLOCKS, SPLIT, WORLD), flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.98))
    ddp = model
    if WORLD > 1:
        import torch.distributed as dist
        dist.init_process_group("gloo", init_method=os.environ["NIRNAY_DDP_INIT"], rank=RANK, world_size=WORLD)
        ddp = torch.nn.parallel.DistributedDataParallel(model)
    steps_per_epoch = math.ceil(len(tr) / BATCH / WORLD)
    total = int(steps_per_epoch * EPOCHS)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    log = {"run": RUN, "tag": TAG, "fusion": FUSION, "blocks": BLOCKS, "batch": BATCH, "world": WORLD, "lr": LR,
           "adapter_lr": ADAPTER_LR, "epochs": EPOCHS, "evals": []}
    best = (-1.0, None)
    step, t0, run_loss = 0, time.time(), 0.0
    model.train()
    ep = 0

    def evaluate():
        m = metrics(collect_logits(model, st, dev))
        m.update({"step": step, "loss": round(run_loss, 4), "elapsed_min": round((time.time() - t0) / 60, 1)})
        log["evals"].append({k: v for k, v in m.items() if k != "by_src"})
        print("\n  dev acc %.4f macro %.4f nll %.4f" % (m["acc"], m["macro_acc"], m["nll"]), flush=True)
        json.dump(log, open(os.path.join(RUND, "log.json"), "w"), indent=2)
        return m

    if os.environ.get("NIRNAY_EXPORT_ONLY") == "1":   # calibrate + export a saved checkpoint (NIRNAY_INIT), no training
        assert WORLD == 1 and os.environ.get("NIRNAY_INIT"), "export-only needs NIRNAY_INIT and a single process"
        best = (-1.0, None)
        m = evaluate()
        best = (m["macro_acc"], m)
        torch.save(model.state_dict(), os.path.join(RUND, "model_best.pt"))
        total = 0
    while step < total:
        # every rank builds the identical batch list (same seed) and takes its own slice; equal counts keep
        # the all-reduce in lockstep
        bl = batches_by_length(st, tr, BATCH, rng)
        bl = bl[:len(bl) // WORLD * WORLD]
        for batch in bl[RANK::WORLD]:
            xb, yb = make_batch(st, batch, rng, True)
            logits = ddp(*xb)
            loss = loss_fn(logits, yb, xb[7])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * float(loss) if step > 1 else float(loss)
            if step % 50 == 0 and RANK == 0:
                el = time.time() - t0
                print("\r step %d/%d  loss %.4f  %.2f s/step  ETA %.0f min   " %
                      (step, total, run_loss, el / step, el / step * (total - step) / 60), end="", flush=True)
            if step % EVAL_EVERY == 0 or step >= total:
                if RANK == 0:
                    m = evaluate()
                    if m["macro_acc"] > best[0]:
                        best = (m["macro_acc"], m)
                        torch.save(model.state_dict(), os.path.join(RUND, "model_best.pt"))
                if WORLD > 1:
                    torch.distributed.barrier()
            if step >= total:
                break
        ep += 1
    if RANK != 0:
        torch.distributed.barrier()
        return
    print("\nbest dev macro acc %.4f" % best[0], flush=True)
    model.load_state_dict(torch.load(os.path.join(RUND, "model_best.pt"), weights_only=True))
    cal_rows = collect_logits(model, st, cal)
    temps = fit_temperatures(cal_rows)
    print("temperatures", temps, flush=True)
    m_raw, m_cal = metrics(cal_rows), metrics(cal_rows, temps)
    log["best_dev"] = best[1]
    log["cal_raw"] = {k: v for k, v in m_raw.items() if k != "by_src"}
    log["cal_calibrated"] = {k: v for k, v in m_cal.items() if k != "by_src"}
    log["temperatures"] = temps
    json.dump(log, open(os.path.join(RUND, "log.json"), "w"), indent=2)
    export(model, enc, meta, temps, log)
    if WORLD > 1:
        torch.distributed.barrier()


def export(model, enc, meta, temps, log):
    from safetensors.torch import save_file
    out = os.environ.get("NIRNAY_EXPORT", os.path.join(ROOT, "models", "nirnay-" + RUN))
    os.makedirs(out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in enc.state_dict().items()}, os.path.join(out, "encoder.safetensors"))
    save_file({k: v.contiguous() for k, v in model.head.state_dict().items()}, os.path.join(out, "head.safetensors"))
    if model.adapter is not None:
        save_file({k: v.contiguous() for k, v in model.adapter.state_dict().items()}, os.path.join(out, "adapter.safetensors"))
    shutil.copy(os.path.join(meta["source_dir"], "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    cfg = {"name": "nirnay-" + RUN, "version": "1.0.0", "backbone": meta["backbone"], "encoder": enc.cfg.to_dict(),
           "head": {"blocks": BLOCKS, "heads": enc.cfg.heads, "ff": enc.cfg.ff}, "split_layer": SPLIT, "fusion": FUSION,
           "special_ids": meta["special_ids"],
           "limits": {"state_tokens": 512, "question_tokens": meta["limits"]["question_tokens"],
                      "option_tokens": meta["limits"]["option_tokens"], "max_fused_options": 128, "window_overlap": 64},
           "threads": 8, "temperature_by_bucket": temps,
           "training": {"records": meta["n_records"], "dev_macro_acc": round(log["best_dev"]["macro_acc"], 4),
                        "dev_acc": round(log["best_dev"]["acc"], 4), "step": log["best_dev"]["step"]}}
    json.dump(cfg, open(os.path.join(out, "nirnay_config.json"), "w", encoding="utf-8"), indent=2)
    print("exported ->", out, flush=True)


if __name__ == "__main__":
    main()
