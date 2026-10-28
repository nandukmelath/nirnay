"""Train Nirnay on a CUDA GPU (Kaggle, Colab, any box) without the precompute cache.

The CPU pipeline (precompute.py + train_head.py) stores the frozen encoder's layer-SPLIT token states for every
training text on disk (~40 GB), because recomputing them on a CPU is the expensive part. On a GPU that is cheap:
  * questions and options (few, reused by many records) are encoded once at start-up exactly as precompute.py does:
    the full frozen encoder, keeping the layer-SPLIT states and the pooled final-layer vector;
  * states are encoded per batch up to layer SPLIT (their pooled vectors are zeros, as in the mid-only cache).
Model, loss, batching, dev selection, temperature calibration and export are train_head.py's own code, so the result
is the model the CPU pipeline trains, only faster. Mixed precision (fp16 autocast + loss scaling) on the GPU.

  NIRNAY_RECORDS=records.jsonl NIRNAY_BACKBONE=BAAI/bge-base-en-v1.5 NIRNAY_POOL=cls NIRNAY_PREFIX=none
  NIRNAY_TRAIN_STATE_TOKENS=256 NIRNAY_SPLIT=8 NIRNAY_MAX_STATE=256 NIRNAY_EPOCHS=1.6 NIRNAY_RUN=v3
  NIRNAY_EXPORT=out/nirnay-v3 python train/train_gpu.py

NIRNAY_DEVICE=cpu runs it on a CPU (tests); NIRNAY_LIMIT_RECORDS=n reads only the first n records.
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("NIRNAY_FUSION", "deep")
os.environ.setdefault("NIRNAY_SPLIT", "8")
import train_head as th  # noqa: E402  (reads the NIRNAY_* hyper-parameters from the environment)
from nirnay.encoder import load_hf_encoder  # noqa: E402
from nirnay.model import QTYPES, DeepFusionHead  # noqa: E402
from nirnay.render import render_question, render_state  # noqa: E402

BACKBONE = os.environ.get("NIRNAY_BACKBONE", "intfloat/multilingual-e5-small")
POOL = os.environ.get("NIRNAY_POOL", "mean")
PREFIX = os.environ.get("NIRNAY_PREFIX", "query: ")
PREFIX = "" if PREFIX.strip().lower() == "none" else PREFIX
REC = os.environ.get("NIRNAY_RECORDS", os.path.join(HERE, "data", "records.jsonl"))
LIMIT = int(os.environ.get("NIRNAY_LIMIT_RECORDS", "0"))
LIM = {"state_tokens": int(os.environ.get("NIRNAY_TRAIN_STATE_TOKENS", "320")), "question_tokens": 64, "option_tokens": 32}
DEV = torch.device(os.environ.get("NIRNAY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
AMP = DEV.type == "cuda"
TAB = torch.float16 if AMP else torch.float32     # dtype of the question / option state tables
CHUNK_TOKENS = 16384                               # padded tokens per encoder call at start-up
# Attention-memory budget per forward pass (sum over samples of the fused layers' score elements per head). A batch
# over budget is split into micro-batches whose gradients accumulate into one optimizer step: the same update as
# the whole batch (padding is masked), with bounded GPU memory. ~8M fits a 16 GB T4 with room to spare.
MICRO_BUDGET = float(os.environ.get("NIRNAY_MICRO_BUDGET", "8e6"))
assert th.FUSION == "deep" and th.SPLIT > 0, "train_gpu.py trains deep fusion (NIRNAY_FUSION=deep, NIRNAY_SPLIT>0)"


def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=AMP)


class LiveStore:
    """train_head.Store's interface (meta, index, off, vec) with token ids instead of cached states."""

    def __init__(self, enc, tok):
        probe = tok.encode("a", add_special_tokens=True).ids
        self.cls, self.sep, self.pad = probe[0], probe[-1], enc.cfg.pad_id
        self.prefix_ids = tok.encode(PREFIX, add_special_tokens=False).ids if PREFIX else []
        t0 = time.time()
        states, questions, options, index = {}, {}, {}, []
        with open(REC, encoding="utf-8") as f:   # exactly precompute.py's numbering and canonical option order
            for line in f:
                r = json.loads(line)
                try:
                    nq, qtext, otexts = render_question("q", r["q"])
                except Exception:
                    continue
                perm = sorted(range(len(otexts)), key=lambda i: (otexts[i], i))
                st = render_state(r["state"])
                si = states.setdefault(st, len(states))
                qi = questions.setdefault(qtext, len(questions))
                oi = [options.setdefault(otexts[i], len(options)) for i in perm]
                index.append({"s": si, "q": qi, "o": oi, "y": perm.index(int(r["y"])), "t": QTYPES[nq["type"]],
                              "src": r["src"], "split": r["split"]})
                if LIMIT and len(index) >= LIMIT:
                    break
        self.index = index
        print("records %d | unique states %d questions %d options %d | %.0fs" %
              (len(index), len(states), len(questions), len(options), time.time() - t0), flush=True)
        self.ids, self.off = {}, {}
        for name, texts, budget in (("states", states, LIM["state_tokens"]), ("questions", questions, LIM["question_tokens"]),
                                    ("options", options, LIM["option_tokens"])):
            room = budget - 2 - len(self.prefix_ids)
            seqs = [[self.cls] + self.prefix_ids + e.ids[:max(1, room)] + [self.sep]
                    for e in tok.encode_batch(list(texts), add_special_tokens=False)]
            lens = np.array([len(s) for s in seqs], dtype=np.int64)
            off = np.zeros(len(seqs) + 1, dtype=np.int64)
            off[1:] = np.cumsum(lens)
            self.ids[name] = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])
            self.off[name] = off
            print("  tokenized %-9s %7d texts %10d tokens" % (name, len(seqs), int(off[-1])), flush=True)
        d = enc.cfg.hidden
        self.vec = {"states": np.zeros((len(states), d), np.float32)}   # mid-only states: no pooled vector
        self.mid, self.vecd = {}, {}
        for name in ("questions", "options"):
            self._encode_table(enc, name)
        self.meta = {"backbone": BACKBONE, "pool": POOL, "prefix": PREFIX, "limits": LIM, "d": d, "mid_layer": th.SPLIT,
                     "state_mid_only": True, "n_records": len(index), "n_states": len(states),
                     "n_questions": len(questions), "n_options": len(options),
                     "special_ids": {"cls": self.cls, "sep": self.sep, "pad": self.pad}, "encoder": enc.cfg.to_dict(),
                     "source_dir": enc.source_dir}

    def batch_ids(self, name, items):
        off = self.off[name]
        lens = [int(off[i + 1] - off[i]) for i in items]
        L = max(lens)
        ids = torch.full((len(items), L), self.pad, dtype=torch.long)
        mask = torch.zeros((len(items), L), dtype=torch.bool)
        for r, (i, n) in enumerate(zip(items, lens)):
            ids[r, :n] = torch.from_numpy(self.ids[name][off[i]:off[i] + n])
            mask[r, :n] = True
        return ids.to(DEV, non_blocking=True), mask.to(DEV, non_blocking=True)

    @torch.no_grad()
    def _encode_table(self, enc, name):
        """Full frozen encoder over every question / option: layer-SPLIT states + pooled final vector."""
        t0 = time.time()
        off = self.off[name]
        n = len(off) - 1
        lens = off[1:] - off[:-1]
        mid = torch.zeros((int(off[-1]), enc.cfg.hidden), dtype=TAB, device=DEV)
        vec = torch.zeros((n, enc.cfg.hidden), dtype=torch.float32, device=DEV)
        order = np.argsort(lens, kind="stable")
        i = 0
        while i < n:
            j = i
            while j < n and lens[order[j]] * (j - i + 1) <= max(CHUNK_TOKENS, lens[order[i]]):
                j += 1
            j = max(j, i + 1)
            idx = order[i:j].tolist()
            ids, mask = self.batch_ids(name, idx)
            with autocast():
                H, M = enc(ids, mask, keep_layer=th.SPLIT)
            vec[idx] = enc.pool(H.float(), mask)
            for r, t in enumerate(idx):
                mid[off[t]:off[t + 1]] = M[r, :lens[t]].to(TAB)
            i = j
        self.mid[name], self.vecd[name] = mid, vec
        self.vec[name] = vec.cpu().numpy()
        print("  encoded   %-9s %7d texts in %.0fs" % (name, n, time.time() - t0), flush=True)

    def rows(self, n, i):   # train_head.Store compatibility (questions / options only)
        return self.mid[n][self.off[n][i]:self.off[n][i + 1]]


def make_batch(st, enc, recs, rng):
    """train_head.make_batch, with the states encoded live and every tensor on the device."""
    d = st.meta["d"]
    B = len(recs)
    opts, golds = [], []
    for r in recs:
        o, y = list(r["o"]), r["y"]
        if len(o) > th.MAX_OPTS:
            keep = [y] + rng.sample([i for i in range(len(o)) if i != y], th.MAX_OPTS - 1)
            keep.sort()
            y = keep.index(y)
            o = [o[i] for i in keep]
        opts.append(o)
        golds.append(y)
    ids, mask = st.batch_ids("states", [r["s"] for r in recs])
    with torch.no_grad(), autocast():
        S = enc(ids, mask, stop_layer=th.SPLIT)
    n = min(th.MAX_STATE, S.shape[1])
    sm = mask[:, :n]
    S = (S[:, :n] * sm.unsqueeze(-1)).to(TAB)   # zeros at padding, exactly as the cached batches
    qo, oo = st.off["questions"], st.off["options"]
    Lq = max(int(qo[r["q"] + 1] - qo[r["q"]]) for r in recs)
    K = max(len(o) for o in opts)
    Lo = max(int(oo[i + 1] - oo[i]) for o in opts for i in o)
    Q = torch.zeros((B, Lq, d), dtype=TAB, device=DEV)
    qm = torch.zeros((B, Lq), dtype=torch.bool, device=DEV)
    O = torch.zeros((B, K, Lo, d), dtype=TAB, device=DEV)
    om = torch.zeros((B, K, Lo), dtype=torch.bool, device=DEV)
    ok = torch.zeros((B, K), dtype=torch.bool, device=DEV)
    ov = torch.zeros((B, K, d), dtype=torch.float32, device=DEV)
    for b, (r, o) in enumerate(zip(recs, opts)):
        x = st.rows("questions", r["q"])
        Q[b, :len(x)] = x
        qm[b, :len(x)] = True
        for j, i in enumerate(o):
            x = st.rows("options", i)
            O[b, j, :len(x)] = x
            om[b, j, :len(x)] = True
        ok[b, :len(o)] = True
        ov[b, :len(o)] = st.vecd["options"][o]
    sv = torch.zeros((B, d), dtype=torch.float32, device=DEV)
    t = torch.tensor([r["t"] for r in recs], device=DEV)
    return (S, sm, Q, qm, O, om, ok, t, sv, ov), torch.tensor(golds, device=DEV)


def micro_batches(st, recs, budget):
    """Split a batch into near-equal parts whose attention cost stays under `budget`."""
    so, qo, oo = st.off["states"], st.off["questions"], st.off["options"]
    n = min(th.MAX_STATE, max(int(so[r["s"] + 1] - so[r["s"]]) for r in recs))
    lq = max(int(qo[r["q"] + 1] - qo[r["q"]]) for r in recs)
    k = max(min(len(r["o"]), th.MAX_OPTS) for r in recs)
    lo = max(int(oo[i + 1] - oo[i]) for r in recs for i in r["o"])
    per = k * lo * (n + lq + lo) + lq * (n + lq) + n * n
    parts = max(1, math.ceil(len(recs) * per / budget))
    size = math.ceil(len(recs) / parts)
    return [recs[i:i + size] for i in range(0, len(recs), size)]


@torch.no_grad()
def collect_logits(model, st, enc, recs, bs=128):
    model.eval()
    rng = random.Random(0)
    res = []
    recs = sorted(recs, key=lambda r: th.shape_key(st, r))
    for i in range(0, len(recs), bs):
        part = [r for r in recs[i:i + bs] if len(r["o"]) <= th.MAX_OPTS]   # as train_head: longer ones are skipped
        if not part:
            continue
        for mb in micro_batches(st, part, 3 * MICRO_BUDGET):   # no activations kept: a larger budget is safe
            xb, yb = make_batch(st, enc, mb, rng)
            with autocast():
                lg = model(*xb)
            for r, z, y in zip(mb, lg.float().cpu().numpy(), yb.tolist()):
                res.append((r, z, y))
    model.train()
    return res


def main():
    from tokenizers import Tokenizer
    torch.manual_seed(th.SEED)
    rng = random.Random(th.SEED)
    torch.set_num_threads(int(os.environ.get("NIRNAY_THREADS", "4")))
    os.makedirs(th.RUND, exist_ok=True)
    print("device", DEV, torch.cuda.get_device_name(DEV) if DEV.type == "cuda" else "", "| amp", AMP, flush=True)
    enc = load_hf_encoder(BACKBONE, POOL, PREFIX).to(DEV).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    tok = Tokenizer.from_file(os.path.join(enc.source_dir, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    st = LiveStore(enc, tok)
    tr = [r for r in st.index if r["split"] == "train"]
    dev = [r for r in st.index if r["split"] == "dev"]
    cal = [r for r in st.index if r["split"] == "cal"]
    if os.environ.get("NIRNAY_DEV_LIMIT"):
        dev = dev[:int(os.environ["NIRNAY_DEV_LIMIT"])]
    print("train %d dev %d cal %d" % (len(tr), len(dev), len(cal)), flush=True)
    d = enc.cfg.hidden
    head = DeepFusionHead(list(enc.layers)[th.SPLIT:], d, pool=POOL)
    pre = [p for nm, p in head.named_parameters() if nm.startswith("layers.")]
    new = [p for nm, p in head.named_parameters() if not nm.startswith("layers.")]
    for p in pre + new:
        p.requires_grad_(True)
    model = th.Model(head, None, POOL).to(DEV)
    if os.environ.get("NIRNAY_INIT"):
        model.load_state_dict(torch.load(os.environ["NIRNAY_INIT"], map_location="cpu", weights_only=True))
        print("warm start from", os.environ["NIRNAY_INIT"], flush=True)
    opt = torch.optim.AdamW([{"params": new, "lr": th.LR}, {"params": pre, "lr": th.ADAPTER_LR}],
                            weight_decay=0.01, betas=(0.9, 0.98))
    scaler = (torch.amp.GradScaler("cuda", enabled=AMP) if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=AMP))
    total = int(math.ceil(len(tr) / th.BATCH) * th.EPOCHS)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / th.WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    log = {"run": th.RUN, "backbone": BACKBONE, "fusion": th.FUSION, "split": th.SPLIT, "batch": th.BATCH, "lr": th.LR,
           "adapter_lr": th.ADAPTER_LR, "epochs": th.EPOCHS, "device": str(DEV), "amp": AMP, "evals": []}
    best = (-1.0, None)
    step, t0, run_loss = 0, time.time(), 0.0
    model.train()

    def evaluate():
        m = th.metrics(collect_logits(model, st, enc, dev))
        m.update({"step": step, "loss": round(run_loss, 4), "elapsed_min": round((time.time() - t0) / 60, 1)})
        log["evals"].append({k: v for k, v in m.items() if k != "by_src"})
        print("\n  dev acc %.4f macro %.4f nll %.4f  (step %d, %.0f min)" % (m["acc"], m["macro_acc"], m["nll"], step,
                                                                          (time.time() - t0) / 60), flush=True)
        json.dump(log, open(os.path.join(th.RUND, "log.json"), "w"), indent=2)
        return m

    while step < total:
        for batch in th.batches_by_length(st, tr, th.BATCH, rng):
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for mb in micro_batches(st, batch, MICRO_BUDGET):   # gradient accumulation: one step per batch
                xb, yb = make_batch(st, enc, mb, rng)
                with autocast():
                    logits = model(*xb)
                part = th.loss_fn(logits.float(), yb, xb[7]) * (len(mb) / len(batch))
                scaler.scale(part).backward()
                loss += part.detach()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * float(loss) if step > 1 else float(loss)
            if step % 100 == 0:
                el = time.time() - t0
                print("step %d/%d  loss %.4f  %.3f s/step  ETA %.0f min" %
                      (step, total, run_loss, el / step, el / step * (total - step) / 60), flush=True)
            if step % th.EVAL_EVERY == 0 or step >= total:
                m = evaluate()
                if m["macro_acc"] > best[0]:
                    best = (m["macro_acc"], m)
                    torch.save(model.state_dict(), os.path.join(th.RUND, "model_best.pt"))
            if step >= total:
                break
    print("\nbest dev macro acc %.4f" % best[0], flush=True)
    model.load_state_dict(torch.load(os.path.join(th.RUND, "model_best.pt"), map_location=DEV, weights_only=True))
    cal_rows = collect_logits(model, st, enc, cal)
    temps = th.fit_temperatures(cal_rows)
    print("temperatures", temps, flush=True)
    m_raw, m_cal = th.metrics(cal_rows), th.metrics(cal_rows, temps)
    log["best_dev"] = best[1]
    log["cal_raw"] = {k: v for k, v in m_raw.items() if k != "by_src"}
    log["cal_calibrated"] = {k: v for k, v in m_cal.items() if k != "by_src"}
    log["temperatures"] = temps
    log["train_minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(log, open(os.path.join(th.RUND, "log.json"), "w"), indent=2)
    model.cpu()
    enc.cpu()
    th.export(model, enc, st.meta, temps, log)


if __name__ == "__main__":
    main()
