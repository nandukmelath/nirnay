"""Run the frozen encoder once over every unique training text and store the token states.

Outputs in train/cache/<tag>/:
  states.f16 / questions.f16 / options.f16   token states, rows of d (float16 memmaps)
  *_mid.f16                                   the same at layer NIRNAY_SAVE_MID (what deep fusion trains on)
  *_off.npy  [n+1] row offsets,  *_vec.npy [n, d] pooled L2-normalised embeddings (float32)
  index.jsonl one line per record: state / question / option ids (canonical option order), gold, type, src, split
  meta.json   backbone, limits, counts

Options for big backbones:
  NIRNAY_STATE_MID_ONLY=1   states stop at the mid layer (deep fusion never needs their frozen top layers)
  NIRNAY_SHARD=i/n          run n processes (one per CPU socket); each encodes every n-th state (by length)
                            into the same memory-mapped file; shard 0 also does questions/options and the index

  python train\\precompute.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from nirnay.encoder import load_hf_encoder  # noqa: E402
from nirnay.model import QTYPES  # noqa: E402
from nirnay.render import render_question, render_state  # noqa: E402

BACKBONE = os.environ.get("NIRNAY_BACKBONE", "intfloat/multilingual-e5-small")
POOL = os.environ.get("NIRNAY_POOL", "mean")
PREFIX = os.environ.get("NIRNAY_PREFIX", "query: ")
PREFIX = "" if PREFIX.strip().lower() == "none" else PREFIX   # Windows drops empty env vars
TAG = os.environ.get("NIRNAY_TAG", "e5s")
REC = os.environ.get("NIRNAY_RECORDS", os.path.join(HERE, "data", "records.jsonl"))
OUTD = os.path.join(HERE, "cache", TAG)
LIM = {"state_tokens": int(os.environ.get("NIRNAY_TRAIN_STATE_TOKENS", "320")), "question_tokens": 64, "option_tokens": 32}
MID = int(os.environ.get("NIRNAY_SAVE_MID", "8"))   # also keep this layer's output (0 = off)
STATE_MID_ONLY = os.environ.get("NIRNAY_STATE_MID_ONLY") == "1"
SHARD, NSHARD = (int(x) for x in os.environ.get("NIRNAY_SHARD", "0/1").split("/"))


def wait_for(path, what):
    t = time.time()
    while not os.path.exists(path):
        time.sleep(2)
        if time.time() - t > 6 * 3600:
            raise SystemExit("timed out waiting for " + what)


def main():
    from tokenizers import Tokenizer
    torch.set_num_threads(int(os.environ.get("NIRNAY_THREADS", "16")))
    os.makedirs(OUTD, exist_ok=True)
    if NSHARD > 1:
        assert MID and STATE_MID_ONLY, "sharding is supported for NIRNAY_STATE_MID_ONLY=1 runs"
    enc = load_hf_encoder(BACKBONE, POOL, PREFIX)
    tok = Tokenizer.from_file(os.path.join(enc.source_dir, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    probe = tok.encode("a", add_special_tokens=True).ids
    cls_id, sep_id = probe[0], probe[-1]
    pad_id = enc.cfg.pad_id
    prefix_ids = tok.encode(PREFIX, add_special_tokens=False).ids if PREFIX else []
    print("backbone", BACKBONE, "cls/sep/pad", cls_id, sep_id, pad_id, "prefix", prefix_ids,
          "shard %d/%d" % (SHARD, NSHARD), flush=True)

    def ids_for(text, budget):
        room = budget - 2 - len(prefix_ids)
        return [cls_id] + prefix_ids + tok.encode(text, add_special_tokens=False).ids[:max(1, room)] + [sep_id]

    states, questions, options = {}, {}, {}
    index = []
    t0 = time.time()
    with open(REC, encoding="utf-8") as f:
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
    print("records %d | unique states %d questions %d options %d | %.0fs" %
          (len(index), len(states), len(questions), len(options), time.time() - t0), flush=True)

    def encode_all(texts, budget, name, mid_only=False):
        seqs = [ids_for(t, budget) for t in texts]
        lens = np.array([len(s) for s in seqs], dtype=np.int64)
        off = np.zeros(len(seqs) + 1, dtype=np.int64)
        off[1:] = np.cumsum(lens)
        d = enc.cfg.hidden
        rows = (int(off[-1]), d)
        sharded = mid_only and NSHARD > 1
        created = os.path.join(OUTD, name + "_mid.created")
        if sharded and SHARD != 0:
            wait_for(created, name + " memmap")
            mm = None
            mm_mid = np.lib.format.open_memmap(os.path.join(OUTD, name + "_mid.f16.npy"), mode="r+")
            assert mm_mid.shape == rows, (mm_mid.shape, rows)
        else:
            if sharded and os.path.exists(created):
                os.remove(created)
            mm = None if mid_only else np.lib.format.open_memmap(os.path.join(OUTD, name + ".f16.npy"), mode="w+",
                                                                 dtype=np.float16, shape=rows)
            mm_mid = np.lib.format.open_memmap(os.path.join(OUTD, name + "_mid.f16.npy"), mode="w+", dtype=np.float16,
                                               shape=rows) if MID else None
            if sharded:
                mm_mid.flush()
                open(created, "w").close()
        vec = np.zeros((len(seqs), d), dtype=np.float32)
        order = np.argsort(lens, kind="stable")
        if sharded:
            order = order[SHARD::NSHARD]
        total = int(lens[order].sum())
        i, done, t1, last = 0, 0, time.time(), 0.0
        budget_tok = 12288
        with torch.no_grad():
            while i < len(order):
                j = i
                while j < len(order) and lens[order[j]] * (j - i + 1) <= max(budget_tok, lens[order[i]]):
                    j += 1
                j = max(j, i + 1)
                idx = order[i:j]
                Lm = int(lens[idx].max())
                ids = torch.full((len(idx), Lm), pad_id, dtype=torch.long)
                mask = torch.zeros((len(idx), Lm), dtype=torch.bool)
                for rr, t in enumerate(idx):
                    ids[rr, :lens[t]] = torch.tensor(seqs[t])
                    mask[rr, :lens[t]] = True
                if mid_only:
                    Mn = enc(ids, mask, stop_layer=MID).numpy().astype(np.float16)
                    for rr, t in enumerate(idx):
                        mm_mid[off[t]:off[t + 1]] = Mn[rr, :lens[t]]
                else:
                    if MID:
                        H, Mh = enc(ids, mask, keep_layer=MID)
                        Mn = Mh.numpy().astype(np.float16)
                    else:
                        H = enc(ids, mask)
                    V = enc.pool(H, mask)
                    Hn = H.numpy().astype(np.float16)
                    for rr, t in enumerate(idx):
                        mm[off[t]:off[t + 1]] = Hn[rr, :lens[t]]
                        if MID:
                            mm_mid[off[t]:off[t + 1]] = Mn[rr, :lens[t]]
                    vec[idx] = V.numpy()
                done += int(lens[idx].sum())
                i = j
                if time.time() - last > 5 or i >= len(order):
                    last = time.time()
                    el = time.time() - t1
                    print("\r  [%s] %d/%d texts  %.1fk tok/s  ETA %.0f min   " %
                          (name, i, len(order), done / max(el, 1e-9) / 1000,
                           (total - done) / max(done / max(el, 1e-9), 1e-9) / 60), end="", flush=True)
        if mm is not None:
            mm.flush()
            del mm
        if mm_mid is not None:
            mm_mid.flush()
            del mm_mid
        if sharded:
            open(os.path.join(OUTD, "%s.done.%d" % (name, SHARD)), "w").close()
            if SHARD != 0:
                print("\n  [%s] shard %d: %d texts, %d tokens in %.1f min" % (name, SHARD, len(order), total, (time.time() - t1) / 60), flush=True)
                return
            for s in range(NSHARD):
                wait_for(os.path.join(OUTD, "%s.done.%d" % (name, s)), "%s shard %d" % (name, s))
        np.save(os.path.join(OUTD, name + "_off.npy"), off)
        np.save(os.path.join(OUTD, name + "_vec.npy"), vec)
        print("\n  [%s] %d texts, %d tokens in %.1f min" % (name, len(seqs), int(off[-1]), (time.time() - t1) / 60), flush=True)

    if SHARD == 0:
        encode_all(list(questions), LIM["question_tokens"], "questions")
        encode_all(list(options), LIM["option_tokens"], "options")
    encode_all(list(states), LIM["state_tokens"], "states", mid_only=STATE_MID_ONLY)
    if SHARD != 0:
        return
    with open(os.path.join(OUTD, "index.jsonl"), "w", encoding="utf-8") as f:
        for r in index:
            f.write(json.dumps(r) + "\n")
    json.dump({"backbone": BACKBONE, "pool": POOL, "prefix": PREFIX, "limits": LIM, "d": enc.cfg.hidden, "mid_layer": MID,
               "state_mid_only": STATE_MID_ONLY,
               "n_records": len(index), "n_states": len(states), "n_questions": len(questions),
               "n_options": len(options), "special_ids": {"cls": cls_id, "sep": sep_id, "pad": pad_id},
               "encoder": enc.cfg.to_dict(), "source_dir": enc.source_dir},
              open(os.path.join(OUTD, "meta.json"), "w"), indent=2)
    print("done ->", OUTD)


if __name__ == "__main__":
    main()
