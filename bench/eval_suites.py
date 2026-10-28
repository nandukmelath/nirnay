"""Run one model over the frozen benchmark suites (bench/data/suites.jsonl).

  python bench\\eval_suites.py laya:laya              Laya English checkpoint
  python bench\\eval_suites.py laya:laya-multilingual
  python bench\\eval_suites.py laya:laya-typed-decisions
  python bench\\eval_suites.py nirnay:models\\nirnay-v1

Both models are called through their public batch API with their shipped calibration, exactly as
a user would. Writes bench/results/suites__<model>.json with per-suite accuracy, macro-F1, ECE,
Brier, NLL (+ MAE for ordinal), throughput, an option-order flip test and a determinism check.
"""
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
SUITES = os.environ.get("BENCH_SUITES_FILE") or os.path.join(HERE, "data", "suites.jsonl")
OUT_TAG = os.environ.get("BENCH_OUT_TAG", "")   # e.g. "multi" -> results/suites_multi__<kind>__<label>.json


class Adapter:
    def __init__(self, spec: str, threads: int):
        kind, _, name = spec.partition(":")
        self.kind, self.name = kind, name
        if kind == "laya":
            import torch
            torch.set_num_threads(threads)
            import laya
            self.m = laya.load(name, device="cpu")
            self.label = name
        elif kind == "nirnay":
            import nirnay
            path = name if os.path.isabs(name) else os.path.join(ROOT, name)
            self.m = nirnay.load(path, threads=threads)
            self.label = os.path.basename(path.rstrip("\\/")) + os.environ.get("BENCH_LABEL_SUFFIX", "")
        else:
            raise SystemExit("model spec must be laya:<name> or nirnay:<dir>")

    def batch(self, states, questions):
        return self.m.predict_batch(states, questions)


def probs_of(ans, q):
    t = q["type"]
    if t == "noul":
        p1 = float(ans["noul"])
        return np.array([1 - p1, p1])
    if t == "choice":
        keys = list(q["criteria"].keys()) if isinstance(q["criteria"], dict) else list(q["criteria"])
        return np.array([float(ans["probabilities"][k]) for k in keys])
    return np.array([float(ans["probabilities"][str(i)]) for i in range(len(q["criteria"]))])


def suite_metrics(P, gold, qtype):
    P = np.clip(np.asarray(P, dtype=np.float64), 0, 1)
    P = P / P.sum(1, keepdims=True)
    gold = np.asarray(gold)
    pred = P.argmax(1)
    conf = P.max(1)
    correct = (pred == gold).astype(float)
    K = P.shape[1]
    f1s = []
    for c in range(K):
        tp = ((pred == c) & (gold == c)).sum()
        fp = ((pred == c) & (gold != c)).sum()
        fn = ((pred != c) & (gold == c)).sum()
        if tp + fn == 0:
            continue
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    edges = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    onehot = np.eye(K)[gold]
    m = {"n": int(len(gold)), "accuracy": round(float(correct.mean()), 4), "macro_f1": round(float(np.mean(f1s)), 4),
         "ece": round(float(ece), 4), "brier": round(float(((P - onehot) ** 2).sum(1).mean()), 4),
         "nll": round(float(-np.log(np.clip(P[np.arange(len(gold)), gold], 1e-12, 1)).mean()), 4)}
    if qtype == "score":
        m["mae"] = round(float(np.abs((P * np.arange(K)).sum(1) - gold).mean()), 4)
    return m


def permute_q(q, rng):
    q = json.loads(json.dumps(q))
    if q["type"] == "choice" and isinstance(q["criteria"], dict):
        items = list(q["criteria"].items())
        rng.shuffle(items)
        q["criteria"] = dict(items)
    return q


def main():
    spec = sys.argv[1]
    threads = int(os.environ.get("BENCH_THREADS", "8"))
    only = set(filter(None, os.environ.get("BENCH_SUITES", "").split(",")))
    cases = [json.loads(l) for l in open(SUITES, encoding="utf-8")]
    by_suite = defaultdict(list)
    for c in cases:
        if not only or c["suite"] in only:
            by_suite[c["suite"]].append(c)
    t = time.perf_counter()
    ad = Adapter(spec, threads)
    load_s = time.perf_counter() - t
    out = {"model": spec, "label": ad.label, "threads": threads, "load_s": round(load_s, 1),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "suites": {}}
    path = os.path.join(HERE, "results", "suites%s__%s__%s.json" % ("_" + OUT_TAG if OUT_TAG else "", ad.kind, ad.label))
    if os.environ.get("BENCH_MERGE") and os.path.exists(path):   # resume: keep suites already measured
        prev = json.load(open(path))
        out["suites"] = prev.get("suites", {})
        by_suite = {k: v for k, v in by_suite.items() if k not in out["suites"]}
        print("merge: keeping %d measured suites, running %d" % (len(out["suites"]), len(by_suite)), flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for name, cs in by_suite.items():
        qs = cs[0]["questions"]
        qid = next(iter(qs))
        q = qs[qid]
        assert all(c["questions"] == qs for c in cs), "suite questions must be identical across cases"
        t = time.perf_counter()
        res = ad.batch([c["state"] for c in cs], qs)
        secs = time.perf_counter() - t
        P = [probs_of(r["answers"][qid], q) for r in res]
        m = suite_metrics(P, [c["gold"] for c in cs], q["type"])
        m["ms_per_case"] = round(1000 * secs / len(cs), 2)
        m["meta"] = cs[0]["meta"]
        # option-order flip test (choice suites, first 200 cases, one random permutation per case)
        if q["type"] == "choice":
            rng = random.Random(13)
            sub = cs[:200]
            flips = 0
            for c, r0 in zip(sub, res[:200]):
                pq = {qid: permute_q(q, rng)}
                r1 = ad.batch([c["state"]], pq)[0]
                flips += int(r1["answers"][qid]["choice"] != r0["answers"][qid]["choice"])
            m["order_flip_rate"] = round(flips / len(sub), 4)
        out["suites"][name] = m
        print("  %-28s acc %.3f  f1 %.3f  ECE %.3f  brier %.3f %s %7.1f ms/case%s" % (
            name, m["accuracy"], m["macro_f1"], m["ece"], m["brier"],
            (" mae %.3f" % m["mae"]) if "mae" in m else "", m["ms_per_case"],
            ("  flip %.3f" % m["order_flip_rate"]) if "order_flip_rate" in m else ""), flush=True)
        json.dump(out, open(path, "w"), indent=2)
    # determinism: same inputs again -> identical answers (exact JSON equality)
    sample = [c for cs in by_suite.values() for c in cs[:25]]
    first = [json.dumps(ad.batch([c["state"]], c["questions"])[0]["answers"], sort_keys=True) for c in sample]
    second = [json.dumps(ad.batch([c["state"]], c["questions"])[0]["answers"], sort_keys=True) for c in sample]
    out["determinism_repeat_identical"] = round(sum(a == b for a, b in zip(first, second)) / len(sample), 4)
    out["determinism_n"] = len(sample)
    json.dump(out, open(path, "w"), indent=2)
    print("determinism (repeat): %.4f identical over %d calls" % (out["determinism_repeat_identical"], len(sample)))
    print("wrote", path)


if __name__ == "__main__":
    main()
