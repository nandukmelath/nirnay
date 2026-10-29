"""Option-order check: one benchmark suite with its options in several fixed orders.

  python bench\\order_check.py laya:laya-typed-decisions 0,1,2      (seed 0 = the suite's own order)
  python bench\\order_check.py nirnay:models\\nirnay-v2 0,1,2,3,4,5
  python bench\\order_check.py --summary

Seed s > 0 shuffles the options with random.Random(s); the same order is used for every case. Each run
writes bench/results/order_check/<suite>__<label>__s<seed>.json with the probabilities in the suite's own
option order, so --summary can compare the orders case by case: accuracy per order, how many answers
change, the accuracy of the probabilities averaged over all orders, and the cases right in any / every order.
"""
import glob
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_suites import Adapter, probs_of  # noqa: E402

SUITE = os.environ.get("ORDER_SUITE", "jev.banking77_full")
OUT = os.path.join(HERE, "results", "order_check")


def run(spec, seeds, threads):
    cases = [json.loads(l) for l in open(os.path.join(HERE, "data", "suites.jsonl"), encoding="utf-8")]
    cs = [c for c in cases if c["suite"] == SUITE]
    qs = cs[0]["questions"]
    qid = next(iter(qs))
    q = qs[qid]
    assert q["type"] == "choice", "order check needs a choice suite"
    ad = Adapter(spec, threads)
    os.makedirs(OUT, exist_ok=True)
    for seed in seeds:
        items = list(q["criteria"].items()) if isinstance(q["criteria"], dict) else list(q["criteria"])
        if seed:
            random.Random(seed).shuffle(items)
        crit = dict(items) if isinstance(q["criteria"], dict) else items
        t = time.perf_counter()
        res = ad.batch([c["state"] for c in cs], {qid: dict(q, criteria=crit)})
        secs = time.perf_counter() - t
        P = [probs_of(r["answers"][qid], q).tolist() for r in res]   # columns in the suite's own order
        gold = [c["gold"] for c in cs]
        acc = float(np.mean([int(np.argmax(p)) == g for p, g in zip(P, gold)]))
        order = [k for k, _ in items] if isinstance(q["criteria"], dict) else items
        path = os.path.join(OUT, "%s__%s__s%d.json" % (SUITE, ad.label, seed))
        json.dump({"suite": SUITE, "model": spec, "label": ad.label, "seed": seed, "order": order, "gold": gold,
                   "P": P, "accuracy": acc, "secs": round(secs, 1)}, open(path, "w"))
        print("%s seed %d: acc %.4f (%.0f s, %.0f ms/case)" % (ad.label, seed, acc, secs, 1000 * secs / len(cs)),
              flush=True)


def summary():
    runs = {}
    for path in sorted(glob.glob(os.path.join(OUT, "%s__*__s*.json" % SUITE))):
        d = json.load(open(path))
        runs.setdefault(d["label"], {})[d["seed"]] = d
    out = {"suite": SUITE, "models": {}}
    for label, by_seed in runs.items():
        seeds = sorted(by_seed)
        gold = np.array(by_seed[seeds[0]]["gold"])
        P = np.stack([np.array(by_seed[s]["P"]) for s in seeds])          # orders x cases x options
        P = np.clip(P, 0, None)
        P = P / P.sum(2, keepdims=True)
        pred = P.argmax(2)                                                  # orders x cases
        right = pred == gold[None, :]
        accs = right.mean(1)
        base = seeds.index(0) if 0 in seeds else 0
        flips = [float((pred[i] != pred[base]).mean()) for i in range(len(seeds)) if i != base]
        m = {"orders": len(seeds), "seeds": seeds, "n": int(len(gold)),
             "accuracy_by_order": {str(s): round(float(a), 4) for s, a in zip(seeds, accs)},
             "accuracy_mean": round(float(accs.mean()), 4), "accuracy_min": round(float(accs.min()), 4),
             "accuracy_max": round(float(accs.max()), 4),
             "answer_changed_vs_own_order": round(float(np.mean(flips)), 4) if flips else None,
             "same_answer_in_every_order": round(float((pred == pred[:1]).all(0).mean()), 4),
             "accuracy_probs_averaged_over_orders": round(float((P.mean(0).argmax(1) == gold).mean()), 4),
             "right_in_any_order": round(float(right.any(0).mean()), 4),
             "right_in_every_order": round(float(right.all(0).mean()), 4)}
        out["models"][label] = m
        print("%-24s orders %d  acc %.3f (min %.3f max %.3f)  changed %s  stable %.3f  prob-avg %.3f  any %.3f  every %.3f"
              % (label, m["orders"], m["accuracy_mean"], m["accuracy_min"], m["accuracy_max"],
                 m["answer_changed_vs_own_order"], m["same_answer_in_every_order"],
                 m["accuracy_probs_averaged_over_orders"], m["right_in_any_order"], m["right_in_every_order"]))
    path = os.path.join(HERE, "results", "order_check_%s.json" % SUITE)
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary()
    else:
        run(sys.argv[1], [int(s) for s in sys.argv[2].split(",")], int(os.environ.get("BENCH_THREADS", "8")))
