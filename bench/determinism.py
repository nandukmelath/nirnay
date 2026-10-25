"""Determinism across thread counts: the same cases answered with 1 / 4 / 8 / 16 threads must give
byte-identical answers (probabilities are reported to 4 decimals, as Laya does).

  python bench\\determinism.py nirnay:models\\nirnay-v1
  python bench\\determinism.py laya:laya-multilingual        (slow; uses 60 cases)
"""
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from eval_suites import Adapter  # noqa: E402


def main():
    spec = sys.argv[1]
    threads = [int(x) for x in os.environ.get("DET_THREADS", "1,4,8,16").split(",")]
    per_suite = int(os.environ.get("DET_PER_SUITE", "25" if spec.startswith("nirnay") else "5"))
    cases = [json.loads(l) for l in open(os.path.join(HERE, "data", "suites.jsonl"), encoding="utf-8")]
    by = defaultdict(list)
    for c in cases:
        by[c["suite"]].append(c)
    sample = [c for cs in by.values() for c in cs[:per_suite]]
    import torch
    runs = {}
    for th in threads:
        ad = Adapter(spec, th)            # reload: ONNX Runtime fixes its thread pool at session creation
        torch.set_num_threads(th)
        runs[th] = [json.dumps(ad.batch([c["state"]], c["questions"])[0]["answers"], sort_keys=True) for c in sample]
        print("threads %2d done" % th, flush=True)
        del ad
    base = runs[threads[0]]
    same = sum(all(runs[t][i] == base[i] for t in threads) for i in range(len(sample)))
    # how big are the differences that do occur? (decision changes vs last-digit probability changes)
    answer_changes, max_prob_diff = 0, 0.0
    for i in range(len(sample)):
        a0 = json.loads(base[i])
        for t in threads[1:]:
            a1 = json.loads(runs[t][i])
            for qid in a0:
                x, y = a0[qid], a1[qid]
                if x.get("choice") != y.get("choice") or ((x.get("noul", 0) >= 0.5) != (y.get("noul", 0) >= 0.5)):
                    answer_changes += 1
                px = x.get("probabilities") or {"p": x.get("noul")}
                py = y.get("probabilities") or {"p": y.get("noul")}
                max_prob_diff = max(max_prob_diff, max(abs(px[k] - py[k]) for k in px))
    res = {"model": spec, "threads": threads, "cases": len(sample), "identical_rate": round(same / len(sample), 4),
           "answer_changes": answer_changes, "max_probability_diff": round(max_prob_diff, 6)}
    name = os.environ.get("DET_OUT") or ("nirnay_determinism.json" if spec.startswith("nirnay")
                                         else "laya_determinism__%s.json" % spec.split(":")[1])
    json.dump(res, open(os.path.join(HERE, "results", name), "w"), indent=2)
    print(res)


if __name__ == "__main__":
    main()
