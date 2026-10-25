"""Few-shot adaptation without training: Nirnay.learn() with k labelled examples per label.

Banking77 (77 intents, held out of Nirnay's training): examples come from the TRAIN split, the
evaluation is the same 400 TEST cases as the benchmark suite. Also run on DAIR emotion.
Reports accuracy and the wall time `learn()` took (encoding the examples included).

  python bench\\fewshot.py [model_dir]
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import nirnay  # noqa: E402
from datasets import load_dataset  # noqa: E402


def run_task(m, suite, train_rows, label_of, ks=(0, 2, 4, 8, 16)):
    cases = [json.loads(l) for l in open(os.path.join(HERE, "data", "suites.jsonl"), encoding="utf-8")]
    cases = [c for c in cases if c["suite"] == suite]
    qid = next(iter(cases[0]["questions"]))
    q = cases[0]["questions"][qid]
    keys = list(q["criteria"].keys())
    by_label = defaultdict(list)
    for r in train_rows:
        by_label[label_of(r)].append(r)
    out = {}
    rng = random.Random(13)
    for k in ks:
        m.memory.forget()
        t_learn = 0.0
        info = {}
        if k:
            ex = []
            for lab in keys:
                pool = by_label.get(lab, [])
                for r in rng.sample(pool, min(k, len(pool))):
                    ex.append((r["state"], lab))
            t = time.perf_counter()
            info = m.learn(qid, q, ex)
            t_learn = time.perf_counter() - t
        res = m.predict_batch([c["state"] for c in cases], {qid: q})
        acc = sum(int(r["answers"][qid]["choice"] == keys[c["gold"]]) for r, c in zip(res, cases)) / len(cases)
        out[str(k)] = {"accuracy": round(acc, 4), "learn_seconds": round(t_learn, 2), **info}
        print("  %-20s k=%2d  acc %.3f  learn %.1fs  %s" % (suite, k, acc, t_learn, info), flush=True)
    m.memory.forget()
    return out


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "nirnay-v1")
    m = nirnay.load(model_dir)
    res = {"model": m.name}
    d = load_dataset("mteb/banking77", split="train")
    rows = [{"state": {"message": r["text"]}, "label": r["label_text"].replace("_", " ")} for r in d]
    res["banking77"] = run_task(m, "jev.banking77_full", rows, lambda r: r["label"])
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    d = load_dataset("dair-ai/emotion", "split", split="train")
    rows = [{"state": {"text": r["text"]}, "label": names[int(r["label"])]} for r in d]
    res["emotion"] = run_task(m, "jev.emotion", rows, lambda r: r["label"])
    out = os.environ.get("FEWSHOT_OUT", "fewshot.json")
    json.dump(res, open(os.path.join(HERE, "results", out), "w"), indent=2)
    print("wrote bench/results/" + out)


if __name__ == "__main__":
    main()
