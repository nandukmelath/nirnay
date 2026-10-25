"""Cold start: fresh process -> import -> load weights (already on disk) -> first answer.
Each model is measured 3 times in a new Python process; median reported.

  python bench\\cold_start.py
"""
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

SNIPPET = r'''
import os, time, json
t0 = time.perf_counter()
os.environ.setdefault("USE_TF", "0")
KIND, NAME = %r, %r
st = {"ticket": "My card was charged twice, I want a refund"}
q = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "technical": None}}}
if KIND == "laya":
    import torch; torch.set_num_threads(8)
    import laya
    m = laya.load(NAME, device="cpu")
    t1 = time.perf_counter()
    m.system_one(st, q)
else:
    import nirnay
    m = nirnay.load(NAME, threads=8)
    t1 = time.perf_counter()
    m.predict(st, q)
t2 = time.perf_counter()
print(json.dumps({"load_s": t1 - t0, "first_answer_s": t2 - t0}))
'''


def run(kind, name):
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=ROOT)
    out = subprocess.run([PY, "-c", SNIPPET % (kind, name)], capture_output=True, text=True, env=env, cwd=ROOT)
    line = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not line:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(line[-1])


def main():
    models = [("nirnay", os.path.join(ROOT, "models", n)) for n in os.environ.get("COLD_NIRNAY", "nirnay-v1").split(",")]
    models += [("laya", "laya"), ("laya", "laya-multilingual"), ("laya", "laya-typed-decisions")]
    only = os.environ.get("COLD_ONLY")
    out_path = os.path.join(HERE, "results", "cold_start.json")
    res = json.load(open(out_path)) if (only and os.path.exists(out_path)) else {}
    if only:
        models = [m for m in models if m[0] == only]
    for kind, name in models:
        runs = [run(kind, name) for _ in range(3)]
        key = "%s:%s" % (kind, os.path.basename(name))
        res[key] = {"first_answer_s": round(statistics.median(r["first_answer_s"] for r in runs), 2),
                    "load_s": round(statistics.median(r["load_s"] for r in runs), 2)}
        print(key, res[key], flush=True)
    json.dump(res, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
