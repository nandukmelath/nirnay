"""Laya CPU latency on this machine, using the workload from Laya's own
research/scripts/bench_latency.py (same state, same questions, same 1/5/10/50 counts).

Each checkpoint is timed at several torch intra-op thread counts so Laya gets its
best setting here. Writes bench/results/laya_latency.json.

  python bench\\bench_latency_laya.py
"""
import gc
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

import laya  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "laya_latency.json")

# verbatim from laya research/scripts/bench_latency.py
STATE_EN = {"ticket": {"subject": "Payout failing", "messages": [{"from": "customer",
            "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"}}


def qs(n):
    return {("q%d" % i): (Q_NOUL if i % 2 else Q_CHOICE) for i in range(n)}


def timed(fn, warmup=2, reps=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 2),
            "p95_ms": round(float(np.percentile(ts, 95)), 2),
            "mean_ms": round(float(np.mean(ts)), 2),
            "min_ms": round(float(np.min(ts)), 2)}


def main():
    threads = [int(x) for x in os.environ.get("BENCH_THREADS", "8,16").split(",")]
    models = os.environ.get("BENCH_MODELS", "laya,laya-multilingual,laya-typed-decisions").split(",")
    onnx_dir = os.environ.get("LAYA_ONNX_DIR")   # Laya's own ONNX export (laya-dotnet/tools/export_onnx.py)
    if onnx_dir:
        global OUT
        OUT = os.path.join(HERE, "results", "laya_latency_onnx.json")
        threads = [0]                              # ONNXAgent uses ONNX Runtime's default thread pool
    proc = psutil.Process()
    res = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "cpu",
                    "cpu": "2x Xeon E5-2630 v3 (HP Z840)", "torch": torch.__version__,
                    "laya": laya.__version__, "threads_tried": threads}, "models": {}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for mname in models:
        torch.set_num_threads(max(max(threads), 1))
        rss0 = proc.memory_info().rss
        t = time.perf_counter()
        if onnx_dir:
            short = {"laya": "english", "laya-multilingual": "multilingual", "laya-typed-decisions": "typed-decisions"}[mname]
            ag = laya.load(mname, backend="onnx", onnx_path=os.path.join(onnx_dir, short, "model.onnx"))
        else:
            ag = laya.load(mname, device="cpu")
        load_ms = round((time.perf_counter() - t) * 1000, 1)
        rss1 = proc.memory_info().rss
        entry = {"cold_load_ms": load_ms, "rss_after_load_mb": round(rss1 / 2**20, 1),
                 "rss_delta_mb": round((rss1 - rss0) / 2**20, 1), "by_threads": {}}
        print("%s loaded in %.0f ms (+%.0f MB RSS)" % (mname, load_ms, entry["rss_delta_mb"]), flush=True)
        for th in threads:
            if th:
                torch.set_num_threads(th)
            per = {}
            for n in (1, 5, 10, 50):
                q = qs(n)
                r = timed(lambda: ag.system_one(STATE_EN, q), warmup=2, reps=10 if n < 50 else 5)
                r["ms_per_question"] = round(r["p50_ms"] / n, 2)
                per["%d_questions" % n] = r
                print("   %-22s threads=%2d %2dq  p50 %8.1f ms  p95 %8.1f ms  %7.2f ms/q"
                      % (mname, th, n, r["p50_ms"], r["p95_ms"], r["ms_per_question"]), flush=True)
            entry["by_threads"][str(th)] = per
        entry["peak_rss_mb"] = round(proc.memory_info().rss / 2**20, 1)
        res["models"][mname] = entry
        json.dump(res, open(OUT, "w"), indent=2)
        del ag
        gc.collect()
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
