"""Nirnay CPU latency on this machine with the exact Laya latency workload
(see bench_latency_laya.py). Writes bench/results/nirnay_latency.json.

Reports both regimes honestly:
  cold_schema  question/option encodings NOT cached (first call with a new schema)
  warm         schema cached (every later call with the same questions; the normal case)

  python bench\\bench_latency_nirnay.py [model_dir]
"""
import json
import os
import sys
import time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from bench_latency_laya import STATE_EN, qs  # noqa: E402  (identical workload)

import nirnay  # noqa: E402

OUT = os.path.join(HERE, "results", "nirnay_latency%s.json" % (("_" + os.environ["BENCH_TAG"]) if os.environ.get("BENCH_TAG") else ""))


def timed(fn, warmup=2, reps=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2),
            "mean_ms": round(float(np.mean(ts)), 2), "min_ms": round(float(np.min(ts)), 2)}


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "nirnay-v1")
    threads = [int(x) for x in os.environ.get("BENCH_THREADS", "1,4,8,16").split(",")]
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t = time.perf_counter()
    m = nirnay.load(model_dir, threads=max(threads))
    load_ms = round((time.perf_counter() - t) * 1000, 1)
    res = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "cpu": "2x Xeon E5-2630 v3 (HP Z840)",
                    "model": m.name, "model_dir": model_dir},
           "cold_load_ms": load_ms, "rss_delta_mb": round((proc.memory_info().rss - rss0) / 2**20, 1),
           "disk_mb": round(sum(os.path.getsize(os.path.join(model_dir, f)) for f in os.listdir(model_dir)) / 2**20, 1),
           "by_threads": {}}
    print("%s loaded in %.0f ms (+%.0f MB RSS, %.0f MB on disk, backend %s)" % (
        m.name, load_ms, res["rss_delta_mb"], res["disk_mb"], m.backend))
    res["backend"] = m.backend
    import torch
    for th in threads:
        m = nirnay.load(model_dir, threads=th)   # ONNX Runtime fixes its thread pool per session
        torch.set_num_threads(th)
        per = {}
        for n in (1, 5, 10, 50):
            q = qs(n)

            def cold():
                m.cache.d.clear()
                m.predict(STATE_EN, q)

            r_cold = timed(cold)
            r_warm = timed(lambda: m.predict(STATE_EN, q))
            per["%d_questions" % n] = {"warm": r_warm, "cold_schema": r_cold,
                                       "ms_per_question_warm": round(r_warm["p50_ms"] / n, 3)}
            print("   threads=%2d %2dq  warm p50 %7.2f ms (p95 %7.2f)  cold-schema p50 %7.2f ms" %
                  (th, n, r_warm["p50_ms"], r_warm["p95_ms"], r_cold["p50_ms"]), flush=True)
        res["by_threads"][str(th)] = per
    # throughput: many states, same questions (predict_batch)
    m = nirnay.load(model_dir, threads=8)
    torch.set_num_threads(8)
    states = [{"ticket": {"subject": "Case %d" % i, "messages": [{"from": "customer", "text": STATE_EN["ticket"]["messages"][0]["text"][: 80 + 3 * i]}]}}
              for i in range(64)]
    q10 = qs(10)
    m.predict_batch(states[:4], q10)
    t = time.perf_counter()
    m.predict_batch(states, q10)
    el = time.perf_counter() - t
    res["throughput_10q"] = {"states": len(states), "seconds": round(el, 3),
                             "decisions_per_s": round(len(states) * 10 / el, 1),
                             "states_per_s": round(len(states) / el, 1)}
    print("   throughput: %.0f decisions/s (%.1f states/s, 10 questions each)" %
          (res["throughput_10q"]["decisions_per_s"], res["throughput_10q"]["states_per_s"]))
    res["peak_rss_mb"] = round(proc.memory_info().rss / 2**20, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
