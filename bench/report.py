"""Merge benchmark results into REPORT.md (Nirnay v2 + v1 vs Laya measured here, Jev as published).

  python bench\\report.py
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
OUT = os.path.join(ROOT, os.environ.get("REPORT_OUT", "REPORT.md"))

# Published Jev figures (TypeSafe's closed API; not runnable here). Source: Laya repo
# research/scripts/bench_apps.py, which cites AbdelStark/jev-benchmarks v0.1.0 (n=100 each)
# and nibzard/decision-model-benchmark.
JEV = {"jev.ag_news": 0.910, "jev.banking77_full": 0.870, "jev.emotion": 0.480, "extra.sms_spam": 0.930}
JEV_NOTE = {"jev.banking77_full": "72 labels, n=100; independent run 0.763",
            "extra.sms_spam": "independent run, different sample"}
LAYA = [("laya", "laya (en)"), ("laya-multilingual", "laya-ml"), ("laya-typed-decisions", "laya-td")]
NIR = [("nirnay-v2", "Nirnay v2 (default)"), ("nirnay-v1", "Nirnay v1 (fast)")]
ORDER = ["jev.ag_news", "jev.emotion", "jev.banking77_full", "app.support_triage", "app.email_spam", "app.phishing",
         "app.guardrails_jailbreak", "app.moderation_toxicity", "app.rag_relevance", "app.model_routing_domain",
         "extra.sst5", "extra.sms_spam"]
TIE = 0.005


def jload(name):
    p = os.path.join(RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt(x, nd=3):
    return "—" if x is None else ("%.*f" % (nd, x))


def acc(d, s):
    return ((d or {}).get("suites", {}).get(s) or {}).get("accuracy")


def tally(mine, theirs):
    w = t = l = 0
    for a, b in zip(mine, theirs):
        if a is None or b is None:
            continue
        if a > b + TIE:
            w += 1
        elif a < b - TIE:
            l += 1
        else:
            t += 1
    return w, t, l


def main():
    lv = ((jload("laya_latency.json") or {}).get("meta") or {}).get("laya", "?")
    nir = {k: jload("suites__nirnay__%s.json" % k) for k, _ in NIR}
    lay = {k: jload("suites__laya__%s.json" % k) for k, _ in LAYA}
    L = ["# Nirnay vs Laya vs Jev — measured on this machine", "",
         "Generated %s. Hardware: HP Z840, 2x Xeon E5-2630 v3 (16 cores / 32 threads, AVX2, no AVX-512), "
         "128 GB DDR4, CPU only (Quadro K2200 unused)." % time.strftime("%Y-%m-%d %H:%M"), "",
         "Laya was installed from PyPI (`laya` %s) with its three published checkpoints and run through its own API "
         "with its shipped calibration. Jev is a closed, paid cloud API: it cannot run on this machine, so its column "
         "is the third-party published figure where one exists." % lv, "",
         "* **Nirnay v2** (default, `models\\nirnay-v2`): bge-base-en-v1.5, deep fusion (top 4 encoder layers trained), "
         "294k questions, 1.6 epochs (overnight, both CPU sockets).",
         "* **Nirnay v1** (fast, `models\\nirnay-v1`): bge-small-en-v1.5, same architecture, 221k questions, 1 epoch.", ""]

    # ------------------------------------------------------------- accuracy
    L += ["## Accuracy (same 400 cases per task, same questions, seed 13)", "",
          "| task | " + " | ".join(l for _, l in NIR) + " | " + " | ".join(l for _, l in LAYA) +
          " | best Laya | Jev (published) | Laya trained on it? |",
          "|---|" + "---|" * (len(NIR) + len(LAYA) + 3)]
    v2s, bests, ens = [], [], []
    for s in ORDER:
        ls = [acc(lay[k], s) for k, _ in LAYA]
        best = max([x for x in ls if x is not None], default=None)
        meta = ((nir["nirnay-v2"] or {}).get("suites", {}).get(s) or {}).get("meta") or {}
        v2s.append(acc(nir["nirnay-v2"], s))
        bests.append(best)
        ens.append(ls[0])
        jt = JEV.get(s)
        L.append("| %s | **%s** | %s | %s | %s | %s | %s |" % (
            s, fmt(v2s[-1]), fmt(acc(nir["nirnay-v1"], s)), " | ".join(fmt(x) for x in ls), fmt(best),
            (fmt(jt) + (" (%s)" % JEV_NOTE[s] if s in JEV_NOTE else "")) if jt else "—",
            "yes" if meta.get("laya_in_training") else "no (held out)"))
    w, t, l = tally(v2s, bests)
    w2, t2, l2 = tally(v2s, ens)
    v1s = [acc(nir["nirnay-v1"], s) for s in ORDER]
    w3, t3, l3 = tally(v2s, v1s)
    L += ["", "Nirnay v2 vs the best of Laya's three checkpoints (picked per task, after the fact): "
          "**%d wins, %d ties (±%.3f), %d losses**." % (w, t, TIE, l),
          "Nirnay v2 vs Laya's default English checkpoint: **%d wins, %d ties, %d losses**." % (w2, t2, l2),
          "Nirnay v2 vs Nirnay v1: %d better, %d same, %d worse." % (w3, t3, l3), ""]

    # ------------------------------------------------------------- calibration
    L += ["## Calibration (ECE, lower is better)", "",
          "| task | Nirnay v2 | Nirnay v1 | best Laya |", "|---|---|---|---|"]
    for s in ORDER:
        le = [((lay[k] or {}).get("suites", {}).get(s) or {}).get("ece") for k, _ in LAYA]
        le = [x for x in le if x is not None]
        L.append("| %s | %s | %s | %s |" % (s, fmt(((nir["nirnay-v2"] or {}).get("suites", {}).get(s) or {}).get("ece")),
                                         fmt(((nir["nirnay-v1"] or {}).get("suites", {}).get(s) or {}).get("ece")),
                                         fmt(min(le) if le else None)))
    L.append("")

    # ------------------------------------------------------------- robustness
    flip = lambda d: [v.get("order_flip_rate") for v in (d or {}).get("suites", {}).values() if v.get("order_flip_rate") is not None]
    avg = lambda xs: (sum(xs) / len(xs)) if xs else None
    cols = [l for _, l in NIR] + [l for _, l in LAYA]
    L += ["## Robustness and determinism", "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    L.append("| option-order flip rate (choice tasks, mean) | %s | %s |" % (
        " | ".join(fmt(avg(flip(nir[k]))) for k, _ in NIR), " | ".join(fmt(avg(flip(lay[k]))) for k, _ in LAYA)))
    L.append("| identical answers on repeat call | %s | %s |" % (
        " | ".join(fmt((nir[k] or {}).get("determinism_repeat_identical")) for k, _ in NIR),
        " | ".join(fmt((lay[k] or {}).get("determinism_repeat_identical")) for k, _ in LAYA)))
    dets = [jload("nirnay_determinism_v2.json"), jload("nirnay_determinism_v1.json")]
    L.append("| byte-identical answers across 1/4/8/16 threads | %s | not measured | not measured | not measured |" % (
        " | ".join(fmt((d or {}).get("identical_rate")) for d in dets)))
    L.append("| decisions changed across 1/4/8/16 threads | %s | | | |" % (
        " | ".join("%s (max prob. diff %s)" % ((d or {}).get("answer_changes"), (d or {}).get("max_probability_diff")) for d in dets)))
    cs = jload("cold_start.json") or {}
    L.append("| cold start: new process -> first answer (s) | %s | %s |" % (
        " | ".join(fmt((cs.get("nirnay:" + k) or {}).get("first_answer_s"), 1) for k, _ in NIR),
        " | ".join(fmt((cs.get("laya:" + k) or {}).get("first_answer_s"), 1) for k, _ in LAYA)))
    L.append("")

    # ------------------------------------------------------------- latency
    ll, lo = jload("laya_latency.json"), jload("laya_latency_onnx.json")
    if ll and lo:   # Laya's best of eager torch (8 / 16 threads) and its own ONNX Runtime backend
        for mname, ent in lo.get("models", {}).items():
            ll["models"].setdefault(mname, {"by_threads": {}})["by_threads"]["onnx"] = ent["by_threads"].get("0", {})
    nl = {"nirnay-v2": jload("nirnay_latency_v2.json"), "nirnay-v1": jload("nirnay_latency_v1.json")}
    L += ["## Latency on this CPU (Laya's own workload: a support ticket, mixed choice / yes-no questions)", "",
          "p50 ms, warm, best setting for each model (Laya: eager torch at 8/16 threads or its ONNX Runtime backend).", "",
          "| questions per call | Nirnay v2 | Nirnay v1 | " + " | ".join(l for _, l in LAYA) +
          " | v2 speed-up vs fastest Laya | v1 speed-up |", "|---|---|---|" + "---|" * len(LAYA) + "---|---|"]
    for n in (1, 5, 10, 50):
        key = "%d_questions" % n

        def best_l(m):
            bt = ((ll or {}).get("models", {}).get(m) or {}).get("by_threads", {})
            xs = [v[key]["p50_ms"] for v in bt.values() if key in v]
            return min(xs) if xs else None

        def best_n(d):
            bt = (d or {}).get("by_threads", {})
            xs = [v[key]["warm"]["p50_ms"] for v in bt.values() if key in v]
            return min(xs) if xs else None
        ls = [best_l(m) for m, _ in LAYA]
        fastest = min([x for x in ls if x], default=None)
        a, b = best_n(nl["nirnay-v2"]), best_n(nl["nirnay-v1"])
        L.append("| %d | **%s** | %s | %s | %s | %s |" % (
            n, fmt(a, 1), fmt(b, 1), " | ".join(fmt(x, 1) for x in ls),
            ("%.1fx" % (fastest / a)) if (fastest and a) else "—", ("%.1fx" % (fastest / b)) if (fastest and b) else "—"))
    L.append("")
    L.append("Batch throughput (10 questions per state): Nirnay v2 %s decisions/s, Nirnay v1 %s decisions/s." % (
        (nl["nirnay-v2"] or {}).get("throughput_10q", {}).get("decisions_per_s"),
        (nl["nirnay-v1"] or {}).get("throughput_10q", {}).get("decisions_per_s")))
    L.append("")
    disk = lambda name: round(sum(os.path.getsize(os.path.join(ROOT, "models", name, f))
                                  for f in os.listdir(os.path.join(ROOT, "models", name))) / 2**20) \
        if os.path.isdir(os.path.join(ROOT, "models", name)) else None
    L += ["## Footprint", "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols),
          "| weights on disk (MB) | %s | 804 | 614 | 804 |" % " | ".join(str(disk(k)) for k, _ in NIR),
          "| RAM added by loading (MB) | %s | %s |" % (
              " | ".join(fmt((nl[k] or {}).get("rss_delta_mb"), 0) for k, _ in NIR),
              " | ".join(fmt(((ll or {}).get("models", {}).get(m) or {}).get("rss_delta_mb"), 0) for m, _ in LAYA)), ""]

    # ------------------------------------------------------------- few-shot
    L += ["## Few-shot adaptation (`learn()`, no training, CPU)", "",
          "Labelled examples per label from the TRAIN split; evaluated on the same 400 TEST cases. Laya cannot do this "
          "without RL fine-tuning on a GPU; its zero-shot score is the reference.", "",
          "| model / task | k = 0 | 2 | 4 | 8 | 16 | learn() time at 16 | Laya (best ckpt) | Jev (published) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for mk, lbl, fname in (("nirnay-v2", "v2", "fewshot_v2.json"), ("nirnay-v1", "v1", "fewshot_v1.json")):
        fs = jload(fname) or {}
        for task, suite in (("banking77", "jev.banking77_full"), ("emotion", "jev.emotion")):
            r = fs.get(task) or {}
            ls = [acc(lay[k], suite) for k, _ in LAYA]
            L.append("| %s %s | %s | %s s | %s | %s |" % (
                lbl, task, " | ".join(fmt((r.get(str(k)) or {}).get("accuracy")) for k in (0, 2, 4, 8, 16)),
                fmt((r.get("16") or {}).get("learn_seconds"), 0), fmt(max([x for x in ls if x is not None], default=None)),
                fmt(JEV.get(suite))))
    L.append("")

    # ------------------------------------------------------------- ablations
    ab = [("nirnay-bges-b2", "v1-small: frozen encoder + 2 fusion blocks"),
          ("nirnay-v1", "v1: deep fusion (bge-small)"),
          ("nirnay-bges-deep-debias", "v1 + opt-in debias"),
          ("nirnay-v2", "v2: deep fusion (bge-base, more data)")]
    abd = [(lbl, jload("suites__nirnay__%s.json" % n)) for n, lbl in ab]
    L += ["## Ablations (same suites)", "", "| task | " + " | ".join(l for l, _ in abd) + " |", "|---|" + "---|" * len(abd)]
    for s in ORDER:
        L.append("| %s | %s |" % (s, " | ".join(fmt(acc(d, s)) for _, d in abd)))
    L.append("")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
