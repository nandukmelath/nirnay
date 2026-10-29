"""Merge benchmark results into REPORT.md (Nirnay models vs Laya measured here, Jev as published).

  python bench\\report.py
  REPORT_NIR="nirnay-v3,nirnay-v2,nirnay-v1" python bench\\report.py      (first model = the headline column)

Models without a results file are left out. The multilingual section appears once
results/suites_multi__*.json exist (MASSIVE, 51 languages; SIB-200, 205 languages).
"""
import json
import os
import statistics
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
MODELS = {   # key -> (column label, result-file tag, description)
    "nirnay-v3": ("Nirnay v3", "v3", "bge-base-en-v1.5, deep fusion; 324k questions: v2's mix plus request routing, short-message "
                                     "spam, request safety (NVIDIA Aegis 2.0), urgency and more phrasings of every question "
                                     "template; warm-started from v2 for 1.25 more epochs on both CPU sockets."),
    "nirnay-multilingual": ("Nirnay multilingual", "ml", "multilingual-e5-base (100 languages), deep fusion; 385k questions: "
                                                         "v3.1's mix plus MTOP intents (6 languages), PAWS-X, NLI in 26 "
                                                         "languages, toxicity in 14, language identification, multilingual "
                                                         "reviews and tweets; ~1.5 epochs on both CPU sockets (a partial "
                                                         "run, then 1.0 epoch warm-started from it)."),
    "nirnay-v31": ("Nirnay v3.1", "v31", "v3 with content-only safety wording (v3's AI-assistant / policy phrasings made it "
                                        "call harmful prompts jailbreaks); same recipe: warm-started from v2, 1.25 epochs."),
    "nirnay-v31-gpu": ("Nirnay v3.1-GPU", "v31gpu", "v3.1's data and architecture, warm-started from v2 for 2 epochs on a "
                                                    "free Kaggle T4 GPU (train/train_gpu.py; frozen lower layers computed live)."),
    "nirnay-v2": ("Nirnay v2", "v2", "bge-base-en-v1.5, deep fusion (top 4 encoder layers trained), 294k questions, 1.6 epochs "
                                     "(overnight, both CPU sockets)."),
    "nirnay-v1": ("Nirnay v1 (fast)", "v1", "bge-small-en-v1.5, same architecture, 221k questions, 1 epoch."),
}
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


def label_coverage():
    """Choice suites whose cases cover fewer labels than every question offers: (suite, labels in gold, offered)."""
    path = os.path.join(HERE, "data", "suites.jsonl")
    if not os.path.exists(path):
        return []
    gold, offered, n = {}, {}, {}
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        q = next(iter(c["questions"].values()))
        if q["type"] == "choice":
            gold.setdefault(c["suite"], set()).add(c["gold"])
            offered[c["suite"]] = len(q["criteria"])
            n[c["suite"]] = n.get(c["suite"], 0) + 1
    return [(s, n[s], len(g), offered[s]) for s, g in gold.items() if len(g) < offered[s]]


def order_section():
    d = jload("order_check_jev.banking77_full.json")
    if not d:
        return []
    lab = dict(LAYA)
    lab.update({k: v[0] for k, v in MODELS.items()})
    L = ["## Option-order check (Banking77, the options in several fixed orders)", "",
         "The same 400 cases with the 77 options in the suite's own (alphabetical) order and in 5 random orders "
         "(`random.Random(seed)`, seeds 1-5; one order for all cases), through each model's public batch API "
         "(`bench/order_check.py`; per-order probabilities in `bench/results/order_check/`). These cases are Laya's "
         "benchmark rows, the first 400 test messages, so they cover 10 of the 77 intents (40 messages each).", "",
         "| model | orders | accuracy by order | mean | same answer in every order | probabilities averaged over "
         "the orders | right in at least one order | right in every order |", "|---|---|---|---|---|---|---|---|"]
    for k in sorted(d["models"], key=lambda k: (not k.startswith("nirnay"), k)):
        m = d["models"][k]
        if m["accuracy_min"] == m["accuracy_max"]:
            by = "%s in every order" % fmt(m["accuracy_min"])
        else:
            by = "%s to %s (own order %s)" % (fmt(m["accuracy_min"]), fmt(m["accuracy_max"]),
                                             fmt(m["accuracy_by_order"].get("0")))
        L.append("| %s | %d | %s | %s | %.1f%% | %s | %s | %s |" % (
            lab.get(k, k), m["orders"], by, fmt(m["accuracy_mean"]), 100 * m["same_answer_in_every_order"],
            fmt(m["accuracy_probs_averaged_over_orders"]), fmt(m["right_in_any_order"]), fmt(m["right_in_every_order"])))
    L.append("")
    return L


def multilingual_section(keys):
    """MASSIVE + SIB-200 per model; per-language MASSIVE table; per-language wins vs Laya-multilingual."""
    res = {k: jload("suites_multi__nirnay__%s.json" % k) for k in keys}
    res = {k: v for k, v in res.items() if v}
    laya = jload("suites_multi__laya__laya-multilingual.json")
    if not res:   # nothing of ours measured yet: no section (a Laya-only table would compare nothing)
        return []
    cols = [(k, MODELS[k][0]) for k in res] + ([("laya-ml", "Laya multilingual")] if laya else [])
    data = dict(res, **({"laya-ml": laya} if laya else {}))

    def per_lang(d, prefix):
        return {s.split(".", 1)[1]: m["accuracy"] for s, m in (d or {}).get("suites", {}).items() if s.startswith(prefix + ".")}

    L = ["## Multilingual (held out of training for every model)", "",
         "* **MASSIVE** intent detection (mteb/amazon_massive_intent, test split): 51 languages, all 60 intents as options, "
         "80 requests per language, the question in English.",
         "* **SIB-200** topic classification (Davlan/sib200, test split): 205 languages and scripts, 7 topics, the same 24 "
         "FLORES sentences in every language.", "",
         "| benchmark | " + " | ".join(l for _, l in cols) + " |", "|---|" + "---|" * len(cols)]
    for prefix, name in (("massive", "MASSIVE, mean of 51 languages"), ("sib200", "SIB-200, mean of 205 languages")):
        row = []
        for k, _ in cols:
            xs = list(per_lang(data[k], prefix).values())
            row.append(fmt(statistics.mean(xs)) if xs else "—")
        L.append("| %s | %s |" % (name, " | ".join(row)))
    for prefix, name in (("massive", "MASSIVE, worst language"), ("sib200", "SIB-200, worst language")):
        row = []
        for k, _ in cols:
            pl = per_lang(data[k], prefix)
            row.append(("%s (%s)" % (fmt(min(pl.values())), min(pl, key=pl.get))) if pl else "—")
        L.append("| %s | %s |" % (name, " | ".join(row)))
    L.append("")
    if laya and "nirnay-multilingual" in res:
        for prefix, name in (("massive", "MASSIVE"), ("sib200", "SIB-200")):
            a, b = per_lang(res["nirnay-multilingual"], prefix), per_lang(laya, prefix)
            common = sorted(set(a) & set(b))
            w, t, l = tally([a[x] for x in common], [b[x] for x in common])
            L.append("%s, per language, Nirnay multilingual vs Laya multilingual: **%d wins, %d ties, %d losses** "
                     "(of %d languages)." % (name, w, t, l, len(common)))
        L.append("")
    langs = sorted(set().union(*[per_lang(data[k], "massive") for k, _ in cols]))
    if langs:
        L += ["### MASSIVE by language", "", "| language | " + " | ".join(l for _, l in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for lg in langs:
            L.append("| %s | %s |" % (lg, " | ".join(fmt(per_lang(data[k], "massive").get(lg)) for k, _ in cols)))
        L.append("")
    return L


def main():
    want = [k.strip() for k in os.environ.get("REPORT_NIR", "nirnay-v31,nirnay-v3,nirnay-multilingual,nirnay-v2,nirnay-v1").split(",")
            if k.strip()]
    nir = {k: jload("suites__nirnay__%s.json" % k) for k in want}
    NIR = [(k, MODELS[k][0]) for k in want if nir.get(k)]
    head = NIR[0][0]
    lv = ((jload("laya_latency.json") or {}).get("meta") or {}).get("laya", "?")
    lay = {k: jload("suites__laya__%s.json" % k) for k, _ in LAYA}
    L = ["# Nirnay vs Laya vs Jev — measured on this machine", "",
         "Generated %s. Hardware: HP Z840, 2x Xeon E5-2630 v3 (16 cores / 32 threads, AVX2, no AVX-512), "
         "128 GB DDR4, CPU only (Quadro K2200 unused)." % time.strftime("%Y-%m-%d %H:%M"), "",
         "Laya was installed from PyPI (`laya` %s) with its three published checkpoints and run through its own API "
         "with its shipped calibration. Jev is a closed, paid cloud API: it cannot run on this machine, so its column "
         "is the third-party published figure where one exists." % lv, ""]
    L += ["* **%s** (`models\\%s`): %s" % (MODELS[k][0], k, MODELS[k][2]) for k, _ in NIR] + [""]

    # ------------------------------------------------------------- accuracy
    L += ["## Accuracy (same 400 cases per task, same questions, seed 13)", "",
          "| task | " + " | ".join(l for _, l in NIR) + " | " + " | ".join(l for _, l in LAYA) +
          " | best Laya | Jev (published) | Laya trained on it? |",
          "|---|" + "---|" * (len(NIR) + len(LAYA) + 3)]
    heads, bests, ens = [], [], []
    for s in ORDER:
        ls = [acc(lay[k], s) for k, _ in LAYA]
        best = max([x for x in ls if x is not None], default=None)
        meta = ((nir[head] or {}).get("suites", {}).get(s) or {}).get("meta") or {}
        heads.append(acc(nir[head], s))
        bests.append(best)
        ens.append(ls[0])
        jt = JEV.get(s)
        cells = ["**%s**" % fmt(acc(nir[k], s)) if k == head else fmt(acc(nir[k], s)) for k, _ in NIR]
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            s, " | ".join(cells), " | ".join(fmt(x) for x in ls), fmt(best),
            (fmt(jt) + (" (%s)" % JEV_NOTE[s] if s in JEV_NOTE else "")) if jt else "—",
            "yes" if meta.get("laya_in_training") else "no (held out)"))
    hl = MODELS[head][0]
    w, t, l = tally(heads, bests)
    w2, t2, l2 = tally(heads, ens)
    L += ["", "%s vs the best of Laya's three checkpoints (picked per task, after the fact): "
          "**%d wins, %d ties (±%.3f), %d losses**." % (hl, w, t, TIE, l),
          "%s vs Laya's default English checkpoint: **%d wins, %d ties, %d losses**." % (hl, w2, t2, l2)]
    for k, lab in NIR[1:]:
        w3, t3, l3 = tally(heads, [acc(nir[k], s) for s in ORDER])
        L.append("%s vs %s: %d better, %d same, %d worse." % (hl, lab, w3, t3, l3))
    L.append("")
    for s, n, k, offered in label_coverage():
        L.append("`%s`: the %d cases cover %d of the %d labels offered; every label is offered in every question."
                 % (s, n, k, offered))
    L.append("")

    # ------------------------------------------------------------- calibration
    L += ["## Calibration (ECE, lower is better)", "",
          "| task | " + " | ".join(l for _, l in NIR) + " | best Laya |", "|---|" + "---|" * (len(NIR) + 1)]
    for s in ORDER:
        le = [((lay[k] or {}).get("suites", {}).get(s) or {}).get("ece") for k, _ in LAYA]
        le = [x for x in le if x is not None]
        L.append("| %s | %s | %s |" % (s, " | ".join(fmt(((nir[k] or {}).get("suites", {}).get(s) or {}).get("ece")) for k, _ in NIR),
                                     fmt(min(le) if le else None)))
    L.append("")

    # ------------------------------------------------------------- robustness
    flip = lambda d: [v.get("order_flip_rate") for v in (d or {}).get("suites", {}).values() if v.get("order_flip_rate") is not None]
    avg = lambda xs: (sum(xs) / len(xs)) if xs else None
    cols = [l for _, l in NIR] + [l for _, l in LAYA]
    nm = " | ".join("not measured" for _ in LAYA)
    L += ["## Robustness and determinism", "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    L.append("| option-order flip rate (choice tasks, mean) | %s | %s |" % (
        " | ".join(fmt(avg(flip(nir[k]))) for k, _ in NIR), " | ".join(fmt(avg(flip(lay[k]))) for k, _ in LAYA)))
    L.append("| identical answers on repeat call | %s | %s |" % (
        " | ".join(fmt((nir[k] or {}).get("determinism_repeat_identical")) for k, _ in NIR),
        " | ".join(fmt((lay[k] or {}).get("determinism_repeat_identical")) for k, _ in LAYA)))
    dets = [jload("nirnay_determinism_%s.json" % MODELS[k][1]) for k, _ in NIR]
    L.append("| byte-identical answers across 1/4/8/16 threads | %s | %s |" % (
        " | ".join(fmt((d or {}).get("identical_rate")) for d in dets), nm))
    L.append("| decisions changed across 1/4/8/16 threads | %s | %s |" % (
        " | ".join(("%s (max prob. diff %s)" % (d.get("answer_changes"), d.get("max_probability_diff"))) if d else "—"
                   for d in dets), nm))
    cs = jload("cold_start.json") or {}
    L.append("| cold start: new process -> first answer (s) | %s | %s |" % (
        " | ".join(fmt((cs.get("nirnay:" + k) or {}).get("first_answer_s"), 1) for k, _ in NIR),
        " | ".join(fmt((cs.get("laya:" + k) or {}).get("first_answer_s"), 1) for k, _ in LAYA)))
    L.append("")
    L += order_section()

    # ------------------------------------------------------------- latency
    ll, lo = jload("laya_latency.json"), jload("laya_latency_onnx.json")
    if ll and lo:   # Laya's best of eager torch (8 / 16 threads) and its own ONNX Runtime backend
        for mname, ent in lo.get("models", {}).items():
            ll["models"].setdefault(mname, {"by_threads": {}})["by_threads"]["onnx"] = ent["by_threads"].get("0", {})
    nl = {k: jload("nirnay_latency_%s.json" % MODELS[k][1]) for k, _ in NIR}
    L += ["## Latency on this CPU (Laya's own workload: a support ticket, mixed choice / yes-no questions)", "",
          "p50 ms, warm, best setting for each model (Laya: eager torch at 8/16 threads or its ONNX Runtime backend).", "",
          "| questions per call | " + " | ".join(l for _, l in NIR) + " | " + " | ".join(l for _, l in LAYA) +
          " | " + " | ".join("%s speed-up vs fastest Laya" % l for _, l in NIR) + " |",
          "|---|" + "---|" * (2 * len(NIR) + len(LAYA))]
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
        ns = [best_n(nl[k]) for k, _ in NIR]
        L.append("| %d | %s | %s | %s |" % (
            n, " | ".join(("**%s**" % fmt(x, 1)) if k == head else fmt(x, 1) for x, (k, _) in zip(ns, NIR)),
            " | ".join(fmt(x, 1) for x in ls),
            " | ".join(("%.1fx" % (fastest / x)) if (fastest and x) else "—" for x in ns)))
    L.append("")
    L.append("Batch throughput (10 questions per state): " + ", ".join(
        "%s %s decisions/s" % (lab, (nl[k] or {}).get("throughput_10q", {}).get("decisions_per_s")) for k, lab in NIR) + ".")
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
    for k, lab in NIR:
        fs = jload("fewshot_%s.json" % MODELS[k][1]) or {}
        for task, suite in (("banking77", "jev.banking77_full"), ("emotion", "jev.emotion")):
            r = fs.get(task) or {}
            if not r:
                continue
            ls = [acc(lay[x], suite) for x, _ in LAYA]
            L.append("| %s %s | %s | %s s | %s | %s |" % (
                lab, task, " | ".join(fmt((r.get(str(q)) or {}).get("accuracy")) for q in (0, 2, 4, 8, 16)),
                fmt((r.get("16") or {}).get("learn_seconds"), 0), fmt(max([x for x in ls if x is not None], default=None)),
                fmt(JEV.get(suite))))
    L.append("")

    # ------------------------------------------------------------- multilingual
    L += multilingual_section([k for k in want if k in MODELS])

    # ------------------------------------------------------------- ablations
    ab = [("nirnay-bges-b2", "v1-small: frozen encoder + 2 fusion blocks"),
          ("nirnay-v1", "v1: deep fusion (bge-small)"),
          ("nirnay-bges-deep-debias", "v1 + opt-in debias"),
          ("nirnay-v2", "v2: deep fusion (bge-base, more data)"),
          ("nirnay-v3", "v3: v2 + routing, short messages, safety, urgency, phrasings"),
          ("nirnay-v31", "v3.1: v3 with content-only safety wording")]
    abd = [(lbl, jload("suites__nirnay__%s.json" % n)) for n, lbl in ab]
    abd = [(lbl, d) for lbl, d in abd if d]
    L += ["## Ablations (same suites)", "", "| task | " + " | ".join(l for l, _ in abd) + " |", "|---|" + "---|" * len(abd)]
    for s in ORDER:
        L.append("| %s | %s |" % (s, " | ".join(fmt(acc(d, s)) for _, d in abd)))
    L.append("")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
