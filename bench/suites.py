"""Benchmark suites, built once and frozen to bench/data/suites.jsonl.

The "jev.*" and "app.*" suites reproduce Laya's research/scripts/bench_apps.py build()
case-for-case (same datasets, same filtering, same order, same seed 13, N=400), so Laya
and Nirnay answer byte-identical states and questions. "extra.*" suites are additional
held-out tasks (neither model trains on them):
  extra.sst5       ordinal 5-level sentiment (the `score` type; Laya's documented weak spot)
  extra.sms_spam   SMS spam (an independent Jev figure exists: 0.930)

  python bench\\suites.py
"""
import json
import os
import random
import sys

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "suites.jsonl")
SEED = 13
N = int(os.environ.get("BENCH_N", "400"))

SUITES = []


def register(name, cases, gold, **meta):
    for i, ((state, qs), g) in enumerate(zip(cases, gold)):
        SUITES.append({"suite": name, "i": i, "state": state, "questions": qs, "gold": g, "meta": meta})
    print("   %-28s %4d cases  %s" % (name, len(cases), meta.get("note", "")), flush=True)


def choice_q(qid, instructions, keys, gold_key):
    crit = {k: None for k in keys}
    return {qid: {"type": "choice", "instructions": instructions, "criteria": crit}}, keys.index(gold_key)


def email_state(subject, body):
    from laya import email_state as es  # the exact cleaning Laya's own benchmark applies
    return es(subject, body)


def build_laya_suites():
    rng = random.Random(SEED)

    d = load_dataset("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    keys = list(crit)
    cases, gold = [], []
    for r in list(d)[:N]:
        cases.append(({"article": r["text"]},
                      {"topic": {"type": "choice", "instructions": "What is the topic of `article`?",
                                 "criteria": dict(crit)}}))
        gold.append(keys.index(keys[int(r["label"])]))
    register("jev.ag_news", cases, gold, note="Jev 0.910", jev="ag_news", laya_in_training=True)

    d = load_dataset("dair-ai/emotion", "split", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    cases, gold = [], []
    for r in list(d)[:N]:
        cases.append(({"text": r["text"]},
                      {"emotion": {"type": "choice",
                                   "instructions": "Which emotion is most strongly expressed in `text`?",
                                   "criteria": {n: None for n in names}}}))
        gold.append(int(r["label"]))
    register("jev.emotion", cases, gold, note="Jev 0.480", jev="emotion", laya_in_training=False)

    d = load_dataset("mteb/banking77", split="test")
    labels = sorted(set(d["label_text"]))
    cases, gold = [], []
    for r in list(d)[:N]:
        qs, gi = choice_q("intent", "Which banking intent does `message` express?",
                          [x.replace("_", " ") for x in labels], r["label_text"].replace("_", " "))
        cases.append(({"message": r["text"]}, qs))
        gold.append(gi)
    register("jev.banking77_full", cases, gold, note="%d labels | Jev 0.870 (72 labels)" % len(labels),
             jev="banking77", laya_in_training=False, n_labels=len(labels))

    d = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    QUEUES = {"Technical Support": "technical problems, bugs, outages, integrations",
              "Product Support": "help using a product or feature",
              "Customer Service": "general account or service questions",
              "IT Support": "internal IT, devices, access, networks",
              "Billing and Payments": "invoices, charges, refunds, payment methods",
              "Returns and Exchanges": "returning or exchanging an item",
              "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
              "Sales and Pre-Sales": "pricing, quotes, buying",
              "Human Resources": "employment, payroll, leave, hiring",
              "General Inquiry": "anything else"}
    keys = list(QUEUES)
    cases, gold = [], []
    for r in d:
        if r.get("language") != "en" or r.get("queue") not in QUEUES or not r.get("body"):
            continue
        cases.append(({"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]},
                      {"queue": {"type": "choice", "instructions": "Which support queue should handle this ticket?",
                                 "criteria": dict(QUEUES)}}))
        gold.append(keys.index(r["queue"]))
        if len(cases) >= N:
            break
    register("app.support_triage", cases, gold, note="10-way queue routing", laya_in_training=True)

    d = load_dataset("SetFit/enron_spam", split="test")
    cases, gold = [], []
    for r in list(d)[:N]:
        st = email_state(r.get("subject") or "", (r.get("message") or "")[:3000])
        cases.append((st, {"is_spam": {"type": "noul", "instructions": "Is this email unsolicited spam or bulk marketing?"}}))
        gold.append(int(r["label"]))
    register("app.email_spam", cases, gold, note="enron spam", laya_in_training=True)

    d = load_dataset("zefang-liu/phishing-email-dataset", split="train")
    rows = [r for r in list(d)[:6000]
            if (r.get("Email Text") or "").strip() and r.get("Email Type") in ("Safe Email", "Phishing Email")]
    rng.shuffle(rows)
    cases, gold = [], []
    for r in rows[:N]:
        cases.append(({"email": r["Email Text"][:3000]},
                      {"is_phishing": {"type": "noul",
                                       "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
                                       "criteria": {"true": "phishing, scam, or fraud",
                                                    "false": "a legitimate email (even if promotional)"}}}))
        gold.append(int(r["Email Type"] == "Phishing Email"))
    register("app.phishing", cases, gold, note="phishing emails", laya_in_training=True)

    d = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    rows = [r for r in d if (r.get("user_input") or "").strip()]
    jb = [r for r in rows if int(r.get("jailbreaking", 0)) == 1][:N // 2]
    nj = [r for r in rows if int(r.get("jailbreaking", 0)) == 0][:N - len(jb)]
    mix = jb + nj
    rng.shuffle(mix)
    cases, gold = [], []
    for r in mix:
        cases.append(({"prompt": r["user_input"][:3000]},
                      {"jailbreak": {"type": "noul",
                                     "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?"}}))
        gold.append(int(r["jailbreaking"]))
    register("app.guardrails_jailbreak", cases, gold, note="toxic-chat jailbreaking (held out)", laya_in_training=False)
    tox = [r for r in rows if int(r.get("toxicity", 0)) == 1][:N // 2]
    ntox = [r for r in rows if int(r.get("toxicity", 0)) == 0][:N - len(tox)]
    mix2 = tox + ntox
    rng.shuffle(mix2)
    cases, gold = [], []
    for r in mix2:
        cases.append(({"post": r["user_input"][:3000]},
                      {"toxic": {"type": "noul",
                                 "instructions": "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?"}}))
        gold.append(int(r["toxicity"]))
    register("app.moderation_toxicity", cases, gold, note="toxic-chat toxicity (held out)", laya_in_training=False)

    d = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    cases, gold = [], []
    for r in d:
        texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
        pos = [t for t, s in zip(texts, sel) if s == 1]
        neg = [t for t, s in zip(texts, sel) if s == 0]
        if not pos or not neg:
            continue
        take_pos = len(cases) % 2 == 0
        p = rng.choice(pos if take_pos else neg)
        cases.append(({"query": r["query"], "passage": p},
                      {"relevant": {"type": "noul", "instructions": "Does `passage` help answer `query`?"}}))
        gold.append(1 if take_pos else 0)
        if len(cases) >= N:
            break
    register("app.rag_relevance", cases, gold, note="MS MARCO relevance", laya_in_training=True)

    DOM = {"code": "software engineering, programming, refactoring, architecture, debugging",
           "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
           "writing": "creative writing, essays, emails, blog posts, copywriting",
           "factual_lookup": "facts, definitions, trivia, history",
           "data_analysis": "statistics, SQL, data manipulation, metrics",
           "chitchat": "casual conversation, greetings, small talk"}
    keys = list(DOM)
    pool = []
    g = load_dataset("openai/gsm8k", "main", split="test")
    pool += [(r["question"], "math_or_logic") for r in list(g)[:N // 3]]
    m = load_dataset("google-research-datasets/mbpp", "full", split="test")
    pool += [(r["text"], "code") for r in list(m)[:N // 3]]
    t = load_dataset("fancyzhx/ag_news", split="test")
    pool += [(r["text"][:400], "factual_lookup") for r in list(t)[:N // 3]]
    rng.shuffle(pool)
    cases, gold = [], []
    for text, dom in pool[:N]:
        cases.append(({"request": text},
                      {"domain": {"type": "choice", "instructions": "What domain does `request` belong to?",
                                  "criteria": dict(DOM)}}))
        gold.append(keys.index(dom))
    register("app.model_routing_domain", cases, gold, note="gsm8k/mbpp/ag_news -> domain", laya_in_training=False)


def build_extra_suites():
    rng = random.Random(SEED + 1)
    d = load_dataset("SetFit/sst5", split="test")
    levels = ["very negative", "negative", "neutral", "positive", "very positive"]
    rows = list(d)
    rng.shuffle(rows)
    cases, gold = [], []
    for r in rows[:N]:
        cases.append(({"review": r["text"]},
                      {"sentiment": {"type": "score", "instructions": "How positive is the sentiment of `review`?",
                                     "criteria": list(levels)}}))
        gold.append(int(r["label"]))
    register("extra.sst5", cases, gold, note="ordinal 5-level (score type, held out)", laya_in_training=False)

    d = load_dataset("ucirvine/sms_spam", split="train")
    rows = list(d)
    rng.shuffle(rows)
    spam = [r for r in rows if int(r["label"]) == 1][:N // 2]
    ham = [r for r in rows if int(r["label"]) == 0][:N - len(spam)]
    mix = spam + ham
    rng.shuffle(mix)
    cases, gold = [], []
    for r in mix:
        cases.append(({"sms": r["sms"].strip()},
                      {"is_spam": {"type": "noul", "instructions": "Is `sms` spam?"}}))
        gold.append(int(r["label"]))
    register("extra.sms_spam", cases, gold, note="SMS spam (held out) | independent Jev 0.930", laya_in_training=False)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("=== building suites (N=%d) ===" % N, flush=True)
    build_laya_suites()
    build_extra_suites()
    with open(OUT, "w", encoding="utf-8") as f:
        for r in SUITES:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d cases -> %s" % (len(SUITES), OUT))


if __name__ == "__main__":
    sys.exit(main())
