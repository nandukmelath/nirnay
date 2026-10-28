"""Multilingual benchmark suites, built once and frozen to bench/data/suites_multi.jsonl.

  massive.<lang>  MASSIVE intent detection (mteb/amazon_massive_intent, test split): 51 languages, the 60 intents
                  as options, an English question over a request written in that language. 80 cases per language.
  sib200.<lang>   SIB-200 topic classification (Davlan/sib200, test split): 205 languages and scripts, 7 topics.
                  The same 24 FLORES sentences in every language, so languages are compared on identical content.

Both datasets are held out of every Nirnay training build (as MASSIVE is for Laya).
  python bench\\suites_multi.py
"""
import json
import os
import random

from datasets import get_dataset_config_names, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "suites_multi.jsonl")
SEED = 13
N_MASSIVE = int(os.environ.get("BENCH_N_MASSIVE", "80"))
N_SIB = int(os.environ.get("BENCH_N_SIB", "24"))
SUITES = []


def register(name, cases, gold, **meta):
    for i, ((state, qs), g) in enumerate(zip(cases, gold)):
        SUITES.append({"suite": name, "i": i, "state": state, "questions": qs, "gold": g, "meta": meta})


def build_massive():
    langs = [c for c in get_dataset_config_names("mteb/amazon_massive_intent") if c not in ("default", "all")]
    intents = sorted(set(load_dataset("mteb/amazon_massive_intent", "en", split="train")["label"]))
    labels = [x.replace("_", " ") for x in intents]
    crit = {k: None for k in labels}
    for lang in langs:
        rows = list(load_dataset("mteb/amazon_massive_intent", lang, split="test"))
        random.Random("%s-%s" % (SEED, lang)).shuffle(rows)
        cases, gold = [], []
        for r in rows[:N_MASSIVE]:
            cases.append(({"utterance": r["text"]},
                          {"intent": {"type": "choice", "instructions": "Which intent does `utterance` express?",
                                      "criteria": dict(crit)}}))
            gold.append(labels.index(r["label"].replace("_", " ")))
        register("massive." + lang, cases, gold, note="MASSIVE %s, %d intents (held out)" % (lang, len(labels)),
                 lang=lang, laya_in_training=False, n_labels=len(labels))
    print("massive: %d languages x %d cases, %d intents" % (len(langs), N_MASSIVE, len(labels)), flush=True)


def build_sib200():
    langs = get_dataset_config_names("Davlan/sib200")
    topics = None
    broken = []
    for lang in langs:
        try:
            d = load_dataset("Davlan/sib200", lang, split="test")
        except Exception:   # a few configs point at files missing from the Hub repo (e.g. nqo_Nkoo)
            broken.append(lang)
            continue
        rows = sorted(d, key=lambda r: int(r["index_id"]))[:N_SIB]
        if topics is None:
            topics = sorted(set(load_dataset("Davlan/sib200", lang, split="train")["category"]))
        cases, gold = [], []
        for r in rows:
            cases.append(({"text": r["text"]},
                          {"topic": {"type": "choice", "instructions": "What is the topic of `text`?",
                                     "criteria": {t: None for t in topics}}}))
            gold.append(topics.index(r["category"]))
        register("sib200." + lang, cases, gold, note="SIB-200 %s, %d topics (held out)" % (lang, len(topics)),
                 lang=lang, laya_in_training=False)
    print("sib200: %d languages x %d cases, topics %s | skipped (broken on the Hub): %s" % (
        len(langs) - len(broken), N_SIB, topics, broken or "none"), flush=True)


def main():
    build_massive()
    build_sib200()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in SUITES:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d cases in %d suites -> %s" % (len(SUITES), len({r["suite"] for r in SUITES}), OUT))


if __name__ == "__main__":
    main()
