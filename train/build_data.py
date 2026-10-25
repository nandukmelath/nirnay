"""Build Nirnay's multi-task decision corpus from public datasets.

Every record is one typed question about one state, in the Jev/Laya question spec:
  {"src", "state", "q": {"type", "instructions", "criteria"}, "y": gold option index, "split"}

Held out of training entirely (they are benchmark suites, mirroring Laya's own held-out policy):
  dair-ai/emotion, mteb/banking77, lmsys/toxic-chat (jailbreak + toxicity), openai/gsm8k,
  google-research-datasets/mbpp (model routing), SetFit/sst5, ucirvine/sms_spam, XNLI, MASSIVE,
  and every jailbreak / prompt-injection dataset.
Datasets Laya also trained on (ag_news, enron spam, phishing, support tickets, MS MARCO) contribute
only rows that are not in the benchmark cases, and any state whose text appears in a benchmark
suite is dropped.

  python train\\build_data.py          (NIRNAY_SCALE=0.1 for a quick smoke build)
"""
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.environ.get("NIRNAY_RECORDS_OUT", os.path.join(HERE, "data", "records.jsonl"))
SUITES = os.path.join(ROOT, "bench", "data", "suites.jsonl")
SCALE = float(os.environ.get("NIRNAY_SCALE", "1.0"))
SEED = 1234
# per-source multipliers, e.g. NIRNAY_SRC_SCALE="ag_news=2.86,msmarco=2.86"  (source = function name minus src_)
SRC_SCALE = {k: float(v) for k, v in (p.split("=") for p in os.environ.get("NIRNAY_SRC_SCALE", "").split(",") if "=" in p)}
CURRENT = {"src": ""}


def n_(k):
    return max(50, int(k * SCALE * SRC_SCALE.get(CURRENT["src"], 1.0)))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()[:300]


def leaves(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from leaves(v)


EVAL_KEYS = set()


def load_eval_keys():
    if not os.path.exists(SUITES):
        sys.exit("build the benchmark suites first: python bench\\suites.py")
    for line in open(SUITES, encoding="utf-8"):
        r = json.loads(line)
        for s in leaves(r["state"]):
            if len(s) >= 20:
                EVAL_KEYS.add(norm(s))
    print("eval texts held out:", len(EVAL_KEYS), flush=True)


def leaks(state) -> bool:
    return any(len(s) >= 20 and norm(s) in EVAL_KEYS for s in leaves(state))


# ------------------------------------------------------------------ question helpers
FIELDS_GENERIC = ["text", "message", "content", "input", "document"]


def wrap(rng, text, fields=FIELDS_GENERIC, p_plain=0.35):
    """-> (state, reference used in instructions)."""
    if rng.random() < p_plain:
        return text, rng.choice(["the text", "this text", "the input", "this message"])
    f = rng.choice(fields)
    return {f: text}, "`%s`" % f


def pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def choice_q(rng, ins, labels, gold, descs=None, kmax=None, p_full=0.5, p_desc=0.5, kmin=2):
    n = len(labels)
    cap = min(n, kmax or n)
    if n > cap or rng.random() > p_full:
        k = rng.randint(min(kmin, cap), cap)
        others = rng.sample([i for i in range(n) if i != gold], k - 1)
        idx = others + [gold]
    else:
        idx = list(range(n))
    rng.shuffle(idx)
    use_desc = descs is not None and rng.random() < p_desc
    crit = {labels[i]: (descs.get(labels[i]) if use_desc else None) for i in idx}
    return {"type": "choice", "instructions": ins, "criteria": crit}, idx.index(gold)


def noul_q(ins, crit=None):
    q = {"type": "noul", "instructions": ins}
    if crit:
        q["criteria"] = crit
    return q


def noul_from_class(rng, templates, ref, labels, gold, shown=None):
    """Binary question about one class: 50% the gold class (yes), 50% another class (no)."""
    if rng.random() < 0.5:
        lab, y = labels[gold], 1
    else:
        lab, y = labels[rng.choice([i for i in range(len(labels)) if i != gold])], 0
    lab = (shown or {}).get(lab, lab)
    return noul_q(pick(rng, templates).format(ref=ref, label=lab)), y


# ------------------------------------------------------------------ sources
TOPIC_INS = ["What is the topic of {ref}?", "Which category best describes {ref}?", "What is {ref} mainly about?",
             "Classify {ref} by subject.", "Pick the topic that fits {ref}.", "What subject area does {ref} cover?",
             "Which section would {ref} be filed under?"]
TOPIC_NOUL = ["Is {ref} about {label}?", "Does {ref} belong to the {label} category?", "Is the topic of {ref} {label}?"]


def src_ag_news(rng):
    d = load_dataset("fancyzhx/ag_news", split="train")
    variants = [
        (["world", "sports", "business", "sci_tech"], {"world": "world news and international politics", "sports": "sports",
                                                        "business": "business and economy", "sci_tech": "science and technology"}),
        (["World", "Sports", "Business", "Sci/Tech"], {"World": "international news", "Sports": "games, teams and athletes",
                                                       "Business": "companies, markets and the economy",
                                                       "Sci/Tech": "technology, science and computing"}),
        (["international", "sport", "economy", "technology"], None),
    ]
    rows = list(d.shuffle(seed=SEED).select(range(n_(14000))))
    for r in rows:
        labels, descs = pick(rng, variants)
        st, ref = wrap(rng, r["text"], ["article", "headline", "text", "news"])
        if rng.random() < 0.8:
            q, y = choice_q(rng, pick(rng, TOPIC_INS).format(ref=ref), labels, r["label"], descs, p_full=0.8)
        else:
            q, y = noul_from_class(rng, TOPIC_NOUL, ref, labels, r["label"], descs)
        yield "ag_news", st, q, y


def src_dbpedia(rng):
    d = load_dataset("fancyzhx/dbpedia_14", split="train").shuffle(seed=SEED).select(range(n_(9000)))
    names = d.features["label"].names
    nice = {"Company": "company", "EducationalInstitution": "educational institution", "Artist": "artist",
            "Athlete": "athlete", "OfficeHolder": "politician or office holder", "MeanOfTransportation": "vehicle or means of transportation",
            "Building": "building", "NaturalPlace": "natural place", "Village": "village", "Animal": "animal",
            "Plant": "plant", "Album": "music album", "Film": "film", "WrittenWork": "book or written work"}
    labels = [nice[n] for n in names]
    ins = ["What kind of entity does {ref} describe?", "Which category does the subject of {ref} belong to?",
           "What is {ref} about?", "Classify the entity described in {ref}."]
    for r in d:
        text = (r["title"] + ". " + r["content"]).strip()
        st, ref = wrap(rng, text, ["description", "text", "article", "entry"])
        if rng.random() < 0.8:
            q, y = choice_q(rng, pick(rng, ins).format(ref=ref), labels, r["label"], None, p_full=0.5)
        else:
            q, y = noul_from_class(rng, ["Does {ref} describe a {label}?", "Is the subject of {ref} a {label}?"],
                                   ref, labels, r["label"])
        yield "dbpedia", st, q, y


def src_yahoo(rng):
    d = load_dataset("community-datasets/yahoo_answers_topics", split="train").shuffle(seed=SEED).select(range(n_(9000)))
    labels = [n.replace("&", "and") for n in d.features["topic"].names]
    ins = ["Which forum category fits {ref}?", "What is the topic of {ref}?", "Which section of a Q&A site should {ref} go to?",
           "What subject is {ref} about?"]
    for r in d:
        text = (r["question_title"] + " " + (r["question_content"] or "")).strip()
        if rng.random() < 0.4 and r["best_answer"]:
            st = {"question": text, "answer": r["best_answer"][:1200]}
            ref = "this question and answer"
        else:
            st, ref = wrap(rng, text, ["question", "post", "text"])
        q, y = choice_q(rng, pick(rng, ins).format(ref=ref), labels, r["topic"], None, p_full=0.6)
        yield "yahoo", st, q, y


def src_20ng(rng):
    d = load_dataset("SetFit/20_newsgroups", split="train").shuffle(seed=SEED)
    names = sorted(set(d["label_text"]))
    out = 0
    for r in d:
        if len(r["text"].strip()) < 40:
            continue
        st, ref = wrap(rng, r["text"][:2500], ["post", "message", "text"])
        q, y = choice_q(rng, pick(rng, ["Which newsgroup was {ref} posted to?", "Which discussion group does {ref} belong to?",
                                       "What is the topic of {ref}?"]).format(ref=ref),
                        names, names.index(r["label_text"]), None, kmax=12, p_full=0.2)
        yield "20ng", st, q, y
        out += 1
        if out >= n_(4000):
            break


SENT_INS = ["What is the sentiment of {ref}?", "How does the author of {ref} feel?", "Is {ref} positive or negative?",
            "Classify the opinion expressed in {ref}.", "What attitude does {ref} express?"]


def src_sst2(rng):
    d = load_dataset("stanfordnlp/sst2", split="train").shuffle(seed=SEED).select(range(n_(7000)))
    variants = [(["negative", "positive"], None), (["bad", "good"], None),
                (["negative", "positive"], {"negative": "the writer dislikes it", "positive": "the writer likes it"})]
    for r in d:
        st, ref = wrap(rng, r["sentence"], ["review", "text", "comment", "sentence"])
        u = rng.random()
        if u < 0.6:
            labels, descs = pick(rng, variants)
            q, y = choice_q(rng, pick(rng, SENT_INS).format(ref=ref), labels, r["label"], descs, p_full=1.0)
        elif u < 0.8:
            q, y = noul_q(pick(rng, ["Is {ref} positive?", "Does {ref} express a favourable opinion?",
                                     "Did the writer of {ref} like it?"]).format(ref=ref)), r["label"]
        else:
            q, y = noul_q(pick(rng, ["Is {ref} negative?", "Does {ref} express criticism?",
                                     "Is the writer of {ref} unhappy?"]).format(ref=ref)), 1 - r["label"]
        yield "sst2", st, q, y


def src_imdb(rng):
    d = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=SEED).select(range(n_(4000)))
    for r in d:
        st, ref = wrap(rng, r["text"].replace("<br />", " ")[:3000], ["review", "text", "comment"])
        if rng.random() < 0.6:
            q, y = choice_q(rng, pick(rng, SENT_INS).format(ref=ref), ["negative", "positive"], r["label"], None, p_full=1.0)
        else:
            q, y = noul_q(pick(rng, ["Did the reviewer enjoy the film in {ref}?", "Is {ref} a positive review?",
                                     "Would the author of {ref} recommend the movie?"]).format(ref=ref)), r["label"]
        yield "imdb", st, q, y


STAR_LEVELS = [
    ["very negative", "negative", "mixed or neutral", "positive", "very positive"],
    ["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"],
    ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"],
    ["awful", "bad", "okay", "good", "great"],
]
SCORE_INS = ["How positive is {ref}?", "Rate the sentiment of {ref}.", "How satisfied is the writer of {ref}?",
             "What rating best fits {ref}?", "How favourable is {ref}?"]


def _rating(rng, src, text, label, fields):
    st, ref = wrap(rng, text, fields)
    u = rng.random()
    if u < 0.65:
        levels = list(pick(rng, STAR_LEVELS))
        q = {"type": "score", "instructions": pick(rng, SCORE_INS).format(ref=ref), "criteria": levels}
        return src, st, q, label
    if u < 0.85:
        labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
        q, y = choice_q(rng, pick(rng, ["How many stars did the writer of {ref} give?", "Which star rating matches {ref}?"]).format(ref=ref),
                        labels, label, None, p_full=1.0)
        return src, st, q, y
    if label == 2:
        return None
    return src, st, noul_q(pick(rng, ["Is {ref} positive overall?", "Is the writer of {ref} happy with it?"]).format(ref=ref)), int(label > 2)


def src_yelp(rng):
    d = load_dataset("Yelp/yelp_review_full", split="train").shuffle(seed=SEED).select(range(n_(10000)))
    for r in d:
        rec = _rating(rng, "yelp", r["text"][:2500], r["label"], ["review", "text", "feedback"])
        if rec:
            yield rec


def src_amazon_multi(rng):
    # the repo still has a loading script; read the Hub's auto-converted parquet instead
    d = load_dataset("parquet", data_files="hf://datasets/mteb/amazon_reviews_multi@refs%2Fconvert%2Fparquet/all_languages/train/*.parquet",
                     split="train").shuffle(seed=SEED).select(range(n_(9000)))
    for r in d:
        rec = _rating(rng, "amazon_multi", r["text"][:2000], int(r["label"]), ["review", "text", "feedback"])
        if rec:
            yield rec


def src_tweet_sent_multi(rng):
    d = load_dataset("parquet", data_files="hf://datasets/cardiffnlp/tweet_sentiment_multilingual@refs%2Fconvert%2Fparquet/all/train/*.parquet",
                     split="train").shuffle(seed=SEED).select(range(n_(7000)))
    variants = [["negative", "neutral", "positive"], ["bad", "neutral", "good"]]
    for r in d:
        st, ref = wrap(rng, r["text"], ["tweet", "post", "text"])
        q, y = choice_q(rng, pick(rng, SENT_INS).format(ref=ref), pick(rng, variants), int(r["label"]), None, p_full=1.0)
        yield "tweet_sent_multi", st, q, y


EMO_INS = ["Which emotion is expressed in {ref}?", "What emotion does the writer of {ref} feel most?",
           "Identify the main emotion in {ref}.", "What is the emotional tone of {ref}?", "How does the writer of {ref} feel?"]


def src_go_emotions(rng):
    d = load_dataset("google-research-datasets/go_emotions", "simplified", split="train").shuffle(seed=SEED)
    names = d.features["labels"].feature.names
    out = 0
    for r in d:
        if len(r["labels"]) != 1:
            continue
        g = r["labels"][0]
        st, ref = wrap(rng, r["text"], ["comment", "text", "message", "post"])
        if rng.random() < 0.75:
            q, y = choice_q(rng, pick(rng, EMO_INS).format(ref=ref), names, g, None, kmax=12, p_full=0.15, kmin=3)
        else:
            q, y = noul_from_class(rng, ["Does {ref} express {label}?", "Is the writer of {ref} feeling {label}?"],
                                   ref, names, g)
        yield "go_emotions", st, q, y
        out += 1
        if out >= n_(9000):
            break


def src_tweet_emotion(rng):
    d = load_dataset("cardiffnlp/tweet_eval", "emotion", split="train").shuffle(seed=SEED)
    names = d.features["label"].names
    for r in d.select(range(min(len(d), n_(3200)))):
        st, ref = wrap(rng, r["text"], ["tweet", "text", "post"])
        q, y = choice_q(rng, pick(rng, EMO_INS).format(ref=ref), names, r["label"], None, p_full=1.0)
        yield "tweet_emotion", st, q, y


INTENT_INS = ["What does the user want in {ref}?", "Which intent does {ref} express?", "Classify the request in {ref}.",
              "What is the purpose of {ref}?", "Which action is being asked for in {ref}?", "What is {ref} asking for?"]


def src_clinc(rng):
    d = load_dataset("clinc/clinc_oos", "plus", split="train").shuffle(seed=SEED)
    names = d.features["intent"].names
    shown = {n: ("out of scope: none of the listed requests" if n == "oos" else n.replace("_", " ")) for n in names}
    labels = [shown[n] for n in names]
    out = 0
    for r in d:
        st, ref = wrap(rng, r["text"], ["utterance", "message", "query", "request"])
        u = rng.random()
        if u < 0.75:
            q, y = choice_q(rng, pick(rng, INTENT_INS).format(ref=ref), labels, r["intent"], None, kmax=48, p_full=0.0, kmin=4)
        elif u < 0.83:
            q, y = choice_q(rng, pick(rng, INTENT_INS).format(ref=ref), labels, r["intent"], None, kmax=150, p_full=1.0)
        else:
            if names[r["intent"]] == "oos":
                continue
            q, y = noul_from_class(rng, ["Does {ref} ask to {label}?", "Is {ref} a request about {label}?",
                                         "Does the user in {ref} want {label}?"], ref, labels, r["intent"])
        yield "clinc", st, q, y
        out += 1
        if out >= n_(14000):
            break


def src_bitext(rng):
    d = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train").shuffle(seed=SEED)
    intents = sorted(set(d["intent"]))
    cats = sorted(set(d["category"]))
    il = [x.replace("_", " ") for x in intents]
    cl = [x.lower() for x in cats]
    for r in d.select(range(n_(7000))):
        st, ref = wrap(rng, r["instruction"], ["message", "ticket", "request", "customer_message"])
        u = rng.random()
        if u < 0.55:
            q, y = choice_q(rng, pick(rng, INTENT_INS).format(ref=ref), il, intents.index(r["intent"]), None, kmax=27, p_full=0.4)
        elif u < 0.8:
            q, y = choice_q(rng, pick(rng, ["Which support category does {ref} fall under?", "Which department should handle {ref}?"]).format(ref=ref),
                            cl, cats.index(r["category"]), None, p_full=0.7)
        else:
            q, y = noul_from_class(rng, ["Does the customer in {ref} want to {label}?", "Is {ref} about {label}?"],
                                   ref, il, intents.index(r["intent"]))
        yield "bitext", st, q, y


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


def src_support_tickets(rng):
    d = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    # skip the rows the benchmark uses: the first 400 English rows that pass its filter (we skip 1500)
    seen_en = 0
    keep = []
    for i, r in enumerate(d):
        if not r.get("body") or r.get("queue") not in QUEUES:
            continue
        if r.get("language") == "en":
            seen_en += 1
            if seen_en <= 1500:
                continue
        keep.append(r)
    rng.shuffle(keep)
    keys = list(QUEUES)
    pri_levels = {"low": 0, "medium": 1, "high": 2}
    types = sorted(set(r["type"] for r in keep if r.get("type")))
    for r in keep[:n_(9000)]:
        st = {"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]}
        u = rng.random()
        if u < 0.6:
            q, y = choice_q(rng, pick(rng, ["Which support queue should handle this ticket?", "Which team should this ticket go to?",
                                            "Route this ticket to the right queue."]),
                            keys, keys.index(r["queue"]), QUEUES, p_full=0.8, p_desc=0.7)
        elif u < 0.8 and str(r.get("priority", "")).lower() in pri_levels:
            q = {"type": "score", "instructions": pick(rng, ["How urgent is this ticket?", "What priority should this ticket get?"]),
                 "criteria": ["low: can wait", "medium: should be handled soon", "high: needs immediate attention"]}
            y = pri_levels[str(r["priority"]).lower()]
        elif r.get("type") in types and len(types) >= 2:
            q, y = choice_q(rng, "What type of ticket is this?", types, types.index(r["type"]), None, p_full=1.0)
        else:
            q, y = noul_from_class(rng, ["Should this ticket go to {label}?", "Is this ticket for the {label} queue?"],
                                   "", keys, keys.index(r["queue"]))
        yield "support_tickets", st, q, y


def _email_state(subject, body):
    from laya import email_state
    return email_state(subject, body)


SPAM_INS = ["Is this email spam?", "Is this email unsolicited bulk marketing?", "Is this message junk mail?",
            "Should this email go to the spam folder?"]
# medium-agnostic wording, so "spam" is not learned as an email-only concept
SPAM_GENERIC = ["Is {ref} spam?", "Is {ref} unsolicited advertising?", "Is {ref} a spam message?",
                "Is {ref} junk or a mass-marketing message?", "Would a spam filter be right to block {ref}?"]
SPAM_FIELDS = ["message", "text", "content", "msg", "body", "post", "sms", "note"]


def src_enron(rng):
    d = load_dataset("SetFit/enron_spam", split="train").shuffle(seed=SEED).select(range(n_(7000)))
    for r in d:
        if rng.random() < 0.35:   # body only, under a random field name and generic wording
            body = (r.get("message") or r.get("subject") or "").strip()[:1500]
            if not body:
                continue
            st, ref = wrap(rng, body, SPAM_FIELDS, p_plain=0.3)
            yield "enron_spam", st, noul_q(pick(rng, SPAM_GENERIC).format(ref=ref)), int(r["label"])
            continue
        st = _email_state(r.get("subject") or "", (r.get("message") or "")[:3000])
        if rng.random() < 0.7:
            q, y = noul_q(pick(rng, SPAM_INS)), int(r["label"])
        else:
            q, y = choice_q(rng, "Is this email spam or a legitimate message?", ["legitimate", "spam"], int(r["label"]),
                            {"legitimate": "normal personal or business email", "spam": "unsolicited bulk or marketing email"},
                            p_full=1.0)
        yield "enron_spam", st, q, y


def src_phishing(rng):
    d = load_dataset("zefang-liu/phishing-email-dataset", split="train")
    rows = [r for r in list(d)[6000:] if (r.get("Email Text") or "").strip()
            and r.get("Email Type") in ("Safe Email", "Phishing Email")]
    rng.shuffle(rows)
    ins = ["Is this email a phishing or scam attempt?", "Is `email` trying to steal money, passwords or personal data?",
           "Is this a fraudulent email?", "Is `email` a scam?"]
    generic = ["Is {ref} a scam?", "Is {ref} trying to trick the reader into giving money or passwords?",
               "Is {ref} fraudulent?", "Is {ref} a phishing attempt?"]
    for r in rows[:n_(6000)]:
        y = int(r["Email Type"] == "Phishing Email")
        if rng.random() < 0.3:
            st, ref = wrap(rng, r["Email Text"][:1500], SPAM_FIELDS, p_plain=0.3)
            yield "phishing", st, noul_q(pick(rng, generic).format(ref=ref)), y
            continue
        st = {"email": r["Email Text"][:3000]} if rng.random() < 0.7 else r["Email Text"][:3000]
        if rng.random() < 0.75:
            crit = {"true": "phishing, scam, or fraud", "false": "a legitimate email"} if rng.random() < 0.5 else None
            q = noul_q(pick(rng, ins), crit)
        else:
            q, y = choice_q(rng, "What kind of email is this?", ["legitimate", "phishing"], y,
                            {"legitimate": "a normal email", "phishing": "a scam or credential-theft attempt"}, p_full=1.0)
        yield "phishing", st, q, y


NLI_DESC = {"entailment": "the hypothesis must be true given the premise",
            "neutral": "the hypothesis may or may not be true",
            "contradiction": "the hypothesis must be false given the premise"}


def _nli(rng, src, prem, hyp, label):
    if rng.random() < 0.7:
        st = {"premise": prem, "hypothesis": hyp}
        P, H = "`premise`", "`hypothesis`"
    else:
        st = "Premise: %s\nHypothesis: %s" % (prem, hyp)
        P, H = "the premise", "the hypothesis"
    names = ["entailment", "neutral", "contradiction"]
    u = rng.random()
    if u < 0.5:
        q, y = choice_q(rng, pick(rng, ["How does %s relate to %s?" % (H, P), "Given %s, is %s true, false or undetermined?" % (P, H)]),
                        names, label, NLI_DESC, p_full=1.0, p_desc=0.7)
    elif u < 0.8:
        q, y = noul_q(pick(rng, ["Does %s imply %s?" % (P, H), "If %s is true, must %s also be true?" % (P, H)])), int(label == 0)
    else:
        q, y = noul_q(pick(rng, ["Does %s contradict %s?" % (H, P), "Are %s and %s incompatible?" % (P, H)])), int(label == 2)
    return src, st, q, y


def src_mnli(rng):
    d = load_dataset("nyu-mll/glue", "mnli", split="train").shuffle(seed=SEED).select(range(n_(11000)))
    for r in d:
        yield _nli(rng, "mnli", r["premise"], r["hypothesis"], r["label"])


def src_snli(rng):
    d = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=SEED).select(range(n_(7000)))
    for r in d:
        if r["label"] in (0, 1, 2):
            yield _nli(rng, "snli", r["premise"], r["hypothesis"], r["label"])


def _pair(rng, src, a, b, y, kind):
    if kind == "q":
        st = {"question1": a, "question2": b}
        ins = ["Are `question1` and `question2` asking the same thing?", "Is `question2` a duplicate of `question1`?"]
    else:
        st = {"sentence1": a, "sentence2": b}
        ins = ["Do `sentence1` and `sentence2` mean the same thing?", "Is `sentence2` a paraphrase of `sentence1`?"]
    return src, st, noul_q(pick(rng, ins)), int(y)


def src_paws(rng):
    d = load_dataset("google-research-datasets/paws", "labeled_final", split="train").shuffle(seed=SEED).select(range(n_(5000)))
    for r in d:
        yield _pair(rng, "paws", r["sentence1"], r["sentence2"], r["label"], "s")


def src_qqp(rng):
    d = load_dataset("nyu-mll/glue", "qqp", split="train").shuffle(seed=SEED).select(range(n_(5000)))
    for r in d:
        yield _pair(rng, "qqp", r["question1"], r["question2"], r["label"], "q")


def src_mrpc(rng):
    d = load_dataset("nyu-mll/glue", "mrpc", split="train")
    for r in d:
        yield _pair(rng, "mrpc", r["sentence1"], r["sentence2"], r["label"], "s")


def src_boolq(rng):
    d = load_dataset("google/boolq", split="train").shuffle(seed=SEED).select(range(n_(9000)))
    for r in d:
        qtext = r["question"].strip()
        qtext = qtext[0].upper() + qtext[1:] + ("" if qtext.endswith("?") else "?")
        if rng.random() < 0.6:
            st = {"passage": r["passage"][:2500]}
            ins = pick(rng, ["According to `passage`: %s" % qtext, "Based on `passage`, %s" % qtext[0].lower() + qtext[1:]])
        else:
            st = {"passage": r["passage"][:2500], "question": qtext}
            ins = "According to `passage`, is the answer to `question` yes?"
        yield "boolq", st, noul_q(ins), int(r["answer"])


def src_race(rng):
    d = load_dataset("ehovy/race", "all", split="train").shuffle(seed=SEED).select(range(n_(9000)))
    for r in d:
        opts = r["options"]
        g = "ABCD".index(r["answer"])
        st = {"article": r["article"][:3000]}
        qt = r["question"].strip()
        if "_" in qt:
            ins = "Fill the blank using `article`: %s" % qt
        else:
            ins = "Based on `article`: %s" % qt
        if rng.random() < 0.6:
            crit = {k: v for k, v in zip("ABCD", opts)}
            q = {"type": "choice", "instructions": ins, "criteria": crit}
            y = g
        else:
            uniq = list(dict.fromkeys(o.strip() for o in opts))
            if len(uniq) != 4:
                continue
            q, y = choice_q(rng, ins, uniq, g, None, p_full=1.0)
        yield "race", st, q, y


def src_csqa(rng):
    d = load_dataset("tau/commonsense_qa", split="train").shuffle(seed=SEED).select(range(n_(9000)))
    for r in d:
        texts = r["choices"]["text"]
        labs = r["choices"]["label"]
        if r["answerKey"] not in labs or len(set(texts)) != len(texts):
            continue
        g = labs.index(r["answerKey"])
        if rng.random() < 0.5:
            st, ins = {"question": r["question"]}, "Which answer to `question` is correct?"
        else:
            st, ins = r["question"], pick(rng, ["Pick the best answer.", "Which option answers the question?"])
        q, y = choice_q(rng, ins, texts, g, None, p_full=1.0)
        yield "commonsense_qa", st, q, y


REL_INS = ["Does `passage` help answer `query`?", "Is `passage` relevant to `query`?",
           "Does `passage` contain the answer to `query`?", "Would `passage` be useful for answering `query`?"]


def src_msmarco(rng):
    d = load_dataset("microsoft/ms_marco", "v1.1", split="train").shuffle(seed=SEED)
    out = 0
    allp = []
    for r in d:
        texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
        pos = [t for t, s in zip(texts, sel) if s == 1]
        neg = [t for t, s in zip(texts, sel) if s == 0]
        if not pos:
            continue
        allp.extend(texts[:2])
        take_pos = rng.random() < 0.5
        if take_pos:
            p, y = rng.choice(pos), 1
        elif neg and rng.random() < 0.8:
            p, y = rng.choice(neg), 0
        elif len(allp) > 100:
            p, y = rng.choice(allp[:-10]), 0
        else:
            continue
        yield "ms_marco", {"query": r["query"], "passage": p}, noul_q(pick(rng, REL_INS)), y
        out += 1
        if out >= n_(14000):
            break


TOX_INS = ["Is {ref} toxic?", "Is {ref} rude or disrespectful?", "Does {ref} contain insults or hateful language?",
           "Would {ref} make someone leave a discussion?", "Is {ref} abusive?"]


def src_civil(rng):
    d = load_dataset("google/civil_comments", split="train").shuffle(seed=SEED)
    pos, neg = [], []
    target = n_(8000)
    for r in d:
        t = r["toxicity"]
        if t >= 0.5 and len(pos) < target // 2:
            pos.append(r)
        elif t < 0.1 and len(neg) < target // 2:
            neg.append(r)
        if len(pos) >= target // 2 and len(neg) >= target // 2:
            break
    rows = pos + neg
    rng.shuffle(rows)
    for r in rows:
        st, ref = wrap(rng, r["text"][:2500], ["comment", "post", "text", "message"])
        y = int(r["toxicity"] >= 0.5)
        if rng.random() < 0.75:
            yield "civil_comments", st, noul_q(pick(rng, TOX_INS).format(ref=ref)), y
        else:
            q, y = choice_q(rng, "Is {ref} acceptable or toxic?".format(ref=ref), ["acceptable", "toxic"], y,
                            {"acceptable": "civil, polite or neutral", "toxic": "rude, insulting or hateful"}, p_full=1.0)
            yield "civil_comments", st, q, y


def src_hate_offensive(rng):
    d = load_dataset("tdavidson/hate_speech_offensive", split="train").shuffle(seed=SEED)
    names = ["hate speech", "offensive", "neither"]
    descs = {"hate speech": "attacks a group based on identity", "offensive": "rude or vulgar but not hate speech",
             "neither": "not offensive"}
    for r in d.select(range(min(len(d), n_(5000)))):
        st, ref = wrap(rng, r["tweet"], ["tweet", "post", "text"])
        if rng.random() < 0.6:
            q, y = choice_q(rng, pick(rng, ["Is {ref} hate speech, offensive, or neither?", "How offensive is {ref}?"]).format(ref=ref),
                            names, r["class"], descs, p_full=1.0)
        else:
            q, y = noul_q(pick(rng, ["Is {ref} offensive?", "Does {ref} contain offensive language?"]).format(ref=ref)), int(r["class"] < 2)
        yield "hate_offensive", st, q, y


def src_tweet_off_hate(rng):
    for cfg, ins in (("offensive", ["Is {ref} offensive?", "Does {ref} contain offensive language?"]),
                     ("hate", ["Does {ref} contain hate speech?", "Does {ref} attack people for who they are?"])):
        d = load_dataset("cardiffnlp/tweet_eval", cfg, split="train").shuffle(seed=SEED)
        for r in d.select(range(min(len(d), n_(4000)))):
            st, ref = wrap(rng, r["text"], ["tweet", "post", "text"])
            yield "tweet_" + cfg, st, noul_q(pick(rng, ins).format(ref=ref)), int(r["label"])


SOURCES = [src_ag_news, src_dbpedia, src_yahoo, src_20ng, src_sst2, src_imdb, src_yelp, src_amazon_multi,
           src_tweet_sent_multi, src_go_emotions, src_tweet_emotion, src_clinc, src_bitext, src_support_tickets,
           src_enron, src_phishing, src_mnli, src_snli, src_paws, src_qqp, src_mrpc, src_boolq, src_race, src_csqa,
           src_msmarco, src_civil, src_hate_offensive, src_tweet_off_hate]


def split_of(state) -> str:
    h = int(hashlib.sha1(json.dumps(state, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "dev" if h < 2 else "cal" if h < 4 else "train"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    load_eval_keys()
    only = set(filter(None, os.environ.get("NIRNAY_SOURCES", "").split(",")))
    skip = set(filter(None, os.environ.get("NIRNAY_SKIP_SOURCES", "").split(",")))
    stats, dropped = Counter(), Counter()
    with open(OUT, "a" if os.environ.get("NIRNAY_APPEND") else "w", encoding="utf-8") as f:
        for fn in SOURCES:
            name = fn.__name__[4:]
            if (only and name not in only) or name in skip:
                continue
            rng = random.Random("%s-%s" % (SEED, name))
            CURRENT["src"] = name
            try:
                for src, st, q, y in fn(rng):
                    if leaks(st):
                        dropped[src] += 1
                        continue
                    f.write(json.dumps({"src": src, "state": st, "q": q, "y": int(y), "split": split_of(st)},
                                       ensure_ascii=False) + "\n")
                    stats[src] += 1
            except Exception as e:  # one broken source must not sink the build
                print("   FAIL %-18s %s" % (name, str(e)[:160]), flush=True)
                continue
            print("   %-18s done  (%s)" % (name, ", ".join("%s=%d" % (k, v) for k, v in stats.items() if k.startswith(name[:5]) or name in k)), flush=True)
    print("records:", sum(stats.values()), dict(stats))
    print("dropped as benchmark overlap:", dict(dropped))


if __name__ == "__main__":
    main()
