"""Pick the frozen backbone: CPU latency + representation quality on NON-benchmark data.

The datasets here (CLINC150, GoEmotions, SST-2, DBpedia-14, Yahoo Answers, TweetEval emotion)
are deliberately not in the Laya-vs-Nirnay benchmark suites, so choosing the encoder here
does not leak into the head-to-head numbers.

For each encoder:
  * latency of one ~120-token state at 8 and 16 threads (fp32, eager torch)
  * zero-shot: cosine(state, label-name) argmax accuracy
  * linear probe: multinomial logistic regression on 64 examples/label (or all) -> test accuracy

  python train\\bakeoff_encoders.py
"""
import json
import os
import random
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "bench", "results", "bakeoff_encoders.json")
SEED = 7
N_TEST = 500

ENCODERS = {
    "intfloat/multilingual-e5-small": {"prefix": "query: ", "pool": "mean"},
    "ibm-granite/granite-embedding-107m-multilingual": {"prefix": "", "pool": "cls"},
    "BAAI/bge-small-en-v1.5": {"prefix": "", "pool": "cls"},
    "ibm-granite/granite-embedding-small-english-r2": {"prefix": "", "pool": "cls"},
    "intfloat/multilingual-e5-base": {"prefix": "query: ", "pool": "mean"},
}

STATE_EN = ("Payout failing. Hi, my Stripe payouts have failed for 3 days and I am losing sales. "
            "Please help ASAP. " * 6)


def tasks():
    rng = random.Random(SEED)
    out = {}

    d = load_dataset("clinc/clinc_oos", "plus")
    names = d["train"].features["intent"].names
    tr = [(r["text"], r["intent"]) for r in d["train"] if names[r["intent"]] != "oos"]
    te = [(r["text"], r["intent"]) for r in d["test"] if names[r["intent"]] != "oos"]
    out["clinc150"] = (names, tr, te)

    d = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = d["train"].features["labels"].feature.names
    tr = [(r["text"], r["labels"][0]) for r in d["train"] if len(r["labels"]) == 1]
    te = [(r["text"], r["labels"][0]) for r in d["test"] if len(r["labels"]) == 1]
    out["go_emotions"] = (names, tr, te)

    d = load_dataset("stanfordnlp/sst2")
    out["sst2"] = (["negative", "positive"], [(r["sentence"], r["label"]) for r in d["train"]],
                   [(r["sentence"], r["label"]) for r in d["validation"]])

    d = load_dataset("fancyzhx/dbpedia_14")
    names = d["train"].features["label"].names
    out["dbpedia14"] = (names, [(r["content"], r["label"]) for r in d["train"].select(range(0, 560000, 40))],
                        [(r["content"], r["label"]) for r in d["test"].select(range(0, 70000, 50))])

    d = load_dataset("community-datasets/yahoo_answers_topics")
    names = d["train"].features["topic"].names
    fmt = lambda r: (r["question_title"] + " " + (r["question_content"] or "")).strip()
    out["yahoo10"] = (names, [(fmt(r), r["topic"]) for r in d["train"].select(range(0, 1400000, 200))],
                      [(fmt(r), r["topic"]) for r in d["test"].select(range(0, 60000, 50))])

    d = load_dataset("cardiffnlp/tweet_eval", "emotion")
    names = d["train"].features["label"].names
    out["tweet_emotion"] = (names, [(r["text"], r["label"]) for r in d["train"]],
                            [(r["text"], r["label"]) for r in d["test"]])

    final = {}
    for k, (names, tr, te) in out.items():
        rng.shuffle(tr)
        rng.shuffle(te)
        # linear-probe train set: up to 64 per label
        per = {}
        sub = []
        for t, y in tr:
            if per.get(y, 0) < 64:
                per[y] = per.get(y, 0) + 1
                sub.append((t, y))
        clean = [n.replace("_", " ").replace("&", "and") for n in names]
        final[k] = (clean, sub, te[:N_TEST])
    return final


class Emb:
    def __init__(self, name, spec):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=torch.float32).eval()
        self.prefix, self.pool = spec["prefix"], spec["pool"]

    @torch.no_grad()
    def encode(self, texts, bs=64, max_len=256):
        outs = []
        order = np.argsort([len(t) for t in texts])
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            batch = [self.prefix + texts[j] for j in idx]
            enc = self.tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            h = self.model(**enc).last_hidden_state
            if self.pool == "mean":
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            else:
                v = h[:, 0]
            outs.append((idx, F.normalize(v, dim=-1)))
        res = torch.zeros(len(texts), outs[0][1].shape[-1])
        for idx, v in outs:
            res[torch.as_tensor(idx)] = v
        return res


def linear_probe(xtr, ytr, xte, yte, n_cls, epochs=300):
    torch.manual_seed(0)
    W = torch.zeros(xtr.shape[1], n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=epochs, line_search_fn="strong_wolfe")
    ytr_t = torch.as_tensor(ytr)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xtr @ W * 10 + b, ytr_t) + 1e-3 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pred = (xte @ W * 10 + b).argmax(-1).numpy()
    return float((pred == np.asarray(yte)).mean())


def latency(emb, threads):
    torch.set_num_threads(threads)
    enc = emb.tok([emb.prefix + STATE_EN], truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        for _ in range(3):
            emb.model(**enc)
        ts = []
        for _ in range(15):
            t = time.perf_counter()
            emb.model(**enc)
            ts.append((time.perf_counter() - t) * 1000)
    return {"tokens": int(enc["input_ids"].shape[1]), "p50_ms": round(float(np.median(ts)), 2)}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    data = tasks()
    print({k: (len(v[0]), len(v[1]), len(v[2])) for k, v in data.items()}, flush=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    only = os.environ.get("BAKEOFF_ONLY")
    for name, spec in ENCODERS.items():
        if (only and only not in name) or (name in res and not only):
            continue
        print("==", name, flush=True)
        emb = Emb(name, spec)
        r = {"params_m": round(sum(p.numel() for p in emb.model.parameters()) / 1e6, 1),
             "latency": {str(t): latency(emb, t) for t in (8, 16)}}
        torch.set_num_threads(16)
        zs, lp = {}, {}
        for k, (names, tr, te) in data.items():
            ye = [y for _, y in te]
            xe = emb.encode([t for t, _ in te])
            xl = emb.encode(names)
            zs[k] = round(float(((xe @ xl.T).argmax(-1).numpy() == np.asarray(ye)).mean()), 4)
            xt = emb.encode([t for t, _ in tr])
            lp[k] = round(linear_probe(xt, [y for _, y in tr], xe, ye, len(names)), 4)
            print("   %-14s zero-shot %.3f  probe %.3f" % (k, zs[k], lp[k]), flush=True)
        r["zero_shot"] = zs
        r["probe"] = lp
        r["zero_shot_mean"] = round(float(np.mean(list(zs.values()))), 4)
        r["probe_mean"] = round(float(np.mean(list(lp.values()))), 4)
        print("   latency", r["latency"], "zs_mean", r["zero_shot_mean"], "probe_mean", r["probe_mean"], flush=True)
        res[name] = r
        json.dump(res, open(OUT, "w"), indent=2)
        del emb


if __name__ == "__main__":
    main()
