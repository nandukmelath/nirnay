"""Nirnay inference engine: Jev/Laya-compatible `predict(state, questions)`.

Per call the state is encoded once (long states in windows); question and option texts are
encoded once per schema and cached; one small fusion head scores every option of every question.
"""
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .encoder import Encoder, EncoderConfig
from .memory import Memory, signature
from .model import QTYPES, DecisionHead, DeepFusionHead, StateAdapter
from .render import SpecError, render_question, render_state

VERSION = "1.0.0"


def temp_bucket(qtype: str, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (qtype, size)


def entropy_confidence(p: np.ndarray) -> float:
    k = len(p)
    if k < 2:
        return 1.0
    ent = -float((p * np.log(np.clip(p, 1e-12, 1.0))).sum())
    return 1.0 - ent / math.log(k)


class _LRU:
    def __init__(self, cap: int):
        self.cap, self.d = cap, OrderedDict()

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        while len(self.d) > self.cap:
            self.d.popitem(last=False)


class Nirnay:
    def __init__(self, model_dir: str, threads: Optional[int] = None, cache_size: int = 8192,
                 max_state_tokens: Optional[int] = None, max_windows: int = 16):
        from safetensors.torch import load_file
        from tokenizers import Tokenizer

        self.dir = model_dir
        self.cfg = json.load(open(os.path.join(model_dir, "nirnay_config.json"), encoding="utf-8"))
        self.threads = int(threads or os.environ.get("NIRNAY_THREADS") or self.cfg.get("threads", 8))
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(True)
        ec = EncoderConfig.from_dict(self.cfg["encoder"])
        self.ecfg = ec
        self.enc = Encoder(ec)
        self.enc.load_state_dict(load_file(os.path.join(model_dir, "encoder.safetensors")))
        self.enc.eval()
        h = self.cfg["head"]
        self.split = int(self.cfg.get("split_layer") or 0)
        self.fusion = self.cfg.get("fusion", "stack")
        self.adapter = None
        if self.fusion == "deep":
            # deep fusion: encoder layers [0, split) are frozen and shared; the fused top layers live in the head
            self.head = DeepFusionHead(list(self.enc.layers)[self.split:], ec.hidden, pool=ec.pooling)
        else:
            self.head = DecisionHead(ec.hidden, h["heads"], h["ff"], h["blocks"])
        self.head.load_state_dict(load_file(os.path.join(model_dir, "head.safetensors")))
        self.head.eval()
        # split (stack) model: the state runs through encoder layers [0, split) then the trained adapter
        if self.split and self.fusion != "deep":
            self.adapter = StateAdapter(list(self.enc.layers)[self.split:], ec.ln_eps)
            self.adapter.load_state_dict(load_file(os.path.join(model_dir, "adapter.safetensors")))
            self.adapter.eval()
        mods = [self.enc, self.head] + ([self.adapter] if self.adapter is not None else [])
        for p in (p for m in mods for p in m.parameters()):
            p.requires_grad_(False)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        sp = self.cfg["special_ids"]
        self.cls_id, self.sep_id, self.pad_id = sp["cls"], sp["sep"], sp["pad"]
        self.prefix_ids = self._ids(ec.prefix) if ec.prefix else []
        lim = self.cfg["limits"]
        self.max_state_tokens = int(max_state_tokens or lim["state_tokens"])
        self.q_tokens, self.o_tokens = int(lim["question_tokens"]), int(lim["option_tokens"])
        self.max_fuse = int(lim.get("max_fused_options", 128))
        self.max_windows = max_windows
        self.window_overlap = int(lim.get("window_overlap", 64))
        self.temps = self.cfg.get("temperature_by_bucket", {})
        self.cache = _LRU(cache_size)
        self.memory = Memory()
        self.name = self.cfg.get("name", "nirnay")
        # opt-in zero-shot debiasing strength (0 = off); see _content_free_bias
        self.debias = float(os.environ.get("NIRNAY_DEBIAS", self.cfg.get("debias", 0.0)))
        self.ort = self.ort_state = None
        if os.environ.get("NIRNAY_BACKEND", "auto") != "torch":
            try:
                from .onnx_backend import OrtEncoder
                p = os.path.join(model_dir, "encoder.onnx")
                if os.path.exists(p) and self.fusion != "deep":
                    self.ort = OrtEncoder(p, self.threads)
                p = os.path.join(model_dir, "state.onnx")
                if (self.adapter is not None or self.fusion == "deep") and os.path.exists(p):
                    self.ort_state = OrtEncoder(p, self.threads)
            except Exception:  # onnxruntime missing or incompatible: stay on torch
                self.ort = self.ort_state = None
        self.backend = "onnxruntime" if (self.ort is not None or self.ort_state is not None) else "torch"

    # ------------------------------------------------------------------ tokenization / encoding
    def _ids(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def _wrap(self, content: List[int]) -> List[int]:
        return [self.cls_id] + self.prefix_ids + content + [self.sep_id]

    def _short_ids(self, text: str, budget: int) -> List[int]:
        room = budget - 2 - len(self.prefix_ids)
        return self._wrap(self._ids(text)[:max(1, room)])

    def _windows(self, text: str) -> Tuple[List[List[int]], int, bool]:
        content = self._ids(text)
        room = self.max_state_tokens - 2 - len(self.prefix_ids)
        if len(content) <= room:
            return [self._wrap(content)], len(content), False
        step = max(1, room - self.window_overlap)
        wins = []
        for s in range(0, len(content), step):
            wins.append(self._wrap(content[s:s + room]))
            if s + room >= len(content) or len(wins) >= self.max_windows:
                break
        return wins, len(content), s + room < len(content)

    @torch.no_grad()
    def _run_encoder(self, seqs: List[List[int]], batch_tokens: int = 8192, state: bool = False):
        """-> list of (hidden [L,d] float32, pooled [d]) in input order; batched by length.
        `state=True` routes through the trained state adapter on a split model."""
        out: List[Any] = [None] * len(seqs)
        order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))
        i = 0
        while i < len(order):
            L = len(seqs[order[i]])
            j = i
            while j < len(order) and len(seqs[order[j]]) * (j - i + 1) <= max(batch_tokens, L):
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            Lm = max(len(seqs[t]) for t in idx)
            ids = torch.full((len(idx), Lm), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), Lm), dtype=torch.bool)
            for r, t in enumerate(idx):
                ids[r, :len(seqs[t])] = torch.tensor(seqs[t])
                mask[r, :len(seqs[t])] = True
            if self.fusion == "deep":
                if state:   # bottom layers only; the fused top layers run in the head
                    if self.ort_state is not None:
                        H, V = self.ort_state(ids, mask)
                    else:
                        H = self.enc(ids, mask, stop_layer=self.split)
                        V = torch.zeros(H.shape[0], H.shape[-1])
                else:       # question / option text: split-layer tokens + frozen final embedding
                    Hf, H = self.enc(ids, mask, keep_layer=self.split)
                    V = self.enc.pool(Hf, mask)
            elif state and self.adapter is not None:
                if self.ort_state is not None:
                    H, V = self.ort_state(ids, mask)
                else:
                    H = self.adapter(self.enc(ids, mask, stop_layer=self.split), mask)
                    V = self.enc.pool(H, mask)
            elif self.ort is not None:
                H, V = self.ort(ids, mask)
            else:
                H = self.enc(ids, mask)
                V = self.enc.pool(H, mask)
            for r, t in enumerate(idx):
                out[t] = (H[r, :len(seqs[t])].clone(), V[r].clone())
            i = j
        return out

    def _encode_cached(self, texts: Sequence[str], budget: int):
        res: List[Any] = [None] * len(texts)
        todo, todo_idx = [], []
        for i, t in enumerate(texts):
            key = (budget, t)
            hit = self.cache.get(key)
            if hit is None:
                todo.append(self._short_ids(t, budget))
                todo_idx.append(i)
            else:
                res[i] = hit
        if todo:
            enc = self._run_encoder(todo)
            for i, e in zip(todo_idx, enc):
                self.cache.put((budget, texts[i]), e)
                res[i] = e
        return res, len(todo)

    @torch.no_grad()
    def _finish_state(self, enc_wins):
        """Window encodings -> (H, v, stream). Deep fusion runs the state through the fused layers
        here, once, and every question of the call reuses it."""
        H = torch.cat([h for h, _ in enc_wins], 0)
        if self.fusion == "deep":
            stream = self.head.state_stream(H[None], torch.ones(1, H.shape[0], dtype=torch.bool))
            return H, stream[1][0], stream
        v = torch.stack([v for _, v in enc_wins], 0).mean(0)
        return H, v / v.norm().clamp_min(1e-12), None

    def encode_state(self, state: Any):
        text = render_state(state)
        wins, n_content, truncated = self._windows(text)
        H, v, stream = self._finish_state(self._run_encoder(wins, state=True))
        return H, v, {"input_tokens": n_content, "windows": len(wins), "truncated": bool(truncated)}, stream

    # ------------------------------------------------------------------ prediction
    def _prepare(self, questions: Dict[str, Dict]):
        if not isinstance(questions, dict) or not questions:
            raise SpecError("questions must be a non-empty object of {id: question}")
        items = []
        for qid, q in questions.items():
            nq, qtext, otexts = render_question(str(qid), q)
            # canonical compute order: the answer is a function of the option *set*
            perm = sorted(range(len(otexts)), key=lambda i: (otexts[i], i))
            items.append({"nq": nq, "qtext": qtext, "otexts": [otexts[i] for i in perm], "perm": perm})
        return items

    @torch.no_grad()
    def _score(self, H, v, items, stream=None):
        qenc, nq_new = self._encode_cached([it["qtext"] for it in items], self.q_tokens)
        flat = [t for it in items for t in it["otexts"]]
        oenc, no_new = self._encode_cached(flat, self.o_tokens)
        d = H.shape[-1]
        B = len(items)
        # very wide questions: fuse only the options closest to the state, the rest get a floor logit
        groups, pos = [], 0
        for it in items:
            k = len(it["otexts"])
            ov = torch.stack([oenc[pos + i][1] for i in range(k)])
            keep = list(range(k))
            if k > self.max_fuse:
                sims = (ov @ v).tolist()
                keep = sorted(sorted(range(k), key=lambda i: (-sims[i], i))[:self.max_fuse])
            groups.append((pos, k, keep))
            pos += k
        Kmax = max(len(g[2]) for g in groups)
        Lq = max(h.shape[0] for h, _ in qenc)
        Lo = max(oenc[g[0] + i][0].shape[0] for g in groups for i in g[2])
        Q = torch.zeros(B, Lq, d)
        qm = torch.zeros(B, Lq, dtype=torch.bool)
        O = torch.zeros(B, Kmax, Lo, d)
        om = torch.zeros(B, Kmax, Lo, dtype=torch.bool)
        ok = torch.zeros(B, Kmax, dtype=torch.bool)
        ovec = torch.zeros(B, Kmax, d)
        qt = torch.tensor([QTYPES[it["nq"]["type"]] for it in items])
        for b, ((h, _), (p0, k, keep)) in enumerate(zip(qenc, groups)):
            Q[b, :h.shape[0]] = h
            qm[b, :h.shape[0]] = True
            for j, i in enumerate(keep):
                oh, ovv = oenc[p0 + i]
                O[b, j, :oh.shape[0]] = oh
                om[b, j, :oh.shape[0]] = True
                ok[b, j] = True
                ovec[b, j] = ovv
        sm = torch.ones(1, H.shape[0], dtype=torch.bool)
        if self.fusion == "deep":
            logits = self.head(H[None], sm, Q, qm, O, om, ok, qt, v[None], ovec, stream=stream).numpy()
        else:
            logits = self.head(H[None], sm, Q, qm, O, om, ok, qt, v[None], ovec).numpy()
        out = []
        for b, (p0, k, keep) in enumerate(groups):
            z = np.full(k, -np.inf, dtype=np.float64)
            z[keep] = logits[b, :len(keep)]
            if len(keep) < k:
                floor = z[keep].min() - 4.0 - math.log(k - len(keep))
                z[[i for i in range(k) if i not in set(keep)]] = floor
            out.append(z)
        return out, {"new_question_texts": nq_new, "new_option_texts": no_new}

    def _content_free_bias(self, it) -> np.ndarray:
        """Contextual calibration (Zhao et al. 2021): the logits the question gets on a content-free
        state measure how much it favours each option regardless of input. Cached per schema."""
        key = ("cf", signature(it["nq"]))
        hit = self.cache.get(key)
        if hit is None:
            zs = []
            for filler in ("N/A", "[empty]", "..."):
                H, v, _, stream = self.encode_state(filler)
                zs.append(self._score(H, v, [it], stream)[0][0])
            hit = np.mean(zs, 0)
            self.cache.put(key, hit)
        return hit

    def _calibrated(self, it, z: np.ndarray, v: Optional[torch.Tensor]) -> np.ndarray:
        nq = it["nq"]
        if self.debias and nq["type"] != "score":
            z = z - self.debias * (self._content_free_bias(it) - self._content_free_bias(it).mean())
        zz = z / float(self.temps.get(temp_bucket(nq["type"], len(z)), 1.0))
        if self.memory.items and v is not None:
            zz = self.memory.adjust(signature(nq), v.numpy().astype(np.float64), zz)
        return zz

    def _answer(self, it, z: np.ndarray, min_confidence: Optional[float], v: Optional[torch.Tensor] = None):
        nq = it["nq"]
        k = len(z)
        zz = self._calibrated(it, z, v)
        p_sorted = np.exp(zz - zz.max())
        p_sorted /= p_sorted.sum()
        p = np.empty(k)
        p[np.asarray(it["perm"])] = p_sorted          # back to the caller's option order
        best = int(np.argmax(p))
        ans_conf = round(float(p[best]), 4)
        keys = nq["keys"]
        if nq["type"] == "choice":
            a = {"type": "choice", "choice": keys[best],
                 "probabilities": {kk: round(float(x), 4) for kk, x in zip(keys, p)},
                 "confidence": round(entropy_confidence(p), 4), "answer_confidence": ans_conf}
        elif nq["type"] == "score":
            a = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4), "choice": keys[best],
                 "legend": {str(i): o[1] for i, o in enumerate(nq["opts"])},
                 "probabilities": {str(i): round(float(x), 4) for i, x in enumerate(p)},
                 "confidence": round(entropy_confidence(p), 4), "answer_confidence": ans_conf}
        else:
            a = {"type": "noul", "noul": round(float(p[1]), 4),
                 "confidence": round(max(float(p[1]), 1 - float(p[1])), 4), "answer_confidence": ans_conf}
        if min_confidence is not None:
            a["abstained"] = bool(ans_conf < float(min_confidence))
        return a

    def predict(self, state: Any, questions: Dict[str, Dict], min_confidence: Optional[float] = None) -> Dict:
        t0 = time.perf_counter()
        items = self._prepare(questions)
        H, v, use, stream = self.encode_state(state)
        zs, cache_use = self._score(H, v, items, stream)
        answers = {it["nq"]["id"]: self._answer(it, z, min_confidence, v) for it, z in zip(items, zs)}
        use.update(cache_use)
        use["output_tokens"] = 0
        return {"model": self.name, "answers": answers, "usage": use,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # Laya / Jev aliases
    system_one = predict

    def _encode_states(self, states: List[Any]):
        """Many states in shared encoder batches -> [(H, v, usage), ...]."""
        all_w, spans, metas = [], [], []
        for s in states:
            w, n, tr = self._windows(render_state(s))
            spans.append((len(all_w), len(w)))
            all_w.extend(w)
            metas.append({"input_tokens": n, "windows": len(w), "truncated": bool(tr)})
        enc = self._run_encoder(all_w, state=True)
        out = []
        for (s0, nw), meta in zip(spans, metas):
            H, v, stream = self._finish_state([enc[s0 + i] for i in range(nw)])
            out.append((H, v, meta, stream))
        return out

    def predict_batch(self, states: List[Any], questions: Dict[str, Dict],
                      min_confidence: Optional[float] = None) -> List[Dict]:
        """Same questions over many states: states are encoded in shared batches."""
        items = self._prepare(questions)
        out = []
        for H, v, meta, stream in self._encode_states(states):
            zs, cu = self._score(H, v, items, stream)
            meta.update(cu)
            meta["output_tokens"] = 0
            out.append({"model": self.name,
                        "answers": {it["nq"]["id"]: self._answer(it, z, min_confidence, v) for it, z in zip(items, zs)},
                        "usage": meta})
        return out

    def learn(self, qid: str, question: Dict, examples: Sequence[Tuple[Any, Any]]) -> Dict:
        """Attach labelled examples [(state, answer), ...] to one question schema. `answer` is an option
        label for choice, a level index for score, True/False for noul. Takes effect immediately for every
        later call that asks the same question (same instructions and options, in any order)."""
        items = self._prepare({qid: question})
        it = items[0]
        nq = it["nq"]
        k = len(it["otexts"])
        keys = nq["keys"]
        vecs, labels, base = [], [], []
        origs = []
        for _, ans in examples:
            if nq["type"] == "noul":
                origs.append(1 if (ans is True or str(ans).lower() in ("true", "yes", "1")) else 0)
            elif nq["type"] == "score":
                origs.append(int(ans))
            else:
                if str(ans) not in keys:
                    raise SpecError("example answer %r is not an option of %r" % (ans, qid))
                origs.append(keys.index(str(ans)))
        for (H, v, _, stream), orig in zip(self._encode_states([s for s, _ in examples]), origs):
            z = self._score(H, v, items, stream)[0][0]
            base.append(z / float(self.temps.get(temp_bucket(nq["type"], k), 1.0)))
            vecs.append(v.numpy())
            labels.append(it["perm"].index(orig))
        return self.memory.set(signature(nq), np.stack(vecs).astype(np.float64), np.asarray(labels),
                               np.stack(base), k)

    def raw_logits(self, state: Any, questions: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Uncalibrated logits in the caller's option order (for calibration fitting / evaluation)."""
        items = self._prepare(questions)
        H, v, _, stream = self.encode_state(state)
        zs, _ = self._score(H, v, items, stream)
        res = {}
        for it, z in zip(items, zs):
            zz = np.empty(len(z))
            zz[np.asarray(it["perm"])] = z
            res[it["nq"]["id"]] = (it["nq"]["type"], zz)
        return res


HUB_REPO = "nandukmelath/nirnay"
MODEL_FILES = ("nirnay_config.json", "encoder.safetensors", "head.safetensors", "tokenizer.json")


def default_model_dir() -> str:
    """NIRNAY_MODEL, else the folder named in models\\DEFAULT, else models\\nirnay-v1, else the Hub repo."""
    if os.environ.get("NIRNAY_MODEL"):
        return os.environ["NIRNAY_MODEL"]
    models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    pointer = os.path.join(models, "DEFAULT")
    if os.path.exists(pointer):
        name = open(pointer, encoding="utf-8").read().strip()
        if name and os.path.isdir(os.path.join(models, name)):
            return os.path.join(models, name)
    if os.path.isdir(os.path.join(models, "nirnay-v1")):
        return os.path.join(models, "nirnay-v1")
    return HUB_REPO   # pip-installed package without local models: use the published weights


def _from_hub(repo_id: str, subfolder: Optional[str]) -> str:
    from huggingface_hub import snapshot_download
    pre = (subfolder.strip("/") + "/") if subfolder else ""
    path = snapshot_download(repo_id, allow_patterns=[pre + f for f in MODEL_FILES])
    return os.path.join(path, *pre.strip("/").split("/")) if pre else path


def load(model_dir: Optional[str] = None, subfolder: Optional[str] = None, **kw) -> Nirnay:
    """nirnay.load()                                  -> the default model (local, else the Hub release)
    nirnay.load("fast") / nirnay.load("accurate")      -> v1 / v2
    nirnay.load("nandukmelath/nirnay")                 -> v2 from the Hugging Face Hub (subfolder="fast" for v1)
    nirnay.load(r"D:\\path\\to\\model_dir")             -> any local model directory"""
    if model_dir is None:
        model_dir = default_model_dir()
    elif model_dir in ("fast", "accurate"):
        models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        local = os.path.join(models, "nirnay-v1" if model_dir == "fast" else "nirnay-v2")
        if os.path.isdir(local):
            model_dir = local
        else:
            model_dir, subfolder = HUB_REPO, ("fast" if model_dir == "fast" else None)
    if not os.path.isdir(model_dir) and model_dir.count("/") == 1 and ":" not in model_dir:
        model_dir = _from_hub(model_dir, subfolder)
    elif subfolder:
        model_dir = os.path.join(model_dir, subfolder)
    return Nirnay(model_dir, **kw)
