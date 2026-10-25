"""Few-shot memory: labelled examples for a question schema, used at prediction time.

No gradient training: each example is stored as its frozen state embedding plus the head's own
calibrated logits. At prediction the kNN evidence over stored examples (a soft prototype per
option) is added to the head's logits:  z = z_head / T + lam * knn_logprob(tau).
(lam, tau) are chosen by leave-one-out log-likelihood on the stored examples, so a memory that
does not help (lam = 0) changes nothing.
"""
import hashlib
import json
import math
from typing import Dict, List

import numpy as np

LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TAUS = (0.02, 0.05, 0.1)


def signature(nq: Dict) -> str:
    opts = [[k, d] for k, d in nq["opts"]]
    key = [nq["type"], nq["instructions"], opts if nq["type"] == "score" else sorted(opts)]
    return hashlib.sha1(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def knn_logprob(sims: np.ndarray, labels: np.ndarray, k: int, tau: float) -> np.ndarray:
    """sims [n] (cosine to stored examples), labels [n] -> log p_knn over k options (smoothed)."""
    w = np.exp((sims - sims.max()) / tau)
    mass = np.full(k, 1e-3)
    np.add.at(mass, labels, w)
    return np.log(mass / mass.sum())


class Memory:
    def __init__(self):
        self.items: Dict[str, Dict] = {}

    def set(self, sig: str, vecs: np.ndarray, labels: np.ndarray, base: np.ndarray, k: int) -> Dict:
        """vecs [n,d] L2-normalised, labels [n] option index, base [n,k] calibrated head logits."""
        best = (-1e18, 0.0, TAUS[0])
        sims_all = vecs @ vecs.T
        n = len(labels)
        for tau in TAUS:
            ev = np.zeros((n, k))
            for i in range(n):
                m = np.ones(n, bool)
                m[i] = False
                if m.sum() == 0:
                    continue
                ev[i] = knn_logprob(sims_all[i, m], labels[m], k, tau)
            for lam in LAMS:
                z = base + lam * ev
                z = z - z.max(1, keepdims=True)
                ll = float((z[np.arange(n), labels] - np.log(np.exp(z).sum(1))).sum())
                if ll > best[0] + 1e-9:
                    best = (ll, lam, tau)
        self.items[sig] = {"vecs": vecs.astype(np.float32), "labels": labels.astype(np.int64), "k": k,
                           "lam": best[1], "tau": best[2], "n": n, "loo_ll": best[0] / max(1, n)}
        return {"examples": n, "lam": best[1], "tau": best[2], "loo_nll": round(-best[0] / max(1, n), 4)}

    def adjust(self, sig: str, v: np.ndarray, z_cal: np.ndarray) -> np.ndarray:
        it = self.items.get(sig)
        if it is None or it["lam"] == 0.0 or len(z_cal) != it["k"]:
            return z_cal
        return z_cal + it["lam"] * knn_logprob(it["vecs"] @ v, it["labels"], it["k"], it["tau"])

    def forget(self, sig: str = None):
        if sig is None:
            self.items.clear()
        else:
            self.items.pop(sig, None)
