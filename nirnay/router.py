"""Route each state to the English or the multilingual checkpoint (when both are installed).

English text goes to the English head (bge-small-en backbone, the most accurate on English);
anything else goes to the multilingual head (multilingual-e5-small backbone). The decision is a
pure function of the text, so routing is deterministic too.
"""
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

from .engine import Nirnay
from .render import render_state

_EN = set("the a an and or of to in is are was were be been it this that for on with as at by from not have has "
          "had you your we our they their he she his her i me my do does did can will would should could please "
          "what which who when where why how if but so no yes there here all any".split())
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def analyse(text: str, sample: int = 4000) -> Dict[str, Any]:
    t = text[:sample]
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return {"english": True, "ascii_letters": 1.0, "stopword_rate": 0.0, "reason": "no letters"}
    ascii_l = sum(1 for c in letters if c.isascii())
    diacritic = sum(1 for c in letters if not c.isascii() and unicodedata.name(c, "").startswith("LATIN"))
    words = [w.lower() for w in _WORD.findall(t)][:400]
    sw = sum(1 for w in words if w in _EN) / max(1, len(words))
    ar = ascii_l / len(letters)
    english = ar >= 0.97 and diacritic / len(letters) < 0.01 and (sw >= 0.08 or len(words) < 6)
    return {"english": english, "ascii_letters": round(ar, 3), "stopword_rate": round(sw, 3),
            "reason": "English Latin text" if english else "non-English or mixed text"}


class Router:
    def __init__(self, english_dir: Optional[str] = None, multilingual_dir: Optional[str] = None, **kw):
        from .engine import default_model_dir
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        english_dir = english_dir or default_model_dir()
        multilingual_dir = multilingual_dir or os.environ.get("NIRNAY_MODEL_ML") or os.path.join(root, "nirnay-v1-multilingual")
        self.en = Nirnay(english_dir, **kw)
        self.ml = Nirnay(multilingual_dir, **kw) if os.path.isdir(multilingual_dir) else None

    def pick(self, state: Any):
        a = analyse(render_state(state))
        if a["english"] or self.ml is None:
            return self.en, {"model": self.en.name, **a}
        return self.ml, {"model": self.ml.name, **a}

    def predict(self, state: Any, questions: Dict[str, Dict], min_confidence: Optional[float] = None) -> Dict:
        m, info = self.pick(state)
        r = m.predict(state, questions, min_confidence=min_confidence)
        r["routing"] = info
        return r

    def predict_batch(self, states: List[Any], questions: Dict[str, Dict], min_confidence: Optional[float] = None):
        picks = [self.pick(s) for s in states]
        out: List[Any] = [None] * len(states)
        for model in {id(p[0]): p[0] for p in picks}.values():
            idx = [i for i, p in enumerate(picks) if p[0] is model]
            for i, r in zip(idx, model.predict_batch([states[i] for i in idx], questions, min_confidence=min_confidence)):
                r["routing"] = picks[i][1]
                out[i] = r
        return out

    system_one = predict

    @property
    def threads(self):
        return self.en.threads
