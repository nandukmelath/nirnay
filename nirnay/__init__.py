"""Nirnay: a deterministic, CPU-first decision model (Jev / Laya compatible question spec).

    import nirnay
    m = nirnay.load()                       # models/nirnay-v1
    r = m.predict({"ticket": "my card was charged twice"},
                  {"team": {"type": "choice", "instructions": "Which team should handle this?",
                            "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing"}}})
    r["answers"]["team"]["choice"]          # -> "billing"
"""
from .engine import VERSION, Nirnay, load
from .render import SpecError, render_state

__version__ = VERSION
__all__ = ["Nirnay", "load", "SpecError", "render_state", "__version__"]
