"""Question spec (Jev / Laya compatible) -> texts the encoder sees.

Question spec, per question id:
  {"type": "choice", "instructions": str, "criteria": {label: description-or-None} | [label, ...]}
  {"type": "score",  "instructions": str, "criteria": [level0, level1, ...]}            (ordered, low -> high)
  {"type": "noul",   "instructions": str, "criteria": {"true": str, "false": str}?,   (yes/no -> P(true))
                     "labels": {"true": str, "false": str}?}
Optional on every type: "option_order" (accepted for wire compatibility; answers never depend on order).
"""
import json
from typing import Any, Dict, List, Tuple

TYPE_WORD = {"choice": "choice", "score": "rating", "noul": "yes/no"}
NOUL_DEFAULT = {"false": "no, the statement does not hold", "true": "yes, the statement holds"}


class SpecError(ValueError):
    pass


def _scalar(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def render_state(state: Any, max_list: int = 200) -> str:
    """str -> as is; dict/list -> one `path: value` line per leaf (paths match `field.sub[0]` references)."""
    if isinstance(state, str):
        return state
    lines: List[str] = []

    def walk(x, path):
        if isinstance(x, dict):
            if not x and path:
                lines.append("%s: {}" % path)
            for k, v in x.items():
                walk(v, "%s.%s" % (path, k) if path else str(k))
        elif isinstance(x, (list, tuple)):
            if not x and path:
                lines.append("%s: []" % path)
            for i, v in enumerate(list(x)[:max_list]):
                walk(v, "%s[%d]" % (path, i))
        else:
            s = _scalar(x)
            lines.append("%s: %s" % (path, s) if path else s)

    walk(state, "")
    return "\n".join(lines)


def _label_text(label: str) -> str:
    return str(label).replace("_", " ").strip()


def normalize_question(qid: str, q: Dict) -> Dict:
    if not isinstance(q, dict):
        raise SpecError("question %r must be an object" % qid)
    t = q.get("type")
    if t not in TYPE_WORD:
        raise SpecError("question %r: type must be choice, score or noul (got %r)" % (qid, t))
    ins = str(q.get("instructions") or "").strip()
    if not ins:
        raise SpecError("question %r: instructions are required" % qid)
    crit = q.get("criteria")
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(k): None for k in crit}
        if not isinstance(crit, dict) or len(crit) < 2:
            raise SpecError("choice question %r needs at least 2 options in criteria" % qid)
        keys = [str(k) for k in crit]
        if len(set(keys)) != len(keys):
            raise SpecError("choice question %r has duplicate option labels" % qid)
        opts = [(k, _scalar(v).strip()) for k, v in crit.items()]
    elif t == "score":
        if isinstance(crit, dict):
            crit = ["%s: %s" % (k, _scalar(v)) if v not in (None, "") else str(k) for k, v in crit.items()]
        if not isinstance(crit, (list, tuple)) or len(crit) < 2:
            raise SpecError("score question %r needs an ordered list of at least 2 levels" % qid)
        opts = [(str(i), _scalar(c).strip()) for i, c in enumerate(crit)]
        keys = [k for k, _ in opts]
    else:
        c = crit if isinstance(crit, dict) else {}
        lab = q.get("labels") if isinstance(q.get("labels"), dict) else {}
        opts = [("false", _scalar(lab.get("false") or c.get("false") or NOUL_DEFAULT["false"])),
                ("true", _scalar(lab.get("true") or c.get("true") or NOUL_DEFAULT["true"]))]
        keys = ["false", "true"]
    return {"id": qid, "type": t, "instructions": ins, "keys": keys, "opts": opts, "raw_criteria": crit}


def question_text(nq: Dict) -> str:
    return "%s question: %s" % (TYPE_WORD[nq["type"]], nq["instructions"])


def option_texts(nq: Dict) -> List[str]:
    t, opts = nq["type"], nq["opts"]
    if t == "choice":
        return ["%s: %s" % (_label_text(k), d) if d else _label_text(k) for k, d in opts]
    if t == "score":
        top = len(opts) - 1
        return ["level %d of %d: %s" % (i, top, d) if d else "level %d of %d" % (i, top)
                for i, (_, d) in enumerate(opts)]
    return ["no: %s" % opts[0][1], "yes: %s" % opts[1][1]]


def render_question(qid: str, q: Dict) -> Tuple[Dict, str, List[str]]:
    nq = normalize_question(qid, q)
    return nq, question_text(nq), option_texts(nq)
