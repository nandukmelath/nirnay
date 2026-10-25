"""Command line.

  python -m nirnay.cli "I was charged twice, refund me" --choice billing,technical,sales --ask "Which team?"
  python -m nirnay.cli --state-file ticket.json --questions questions.json
  python -m nirnay.cli --batch states.jsonl --questions questions.json > answers.jsonl
"""
import argparse
import json
import sys

from .engine import load


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nirnay", description="Deterministic decision model (Jev/Laya question spec)")
    ap.add_argument("state", nargs="?", help="state text (or use --state-file / --batch)")
    ap.add_argument("--state-file", help="JSON or text file holding one state")
    ap.add_argument("--batch", help="JSONL file, one state per line -> one JSON answer per line")
    ap.add_argument("--questions", help="JSON file: {id: question}")
    ap.add_argument("--ask", default="Which option fits best?", help="instructions for --choice / --yesno")
    ap.add_argument("--choice", help="comma-separated labels for a quick choice question")
    ap.add_argument("--yesno", action="store_true", help="ask --ask as a yes/no question")
    ap.add_argument("--min-confidence", type=float)
    ap.add_argument("--model", help="model directory")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args(argv)

    if a.questions:
        questions = json.load(open(a.questions, encoding="utf-8"))
    elif a.choice:
        questions = {"answer": {"type": "choice", "instructions": a.ask,
                                "criteria": {c.strip(): None for c in a.choice.split(",") if c.strip()}}}
    elif a.yesno:
        questions = {"answer": {"type": "noul", "instructions": a.ask}}
    else:
        ap.error("give --questions, --choice or --yesno")
    m = load(a.model, threads=a.threads)
    if a.batch:
        states = [json.loads(l) for l in open(a.batch, encoding="utf-8") if l.strip()]
        for r in m.predict_batch(states, questions, min_confidence=a.min_confidence):
            sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")
        return 0
    if a.state_file:
        raw = open(a.state_file, encoding="utf-8").read()
        try:
            state = json.loads(raw)
        except ValueError:
            state = raw
    elif a.state is not None:
        state = a.state
    else:
        state = sys.stdin.read()
    print(json.dumps(m.predict(state, questions, min_confidence=a.min_confidence), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
