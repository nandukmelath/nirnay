"""Nirnay demo Space: typed decisions (choice / yes-no / rating) with calibrated probabilities, on CPU."""
import json
import os
import time

import gradio as gr

import nirnay

REPO = "nandukmelath/nirnay"
THREADS = int(os.environ.get("NIRNAY_THREADS", "2"))
MODELS = {}
NAMES = ["v2 — accurate (default)", "v1 — fast"]


def get(name):
    if name not in MODELS:
        src = os.environ.get("NIRNAY_DEMO_REPO", REPO)   # a local folder for testing, the Hub repo on the Space
        MODELS[name] = nirnay.load(src, subfolder=None if name.startswith("v2") else "fast", threads=THREADS)
    return MODELS[name]


for _n in NAMES:   # download + warm both at startup
    get(_n).predict("warm up", {"q": {"type": "noul", "instructions": "Is this a test?"}})


def parse_options(text):
    crit = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            k, d = line.split(":", 1)
            crit[k.strip()] = d.strip() or None
        else:
            crit[line] = None
    return crit


def parse_state(text):
    text = (text or "").strip()
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def decide(text, qtype, question, options, model_name):
    if not (text or "").strip() or not (question or "").strip():
        raise gr.Error("Enter a text and a question.")
    if qtype == "choice":
        crit = parse_options(options)
        if len(crit) < 2:
            raise gr.Error("Give at least two options, one per line (optionally 'label: description').")
        q = {"type": "choice", "instructions": question, "criteria": crit}
    elif qtype == "rating":
        levels = [l.strip() for l in (options or "").splitlines() if l.strip()]
        if len(levels) < 2:
            raise gr.Error("Give at least two rating levels, lowest first, one per line.")
        q = {"type": "score", "instructions": question, "criteria": levels}
    else:
        q = {"type": "noul", "instructions": question}
    m = get(model_name)
    t = time.perf_counter()
    r = m.predict(parse_state(text), {"q": q})
    ms = (time.perf_counter() - t) * 1000
    a = r["answers"]["q"]
    if qtype == "choice":
        probs = a["probabilities"]
        headline = "**%s** (p = %.3f)" % (a["choice"], a["answer_confidence"])
    elif qtype == "rating":
        probs = {lvl: a["probabilities"][str(i)] for i, lvl in enumerate(q["criteria"])}
        headline = "**%s** — expected level %.2f of %d" % (q["criteria"][int(a["choice"])], a["score"], len(q["criteria"]) - 1)
    else:
        probs = {"yes": a["noul"], "no": round(1 - a["noul"], 4)}
        headline = "**%s** (P(yes) = %.3f)" % ("yes" if a["noul"] >= 0.5 else "no", a["noul"])
    info = "%s · %.0f ms on this Space's 2-vCPU CPU · model %s" % (headline, ms, m.name)
    return probs, info, r


MULTI_STATE = json.dumps({"ticket": {"subject": "Payout failing", "customer_plan": "Pro",
                                     "messages": [{"from": "customer", "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. "
                                                                               "This is the second time this month. Please help ASAP or I will cancel."}]}}, indent=2)
MULTI_Q = json.dumps({
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "payments, payouts, invoices", "technical": "bugs and integrations", "sales": "pricing and plans"}},
    "urgent": {"type": "noul", "instructions": "Is this ticket urgent?"},
    "churn_risk": {"type": "noul", "instructions": "Is the customer threatening to cancel?"},
    "sentiment": {"type": "score", "instructions": "How does the customer feel?",
                  "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]},
    "priority": {"type": "score", "instructions": "What priority should this ticket get?",
                 "criteria": ["low: can wait", "medium: handle this week", "high: handle today", "critical: drop everything"]},
    "repeat_issue": {"type": "noul", "instructions": "Has this problem happened before?"}}, indent=2)


def decide_many(state_text, questions_text, model_name):
    try:
        questions = json.loads(questions_text)
    except ValueError as e:
        raise gr.Error("Questions must be JSON: %s" % e)
    m = get(model_name)
    t = time.perf_counter()
    r = m.predict(parse_state(state_text), questions)
    ms = (time.perf_counter() - t) * 1000
    summary = {qid: (a.get("choice") if a["type"] == "choice" else
                     round(a["score"], 2) if a["type"] == "score" else ("yes" if a["noul"] >= 0.5 else "no"))
               for qid, a in r["answers"].items()}
    info = "%d questions answered in **%.0f ms** with one read of the text (Laya re-reads it once per question)." % (len(questions), ms)
    return summary, info, r


EXAMPLES = [
    ["Hi, I was charged twice for my subscription this month and I want a refund ASAP.", "choice",
     "Which team should handle this ticket?",
     "billing: payments, charges, refunds\ntechnical: bugs, crashes, login problems\nsales: pricing, plans, quotes\naccount: profile, password, settings"],
    ["WINNER!! You have been selected for a cash prize. Reply YES now to claim before midnight.", "yes / no",
     "Is this message spam?", ""],
    ["You are an idiot and nobody wants you in this group.", "yes / no",
     "Is this message toxic or insulting?", ""],
    ["The food was fine but the service was painfully slow and nobody apologised.", "rating",
     "How positive is this review?", "very negative\nnegative\nneutral\npositive\nvery positive"],
    ["I ordered a new card two weeks ago and it still hasn't arrived.", "choice",
     "Which banking request is this?",
     "card arrival\ncard lost or stolen\ncard declined\nexchange rate\ntransfer failed\ntop up failed\nchange PIN\nrefund request\naccount closure\nwrong amount charged\ncash withdrawal issue\nverify identity"],
    ["The new firmware made the drone drift left during hover; calibration did not help.", "choice",
     "What is this text about?", "world news\nsports\nbusiness\nscience and technology"],
]

with gr.Blocks(title="Nirnay — CPU decision model") as demo:
    gr.Markdown(
        "# ⚖️ Nirnay — typed decisions on CPU\n"
        "Ask **choice**, **yes / no** or **rating** questions about any text or JSON and get calibrated probabilities "
        "in one pass — no text generation, deterministic, option order can't change the answer. "
        "Same question format as Laya / Jev (`/v1/systemone`). "
        "[Model card & benchmarks](https://huggingface.co/%s) · "
        "[Code on GitHub](https://github.com/nandukmelath/nirnay)" % REPO)
    with gr.Tab("Decide"):
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text (or JSON)", lines=5)
                qtype = gr.Radio(["choice", "yes / no", "rating"], value="choice", label="Question type")
                question = gr.Textbox(label="Question", value="Which team should handle this ticket?")
                options = gr.Textbox(label="Options — one per line, optional 'label: description' (choice: labels, rating: levels low→high)", lines=6)
                model_name = gr.Dropdown(NAMES, value=NAMES[0], label="Model")
                go = gr.Button("Decide", variant="primary")
            with gr.Column():
                info = gr.Markdown()
                probs = gr.Label(label="Probabilities", num_top_classes=8)
                raw = gr.JSON(label="Full response (Laya / Jev format)")
        go.click(decide, [text, qtype, question, options, model_name], [probs, info, raw])
        gr.Examples(EXAMPLES, [text, qtype, question, options], label="Try an example (then press Decide)")
    with gr.Tab("Many questions, one read"):
        with gr.Row():
            with gr.Column():
                st = gr.Code(MULTI_STATE, language="json", label="State (text or JSON)")
                qs = gr.Code(MULTI_Q, language="json", label="Questions (Laya / Jev spec)")
                model2 = gr.Dropdown(NAMES, value=NAMES[0], label="Model")
                go2 = gr.Button("Answer all", variant="primary")
            with gr.Column():
                info2 = gr.Markdown()
                summ = gr.JSON(label="Answers")
                raw2 = gr.JSON(label="Full response")
        go2.click(decide_many, [st, qs, model2], [summ, info2, raw2])

if __name__ == "__main__":
    demo.launch()
