# Nirnay — a deterministic, CPU-first decision model

*Nirnay* (निर्णय / നിർണയം, "decision, determination") answers typed questions about a piece of
text or JSON — **choice**, **score** (ordinal rating) and **noul** (yes/no) — with calibrated
probabilities, in one pass, without ever generating text. It speaks the same question spec and
HTTP wire protocol as TypeSafe **Jev** and Convai **Laya**, so clients written for either work
unchanged, and it is built to run fast on this HP Z840 (2x Xeon E5-2630 v3, CPU only).

See `REPORT.md` for the head-to-head numbers measured on this machine.

## Results on this machine (summary of REPORT.md)

Two models ship: **v2** (default, `nirnay.load()`) — bge-base backbone, trained overnight; **v1** (`nirnay.load("fast")`)
— bge-small backbone, 2.7x faster again.

Better than Laya here:
* **Speed** — the state is encoded once per call. v2: 146 ms for 1 question, 297 ms for 10, 696 ms for 50;
  v1: 54 / 95 / 258 ms. Laya at its fastest setting: 159 ms / 1.39 s / 6.7 s → v2 **1.1x–9.7x**, v1 **2.9x–26x** faster.
* **Footprint** — v2 527 MB, v1 155 MB on disk (Laya 614–804 MB); cold start ~7 s vs 12–16 s.
* **Option-order robustness** — reordering options never changes an answer (flip rate 0.000 vs Laya 0.14–0.19,
  up to 0.44 on banking77).
* **Accuracy wins** — banking77 (77 intents) 0.610 vs Laya ≤0.492; toxicity 0.660 vs ≤0.535; SST-5 ratings 0.492
  vs ≤0.470; RAG relevance 0.655 vs Laya-en 0.625; email spam 0.995 ≈ 0.998. Against Jev's published figures:
  AG News 0.925 vs 0.910, emotion 0.550 vs 0.480.
* **Instant adaptation** — `learn()` with 16 examples per intent: banking77 **0.853** on CPU (Laya needs RL
  fine-tuning on a GPU; Jev's independently measured figure is 0.763).

Not better (yet): against the best of Laya's three checkpoints, v2 wins 3, ties 1 and loses 8 of 12 tasks
(against Laya's default English checkpoint: 4 wins, 2 ties, 6 losses). The clear gaps are SMS spam
(0.67 vs 0.94), model routing (0.40 vs 0.66), support triage (0.42 vs 0.54), jailbreak (0.70 vs 0.81),
emotion (0.55 vs 0.60) and AG News (0.925 vs 0.953). Laya is a 421M model fully fine-tuned on GPUs;
v2 trains 29M parameters on this CPU (one overnight run).

## Quick start

```bat
RUN.bat demo                 :: one example decision
RUN.bat serve 8000           :: HTTP server, POST /v1/systemone  (Jev / Laya protocol)
RUN.bat mcp                  :: MCP stdio server for Claude Code / Codex / Hermes
RUN.bat cli "text" --choice "a,b,c" --ask "Which fits?"
RUN.bat bench                :: latency on this machine
RUN.bat test                 :: invariants
```

```python
import nirnay
m = nirnay.load()                                   # default (models\DEFAULT -> nirnay-v2); nirnay.load("fast") = v1
r = m.predict(
    {"ticket": {"subject": "Payout failing", "body": "My payouts failed for 3 days, please help ASAP"}},
    {"team":    {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"}},
     "urgent":  {"type": "noul",   "instructions": "Does the customer express urgency?"},
     "severity": {"type": "score", "instructions": "How severe is the problem?",
                  "criteria": ["minor", "moderate", "severe"]}})
r["answers"]["team"]["choice"], r["answers"]["urgent"]["noul"], r["answers"]["severity"]["score"]

m.predict_batch(list_of_states, questions)          # throughput path
m.learn("team", questions["team"], [(state, "billing"), ...])   # few-shot memory, instant, no GPU
```

Answers have the Laya/Jev shape: `choice` / `score` / `noul`, `probabilities`, `confidence`
(1 − normalised entropy), `answer_confidence` (probability of the reported answer; the calibrated
number to gate on), and `abstained` when `min_confidence` is given.

## Architecture

```
 state (text / JSON)          question "<type> question: <instructions>"      options "<label>: <description>"
        |                                  |                                       |
  encoder layers 1-8 (bge-small-en, frozen) — every text encoded on its own; question/option results cached per schema
        |                                  |                                       |
        S8 ───────────────────────────────┼───────────────────────────────────────┤
                                           ▼                                       ▼
  encoder layers 9-12, trained copies, run over [ state ; questions ; options ] with an asymmetric mask:
      state rows     attend to the state only          -> computed ONCE per call, shared by every question
      question rows  attend to state + own question
      option rows    attend to state + their question + own option   (never to other options)
                                           |
  score at each option's first token  +  w_type · cos(state, option)        (frozen-embedding prior)
                                           |
  softmax over the question's options  ->  per-bucket temperature  ->  (+ few-shot memory)
```

So the top of the network is a real cross-encoder (option and question tokens read the state at every
layer, like Laya), but the state never reads the questions — which is what lets one state encoding
serve 1 or 50 questions, and options be encoded independently of each other.

What this fixes relative to Laya (whose code was read, installed and measured here):

| Laya design | consequence | Nirnay |
|---|---|---|
| one full encoder pass **per question** (state re-encoded every time) | 10 questions cost 10x | the state is encoded **once per call**; extra questions cost a few ms |
| all options share a **192/256-token budget** in one sequence | many-label questions get truncated options (banking77: 0.43) | each option is encoded on its own: no budget, hundreds of labels |
| options scored by **position** in a joint sequence | reordering the options changes answers (15–40 % flips) | options are independent and computed in canonical order: **order cannot change the answer** |
| question/option text re-encoded on every call | wasted compute on repeated schemas | cached per schema |
| 421M-parameter ModernBERT-large backbone | ~0.5 s per question on this CPU | 33M encoder; 7.3M trained (top 4 layers + scorer) |
| ships over-confident; temperature refit advised | probabilities need re-calibration | temperatures fitted on a held-out calibration split per (type, option count) |
| adapting to a domain = RL fine-tuning on GPUs (hours) | not possible on this machine | `learn()` few-shot memory, milliseconds, leave-one-out tuned |
| long input = truncation, or sliding re-encodes | | long states are windowed once; options attend across all windows |

Training objective: log score (cross-entropy) on every question plus ranked probability score on
ordinal questions — the strictly proper scoring rules Laya's RLCD rewards via REINFORCE on noisy
logits, optimised here with exact gradients. Layers 1-8 are frozen, so their outputs are computed
once for every training text (`train\precompute.py`) and only the top 4 layers + scorer are trained
— that is what makes training feasible on this CPU. Training runs data-parallel on both sockets
(one process per CPU, gloo all-reduce, batches paired by cost so neither socket waits).

Determinism: own forward pass (no transformers modelling code at inference), fixed thread count,
`torch.use_deterministic_algorithms(True)`, canonical option order, 4-decimal outputs. The
benchmark harness checks identical answers on repeat calls and across 1/4/8/16 threads.

## Reproduce

```bat
.venv\Scripts\python bench\suites.py            :: freeze the 12 benchmark suites (Laya's own cases + 2 extra)
.venv\Scripts\python train\build_data.py        :: 220k decision questions from 29 public datasets
set NIRNAY_BACKBONE=BAAI/bge-small-en-v1.5& set NIRNAY_POOL=cls& set NIRNAY_PREFIX=none& set NIRNAY_TAG=bges
.venv\Scripts\python train\precompute.py        :: frozen encoder over every unique text (also keeps layer-8 states)
jobs\train_deep.cmd                             :: deep-fusion training, one process per CPU socket (DDP)
                                                :: -> train + calibrate + export to models\nirnay-bges-deep
.venv\Scripts\python bench\eval_suites.py nirnay:models\nirnay-v1
.venv\Scripts\python bench\eval_suites.py laya:laya          (and laya-multilingual, laya-typed-decisions)
.venv\Scripts\python bench\bench_latency_nirnay.py & .venv\Scripts\python bench\bench_latency_laya.py
.venv\Scripts\python bench\report.py            :: REPORT.md
```

Held out of training (benchmark suites): dair-ai/emotion, banking77, toxic-chat (jailbreak and
toxicity), gsm8k/mbpp routing, SST-5, SMS spam — plus every benchmark text, removed by exact match.

## Next steps (to close the accuracy gap)

1. **More training on this CPU**: v2 is the 1.6-epoch overnight run (`jobs\overnight.cmd`, dev accuracy was
   still rising slowly). The remaining gaps are mostly held-out task types (SMS spam, routing, jailbreak) that
   need more varied training data rather than more epochs.
2. **Use the Quadro K2200 for training** — needs a current NVIDIA driver (the installed one is from 2015);
   that is a system change to decide on, since ArcGIS / Global Mapper also use the card.
3. **Your own tasks** — for a fixed set of labels you care about, `learn()` with a few labelled examples per
   label is the fastest win (see the few-shot table in REPORT.md).

## Limits

* English backbone (bge-small-en). Laya routes non-English text to a multilingual checkpoint; a
  multilingual Nirnay head (multilingual-e5-small) is prepared in `jobs\multilingual.cmd` but not trained
  in this build, and `nirnay.router.Router` only uses it if `models\nirnay-v1-multilingual` exists.
* The opt-in zero-shot debiasing (`NIRNAY_DEBIAS=1`) helped toxicity / SMS spam but hurt routing and
  jailbreak badly in testing, so it is off by default.
* It decides; it does not reason step by step or write text. Multi-step reasoning squeezed into
  one question is a failure mode for every model of this kind.
* Calibration is fitted on the training mixture; refit temperatures (or use `learn`) on your own
  traffic before trusting probabilities for gating.
* Jev is a closed, paid cloud API, so it was not run here; its figures in REPORT.md are the
  published third-party ones.
