# Nirnay vs Laya vs Jev — measured on this machine

Generated 2026-10-29 22:38. Hardware: HP Z840, 2x Xeon E5-2630 v3 (16 cores / 32 threads, AVX2, no AVX-512), 128 GB DDR4, CPU only (Quadro K2200 unused).

Laya was installed from PyPI (`laya` 0.4.14) with its three published checkpoints and run through its own API with its shipped calibration. Jev is a closed, paid cloud API: it cannot run on this machine, so its column is the third-party published figure where one exists.

* **Nirnay v3.1** (`models\nirnay-v31`): v3 with content-only safety wording (v3's AI-assistant / policy phrasings made it call harmful prompts jailbreaks); same recipe: warm-started from v2, 1.25 epochs.
* **Nirnay multilingual** (`models\nirnay-multilingual`): multilingual-e5-base (100 languages), deep fusion; 385k questions: v3.1's mix plus MTOP intents (6 languages), PAWS-X, NLI in 26 languages, toxicity in 14, language identification, multilingual reviews and tweets; ~1.5 epochs on both CPU sockets (a partial run, then 1.0 epoch warm-started from it).
* **Nirnay v3** (`models\nirnay-v3`): bge-base-en-v1.5, deep fusion; 324k questions: v2's mix plus request routing, short-message spam, request safety (NVIDIA Aegis 2.0), urgency and more phrasings of every question template; warm-started from v2 for 1.25 more epochs on both CPU sockets.
* **Nirnay v2** (`models\nirnay-v2`): bge-base-en-v1.5, deep fusion (top 4 encoder layers trained), 294k questions, 1.6 epochs (overnight, both CPU sockets).
* **Nirnay v1 (fast)** (`models\nirnay-v1`): bge-small-en-v1.5, same architecture, 221k questions, 1 epoch.

## Accuracy (same 400 cases per task, same questions, seed 13)

| task | Nirnay v3.1 | Nirnay multilingual | Nirnay v3 | Nirnay v2 | Nirnay v1 (fast) | laya (en) | laya-ml | laya-td | best Laya | Jev (published) | Laya trained on it? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev.ag_news | **0.925** | 0.943 | 0.922 | 0.925 | 0.882 | 0.950 | 0.935 | 0.953 | 0.953 | 0.910 | yes |
| jev.emotion | **0.595** | 0.485 | 0.590 | 0.550 | 0.515 | 0.593 | 0.537 | 0.600 | 0.600 | 0.480 | no (held out) |
| jev.banking77_full | **0.598** | 0.477 | 0.627 | 0.610 | 0.595 | 0.425 | 0.470 | 0.492 | 0.492 | 0.870 (72 labels, n=100; independent run 0.763) | no (held out) |
| app.support_triage | **0.407** | 0.425 | 0.405 | 0.415 | 0.417 | 0.502 | 0.540 | 0.505 | 0.540 | — | yes |
| app.email_spam | **0.995** | 0.995 | 0.998 | 0.995 | 0.968 | 0.998 | 0.998 | 0.968 | 0.998 | — | yes |
| app.phishing | **0.973** | 0.975 | 0.975 | 0.975 | 0.948 | 0.980 | 0.993 | 0.940 | 0.993 | — | yes |
| app.guardrails_jailbreak | **0.647** | 0.755 | 0.527 | 0.695 | 0.787 | 0.708 | 0.805 | 0.762 | 0.805 | — | no (held out) |
| app.moderation_toxicity | **0.688** | 0.657 | 0.685 | 0.660 | 0.595 | 0.530 | 0.535 | 0.530 | 0.535 | — | no (held out) |
| app.rag_relevance | **0.680** | 0.660 | 0.682 | 0.655 | 0.613 | 0.625 | 0.672 | 0.625 | 0.672 | — | yes |
| app.model_routing_domain | **0.875** | 0.857 | 0.875 | 0.404 | 0.446 | 0.639 | 0.441 | 0.659 | 0.659 | — | no (held out) |
| extra.sst5 | **0.490** | 0.480 | 0.495 | 0.492 | 0.425 | 0.352 | 0.292 | 0.470 | 0.470 | — | no (held out) |
| extra.sms_spam | **0.790** | 0.868 | 0.767 | 0.670 | 0.600 | 0.938 | 0.807 | 0.910 | 0.938 | 0.930 (independent run, different sample) | no (held out) |

Nirnay v3.1 vs the best of Laya's three checkpoints (picked per task, after the fact): **5 wins, 2 ties (±0.005), 5 losses**.
Nirnay v3.1 vs Laya's default English checkpoint: **5 wins, 2 ties, 5 losses**.
Nirnay v3.1 vs Nirnay multilingual: 6 better, 2 same, 4 worse.
Nirnay v3.1 vs Nirnay v3: 2 better, 9 same, 1 worse.
Nirnay v3.1 vs Nirnay v2: 5 better, 4 same, 3 worse.
Nirnay v3.1 vs Nirnay v1 (fast): 9 better, 1 same, 2 worse.

`jev.banking77_full`: the 400 cases cover 10 of the 77 labels offered; every label is offered in every question.
`app.model_routing_domain`: the 399 cases cover 3 of the 6 labels offered; every label is offered in every question.

## Calibration (ECE, lower is better)

| task | Nirnay v3.1 | Nirnay multilingual | Nirnay v3 | Nirnay v2 | Nirnay v1 (fast) | best Laya |
|---|---|---|---|---|---|---|
| jev.ag_news | 0.028 | 0.022 | 0.022 | 0.035 | 0.039 | 0.032 |
| jev.emotion | 0.143 | 0.211 | 0.163 | 0.180 | 0.096 | 0.201 |
| jev.banking77_full | 0.158 | 0.135 | 0.149 | 0.118 | 0.079 | 0.095 |
| app.support_triage | 0.073 | 0.063 | 0.074 | 0.073 | 0.051 | 0.086 |
| app.email_spam | 0.003 | 0.008 | 0.004 | 0.004 | 0.015 | 0.003 |
| app.phishing | 0.021 | 0.005 | 0.016 | 0.014 | 0.022 | 0.006 |
| app.guardrails_jailbreak | 0.042 | 0.054 | 0.149 | 0.067 | 0.060 | 0.087 |
| app.moderation_toxicity | 0.104 | 0.186 | 0.097 | 0.143 | 0.221 | 0.244 |
| app.rag_relevance | 0.050 | 0.073 | 0.061 | 0.034 | 0.040 | 0.042 |
| app.model_routing_domain | 0.034 | 0.053 | 0.034 | 0.064 | 0.294 | 0.089 |
| extra.sst5 | 0.043 | 0.102 | 0.030 | 0.066 | 0.073 | 0.070 |
| extra.sms_spam | 0.137 | 0.060 | 0.177 | 0.241 | 0.296 | 0.031 |

## Robustness and determinism

| | Nirnay v3.1 | Nirnay multilingual | Nirnay v3 | Nirnay v2 | Nirnay v1 (fast) | laya (en) | laya-ml | laya-td |
|---|---|---|---|---|---|---|---|---|
| option-order flip rate (choice tasks, mean) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.148 | 0.188 | 0.137 |
| identical answers on repeat call | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| byte-identical answers across 1/4/8/16 threads | 0.987 | 0.970 | 0.990 | 0.967 | 0.987 | not measured | not measured | not measured |
| decisions changed across 1/4/8/16 threads | 0 (max prob. diff 0.0001) | 0 (max prob. diff 0.0001) | 0 (max prob. diff 0.0001) | 0 (max prob. diff 0.0001) | 0 (max prob. diff 0.0001) | not measured | not measured | not measured |
| cold start: new process -> first answer (s) | — | — | — | 7.4 | 7.1 | 11.7 | 15.8 | 11.8 |

## Option-order check (Banking77, the options in several fixed orders)

The same 400 cases with the 77 options in the suite's own (alphabetical) order and in 5 random orders (`random.Random(seed)`, seeds 1-5; one order for all cases), through each model's public batch API (`bench/order_check.py`; per-order probabilities in `bench/results/order_check/`). These cases are Laya's benchmark rows, the first 400 test messages, so they cover 10 of the 77 intents (40 messages each).

| model | orders | accuracy by order | mean | same answer in every order | probabilities averaged over the orders | right in at least one order | right in every order |
|---|---|---|---|---|---|---|---|
| Nirnay v2 | 6 | 0.610 in every order | 0.610 | 100.0% | 0.610 | 0.610 | 0.610 |
| Nirnay v3.1 | 6 | 0.598 in every order | 0.598 | 100.0% | 0.598 | 0.598 | 0.598 |
| laya-td | 6 | 0.435 to 0.603 (own order 0.492) | 0.493 | 26.8% | 0.552 | 0.695 | 0.228 |

## Latency on this CPU (Laya's own workload: a support ticket, mixed choice / yes-no questions)

p50 ms, warm, best setting for each model (Laya: eager torch at 8/16 threads or its ONNX Runtime backend).

| questions per call | Nirnay v3.1 | Nirnay multilingual | Nirnay v3 | Nirnay v2 | Nirnay v1 (fast) | laya (en) | laya-ml | laya-td | Nirnay v3.1 speed-up vs fastest Laya | Nirnay multilingual speed-up vs fastest Laya | Nirnay v3 speed-up vs fastest Laya | Nirnay v2 speed-up vs fastest Laya | Nirnay v1 (fast) speed-up vs fastest Laya |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **129.3** | 146.2 | 119.9 | 146.0 | 54.2 | 412.8 | 159.0 | 499.8 | 1.2x | 1.1x | 1.3x | 1.1x | 2.9x |
| 5 | **177.4** | 209.7 | 163.5 | 194.5 | 76.1 | 1852.9 | 676.5 | 2346.1 | 3.8x | 3.2x | 4.1x | 3.5x | 8.9x |
| 10 | **231.9** | 296.9 | 219.9 | 296.6 | 94.8 | 3912.4 | 1388.7 | 4302.4 | 6.0x | 4.7x | 6.3x | 4.7x | 14.7x |
| 50 | **654.6** | 828.9 | 632.5 | 695.7 | 258.0 | 20103.0 | 6730.2 | 20691.5 | 10.3x | 8.1x | 10.6x | 9.7x | 26.1x |

Batch throughput (10 questions per state): Nirnay v3.1 54.8 decisions/s, Nirnay multilingual 50.0 decisions/s, Nirnay v3 62.2 decisions/s, Nirnay v2 51.6 decisions/s, Nirnay v1 (fast) 143.3 decisions/s.

## Footprint

| | Nirnay v3.1 | Nirnay multilingual | Nirnay v3 | Nirnay v2 | Nirnay v1 (fast) | laya (en) | laya-ml | laya-td |
|---|---|---|---|---|---|---|---|---|
| weights on disk (MB) | 527 | 1185 | 527 | 527 | 155 | 804 | 614 | 804 |
| RAM added by loading (MB) | 612 | 1496 | 613 | 611 | 240 | 1780 | 387 | 589 |

## Few-shot adaptation (`learn()`, no training, CPU)

Labelled examples per label from the TRAIN split; evaluated on the same 400 TEST cases. Laya cannot do this without RL fine-tuning on a GPU; its zero-shot score is the reference.

| model / task | k = 0 | 2 | 4 | 8 | 16 | learn() time at 16 | Laya (best ckpt) | Jev (published) |
|---|---|---|---|---|---|---|---|---|
| Nirnay v3.1 banking77 | 0.598 | 0.635 | 0.725 | 0.825 | 0.840 | 241 s | 0.492 | 0.870 |
| Nirnay v3.1 emotion | 0.595 | 0.595 | 0.595 | 0.600 | 0.620 | 5 s | 0.600 | 0.480 |
| Nirnay multilingual banking77 | 0.477 | 0.540 | 0.605 | 0.767 | 0.745 | 294 s | 0.492 | 0.870 |
| Nirnay multilingual emotion | 0.485 | 0.485 | 0.485 | 0.485 | 0.485 | 6 s | 0.600 | 0.480 |
| Nirnay v3 banking77 | 0.627 | 0.662 | 0.738 | 0.823 | 0.850 | 238 s | 0.492 | 0.870 |
| Nirnay v3 emotion | 0.590 | 0.590 | 0.590 | 0.588 | 0.615 | 5 s | 0.600 | 0.480 |
| Nirnay v2 banking77 | 0.610 | 0.682 | 0.767 | 0.840 | 0.853 | 246 s | 0.492 | 0.870 |
| Nirnay v2 emotion | 0.550 | 0.550 | 0.550 | 0.557 | 0.570 | 5 s | 0.600 | 0.480 |
| Nirnay v1 (fast) banking77 | 0.595 | 0.662 | 0.723 | 0.775 | 0.835 | 87 s | 0.492 | 0.870 |
| Nirnay v1 (fast) emotion | 0.515 | 0.515 | 0.515 | 0.507 | 0.542 | 2 s | 0.600 | 0.480 |

## Multilingual (held out of training for every model)

* **MASSIVE** intent detection (mteb/amazon_massive_intent, test split): 51 languages, all 60 intents as options, 80 requests per language, the question in English.
* **SIB-200** topic classification (Davlan/sib200, test split): 205 languages and scripts, 7 topics, the same 24 FLORES sentences in every language.

| benchmark | Nirnay v3.1 | Nirnay multilingual | Laya multilingual |
|---|---|---|---|
| MASSIVE, mean of 51 languages | 0.076 | 0.386 | 0.217 |
| SIB-200, mean of 205 languages | 0.411 | 0.670 | 0.632 |
| MASSIVE, worst language | 0.000 (te) | 0.250 (am) | 0.037 (km) |
| SIB-200, worst language | 0.083 (bho_Deva) | 0.083 (dzo_Tibt) | 0.083 (bjn_Arab) |

MASSIVE, per language, Nirnay multilingual vs Laya multilingual: **49 wins, 1 ties, 1 losses** (of 51 languages).
SIB-200, per language, Nirnay multilingual vs Laya multilingual: **105 wins, 32 ties, 68 losses** (of 205 languages).

### MASSIVE by language

| language | Nirnay v3.1 | Nirnay multilingual | Laya multilingual |
|---|---|---|---|
| af | 0.125 | 0.388 | 0.225 |
| am | 0.000 | 0.250 | 0.062 |
| ar | 0.050 | 0.412 | 0.263 |
| az | 0.050 | 0.375 | 0.138 |
| bn | 0.000 | 0.450 | 0.163 |
| cy | 0.013 | 0.325 | 0.138 |
| da | 0.150 | 0.338 | 0.212 |
| de | 0.138 | 0.375 | 0.287 |
| el | 0.013 | 0.362 | 0.188 |
| en | 0.525 | 0.450 | 0.312 |
| es | 0.100 | 0.400 | 0.175 |
| fa | 0.013 | 0.300 | 0.263 |
| fi | 0.062 | 0.400 | 0.263 |
| fr | 0.188 | 0.475 | 0.338 |
| he | 0.000 | 0.375 | 0.212 |
| hi | 0.037 | 0.350 | 0.175 |
| hu | 0.138 | 0.375 | 0.263 |
| hy | 0.037 | 0.312 | 0.138 |
| id | 0.138 | 0.438 | 0.362 |
| is | 0.087 | 0.362 | 0.188 |
| it | 0.100 | 0.438 | 0.438 |
| ja | 0.100 | 0.400 | 0.325 |
| jv | 0.062 | 0.350 | 0.175 |
| ka | 0.000 | 0.400 | 0.087 |
| km | 0.000 | 0.300 | 0.037 |
| kn | 0.000 | 0.300 | 0.062 |
| ko | 0.000 | 0.475 | 0.237 |
| lv | 0.013 | 0.400 | 0.237 |
| ml | 0.000 | 0.463 | 0.087 |
| mn | 0.000 | 0.325 | 0.062 |
| ms | 0.050 | 0.412 | 0.225 |
| my | 0.000 | 0.362 | 0.075 |
| nb | 0.125 | 0.425 | 0.362 |
| nl | 0.188 | 0.525 | 0.275 |
| pl | 0.200 | 0.412 | 0.250 |
| pt | 0.113 | 0.362 | 0.263 |
| ro | 0.150 | 0.375 | 0.237 |
| ru | 0.087 | 0.450 | 0.300 |
| sl | 0.062 | 0.325 | 0.163 |
| sq | 0.062 | 0.338 | 0.188 |
| sv | 0.100 | 0.525 | 0.362 |
| sw | 0.062 | 0.300 | 0.138 |
| ta | 0.013 | 0.412 | 0.125 |
| te | 0.000 | 0.338 | 0.163 |
| th | 0.013 | 0.362 | 0.275 |
| tl | 0.125 | 0.362 | 0.175 |
| tr | 0.113 | 0.425 | 0.350 |
| ur | 0.087 | 0.438 | 0.100 |
| vi | 0.062 | 0.375 | 0.237 |
| zh-CN | 0.062 | 0.338 | 0.412 |
| zh-TW | 0.087 | 0.438 | 0.275 |

## Ablations (same suites)

| task | v1-small: frozen encoder + 2 fusion blocks | v1: deep fusion (bge-small) | v1 + opt-in debias | v2: deep fusion (bge-base, more data) | v3: v2 + routing, short messages, safety, urgency, phrasings | v3.1: v3 with content-only safety wording |
|---|---|---|---|---|---|---|
| jev.ag_news | 0.887 | 0.882 | 0.865 | 0.925 | 0.922 | 0.925 |
| jev.emotion | 0.420 | 0.515 | 0.507 | 0.550 | 0.590 | 0.595 |
| jev.banking77_full | 0.555 | 0.595 | 0.608 | 0.610 | 0.627 | 0.598 |
| app.support_triage | 0.403 | 0.417 | 0.278 | 0.415 | 0.405 | 0.407 |
| app.email_spam | 0.968 | 0.968 | 0.948 | 0.995 | 0.998 | 0.995 |
| app.phishing | 0.950 | 0.948 | 0.963 | 0.975 | 0.975 | 0.973 |
| app.guardrails_jailbreak | 0.680 | 0.787 | 0.468 | 0.695 | 0.527 | 0.647 |
| app.moderation_toxicity | 0.647 | 0.595 | 0.647 | 0.660 | 0.685 | 0.688 |
| app.rag_relevance | 0.547 | 0.613 | 0.512 | 0.655 | 0.682 | 0.680 |
| app.model_routing_domain | 0.471 | 0.446 | 0.078 | 0.404 | 0.875 | 0.875 |
| extra.sst5 | 0.365 | 0.425 | 0.425 | 0.492 | 0.495 | 0.490 |
| extra.sms_spam | 0.580 | 0.600 | 0.657 | 0.670 | 0.767 | 0.790 |

