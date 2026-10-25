# Nirnay vs Laya vs Jev — measured on this machine

Generated 2026-10-25 03:29. Hardware: HP Z840, 2x Xeon E5-2630 v3 (16 cores / 32 threads, AVX2, no AVX-512), 128 GB DDR4, CPU only (Quadro K2200 unused).

Laya was installed from PyPI (`laya` 0.4.14) with its three published checkpoints and run through its own API with its shipped calibration. Jev is a closed, paid cloud API: it cannot run on this machine, so its column is the third-party published figure where one exists.

* **Nirnay v2** (default, `models\nirnay-v2`): bge-base-en-v1.5, deep fusion (top 4 encoder layers trained), 294k questions, 1.6 epochs (overnight, both CPU sockets).
* **Nirnay v1** (fast, `models\nirnay-v1`): bge-small-en-v1.5, same architecture, 221k questions, 1 epoch.

## Accuracy (same 400 cases per task, same questions, seed 13)

| task | Nirnay v2 (default) | Nirnay v1 (fast) | laya (en) | laya-ml | laya-td | best Laya | Jev (published) | Laya trained on it? |
|---|---|---|---|---|---|---|---|---|
| jev.ag_news | **0.925** | 0.882 | 0.950 | 0.935 | 0.953 | 0.953 | 0.910 | yes |
| jev.emotion | **0.550** | 0.515 | 0.593 | 0.537 | 0.600 | 0.600 | 0.480 | no (held out) |
| jev.banking77_full | **0.610** | 0.595 | 0.425 | 0.470 | 0.492 | 0.492 | 0.870 (72 labels, n=100; independent run 0.763) | no (held out) |
| app.support_triage | **0.415** | 0.417 | 0.502 | 0.540 | 0.505 | 0.540 | — | yes |
| app.email_spam | **0.995** | 0.968 | 0.998 | 0.998 | 0.968 | 0.998 | — | yes |
| app.phishing | **0.975** | 0.948 | 0.980 | 0.993 | 0.940 | 0.993 | — | yes |
| app.guardrails_jailbreak | **0.695** | 0.787 | 0.708 | 0.805 | 0.762 | 0.805 | — | no (held out) |
| app.moderation_toxicity | **0.660** | 0.595 | 0.530 | 0.535 | 0.530 | 0.535 | — | no (held out) |
| app.rag_relevance | **0.655** | 0.613 | 0.625 | 0.672 | 0.625 | 0.672 | — | yes |
| app.model_routing_domain | **0.404** | 0.446 | 0.639 | 0.441 | 0.659 | 0.659 | — | no (held out) |
| extra.sst5 | **0.492** | 0.425 | 0.352 | 0.292 | 0.470 | 0.470 | — | no (held out) |
| extra.sms_spam | **0.670** | 0.600 | 0.938 | 0.807 | 0.910 | 0.938 | 0.930 (independent run, different sample) | no (held out) |

Nirnay v2 vs the best of Laya's three checkpoints (picked per task, after the fact): **3 wins, 1 ties (±0.005), 8 losses**.
Nirnay v2 vs Laya's default English checkpoint: **4 wins, 2 ties, 6 losses**.
Nirnay v2 vs Nirnay v1: 9 better, 1 same, 2 worse.

## Calibration (ECE, lower is better)

| task | Nirnay v2 | Nirnay v1 | best Laya |
|---|---|---|---|
| jev.ag_news | 0.035 | 0.039 | 0.032 |
| jev.emotion | 0.180 | 0.096 | 0.201 |
| jev.banking77_full | 0.118 | 0.079 | 0.095 |
| app.support_triage | 0.073 | 0.051 | 0.086 |
| app.email_spam | 0.004 | 0.015 | 0.003 |
| app.phishing | 0.014 | 0.022 | 0.006 |
| app.guardrails_jailbreak | 0.067 | 0.060 | 0.087 |
| app.moderation_toxicity | 0.143 | 0.221 | 0.244 |
| app.rag_relevance | 0.034 | 0.040 | 0.042 |
| app.model_routing_domain | 0.064 | 0.294 | 0.089 |
| extra.sst5 | 0.066 | 0.073 | 0.070 |
| extra.sms_spam | 0.241 | 0.296 | 0.031 |

## Robustness and determinism

| | Nirnay v2 (default) | Nirnay v1 (fast) | laya (en) | laya-ml | laya-td |
|---|---|---|---|---|---|
| option-order flip rate (choice tasks, mean) | 0.000 | 0.000 | 0.148 | 0.188 | 0.137 |
| identical answers on repeat call | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| byte-identical answers across 1/4/8/16 threads | 0.967 | 0.987 | not measured | not measured | not measured |
| decisions changed across 1/4/8/16 threads | 0 (max prob. diff 0.0001) | 0 (max prob. diff 0.0001) | | | |
| cold start: new process -> first answer (s) | 7.4 | 7.1 | 11.7 | 15.8 | 11.8 |

## Latency on this CPU (Laya's own workload: a support ticket, mixed choice / yes-no questions)

p50 ms, warm, best setting for each model (Laya: eager torch at 8/16 threads or its ONNX Runtime backend).

| questions per call | Nirnay v2 | Nirnay v1 | laya (en) | laya-ml | laya-td | v2 speed-up vs fastest Laya | v1 speed-up |
|---|---|---|---|---|---|---|---|
| 1 | **146.0** | 54.2 | 412.8 | 159.0 | 499.8 | 1.1x | 2.9x |
| 5 | **194.5** | 76.1 | 1852.9 | 676.5 | 2346.1 | 3.5x | 8.9x |
| 10 | **296.6** | 94.8 | 3912.4 | 1388.7 | 4302.4 | 4.7x | 14.7x |
| 50 | **695.7** | 258.0 | 20103.0 | 6730.2 | 20691.5 | 9.7x | 26.1x |

Batch throughput (10 questions per state): Nirnay v2 51.6 decisions/s, Nirnay v1 143.3 decisions/s.

## Footprint

| | Nirnay v2 (default) | Nirnay v1 (fast) | laya (en) | laya-ml | laya-td |
|---|---|---|---|---|---|
| weights on disk (MB) | 527 | 155 | 804 | 614 | 804 |
| RAM added by loading (MB) | 611 | 240 | 1780 | 387 | 589 |

## Few-shot adaptation (`learn()`, no training, CPU)

Labelled examples per label from the TRAIN split; evaluated on the same 400 TEST cases. Laya cannot do this without RL fine-tuning on a GPU; its zero-shot score is the reference.

| model / task | k = 0 | 2 | 4 | 8 | 16 | learn() time at 16 | Laya (best ckpt) | Jev (published) |
|---|---|---|---|---|---|---|---|---|
| v2 banking77 | 0.610 | 0.682 | 0.767 | 0.840 | 0.853 | 246 s | 0.492 | 0.870 |
| v2 emotion | 0.550 | 0.550 | 0.550 | 0.557 | 0.570 | 5 s | 0.600 | 0.480 |
| v1 banking77 | 0.595 | 0.662 | 0.723 | 0.775 | 0.835 | 87 s | 0.492 | 0.870 |
| v1 emotion | 0.515 | 0.515 | 0.515 | 0.507 | 0.542 | 2 s | 0.600 | 0.480 |

## Ablations (same suites)

| task | v1-small: frozen encoder + 2 fusion blocks | v1: deep fusion (bge-small) | v1 + opt-in debias | v2: deep fusion (bge-base, more data) |
|---|---|---|---|---|
| jev.ag_news | 0.887 | 0.882 | 0.865 | 0.925 |
| jev.emotion | 0.420 | 0.515 | 0.507 | 0.550 |
| jev.banking77_full | 0.555 | 0.595 | 0.608 | 0.610 |
| app.support_triage | 0.403 | 0.417 | 0.278 | 0.415 |
| app.email_spam | 0.968 | 0.968 | 0.948 | 0.995 |
| app.phishing | 0.950 | 0.948 | 0.963 | 0.975 |
| app.guardrails_jailbreak | 0.680 | 0.787 | 0.468 | 0.695 |
| app.moderation_toxicity | 0.647 | 0.595 | 0.647 | 0.660 |
| app.rag_relevance | 0.547 | 0.613 | 0.512 | 0.655 |
| app.model_routing_domain | 0.471 | 0.446 | 0.078 | 0.404 |
| extra.sst5 | 0.365 | 0.425 | 0.425 | 0.492 |
| extra.sms_spam | 0.580 | 0.600 | 0.657 | 0.670 |

