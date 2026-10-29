@echo off
rem Nirnay-multilingual: records_ml (v3 sources + MTOP, PAWS-X, 26-language NLI, 14-language toxicity, language ID,
rem multilingual reviews and tweets) on intfloat/multilingual-e5-base, deep fusion, 1.2 epochs on both sockets,
rem then the 12 English suites, MASSIVE + SIB-200, determinism, latency, few-shot.
rem Progress: D:\Nirnay\train\ml_status.log
set ST=D:\Nirnay\train\ml_status.log
set PY=D:\Nirnay\.venv\Scripts\python.exe
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set USE_TF=0
cd /d D:\Nirnay
echo %date% %time% START ml >> %ST%
if exist D:\Nirnay\train\ML_FROM_KAGGLE (
  echo %date% %time% multilingual model already trained on Kaggle GPUs: evaluation only >> %ST%
  goto evals
)

echo %date% %time% stage 1/7 build records_ml >> %ST%
rem online for the build: a caller's HF_HUB_OFFLINE=1 (v31.cmd sets it for training) made five sources fail on 10-28
set HF_HUB_OFFLINE=
set NIRNAY_RECORDS_OUT=D:\Nirnay\train\data\records_ml.jsonl
set NIRNAY_SRC_SCALE=ag_news=2.86,support_tickets=2.78,enron=3.14,phishing=2,msmarco=2.86,mtop=1.5
set NIRNAY_SKIP_SOURCES=
start "" /b /wait /affinity FFFF %PY% -u train\build_data.py > train\build_data_ml.log 2>&1
findstr /b /c:"records:" train\build_data_ml.log >nul || goto fail1
rem a failed source must stop the run, not shrink the training set silently
findstr /c:"   FAIL " train\build_data_ml.log >nul && goto fail1
set NIRNAY_RECORDS_OUT=
set NIRNAY_SRC_SCALE=

echo %date% %time% stage 2/7 precompute multilingual-e5-base (2 shards) >> %ST%
if exist D:\Nirnay\train\cache\me5b rmdir /s /q D:\Nirnay\train\cache\me5b
set NIRNAY_BACKBONE=intfloat/multilingual-e5-base
set NIRNAY_POOL=mean
set NIRNAY_PREFIX=
set NIRNAY_TAG=me5b
set NIRNAY_THREADS=8
set NIRNAY_RECORDS=D:\Nirnay\train\data\records_ml.jsonl
set NIRNAY_TRAIN_STATE_TOKENS=256
set NIRNAY_SAVE_MID=8
set NIRNAY_STATE_MID_ONLY=1
set NIRNAY_SHARD=1/2
start "" /b /affinity FFFF0000 %PY% -u train\precompute.py > train\precompute_me5b_s1.log 2>&1
set NIRNAY_SHARD=0/2
start "" /b /wait /affinity FFFF %PY% -u train\precompute.py > train\precompute_me5b.log 2>&1
findstr /c:"done ->" train\precompute_me5b.log >nul || goto fail2
set NIRNAY_SHARD=
timeout /t 30 /nobreak >nul

echo %date% %time% stage 3/7 train deep fusion (DDP, 1.0 epoch, warm start from the 10-28 partial run) >> %ST%
set HF_HUB_OFFLINE=1
set NIRNAY_RUN=me5b-deep2
set NIRNAY_INIT=D:\Nirnay\train\runs\me5b-partial\model_best.pt
set NIRNAY_FUSION=deep
set NIRNAY_SPLIT=8
set NIRNAY_BATCH=64
set NIRNAY_EPOCHS=1.0
set NIRNAY_LR=2e-4
set NIRNAY_ADAPTER_LR=5e-5
set NIRNAY_WARMUP=300
set NIRNAY_EVAL_EVERY=400
set NIRNAY_MAX_STATE=256
set NIRNAY_EXPORT=D:\Nirnay\models\nirnay-multilingual
set WORLD_SIZE=2
set NIRNAY_DDP_INIT=file:///D:/Nirnay/train/ddp_ml.init
if exist D:\Nirnay\train\ddp_ml.init del D:\Nirnay\train\ddp_ml.init
set RANK=1
start "" /b /affinity FFFF %PY% -u train\train_head.py > train\train_me5b_deep_r1.log 2>&1
set RANK=0
start "" /b /wait /affinity FFFF0000 %PY% -u train\train_head.py > train\train_me5b_deep.log 2>&1
findstr /c:"exported ->" train\train_me5b_deep.log >nul || goto fail3
set WORLD_SIZE=
set RANK=
timeout /t 30 /nobreak >nul

:evals
set HF_HUB_OFFLINE=1
echo %date% %time% stage 4/7 evaluate: 12 English suites >> %ST%
set BENCH_THREADS=8
start "" /b /wait /affinity FFFF0000 %PY% -u bench\eval_suites.py nirnay:models\nirnay-multilingual > bench\results\suites_nirnay_nirnay-multilingual.log 2>&1
if exist D:\Nirnay\models\nirnay-v31-gpu\nirnay_config.json start "" /b /wait /affinity FFFF0000 %PY% -u bench\eval_suites.py nirnay:models\nirnay-v31-gpu > bench\results\suites_nirnay_nirnay-v31-gpu.log 2>&1

echo %date% %time% stage 5/7 evaluate: MASSIVE (51 languages) + SIB-200 (205 languages) >> %ST%
set BENCH_SUITES_FILE=D:\Nirnay\bench\data\suites_multi.jsonl
set BENCH_OUT_TAG=multi
start "" /b /wait /affinity FFFF0000 %PY% -u bench\eval_suites.py nirnay:models\nirnay-multilingual > bench\results\suites_multi_nirnay-multilingual.log 2>&1
start "" /b /wait /affinity FFFF0000 %PY% -u bench\eval_suites.py nirnay:models\nirnay-v31 > bench\results\suites_multi_nirnay-v31.log 2>&1
set BENCH_SUITES_FILE=
set BENCH_OUT_TAG=

echo %date% %time% stage 6/7 determinism + latency >> %ST%
set DET_OUT=nirnay_determinism_ml.json
%PY% -u bench\determinism.py nirnay:models\nirnay-multilingual > bench\results\determinism_ml.log 2>&1
set DET_OUT=
set NIRNAY_BACKEND=torch
set BENCH_TAG=ml
set BENCH_THREADS=
%PY% -u bench\bench_latency_nirnay.py D:\Nirnay\models\nirnay-multilingual > bench\results\latency_nirnay_ml.log 2>&1
set BENCH_TAG=

echo %date% %time% stage 7/7 few-shot >> %ST%
set FEWSHOT_OUT=fewshot_ml.json
%PY% -u bench\fewshot.py D:\Nirnay\models\nirnay-multilingual > bench\results\fewshot_ml.log 2>&1
if not exist D:\Nirnay\models\nirnay-v31-gpu\nirnay_config.json goto done
echo %date% %time% extra: determinism, latency, few-shot for nirnay-v31-gpu >> %ST%
set DET_OUT=nirnay_determinism_v31gpu.json
%PY% -u bench\determinism.py nirnay:models\nirnay-v31-gpu > bench\results\determinism_v31gpu.log 2>&1
set DET_OUT=
set BENCH_TAG=v31gpu
%PY% -u bench\bench_latency_nirnay.py D:\Nirnay\models\nirnay-v31-gpu > bench\results\latency_nirnay_v31gpu.log 2>&1
set BENCH_TAG=
set FEWSHOT_OUT=fewshot_v31gpu.json
%PY% -u bench\fewshot.py D:\Nirnay\models\nirnay-v31-gpu > bench\results\fewshot_v31gpu.log 2>&1
:done
echo %date% %time% DONE ml >> %ST%
exit /b 0

:fail1
echo %date% %time% FAILED stage 1 (see train\build_data_ml.log) >> %ST%
exit /b 1
:fail2
echo %date% %time% FAILED stage 2 (see train\precompute_me5b*.log) >> %ST%
exit /b 2
:fail3
echo %date% %time% FAILED stage 3 (see train\train_me5b_deep*.log) >> %ST%
exit /b 3
