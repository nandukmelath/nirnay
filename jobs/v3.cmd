@echo off
rem v3: records_v3 (v2 + routing, short-message spam, safety, urgency, more phrasings), bge-base deep fusion
rem warm-started from v2 for 1.25 epochs on both sockets, then the 12 suites, determinism, latency, few-shot.
rem Progress: D:\Nirnay\train\v3_status.log
set ST=D:\Nirnay\train\v3_status.log
set PY=D:\Nirnay\.venv\Scripts\python.exe
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set USE_TF=0
cd /d D:\Nirnay
echo %date% %time% START v3 >> %ST%

echo %date% %time% stage 1/6 precompute bge-base over records_v3 (2 shards) >> %ST%
if exist D:\Nirnay\train\cache\bgeb3 rmdir /s /q D:\Nirnay\train\cache\bgeb3
set NIRNAY_BACKBONE=BAAI/bge-base-en-v1.5
set NIRNAY_POOL=cls
set NIRNAY_PREFIX=none
set NIRNAY_TAG=bgeb3
set NIRNAY_THREADS=8
set NIRNAY_RECORDS=D:\Nirnay\train\data\records_v3.jsonl
set NIRNAY_TRAIN_STATE_TOKENS=256
set NIRNAY_SAVE_MID=8
set NIRNAY_STATE_MID_ONLY=1
set NIRNAY_SHARD=1/2
start "" /b /affinity FFFF0000 %PY% -u train\precompute.py > train\precompute_bgeb3_s1.log 2>&1
set NIRNAY_SHARD=0/2
start "" /b /wait /affinity FFFF %PY% -u train\precompute.py > train\precompute_bgeb3.log 2>&1
findstr /c:"done ->" train\precompute_bgeb3.log >nul || goto fail1
set NIRNAY_SHARD=
timeout /t 30 /nobreak >nul

echo %date% %time% stage 2/6 train deep fusion, warm start from v2 (DDP, 1.25 epochs) >> %ST%
set HF_HUB_OFFLINE=1
set NIRNAY_RUN=bgeb3-deep
set NIRNAY_INIT=D:\Nirnay\train\runs\bgeb-deep\model_best.pt
set NIRNAY_FUSION=deep
set NIRNAY_SPLIT=8
set NIRNAY_BATCH=64
set NIRNAY_EPOCHS=1.25
set NIRNAY_LR=1e-4
set NIRNAY_ADAPTER_LR=3e-5
set NIRNAY_WARMUP=300
set NIRNAY_EVAL_EVERY=400
set NIRNAY_MAX_STATE=256
set NIRNAY_EXPORT=D:\Nirnay\models\nirnay-v3
set WORLD_SIZE=2
set NIRNAY_DDP_INIT=file:///D:/Nirnay/train/ddp_v3.init
if exist D:\Nirnay\train\ddp_v3.init del D:\Nirnay\train\ddp_v3.init
set RANK=1
start "" /b /affinity FFFF %PY% -u train\train_head.py > train\train_bgeb3_deep_r1.log 2>&1
set RANK=0
start "" /b /wait /affinity FFFF0000 %PY% -u train\train_head.py > train\train_bgeb3_deep.log 2>&1
findstr /c:"exported ->" train\train_bgeb3_deep.log >nul || goto fail2
set WORLD_SIZE=
set RANK=
set NIRNAY_INIT=
timeout /t 30 /nobreak >nul

echo %date% %time% stage 3/6 evaluate on the 12 suites >> %ST%
set BENCH_THREADS=8
start "" /b /wait /affinity FFFF0000 %PY% -u bench\eval_suites.py nirnay:models\nirnay-v3 > bench\results\suites_nirnay_nirnay-v3.log 2>&1

echo %date% %time% stage 4/6 determinism >> %ST%
set DET_OUT=nirnay_determinism_v3.json
%PY% -u bench\determinism.py nirnay:models\nirnay-v3 > bench\results\determinism_v3.log 2>&1
set DET_OUT=

echo %date% %time% stage 5/6 latency (idle machine) >> %ST%
set NIRNAY_BACKEND=torch
set BENCH_TAG=v3
set BENCH_THREADS=
%PY% -u bench\bench_latency_nirnay.py D:\Nirnay\models\nirnay-v3 > bench\results\latency_nirnay_v3.log 2>&1
set BENCH_TAG=

echo %date% %time% stage 6/6 few-shot >> %ST%
set FEWSHOT_OUT=fewshot_v3.json
%PY% -u bench\fewshot.py D:\Nirnay\models\nirnay-v3 > bench\results\fewshot_v3.log 2>&1
echo %date% %time% DONE v3 >> %ST%
exit /b 0

:fail1
echo %date% %time% FAILED stage 1 (see train\precompute_bgeb3*.log) >> %ST%
exit /b 1
:fail2
echo %date% %time% FAILED stage 2 (see train\train_bgeb3_deep*.log) >> %ST%
exit /b 2
