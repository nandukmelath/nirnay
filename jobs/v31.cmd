@echo off
rem v3.1 = v3 with content-only safety wording (v3's AI-assistant / policy phrasings made it call harmful prompts
rem jailbreaks). Same recipe as v3: records rebuilt, bge-base deep fusion warm-started from v2, 1.25 epochs on both
rem sockets, then the 12 suites, determinism, latency, few-shot; then jobs\ml.cmd (evaluation only if the
rem multilingual model already came from Kaggle). Progress: D:\Nirnay\train\v31_status.log
set ST=D:\Nirnay\train\v31_status.log
set PY=D:\Nirnay\.venv\Scripts\python.exe
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set USE_TF=0
cd /d D:\Nirnay
echo %date% %time% START v3.1 >> %ST%

echo %date% %time% stage 1/7 build records_v31 >> %ST%
set NIRNAY_RECORDS_OUT=D:\Nirnay\train\data\records_v31.jsonl
set NIRNAY_SRC_SCALE=ag_news=2.86,support_tickets=2.78,enron=3.14,phishing=2,msmarco=2.86
set NIRNAY_SKIP_SOURCES=amazon_multi,tweet_sent_multi,mtop,paws_x,nli_multi,toxic_multi,lang_id
start "" /b /wait /affinity FFFF %PY% -u train\build_data.py > train\build_data_v31.log 2>&1
findstr /b /c:"records:" train\build_data_v31.log >nul || goto fail1
set NIRNAY_RECORDS_OUT=
set NIRNAY_SRC_SCALE=
set NIRNAY_SKIP_SOURCES=

echo %date% %time% stage 2/7 precompute bge-base over records_v31 (2 shards) >> %ST%
if exist D:\Nirnay\train\cache\bgeb31 rmdir /s /q D:\Nirnay\train\cache\bgeb31
set NIRNAY_BACKBONE=BAAI/bge-base-en-v1.5
set NIRNAY_POOL=cls
set NIRNAY_PREFIX=none
set NIRNAY_TAG=bgeb31
set NIRNAY_THREADS=8
set NIRNAY_RECORDS=D:\Nirnay\train\data\records_v31.jsonl
set NIRNAY_TRAIN_STATE_TOKENS=256
set NIRNAY_SAVE_MID=8
set NIRNAY_STATE_MID_ONLY=1
set NIRNAY_SHARD=1/2
start "" /b /affinity FFFF0000 %PY% -u train\precompute.py > train\precompute_bgeb31_s1.log 2>&1
set NIRNAY_SHARD=0/2
start "" /b /wait /affinity FFFF %PY% -u train\precompute.py > train\precompute_bgeb31.log 2>&1
findstr /c:"done ->" train\precompute_bgeb31.log >nul || goto fail2
set NIRNAY_SHARD=
timeout /t 30 /nobreak >nul

echo %date% %time% stage 3/7 train deep fusion, warm start from v2 (DDP, 1.25 epochs) >> %ST%
set HF_HUB_OFFLINE=1
set NIRNAY_RUN=bgeb31-deep
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
set NIRNAY_EXPORT=D:\Nirnay\models\nirnay-v31
set WORLD_SIZE=2
set NIRNAY_DDP_INIT=file:///D:/Nirnay/train/ddp_v31.init
if exist D:\Nirnay\train\ddp_v31.init del D:\Nirnay\train\ddp_v31.init
set RANK=1
start "" /b /affinity FFFF %PY% -u train\train_head.py > train\train_bgeb31_deep_r1.log 2>&1
set RANK=0
start "" /b /wait /affinity FFFF0000 %PY% -u train\train_head.py > train\train_bgeb31_deep.log 2>&1
findstr /c:"exported ->" train\train_bgeb31_deep.log >nul || goto fail3
set WORLD_SIZE=
set RANK=
set NIRNAY_INIT=
timeout /t 30 /nobreak >nul

echo %date% %time% stage 4/7 evaluate on the 12 suites >> %ST%
set BENCH_THREADS=8
start "" /b /wait /affinity FFFF0000 %PY% -u bench\eval_suites.py nirnay:models\nirnay-v31 > bench\results\suites_nirnay_nirnay-v31.log 2>&1

echo %date% %time% stage 5/7 determinism >> %ST%
set DET_OUT=nirnay_determinism_v31.json
%PY% -u bench\determinism.py nirnay:models\nirnay-v31 > bench\results\determinism_v31.log 2>&1
set DET_OUT=

echo %date% %time% stage 6/7 latency (idle machine) >> %ST%
set NIRNAY_BACKEND=torch
set BENCH_TAG=v31
set BENCH_THREADS=
%PY% -u bench\bench_latency_nirnay.py D:\Nirnay\models\nirnay-v31 > bench\results\latency_nirnay_v31.log 2>&1
set BENCH_TAG=

echo %date% %time% stage 7/7 few-shot >> %ST%
set FEWSHOT_OUT=fewshot_v31.json
%PY% -u bench\fewshot.py D:\Nirnay\models\nirnay-v31 > bench\results\fewshot_v31.log 2>&1
set FEWSHOT_OUT=
echo %date% %time% DONE v3.1; starting jobs\ml.cmd >> %ST%
call D:\Nirnay\jobs\ml.cmd
exit /b 0

:fail1
echo %date% %time% FAILED stage 1 (see train\build_data_v31.log) >> %ST%
exit /b 1
:fail2
echo %date% %time% FAILED stage 2 (see train\precompute_bgeb31*.log) >> %ST%
exit /b 2
:fail3
echo %date% %time% FAILED stage 3 (see train\train_bgeb31_deep*.log) >> %ST%
exit /b 3
