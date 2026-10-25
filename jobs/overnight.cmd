@echo off
rem Overnight run: bge-base backbone + deep fusion, expanded data, 1.6 epochs on both sockets, then eval + report.
rem Progress: D:\Nirnay\train\overnight_status.log
set ST=D:\Nirnay\train\overnight_status.log
set PY=D:\Nirnay\.venv\Scripts\python.exe
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set USE_TF=0
cd /d D:\Nirnay
echo %date% %time% START overnight >> %ST%

echo %date% %time% stage 1/7 build data (records_v2) >> %ST%
set NIRNAY_RECORDS_OUT=D:\Nirnay\train\data\records_v2.jsonl
set NIRNAY_SRC_SCALE=ag_news=2.86,support_tickets=2.78,enron=3.14,phishing=2,msmarco=2.86
set NIRNAY_SKIP_SOURCES=amazon_multi,tweet_sent_multi
start "" /b /wait /affinity FFFF %PY% -u train\build_data.py > train\build_data_v2.log 2>&1
findstr /b /c:"records:" train\build_data_v2.log >nul || goto fail1
set NIRNAY_RECORDS_OUT=
set NIRNAY_SRC_SCALE=
set NIRNAY_SKIP_SOURCES=

echo %date% %time% stage 2/7 precompute bge-base (2 shards) >> %ST%
if exist D:\Nirnay\train\cache\bgeb rmdir /s /q D:\Nirnay\train\cache\bgeb
set NIRNAY_BACKBONE=BAAI/bge-base-en-v1.5
set NIRNAY_POOL=cls
set NIRNAY_PREFIX=none
set NIRNAY_TAG=bgeb
set NIRNAY_THREADS=8
set NIRNAY_RECORDS=D:\Nirnay\train\data\records_v2.jsonl
set NIRNAY_TRAIN_STATE_TOKENS=256
set NIRNAY_SAVE_MID=8
set NIRNAY_STATE_MID_ONLY=1
set NIRNAY_SHARD=1/2
start "" /b /affinity FFFF0000 %PY% -u train\precompute.py > train\precompute_bgeb_s1.log 2>&1
set NIRNAY_SHARD=0/2
start "" /b /wait /affinity FFFF %PY% -u train\precompute.py > train\precompute_bgeb.log 2>&1
findstr /c:"done ->" train\precompute_bgeb.log >nul || goto fail2
set NIRNAY_SHARD=
timeout /t 30 /nobreak >nul

echo %date% %time% stage 3/7 train deep fusion (DDP, 1.6 epochs) >> %ST%
set HF_HUB_OFFLINE=1
set NIRNAY_RUN=bgeb-deep
set NIRNAY_FUSION=deep
set NIRNAY_SPLIT=8
set NIRNAY_BATCH=64
set NIRNAY_EPOCHS=1.6
set NIRNAY_LR=2e-4
set NIRNAY_ADAPTER_LR=5e-5
set NIRNAY_WARMUP=300
set NIRNAY_EVAL_EVERY=400
set NIRNAY_MAX_STATE=256
set NIRNAY_EXPORT=D:\Nirnay\models\nirnay-bgeb-deep
set WORLD_SIZE=2
set NIRNAY_DDP_INIT=file:///D:/Nirnay/train/ddp_bgeb_deep.init
if exist D:\Nirnay\train\ddp_bgeb_deep.init del D:\Nirnay\train\ddp_bgeb_deep.init
set RANK=1
start "" /b /affinity FFFF %PY% -u train\train_head.py > train\train_bgeb_deep_r1.log 2>&1
set RANK=0
start "" /b /wait /affinity FFFF0000 %PY% -u train\train_head.py > train\train_bgeb_deep.log 2>&1
findstr /c:"exported ->" train\train_bgeb_deep.log >nul || goto fail3
set WORLD_SIZE=
set RANK=
timeout /t 30 /nobreak >nul

echo %date% %time% stage 4/7 evaluate on the 12 suites >> %ST%
set BENCH_THREADS=8
start "" /b /wait /affinity FFFF0000 %PY% -u bench\eval_suites.py nirnay:models\nirnay-bgeb-deep > bench\results\suites_nirnay_nirnay-bgeb-deep.log 2>&1

echo %date% %time% stage 5/7 latency (idle machine) >> %ST%
set NIRNAY_BACKEND=torch
set BENCH_TAG=bgeb
%PY% -u bench\bench_latency_nirnay.py D:\Nirnay\models\nirnay-bgeb-deep > bench\results\latency_nirnay_bgeb.log 2>&1
set BENCH_TAG=

echo %date% %time% stage 6/7 few-shot >> %ST%
set FEWSHOT_OUT=fewshot_bgeb.json
%PY% -u bench\fewshot.py D:\Nirnay\models\nirnay-bgeb-deep > bench\results\fewshot_bgeb.log 2>&1

echo %date% %time% stage 7/7 report >> %ST%
set REPORT_NIRNAY=nirnay-bgeb-deep
set REPORT_OUT=REPORT_overnight.md
set REPORT_LATENCY=nirnay_latency_bgeb.json
set REPORT_FEWSHOT=fewshot_bgeb.json
%PY% bench\report.py > bench\results\report_overnight.log 2>&1
echo %date% %time% DONE overnight >> %ST%
exit /b 0

:fail1
echo %date% %time% FAILED stage 1 (see train\build_data_v2.log) >> %ST%
exit /b 1
:fail2
echo %date% %time% FAILED stage 2 (see train\precompute_bgeb*.log) >> %ST%
exit /b 2
:fail3
echo %date% %time% FAILED stage 3 (see train\train_bgeb_deep*.log) >> %ST%
exit /b 3
