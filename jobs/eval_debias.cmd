@echo off
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set BENCH_THREADS=8
set NIRNAY_DEBIAS=1
set BENCH_LABEL_SUFFIX=-debias
cd /d D:\Nirnay
start "" /b /wait /affinity FFFF0000 D:\Nirnay\.venv\Scripts\python.exe -u bench\eval_suites.py nirnay:models\nirnay-bges-deep > D:\Nirnay\bench\results\suites_nirnay_nirnay-bges-deep-debias.log 2>&1
echo exit=%ERRORLEVEL% >> D:\Nirnay\bench\results\suites_nirnay_nirnay-bges-deep-debias.log
