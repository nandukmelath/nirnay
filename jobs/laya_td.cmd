@echo off
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set USE_TF=0
set BENCH_THREADS=8
cd /d D:\Nirnay
start "" /b /wait /affinity FFFF D:\Nirnay\.venv\Scripts\python.exe -u bench\eval_suites.py laya:laya-typed-decisions > D:\Nirnay\bench\results\suites_laya_laya-typed-decisions.log 2>&1
echo laya-typed-decisions exit=%ERRORLEVEL% >> D:\Nirnay\bench\results\suites_laya_done.log
