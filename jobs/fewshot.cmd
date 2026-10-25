@echo off
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set NIRNAY_THREADS=8
cd /d D:\Nirnay
start "" /b /wait /affinity FFFF0000 D:\Nirnay\.venv\Scripts\python.exe -u bench\fewshot.py D:\Nirnay\models\nirnay-v1 > D:\Nirnay\bench\results\fewshot.log 2>&1
echo exit=%ERRORLEVEL% >> D:\Nirnay\bench\results\fewshot.log
