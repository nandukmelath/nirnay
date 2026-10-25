@echo off
rem Multilingual backbone: multilingual-e5-small precompute then 2-block head (socket 0)
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set NIRNAY_BACKBONE=intfloat/multilingual-e5-small
set NIRNAY_POOL=mean
set NIRNAY_PREFIX=query: 
set NIRNAY_TAG=e5s
set NIRNAY_THREADS=8
cd /d D:\Nirnay
start "" /b /wait /affinity FFFF D:\Nirnay\.venv\Scripts\python.exe -u train\precompute.py > D:\Nirnay\train\precompute_e5s.log 2>&1
echo precompute exit=%ERRORLEVEL% >> D:\Nirnay\train\precompute_e5s.log
set HF_HUB_OFFLINE=1
set NIRNAY_RUN=e5s-b2
set NIRNAY_BLOCKS=2
set NIRNAY_BATCH=64
set NIRNAY_EPOCHS=1
set NIRNAY_LR=3e-4
set NIRNAY_WARMUP=300
set NIRNAY_EVAL_EVERY=800
set NIRNAY_EXPORT=D:\Nirnay\models\nirnay-e5s-b2
start "" /b /wait /affinity FFFF D:\Nirnay\.venv\Scripts\python.exe -u train\train_head.py > D:\Nirnay\train\train_e5s_b2.log 2>&1
echo train exit=%ERRORLEVEL% >> D:\Nirnay\train\train_e5s_b2.log
