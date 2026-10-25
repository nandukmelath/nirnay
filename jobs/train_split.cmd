@echo off
rem Split model: trainable copies of encoder layers 9-12 on the state + 2 fusion blocks (socket 1)
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set NIRNAY_TAG=bges
set NIRNAY_RUN=bges-s8
set NIRNAY_SPLIT=8
set NIRNAY_BLOCKS=2
set NIRNAY_BATCH=64
set NIRNAY_EPOCHS=1
set NIRNAY_LR=3e-4
set NIRNAY_ADAPTER_LR=5e-5
set NIRNAY_WARMUP=300
set NIRNAY_EVAL_EVERY=800
set NIRNAY_THREADS=8
set NIRNAY_EXPORT=D:\Nirnay\models\nirnay-bges-s8
cd /d D:\Nirnay
start "" /b /wait /affinity FFFF0000 D:\Nirnay\.venv\Scripts\python.exe -u train\train_head.py > D:\Nirnay\train\train_bges_s8.log 2>&1
echo train exit=%ERRORLEVEL% >> D:\Nirnay\train\train_bges_s8.log
