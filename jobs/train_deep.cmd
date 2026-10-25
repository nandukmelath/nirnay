@echo off
rem Deep fusion (top 4 encoder layers fused over state/question/options), DDP: rank 0 on socket 1, rank 1 on socket 0
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set NIRNAY_TAG=bges
set NIRNAY_RUN=bges-deep
set NIRNAY_FUSION=deep
set NIRNAY_SPLIT=8
set NIRNAY_BATCH=64
set NIRNAY_EPOCHS=1
set NIRNAY_LR=3e-4
set NIRNAY_ADAPTER_LR=8e-5
set NIRNAY_WARMUP=200
set NIRNAY_EVAL_EVERY=400
set NIRNAY_THREADS=8
set NIRNAY_MAX_STATE=256
set NIRNAY_EXPORT=D:\Nirnay\models\nirnay-bges-deep
set WORLD_SIZE=2
set NIRNAY_DDP_INIT=file:///D:/Nirnay/train/ddp_bges_deep.init
cd /d D:\Nirnay
if exist D:\Nirnay\train\ddp_bges_deep.init del D:\Nirnay\train\ddp_bges_deep.init
set RANK=1
start "" /b /affinity FFFF D:\Nirnay\.venv\Scripts\python.exe -u train\train_head.py > D:\Nirnay\train\train_bges_deep_r1.log 2>&1
set RANK=0
start "" /b /wait /affinity FFFF0000 D:\Nirnay\.venv\Scripts\python.exe -u train\train_head.py > D:\Nirnay\train\train_bges_deep.log 2>&1
echo train exit=%ERRORLEVEL% >> D:\Nirnay\train\train_bges_deep.log
