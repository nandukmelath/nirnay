@echo off
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
cd /d D:\Nirnay
D:\Nirnay\.venv\Scripts\python.exe -u bench\determinism.py nirnay:models\nirnay-v1 > bench\results\determinism_nirnay_torch.log 2>&1
echo exit=%ERRORLEVEL% >> bench\results\determinism_nirnay_torch.log
