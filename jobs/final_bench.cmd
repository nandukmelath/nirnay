@echo off
rem Final benchmarks on an idle machine (no affinity, like Laya's latency run)
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set USE_TF=0
set PY=D:\Nirnay\.venv\Scripts\python.exe
cd /d D:\Nirnay
set NIRNAY_BACKEND=torch
set BENCH_TAG=torch
%PY% -u bench\bench_latency_nirnay.py D:\Nirnay\models\nirnay-v1 > bench\results\latency_nirnay_torch.log 2>&1
set NIRNAY_BACKEND=auto
set BENCH_TAG=ort
%PY% -u bench\bench_latency_nirnay.py D:\Nirnay\models\nirnay-v1 > bench\results\latency_nirnay_ort.log 2>&1
set BENCH_TAG=
set LAYA_ONNX_DIR=D:\NirnayScratch\laya_onnx
set BENCH_MODELS=laya,laya-multilingual
%PY% -u bench\bench_latency_laya.py > bench\results\latency_laya_onnx.log 2>&1
set LAYA_ONNX_DIR=
set BENCH_MODELS=
%PY% -u bench\determinism.py nirnay:models\nirnay-v1 > bench\results\determinism_nirnay.log 2>&1
%PY% -u bench\cold_start.py > bench\results\cold_start.log 2>&1
echo final bench done >> bench\results\final_bench_done.log
