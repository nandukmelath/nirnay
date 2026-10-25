@echo off
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set USE_TF=0
cd /d D:\NirnayScratch\laya_repo
start "" /b /wait /affinity FFFF0000 D:\Nirnay\.venv\Scripts\python.exe -u laya-dotnet\tools\export_onnx.py --model multilingual --out D:\NirnayScratch\laya_onnx > D:\Nirnay\bench\results\laya_onnx_export_ml.log 2>&1
echo exit=%ERRORLEVEL% >> D:\Nirnay\bench\results\laya_onnx_export_ml.log
start "" /b /wait /affinity FFFF0000 D:\Nirnay\.venv\Scripts\python.exe -u laya-dotnet\tools\export_onnx.py --model english --out D:\NirnayScratch\laya_onnx > D:\Nirnay\bench\results\laya_onnx_export_en.log 2>&1
echo exit=%ERRORLEVEL% >> D:\Nirnay\bench\results\laya_onnx_export_en.log
