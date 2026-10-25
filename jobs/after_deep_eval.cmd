@echo off
rem after deep training exports: ONNX export, then Nirnay-deep suites (socket 1) || Laya-td remaining suites (socket 0)
:wait
findstr /c:"exported ->" D:\Nirnay\train\train_bges_deep.log >nul 2>&1 && goto go
findstr /c:"Traceback" D:\Nirnay\train\train_bges_deep.log >nul 2>&1 && exit /b 1
timeout /t 30 /nobreak >nul
goto wait
:go
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set USE_TF=0
set BENCH_THREADS=8
cd /d D:\Nirnay
timeout /t 20 /nobreak >nul
D:\Nirnay\.venv\Scripts\python.exe -m nirnay.onnx_backend export D:\Nirnay\models\nirnay-bges-deep > D:\Nirnay\bench\results\onnx_export_deep.log 2>&1
set BENCH_MERGE=1
start "" /b /affinity FFFF D:\Nirnay\.venv\Scripts\python.exe -u bench\eval_suites.py laya:laya-typed-decisions > D:\Nirnay\bench\results\suites_laya_laya-typed-decisions_part2.log 2>&1
set BENCH_MERGE=
start "" /b /wait /affinity FFFF0000 D:\Nirnay\.venv\Scripts\python.exe -u bench\eval_suites.py nirnay:models\nirnay-bges-deep > D:\Nirnay\bench\results\suites_nirnay_nirnay-bges-deep.log 2>&1
echo exit=%ERRORLEVEL% >> D:\Nirnay\bench\results\suites_nirnay_nirnay-bges-deep.log
