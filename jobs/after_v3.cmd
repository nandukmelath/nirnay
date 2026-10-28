@echo off
rem Runs after jobs\v3.cmd: Laya-multilingual on the MASSIVE + SIB-200 suites (the comparison), then jobs\ml.cmd.
rem Waits up to 20 h for "DONE v3" / "FAILED" in train\v3_status.log. Progress: train\after_v3_status.log
set ST=D:\Nirnay\train\after_v3_status.log
set PY=D:\Nirnay\.venv\Scripts\python.exe
set HF_HOME=D:\Nirnay\.hf
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
set USE_TF=0
cd /d D:\Nirnay
echo %date% %time% waiting for v3 >> %ST%
set /a n=0
:wait
findstr /c:"DONE v3" /c:"FAILED" D:\Nirnay\train\v3_status.log >nul 2>&1 && goto go
set /a n+=1
if %n% gtr 1200 goto giveup
timeout /t 60 /nobreak >nul
goto wait
:go
echo %date% %time% v3 finished; Laya-multilingual on MASSIVE + SIB-200 >> %ST%
set BENCH_THREADS=8
set BENCH_SUITES_FILE=D:\Nirnay\bench\data\suites_multi.jsonl
set BENCH_OUT_TAG=multi
start "" /b /wait /affinity FFFF0000 %PY% -u bench\eval_suites.py laya:laya-multilingual > bench\results\suites_multi_laya-multilingual.log 2>&1
echo %date% %time% Laya-multilingual done; starting jobs\ml.cmd >> %ST%
set BENCH_SUITES_FILE=
set BENCH_OUT_TAG=
set BENCH_THREADS=
set HF_HUB_OFFLINE=
call D:\Nirnay\jobs\ml.cmd
echo %date% %time% ml.cmd returned %ERRORLEVEL% >> %ST%
exit /b 0
:giveup
echo %date% %time% gave up waiting for v3 >> %ST%
exit /b 1
