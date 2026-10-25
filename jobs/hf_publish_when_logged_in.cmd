@echo off
rem Publish to Hugging Face as soon as a login token exists (waits up to 24 h), then verify from the Hub.
set LOG=D:\Nirnay\jobs\hf_publish.log
set HF_HOME=
set HF_HUB_OFFLINE=
set TOK=C:\Users\localhost\.cache\huggingface\token
echo %date% %time% waiting for Hugging Face login (token file) >> %LOG%
set /a n=0
:wait
if exist "%TOK%" goto go
set /a n+=1
if %n% gtr 8640 goto giveup
timeout /t 10 /nobreak >nul
goto wait
:go
echo %date% %time% token found, uploading >> %LOG%
timeout /t 5 /nobreak >nul
cd /d D:\Nirnay
D:\Nirnay\.venv\Scripts\python.exe -u jobs\hf_upload.py >> %LOG% 2>&1
if errorlevel 1 goto failed
echo %date% %time% verifying from the Hub >> %LOG%
D:\Nirnay\.venv\Scripts\python.exe -u jobs\hf_verify.py >> %LOG% 2>&1
echo %date% %time% PUBLISH DONE >> %LOG%
exit /b 0
:failed
echo %date% %time% UPLOAD FAILED (see above) >> %LOG%
exit /b 1
:giveup
echo %date% %time% gave up after 24 h without a login >> %LOG%
exit /b 2
