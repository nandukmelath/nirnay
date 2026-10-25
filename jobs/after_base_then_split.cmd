@echo off
rem wait until the base model's suite evaluation has finished, then train the split model on socket 1
:wait
if exist D:\Nirnay\bench\results\suites_nirnay_nirnay-bges-b2.log (
  findstr /c:"exit=" D:\Nirnay\bench\results\suites_nirnay_nirnay-bges-b2.log >nul 2>&1 && goto go
)
timeout /t 30 /nobreak >nul
goto wait
:go
call D:\Nirnay\jobs\train_split.cmd
