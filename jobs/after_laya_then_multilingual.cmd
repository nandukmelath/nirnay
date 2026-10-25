@echo off
rem wait until Laya's typed-decisions evaluation has finished, then run the multilingual pipeline on socket 0
:wait
findstr /c:"laya-typed-decisions exit=" D:\Nirnay\bench\results\suites_laya_done.log >nul 2>&1 && goto go
timeout /t 30 /nobreak >nul
goto wait
:go
call D:\Nirnay\jobs\multilingual.cmd
