@echo off
:wait
findstr /c:"exit=" D:\Nirnay\bench\results\fewshot.log >nul 2>&1 || goto sleep
findstr /c:"wrote D:" D:\Nirnay\bench\results\suites_laya_laya-typed-decisions_part2.log >nul 2>&1 || goto sleep
goto go
:sleep
timeout /t 20 /nobreak >nul
goto wait
:go
timeout /t 15 /nobreak >nul
call D:\Nirnay\jobs\final_bench.cmd
