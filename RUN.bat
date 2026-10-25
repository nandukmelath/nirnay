@echo off
rem Nirnay decision model - quick launcher (CPU only, offline after the first download).
rem   RUN.bat demo              one example call
rem   RUN.bat serve [port]      HTTP server, Jev/Laya /v1/systemone protocol (default port 8000)
rem   RUN.bat mcp               MCP stdio server for local agents
rem   RUN.bat cli ...           command line (see: RUN.bat cli --help)
rem   RUN.bat bench             latency benchmark on this machine
rem   RUN.bat test              invariants (encoder parity, option-order invariance)
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "HF_HOME=%ROOT%.hf"
set "HF_HUB_OFFLINE=1"
set "PYTHONPATH=%ROOT%"
if not exist "%PY%" (
  echo Python environment not found at %PY%
  exit /b 1
)
set "CMD=%~1"
if "%CMD%"=="" set "CMD=demo"
if /i "%CMD%"=="demo" goto demo
if /i "%CMD%"=="serve" goto serve
if /i "%CMD%"=="mcp" goto mcp
if /i "%CMD%"=="cli" goto cli
if /i "%CMD%"=="bench" goto bench
if /i "%CMD%"=="test" goto test
echo unknown command %CMD%
exit /b 2

:demo
"%PY%" -m nirnay.cli "Hi, my card was charged twice for the same order and I want my money back today." --choice "billing,technical support,sales,shipping" --ask "Which team should handle this ticket?"
exit /b %ERRORLEVEL%

:serve
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8000"
"%PY%" -m nirnay.server --port %PORT%
exit /b %ERRORLEVEL%

:mcp
"%PY%" -m nirnay.mcp_server
exit /b %ERRORLEVEL%

:cli
shift
"%PY%" -m nirnay.cli %1 %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%

:bench
"%PY%" "%ROOT%bench\bench_latency_nirnay.py"
exit /b %ERRORLEVEL%

:test
"%PY%" "%ROOT%tests\test_core.py"
exit /b %ERRORLEVEL%
