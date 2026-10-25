# Evaluate one Nirnay model directory over the suites, pinned to a CPU socket.
#   powershell -File bench\run_nirnay_suites.ps1 <model_dir_name> [affinity_hex] [threads]
param([string]$Model = 'nirnay-v1', [string]$Affinity = 'FFFF0000', [string]$Threads = '8')
$env:HF_HOME = 'D:\Nirnay\.hf'; $env:HF_HUB_DISABLE_TELEMETRY = '1'; $env:HF_HUB_OFFLINE = '1'; $env:BENCH_THREADS = $Threads
Set-Location 'D:\Nirnay'
$log = "D:\Nirnay\bench\results\suites_nirnay_$Model.log"
cmd /c "start `"`" /b /wait /affinity $Affinity D:\Nirnay\.venv\Scripts\python.exe -u bench\eval_suites.py nirnay:models\$Model > `"$log`" 2>&1"
"exit=$LASTEXITCODE" | Add-Content $log
