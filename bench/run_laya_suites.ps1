# Run Laya's three checkpoints over the benchmark suites, pinned to CPU socket 0 (logical 0-15)
# so Nirnay's training can use socket 1 at the same time without either disturbing the other.
# `start /affinity` applies the mask at creation, so the venv launcher's child inherits it.
$env:HF_HOME = 'D:\Nirnay\.hf'; $env:HF_HUB_DISABLE_TELEMETRY = '1'; $env:USE_TF = '0'; $env:BENCH_THREADS = '8'
Set-Location 'D:\Nirnay'
foreach ($m in @('laya', 'laya-multilingual', 'laya-typed-decisions')) {
    $log = "D:\Nirnay\bench\results\suites_laya_$m.log"
    cmd /c "start `"`" /b /wait /affinity FFFF D:\Nirnay\.venv\Scripts\python.exe -u bench\eval_suites.py laya:$m > `"$log`" 2>&1"
    "$m exit=$LASTEXITCODE" | Add-Content 'D:\Nirnay\bench\results\suites_laya_done.log'
}
