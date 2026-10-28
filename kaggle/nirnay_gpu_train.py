"""Kaggle GPU job (GPU T4 x2): trains two Nirnay models at once, one per GPU, with train/train_gpu.py.

Input: the private dataset <user>/nirnay-train-data made by jobs/kaggle_gpu.py (code/, records_v31.jsonl,
records_ml.jsonl, v2_model_best.pt). Output in /kaggle/working: nirnay-v31-gpu/, nirnay-multilingual/, one log per
model and the training run logs. With a single GPU the two models train one after the other.
"""
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

W = "/kaggle/working"
SRC = os.path.join(W, "src")


def find(name):
    hits = sorted(glob.glob("/kaggle/input/**/" + name, recursive=True))
    return hits[0] if hits else None


def stage_code():
    script = find("train_gpu.py")
    if script is None:   # code uploaded as an archive that was not unpacked
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            zipfile.ZipFile(z).extractall(os.path.join(W, "unzipped"))
        script = (sorted(glob.glob(W + "/unzipped/**/train_gpu.py", recursive=True)) or [None])[0]
    assert script, "train_gpu.py not found in the attached dataset"
    root = os.path.dirname(os.path.dirname(script))   # .../code (holds nirnay/ and train/)
    shutil.copytree(root, SRC, dirs_exist_ok=True)
    return os.path.join(SRC, "train", "train_gpu.py")


COMMON = {"NIRNAY_FUSION": "deep", "NIRNAY_SPLIT": "8", "NIRNAY_BATCH": "64", "NIRNAY_WARMUP": "300",
          "NIRNAY_EVAL_EVERY": "1000", "NIRNAY_MAX_STATE": "256", "NIRNAY_TRAIN_STATE_TOKENS": "256",
          "NIRNAY_THREADS": "2", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
JOBS = [
    {"name": "multilingual", "NIRNAY_RECORDS": "records_ml.jsonl", "NIRNAY_BACKBONE": "intfloat/multilingual-e5-base",
     "NIRNAY_POOL": "mean", "NIRNAY_PREFIX": "query: ", "NIRNAY_EPOCHS": "2.5", "NIRNAY_LR": "2e-4",
     "NIRNAY_ADAPTER_LR": "5e-5", "NIRNAY_RUN": "multilingual", "NIRNAY_EXPORT": W + "/nirnay-multilingual"},
    {"name": "v31-gpu", "NIRNAY_RECORDS": "records_v31.jsonl", "NIRNAY_BACKBONE": "BAAI/bge-base-en-v1.5",
     "NIRNAY_POOL": "cls", "NIRNAY_PREFIX": "none", "NIRNAY_EPOCHS": "2.0", "NIRNAY_LR": "1e-4",
     "NIRNAY_ADAPTER_LR": "3e-5", "NIRNAY_INIT": "v2_model_best.pt", "NIRNAY_RUN": "v31-gpu",
     "NIRNAY_EXPORT": W + "/nirnay-v31-gpu"},
]


def launch(job, gpu, script):
    env = dict(os.environ, **COMMON)
    for k, v in job.items():
        if k.startswith("NIRNAY_"):
            env[k] = (find(v) or v) if k in ("NIRNAY_RECORDS", "NIRNAY_INIT") else v
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = open(os.path.join(W, job["name"] + ".log"), "w")
    print("launch %s on GPU %s: records %s" % (job["name"], gpu, env["NIRNAY_RECORDS"]), flush=True)
    return subprocess.Popen([sys.executable, "-u", script], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=SRC), log


def last_line(name):
    try:
        lines = [l for l in open(os.path.join(W, name + ".log"), encoding="utf-8", errors="replace").read().splitlines() if l.strip()]
        return lines[-1][:160] if lines else ""
    except OSError:
        return ""


def main():
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
    import torch
    ngpu = torch.cuda.device_count()
    script = stage_code()
    t0 = time.time()
    queue = list(JOBS)
    running = []
    while queue or running:
        while queue and len(running) < max(1, ngpu):
            free = sorted(set(range(max(1, ngpu))) - {g for _, _, _, g in running})[0]
            job = queue.pop(0)
            p, log = launch(job, free, script)
            running.append((job, p, log, free))
        time.sleep(120)
        for job, p, log, g in list(running):
            state = "running" if p.poll() is None else "exit %s" % p.returncode
            print("[%6.1f min] %-12s %-9s %s" % ((time.time() - t0) / 60, job["name"], state, last_line(job["name"])), flush=True)
            if p.poll() is not None:
                log.close()
                running.remove((job, p, log, g))
    for job in JOBS:
        print("=" * 30, job["name"], "-> exported:", os.path.isdir(job["NIRNAY_EXPORT"]), flush=True)
        print(open(os.path.join(W, job["name"] + ".log"), encoding="utf-8", errors="replace").read()[-4000:], flush=True)
    for run in glob.glob(SRC + "/train/runs/*"):   # keep the run logs and best checkpoints next to the models
        shutil.copytree(run, os.path.join(W, "runs", os.path.basename(run)), dirs_exist_ok=True)
    shutil.rmtree(SRC, ignore_errors=True)
    shutil.rmtree(os.path.join(W, "unzipped"), ignore_errors=True)


if __name__ == "__main__":
    main()
