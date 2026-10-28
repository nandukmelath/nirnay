"""Train Nirnay on Kaggle's free GPUs (GPU T4 x2) and bring the models back.

  D:\\Nirnay\\.venv\\Scripts\\python.exe -u jobs\\kaggle_gpu.py        (needs `kaggle auth login` once)

1. stage code + records_v3 / records_ml + v2's checkpoint, and create or version the PRIVATE dataset
   <user>/nirnay-train-data (the records contain third-party dataset text, so they are never made public);
2. push kaggle/nirnay_gpu_train.py as the private script kernel <user>/nirnay-gpu-train on GPU T4 x2;
3. poll until it finishes (it trains the English v3-GPU and the multilingual model, one per GPU);
4. download the output to kaggle_out/ and copy each exported model to models/.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import time

ROOT = r"D:\Nirnay"
KAGGLE = os.path.join(ROOT, ".venv", "Scripts", "kaggle.exe")
STAGE = os.path.join(ROOT, "kaggle_stage")
OUT = os.path.join(ROOT, "kaggle_out")
DATASET, KERNEL = "nirnay-train-data", "nirnay-gpu-train"
FILES = {"records_v31.jsonl": os.path.join(ROOT, "train", "data", "records_v31.jsonl"),
         "records_ml.jsonl": os.path.join(ROOT, "train", "data", "records_ml.jsonl"),
         "v2_model_best.pt": os.path.join(ROOT, "train", "runs", "bgeb-deep", "model_best.pt")}
MODELS = ["nirnay-v31-gpu", "nirnay-multilingual"]


def log(msg):
    print(time.strftime("%d-%m %H:%M:%S"), msg, flush=True)


def kaggle(*args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        raise RuntimeError("kaggle %s failed: %s" % (" ".join(args), out[-800:]))
    return out


def username():
    out = kaggle("config", "view")
    m = re.search(r"username:\s*(\S+)", out)
    if not m:
        raise RuntimeError("could not read the Kaggle username from `kaggle config view`:\n" + out[-400:])
    return m.group(1)


def stage_dataset(user):
    d = os.path.join(STAGE, "data")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "code", "nirnay"))
    os.makedirs(os.path.join(d, "code", "train"))
    for f in glob.glob(os.path.join(ROOT, "nirnay", "*.py")):
        shutil.copy(f, os.path.join(d, "code", "nirnay"))
    for f in ("train_head.py", "train_gpu.py"):
        shutil.copy(os.path.join(ROOT, "train", f), os.path.join(d, "code", "train"))
    for name, src in FILES.items():
        assert os.path.exists(src), "missing " + src
        shutil.copy(src, os.path.join(d, name))
    json.dump({"title": DATASET, "id": "%s/%s" % (user, DATASET), "licenses": [{"name": "other"}]},
              open(os.path.join(d, "dataset-metadata.json"), "w"), indent=2)
    return d


def upload_dataset(user, d):
    ref = "%s/%s" % (user, DATASET)
    exists = subprocess.run([KAGGLE, "datasets", "status", ref], capture_output=True, text=True).returncode == 0
    if exists:
        log("new version of dataset %s" % ref)
        kaggle("datasets", "version", "-p", d, "-m", "training data and code", "--dir-mode", "zip")
    else:
        log("creating private dataset %s" % ref)
        kaggle("datasets", "create", "-p", d, "--dir-mode", "zip")
    for _ in range(120):
        st = kaggle("datasets", "status", ref, check=False).lower()
        if "ready" in st:
            log("dataset ready")
            return
        time.sleep(30)
    raise RuntimeError("dataset not ready after 60 min: " + st)


def push_kernel(user):
    d = os.path.join(STAGE, "kernel")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shutil.copy(os.path.join(ROOT, "kaggle", "nirnay_gpu_train.py"), d)
    json.dump({"id": "%s/%s" % (user, KERNEL), "title": KERNEL, "code_file": "nirnay_gpu_train.py",
               "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
               "enable_internet": True, "dataset_sources": ["%s/%s" % (user, DATASET)], "competition_sources": [],
               "kernel_sources": [], "model_sources": []}, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    log(kaggle("kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4"))


def wait_kernel(user):
    ref = "%s/%s" % (user, KERNEL)
    t0 = time.time()
    while True:
        st = kaggle("kernels", "status", ref, check=False)
        low = st.lower()
        log("status: " + st.splitlines()[-1][:200] if st else "status: ?")
        if any(w in low for w in ("complete", "error", "cancel")):
            return "complete" in low
        if time.time() - t0 > 13 * 3600:
            return False
        time.sleep(300)


def download(user):
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    log("downloading output")
    kaggle("kernels", "output", "%s/%s" % (user, KERNEL), "-p", OUT)
    for m in MODELS:
        src = os.path.join(OUT, m)
        if os.path.exists(os.path.join(src, "nirnay_config.json")):
            dst = os.path.join(ROOT, "models", m)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
            log("model ready -> " + dst)
        else:
            log("no exported model for " + m + " (see kaggle_out logs)")


def main():
    user = username()
    log("Kaggle user: " + user)
    upload_dataset(user, stage_dataset(user))
    push_kernel(user)
    ok = wait_kernel(user)
    download(user)
    log("KAGGLE DONE" if ok else "KAGGLE FINISHED WITH ERRORS")


if __name__ == "__main__":
    main()
