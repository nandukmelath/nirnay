"""Publish Nirnay to the Hugging Face Hub: model repo, demo Space and a collection.

  D:\\Nirnay\\.venv\\Scripts\\hf.exe auth login        (once, with a WRITE token)
  D:\\Nirnay\\.venv\\Scripts\\python.exe D:\\Nirnay\\jobs\\hf_upload.py
"""
import time

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

MODEL_DIR = r"D:\Nirnay\hf_release"
SPACE_DIR = r"D:\Nirnay\hf_space"


def main():
    api = HfApi()
    name = api.whoami()["name"]
    if name != "nandukmelath":
        print("WARNING: logged in as %s; the model card links point to nandukmelath" % name, flush=True)
    model_repo, space_repo = "%s/nirnay" % name, "%s/nirnay-demo" % name

    print("creating model repo", model_repo, flush=True)
    api.create_repo(model_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=MODEL_DIR, repo_id=model_repo, repo_type="model",
                      commit_message="Nirnay v2 (bge-base, default) + v1 (fast/): weights, package, code, benchmarks")
    print("model uploaded: %d files -> https://huggingface.co/%s" % (len(api.list_repo_files(model_repo)), model_repo), flush=True)

    print("creating Space", space_repo, flush=True)
    try:
        api.create_repo(space_repo, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    except HfHubHTTPError as e:  # 402: Gradio / Docker Spaces on free cpu-basic need a PRO plan
        if e.response is None or e.response.status_code != 402:
            raise
        print("Space skipped (402: this account cannot host a Gradio Space):", str(e).splitlines()[-1], flush=True)
        space_repo = None
    if space_repo:
        api.upload_folder(folder_path=SPACE_DIR, repo_id=space_repo, repo_type="space",
                          ignore_patterns=["__pycache__/*", "*.pyc"],
                          commit_message="Nirnay demo: typed decisions on CPU")
        print("Space uploaded -> https://huggingface.co/spaces/%s" % space_repo, flush=True)

    try:
        coll = api.create_collection(
            title="Nirnay: fast decision models for CPU", namespace=name, exists_ok=True,
            description="Typed decisions (choice / yes-no / rating) with calibrated probabilities, built for CPUs. "
                        "Laya / Jev compatible.")
        api.add_collection_item(coll.slug, item_id=model_repo, item_type="model", exists_ok=True)
        if space_repo:
            api.add_collection_item(coll.slug, item_id=space_repo, item_type="space", exists_ok=True)
        print("collection -> https://huggingface.co/collections/%s" % coll.slug, flush=True)
    except Exception as e:  # a collection is a nice-to-have
        print("collection skipped:", str(e)[:200], flush=True)

    if not space_repo:
        return
    # wait for the Space to build and start (first build installs torch: ~5-10 min)
    t0 = time.time()
    stage = None
    while time.time() - t0 < 1800:
        stage = api.get_space_runtime(space_repo).stage
        if stage in ("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            break
        time.sleep(30)
    print("Space stage after %.0f min: %s" % ((time.time() - t0) / 60, stage), flush=True)


if __name__ == "__main__":
    main()
