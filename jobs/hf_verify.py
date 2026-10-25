"""After publishing: download nandukmelath/nirnay into an EMPTY cache and run real predictions,
then check that the pip-install link to the wheel resolves."""
import os
import shutil
import tempfile
import urllib.request

tmp = tempfile.mkdtemp(prefix="nirnay_hub_")
os.environ["HF_HOME"] = tmp
os.environ.pop("HF_HUB_OFFLINE", None)

import nirnay  # noqa: E402

REPO = "nandukmelath/nirnay"
q = {"team": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "payments and refunds", "technical": "bugs", "sales": "pricing"}},
     "angry": {"type": "noul", "instructions": "Is the customer angry?"}}
try:
    for sub in (None, "fast"):
        m = nirnay.load(REPO, subfolder=sub, threads=8)
        r = m.predict("This is the third time the app crashed and deleted my work. Fix it now!", q)
        a = r["answers"]
        print("hub load OK: %-10s team=%s (%.3f)  angry P(yes)=%.3f" % (
            m.name, a["team"]["choice"], a["team"]["answer_confidence"], a["angry"]["noul"]))
    url = "https://huggingface.co/%s/resolve/main/dist/nirnay-1.0.0-py3-none-any.whl" % REPO
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as resp:
        print("wheel link:", resp.status, url)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
