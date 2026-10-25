"""HTTP server speaking the Jev `/v1/systemone` wire protocol (the one laya-serve also speaks),
so an existing Jev or Laya client only needs its base URL changed.

  python -m nirnay.server --port 8000           (NIRNAY_API_KEY=... to require a bearer token)

  POST /v1/systemone        {"state": ..., "questions": {...}, "min_confidence"?: float}
  POST /v1/systemone/batch  {"states": [...], "questions": {...}, "min_confidence"?: float}
  GET  /health
"""
import argparse
import hmac
import os
import threading
from typing import Any, Dict, List, Optional

from .engine import load
from .render import SpecError

MAX_STATES = int(os.environ.get("NIRNAY_MAX_BATCH_STATES", "1024"))
MAX_OPTIONS = int(os.environ.get("NIRNAY_MAX_OPTIONS", "1000"))


def create_app(model_dir: Optional[str] = None):
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    model = load(model_dir)
    lock = threading.Lock()   # one forward at a time: torch already uses every configured thread
    api_key = os.environ.get("NIRNAY_API_KEY")
    app = FastAPI(title="Nirnay decision model", version=model.cfg.get("version", "1"))

    class One(BaseModel):
        state: Any
        questions: Dict[str, Dict[str, Any]]
        min_confidence: Optional[float] = None
        model: Optional[str] = None

    class Many(BaseModel):
        states: List[Any]
        questions: Dict[str, Dict[str, Any]]
        min_confidence: Optional[float] = None
        model: Optional[str] = None

    def auth(authorization: Optional[str]):
        if api_key and not hmac.compare_digest(authorization or "", "Bearer " + api_key):
            raise HTTPException(status_code=401, detail="missing or wrong bearer token")

    def check(questions):
        for qid, q in questions.items():
            c = q.get("criteria") if isinstance(q, dict) else None
            if isinstance(c, (dict, list)) and len(c) > MAX_OPTIONS:
                raise HTTPException(status_code=422, detail="question %r has more than %d options" % (qid, MAX_OPTIONS))

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model.name, "threads": model.threads,
                "cached_texts": len(model.cache.d)}

    @app.post("/v1/systemone")
    def systemone(req: One, authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        if req.state is None:
            raise HTTPException(status_code=400, detail="state is required")
        check(req.questions)
        try:
            with lock:
                return model.predict(req.state, req.questions, min_confidence=req.min_confidence)
        except SpecError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/v1/systemone/batch")
    def batch(req: Many, authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        if not req.states or len(req.states) > MAX_STATES:
            raise HTTPException(status_code=400, detail="states must hold 1..%d items" % MAX_STATES)
        check(req.questions)
        try:
            with lock:
                return {"results": model.predict_batch(req.states, req.questions, min_confidence=req.min_confidence)}
        except SpecError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description="Nirnay HTTP server (Jev /v1/systemone protocol)")
    ap.add_argument("--host", default=os.environ.get("NIRNAY_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("NIRNAY_PORT", "8000")))
    ap.add_argument("--model", default=None, help="model directory (default models/nirnay-v1)")
    a = ap.parse_args()
    uvicorn.run(create_app(a.model), host=a.host, port=a.port, workers=1)


if __name__ == "__main__":
    main()
