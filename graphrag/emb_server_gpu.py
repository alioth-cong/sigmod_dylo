#!/usr/bin/env python3
"""OpenAI-compatible /v1/embeddings served on the GPU with micro-batch merging.

Removes two bottlenecks of emb_server.py:
  1) CPU inference -- bge-small is only 130 MB and easily shares a GPU with vLLM;
  2) a global lock serializing all requests, which defeats GraphRAG's concurrent_requests=25.

Here a single background thread consumes a queue: concurrent requests are merged into large batches for the GPU, avoiding
sentence-transformers' thread unsafety while getting batch throughput. Embeddings are identical to the CPU version
(same model, same normalization), so no experimental semantics change.
"""
import os
import queue
import threading
from typing import List, Union

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-en-v1.5")
PORT = int(os.environ.get("EMB_PORT", "8011"))
DEV = os.environ.get("EMB_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH = int(os.environ.get("EMB_BATCH", "256"))
WAIT_MS = float(os.environ.get("EMB_WAIT_MS", "8"))

app = FastAPI()
print(f"loading {MODEL} on {DEV} ...", flush=True)
_m = SentenceTransformer(MODEL, device=DEV)
_m.max_seq_length = 512
_q: "queue.Queue" = queue.Queue()


def _worker() -> None:
    """Collect requests arriving within a short window into one batch before inference."""
    while True:
        item = _q.get()
        items = [item]
        # wait a moment longer to collect concurrently arriving requests
        deadline = threading.Event()
        threading.Timer(WAIT_MS / 1000.0, deadline.set).start()
        while not deadline.is_set():
            try:
                items.append(_q.get_nowait())
            except queue.Empty:
                if deadline.wait(WAIT_MS / 1000.0):
                    break
        flat, spans = [], []
        for texts, _ev, _out in items:
            spans.append((len(flat), len(flat) + len(texts)))
            flat.extend(texts)
        try:
            v = _m.encode(flat, normalize_embeddings=True, batch_size=BATCH,
                          show_progress_bar=False, convert_to_numpy=True)
            for (a, b), (_t, ev, out) in zip(spans, items):
                out.append(v[a:b])
                ev.set()
        except Exception as e:  # wake the caller even on failure, otherwise the request hangs forever
            for _t, ev, out in items:
                out.append(e)
                ev.set()


threading.Thread(target=_worker, daemon=True).start()
print("ready", flush=True)


class Req(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL


@app.get("/v1/models")
def models():
    return {"object": "list",
            "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]}


@app.post("/v1/embeddings")
def embed(r: Req):
    texts = [r.input] if isinstance(r.input, str) else r.input
    ev, out = threading.Event(), []
    _q.put((texts, ev, out))
    ev.wait()
    res = out[0]
    if isinstance(res, Exception):
        raise res
    return {"object": "list", "model": r.model,
            "data": [{"object": "embedding", "index": i, "embedding": e.tolist()}
                     for i, e in enumerate(res)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
