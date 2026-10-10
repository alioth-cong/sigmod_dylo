#!/usr/bin/env python3
"""OpenAI-compatible /v1/embeddings on CPU, for GraphRAG's embedding stage.

Runs on CPU deliberately: the embedding model is 130 MB and the GPUs are in use
by another user. GraphRAG only needs this endpoint to answer /v1/embeddings and
/v1/models, so nothing else is implemented.
"""
import os, threading
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-en-v1.5")
app = FastAPI()
_lock = threading.Lock()
print(f"loading {MODEL} on CPU ...", flush=True)
_m = SentenceTransformer(MODEL, device="cpu")
_m.max_seq_length = 512
print("ready", flush=True)

class Req(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]}

@app.post("/v1/embeddings")
def embed(r: Req):
    texts = [r.input] if isinstance(r.input, str) else r.input
    with _lock:                       # sentence-transformers is not thread-safe
        v = _m.encode(texts, normalize_embeddings=True, batch_size=64,
                      show_progress_bar=False, convert_to_numpy=True)
    return {"object": "list", "model": r.model,
            "data": [{"object": "embedding", "index": i, "embedding": e.tolist()}
                     for i, e in enumerate(v)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("EMB_PORT", "8001")),
                log_level="warning")
