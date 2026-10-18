"""OpenAI 兼容的 /v1/embeddings，跑在 GPU 上并做微批合并。

替换 emb_server.py 的两个瓶颈：
  1) CPU 推理 —— bge-small 只有 130 MB，与 vLLM 共享一张卡绰绰有余；
  2) 一把全局锁把所有请求串行化，使 GraphRAG 的 concurrent_requests=25 完全失效。

这里改为单后台线程消费队列：并发请求先合并成大批再送 GPU，既避开
sentence-transformers 的线程不安全，又拿到批处理吞吐。嵌入结果与 CPU 版一致
（同模型、同 normalize），不改变任何实验语义。
"""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import os
import queue
import threading
from typing import List, Union
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
MODEL = os.environ.get('EMB_MODEL', 'BAAI/bge-small-en-v1.5')
PORT = int(os.environ.get('EMB_PORT', '8011'))
DEV = os.environ.get('EMB_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
BATCH = int(os.environ.get('EMB_BATCH', '256'))
WAIT_MS = float(os.environ.get('EMB_WAIT_MS', '8'))
app = FastAPI()
print(f'loading {MODEL} on {DEV} ...', flush=True)
_m = SentenceTransformer(MODEL, device=DEV)
_m.max_seq_length = 512
_q: 'queue.Queue' = queue.Queue()

def _worker() -> None:
    """把短时间内到达的请求攒成一批再推理。"""
    while True:
        item = _q.get()
        items = [item]
        deadline = threading.Event()
        threading.Timer(WAIT_MS / 1000.0, deadline.set).start()
        while not deadline.is_set():
            try:
                items.append(_q.get_nowait())
            except queue.Empty:
                if deadline.wait(WAIT_MS / 1000.0):
                    break
        (flat, spans) = ([], [])
        for (texts, _ev, _out) in items:
            spans.append((len(flat), len(flat) + len(texts)))
            flat.extend(texts)
        try:
            v = _m.encode(flat, normalize_embeddings=True, batch_size=BATCH, show_progress_bar=False, convert_to_numpy=True)
            for ((a, b), (_t, ev, out)) in zip(spans, items):
                out.append(v[a:b])
                ev.set()
        except Exception as e:
            for (_t, ev, out) in items:
                out.append(e)
                ev.set()
threading.Thread(target=_worker, daemon=True).start()
print('ready', flush=True)

class Req(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL

@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': MODEL, 'object': 'model', 'owned_by': 'local'}]}

@app.post('/v1/embeddings')
def embed(r: Req):
    texts = [r.input] if isinstance(r.input, str) else r.input
    (ev, out) = (threading.Event(), [])
    _q.put((texts, ev, out))
    ev.wait()
    res = out[0]
    if isinstance(res, Exception):
        raise res
    return {'object': 'list', 'model': r.model, 'data': [{'object': 'embedding', 'index': i, 'embedding': e.tolist()} for (i, e) in enumerate(res)], 'usage': {'prompt_tokens': 0, 'total_tokens': 0}}
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='warning')
