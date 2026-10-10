#!/usr/bin/env python3
"""Truncating proxy in front of the embedding service.

bge-small-en-v1.5 has a maximum sequence length of 512. GraphRAG's generate_text_embeddings
neither truncates nor sends truncate_prompt_tokens, so on overlong text vLLM returns 400,
the workflow fails -> the whole update pipeline aborts -> none of the 8 subsequent update_* merge workflows
run, so the increment is computed but never merged back into output/ (measured: documents stuck at 304).

This proxy forwards requests unchanged and only adds truncate_prompt_tokens. It listens on the original port
and vLLM moves to a backend port, so settings.yaml needs no change.

Usage: embed_truncate_proxy.py --port 8011 --backend http://127.0.0.1:8013 --max-tokens 512
"""
import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(backend: str, max_tokens: int):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence the access log
            pass

        def _relay(self, method: str):
            body = b""
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                body = self.rfile.read(n)
            # inject the truncation parameter only into embeddings requests; forward everything else unchanged
            if "/embeddings" in self.path and body:
                try:
                    d = json.loads(body)
                    d.setdefault("truncate_prompt_tokens", max_tokens)
                    body = json.dumps(d).encode()
                except Exception:  # noqa: BLE001
                    pass
            req = urllib.request.Request(
                backend + self.path, data=body if body else None, method=method,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data, code = r.read(), r.status
            except urllib.error.HTTPError as e:
                data, code = e.read(), e.code
            except Exception as e:  # noqa: BLE001
                data = json.dumps({"error": {"message": f"proxy: {e}"}}).encode()
                code = 502
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self._relay("POST")

        def do_GET(self):
            self._relay("GET")

    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--backend", default="http://127.0.0.1:8013")
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.backend, a.max_tokens))
    print(f"truncating proxy :{a.port} -> {a.backend} (truncate_prompt_tokens={a.max_tokens})",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
