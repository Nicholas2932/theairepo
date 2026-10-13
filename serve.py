"""Minimal local API (stretch, SPEC §6.9). Stdlib only, binds 127.0.0.1.

    python serve.py --port 8000
    POST /answer  {"question": "..."}  ->  {"answer_label", "answer", "citations"}
    GET  /health                       ->  {"status": "ok"}

At startup the pipeline runs up to INDEX_BUILT (no artifacts are written).
Each request then uses the same retrieval and answering functions as the
batch pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from rag import answering, chunking, config, loader, retrieval
from rag.stages import Pipeline


def build_index(kb_dir: Path, queries_path: Path, strategy: str):
    pipe = Pipeline()
    corpus = loader.load_stage(pipe, kb_dir, queries_path)
    chunkset = chunking.chunk_stage(pipe, corpus, strategy, None)
    return retrieval.build_index_stage(pipe, chunkset)


def make_handler(index, top_k: int, client):
    class Handler(BaseHTTPRequestHandler):
        server_version = "mini-rag/1.0"

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/answer":
                self._send(405, {"error": "use POST"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/answer":
                self._send(404 if self.path != "/health" else 405, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > config.API_MAX_BODY_BYTES:
                self._send(400, {"error": "request body missing or too large"})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send(400, {"error": "body must be JSON"})
                return
            question = body.get("question") if isinstance(body, dict) else None
            if not isinstance(question, str) or not question.strip():
                self._send(400, {"error": "'question' must be a non-empty string"})
                return
            qr = retrieval.retrieve_query(index, "api", question, top_k)
            record, _ = answering.answer_query(qr, client)
            self._send(200, {"answer_label": record["answer_label"], "answer": record["answer"],
                             "citations": record["citations"]})

        def log_message(self, fmt, *args):
            sys.stderr.write(f"[serve] {self.address_string()} {fmt % args}\n")

    return Handler


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=config.API_DEFAULT_PORT)
    p.add_argument("--kb", type=Path, default=config.DEFAULT_KB_DIR)
    p.add_argument("--queries", type=Path, default=config.DEFAULT_QUERIES_PATH,
                   help="needed only because the load stage validates the full input set")
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.add_argument("--llm", choices=("none",) + config.LLM_PROVIDERS, default="none")
    p.add_argument("--llm-model", default=None)
    args = p.parse_args(argv)

    client = None
    if args.llm != "none":
        from rag import llm
        client = llm.make_client(args.llm, args.llm_model)
    index = build_index(args.kb, args.queries, config.PRIMARY_CHUNKING)
    server = HTTPServer((config.API_HOST, args.port), make_handler(index, args.top_k, client))
    print(f"serving on http://{config.API_HOST}:{args.port} ({index.n} chunks indexed)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
