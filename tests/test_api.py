"""AC-API-1, AC-API-2 (stretch). Server runs in-process on an ephemeral port."""

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

import serve
from conftest import SAMPLE_KB, SAMPLE_QUERIES, load
from rag import config


@pytest.fixture(scope="module")
def base_url():
    index = serve.build_index(SAMPLE_KB, SAMPLE_QUERIES, config.PRIMARY_CHUNKING)
    server = HTTPServer(("127.0.0.1", 0), serve.make_handler(index, config.TOP_K, None))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(url, data: bytes):
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_ac_api_1_matches_batch(base_url, sample_out, sample_corpus):
    _, queries = sample_corpus
    batch = {a["query_id"]: a for a in load(sample_out, "answers.json")}
    for q in queries:
        status, body = post(base_url + "/answer", json.dumps({"question": q.question}).encode())
        assert status == 200
        assert set(body) == {"answer_label", "answer", "citations"}
        b = batch[q.query_id]
        assert (body["answer_label"], body["answer"], body["citations"]) == \
            (b["answer_label"], b["answer"], b["citations"])


@pytest.mark.parametrize("payload", [b"not json", b"{}", b'{"question": ""}', b'{"question": 5}', b"[1]"])
def test_ac_api_2_bad_body_400_and_server_survives(base_url, payload):
    assert post(base_url + "/answer", payload)[0] == 400
    with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
        assert json.loads(r.read()) == {"status": "ok"}


def test_unknown_path_404(base_url):
    assert post(base_url + "/nope", b"{}")[0] == 404
