"""AC-RET-1..3."""

import ast
import math

from conftest import REPO, load
from rag import config, retrieval
from rag.textproc import norm_title


def test_ac_ret_1_shape_and_order(sample_out):
    n_chunks = len(load(sample_out, "chunks.json"))
    for e in load(sample_out, "retrieval.json"):
        top = e["top_k"]
        assert len(top) >= min(3, n_chunks)
        assert [h["rank"] for h in top] == list(range(1, len(top) + 1))
        scores = [h["score"] for h in top]
        assert all(isinstance(s, float) and math.isfinite(s) for s in scores)
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)


def test_ac_ret_1_tiny_corpus_returns_all_chunks(sample_corpus):
    from rag import chunking
    docs, _ = sample_corpus
    chunks = chunking.chunk_documents(docs[:1], "paragraph")
    qr = retrieval.retrieve_query(retrieval.build_index(chunks), "x", "anything at all")
    assert len(qr.hits) == min(3, len(chunks))


def test_ac_ret_2_sample_hit_rate(sample_out, sample_corpus):
    _, queries = sample_corpus
    assert load(sample_out, "eval.json")["summary"]["top3_hit_rate"] == 1.0
    by_q = {e["query_id"]: e for e in load(sample_out, "retrieval.json")}
    for q in queries:
        top3 = {norm_title(h["doc_title"]) for h in by_q[q.query_id]["top_k"][:3]}
        assert {norm_title(t) for t in q.expected_doc_titles} <= top3


def test_ac_ret_3_retrieval_never_reads_ground_truth(sample_index, sample_corpus):
    src = (REPO / "rag" / "retrieval.py").read_text(encoding="utf-8")
    assert "expected_doc_titles" not in src and "ground_truth" not in src

    class Q:  # a query object exposing only id + question
        __slots__ = ("query_id", "question")

        def __init__(self, q):
            self.query_id, self.question = q.query_id, q.question

    _, queries = sample_corpus
    full = retrieval.retrieve_all(sample_index, queries)
    bare = retrieval.retrieve_all(sample_index, [Q(q) for q in queries])
    assert retrieval.retrieval_to_json(full) == retrieval.retrieval_to_json(bare)


def test_query_wording_variants_still_hit(sample_index):
    qr = retrieval.retrieve_query(sample_index, "x", "withdrawn demo profits?")
    assert qr.hits[0].doc_title == next(c.doc_title for c in sample_index.chunks
                                        if "demo" in c.text.lower())


def test_score_components_recorded(sample_out):
    for e in load(sample_out, "retrieval.json"):
        for h in e["top_k"]:
            assert set(h["score_components"]) == set(config.FUSION_WEIGHTS)
