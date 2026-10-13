"""AC-EVAL-1..3."""

import ast

import pytest

from conftest import REPO, load
from rag import evaluation, vocab
from rag.loader import Query

FIELDS = ("query_id", "expected_doc_titles", "retrieved_doc_titles_top3", "retrieval_status",
          "matched_expected_title", "explanation")


def test_ac_eval_1_structure(sample_out, sample_corpus):
    ev = load(sample_out, "eval.json")
    assert [r["query_id"] for r in ev["results"]] == [q.query_id for q in sample_corpus[1]]
    for r in ev["results"]:
        assert all(f in r for f in FIELDS)
        assert r["retrieval_status"] in vocab.RETRIEVAL_STATUS
        assert len(r["retrieved_doc_titles_top3"]) <= 3
    s = ev["summary"]
    for k in ("top3_hit_rate", "total_queries", "hits", "partial_hits", "misses"):
        assert k in s
    assert s["hits"] + s["partial_hits"] + s["misses"] == s["total_queries"] == len(ev["results"])


def entry(*titles):
    return {"query_id": "q", "top_k": [{"rank": i + 1, "doc_title": t, "chunk_text": "x"}
                                       for i, t in enumerate(titles)]}


def query(*expected):
    return Query("q", "question?", tuple(expected), ())


KB = {"alpha", "beta", "gamma", "delta"}


@pytest.mark.parametrize("ent, q, status, matched, explanation", [
    (entry("Alpha", "Beta", "Gamma"), query("Alpha"), vocab.HIT, True, "Expected title found at rank 1"),
    (entry("Beta", "Alpha", "Gamma"), query("alpha "), vocab.HIT, True, "Expected title found at rank 2"),
    (entry("Alpha", "Beta", "Gamma"), query("Alpha", "Beta"), vocab.HIT, True, "All 2 expected titles found in top 3"),
    (entry("Alpha", "Gamma", "Gamma"), query("Alpha", "Beta"), vocab.PARTIAL_HIT, True,
     '1 of 2 expected titles found in top 3 (missing: "Beta")'),
    (entry("Gamma", "Delta", "Gamma", "Alpha"), query("Alpha"), vocab.MISS, False,
     'Expected title "Alpha" not in top 3 (retrieved: "Gamma", "Delta")'),
    (entry("Alpha", "Beta", "Gamma"), query("Omega"), vocab.MISS, False,
     'Expected title "Omega" not present in knowledge base'),
    (entry("Alpha", "Beta", "Gamma"), query(), vocab.HIT, None,
     "No expected titles given; counted as a vacuous hit and excluded from the hit rate"),
], ids=["hit-r1", "hit-r2-normalised", "hit-multi", "partial", "miss", "not-in-kb", "vacuous"])
def test_ac_eval_2_table(ent, q, status, matched, explanation):
    r = evaluation.evaluate_query(ent, q, KB)
    assert (r["retrieval_status"], r["matched_expected_title"], r["explanation"]) == \
        (status, matched, explanation)


def test_ac_eval_2_summary_excludes_vacuous():
    results = [evaluation.evaluate_query(entry("Alpha", "Beta", "Gamma"), q, KB)
               for q in (query("Alpha"), query("Omega"), query())]
    s = evaluation.summarise(results)
    assert s["total_queries"] == 3 and s["answerable_queries"] == 2
    assert s["hits"] == 2 and s["misses"] == 1 and s["vacuous_hits"] == 1
    assert s["top3_hit_rate"] == 0.5


def test_ac_eval_3_no_llm_or_answering_imports():
    for name in ("evaluation.py", "comparison.py"):
        tree = ast.parse((REPO / "rag" / name).read_text(encoding="utf-8"))
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                names.add(n.module or "")
                names |= {a.name for a in n.names}
            elif isinstance(n, ast.Import):
                names |= {a.name for a in n.names}
        assert not names & {"llm", "answering", "rag.llm", "rag.answering"}, name
