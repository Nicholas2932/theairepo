"""AC-GRD-1, AC-CMP-1, AC-CMP-2."""

from conftest import load
from rag import grounding, vocab


def test_ac_grd_1_sample_passes(sample_out):
    g = load(sample_out, "grounding_check.json")
    assert g["summary"]["failed"] == 0
    assert all(r["status"] == vocab.GROUNDING_PASS for r in g["results"])
    assert len(g["results"]) == len(load(sample_out, "answers.json"))


def _first(sample_out):
    a = load(sample_out, "answers.json")[0]
    e = next(e for e in load(sample_out, "retrieval.json") if e["query_id"] == a["query_id"])
    return a, e["top_k"]


def test_ac_grd_1_injected_number_fails(sample_out):
    a, top = _first(sample_out)
    cit = a["citations"][0]
    tampered = dict(a, answer=a["answer"].replace(cit, f"It costs 77 USD. {cit}", 1))
    r = grounding.check_answer(tampered, top)
    assert r["status"] == vocab.GROUNDING_FAIL


def test_ac_grd_1_unsupported_sentence_fails(sample_out):
    a, top = _first(sample_out)
    tampered = dict(a, answer=f"Customers receive a complimentary hamper. {a['answer']}")
    assert grounding.check_answer(tampered, top)["status"] == vocab.GROUNDING_FAIL


def test_insufficient_is_not_applicable(sample_out):
    a, top = _first(sample_out)
    refusal = dict(a, answer_label=vocab.INSUFFICIENT_CONTEXT, answer=vocab.INSUFFICIENT_ANSWER_TEXT,
                   citations=[], used_chunk_ids=[])
    assert grounding.check_answer(refusal, top)["status"] == vocab.GROUNDING_NOT_APPLICABLE


def test_ac_cmp_1_comparison_structure(sample_out):
    c = load(sample_out, "chunking_comparison.json")
    names = [s["name"] for s in c["strategies"]]
    assert len(names) >= 2 and set(names) == set(vocab.CHUNKING_STRATEGIES)
    for s in c["strategies"]:
        assert {"top3_hit_rate", "hits", "partial_hits", "misses", "mrr"} <= set(s["summary"])
        assert s["num_chunks"] > 0
    assert c["tradeoff_explanation"].strip() and c["caveat"].strip()


def test_ac_cmp_2_primary_matches_eval(sample_out):
    c = load(sample_out, "chunking_comparison.json")
    primary = next(s for s in c["strategies"] if s["name"] == c["primary_strategy"])
    ev = load(sample_out, "eval.json")["summary"]
    for k, v in primary["summary"].items():
        assert ev[k] == v, k
    assert primary["num_chunks"] == len(load(sample_out, "chunks.json"))
