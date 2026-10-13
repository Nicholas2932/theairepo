"""AC-ANS-1..6 and AC-LLM-2 (with a stub client, no network)."""

import ast
import hashlib
import json

import pytest

from conftest import FIXTURES, REPO, SAMPLE_KB, load, run
from rag import answering, llm, retrieval, vocab
from rag.textproc import content_terms


def test_ac_ans_1_labels_and_citation_counts(sample_out):
    for a in load(sample_out, "answers.json"):
        assert a["answer_label"] in vocab.ANSWER_LABELS
        n = len(a["citations"])
        if a["answer_label"] == vocab.INSUFFICIENT_CONTEXT:
            assert n == 0 and a["used_chunk_ids"] == []
        else:
            assert n >= vocab.MIN_CITATIONS[a["answer_label"]]


def test_ac_ans_2_citations_retrieved_and_title_matched(sample_out):
    by_q = {e["query_id"]: {h["chunk_id"]: h for h in e["top_k"]}
            for e in load(sample_out, "retrieval.json")}
    for a in load(sample_out, "answers.json"):
        for c in a["citations"]:
            title, cid = vocab.parse_citation(c)
            assert cid in by_q[a["query_id"]]
            assert by_q[a["query_id"]][cid]["doc_title"] == title
            assert c in a["answer"]


def test_ac_ans_3_used_chunk_ids_match_citations(sample_out):
    for a in load(sample_out, "answers.json"):
        assert a["used_chunk_ids"] == [vocab.parse_citation(c)[1] for c in a["citations"]]


def test_ac_ans_4_sample_answers_cover_ground_truth(sample_out, sample_corpus):
    _, queries = sample_corpus
    answers = {a["query_id"]: a for a in load(sample_out, "answers.json")}
    for q in queries:
        a = answers[q.query_id]
        assert a["answer_label"] == vocab.GROUNDED_ANSWER, q.query_id
        got = set(content_terms(a["answer"]))
        for point in q.ground_truth_answer_points:
            want = set(content_terms(point))
            assert len(want & got) / len(want) >= 0.8, (q.query_id, point, a["answer"])


def test_ac_ans_5_answering_has_no_kb_access():
    tree = ast.parse((REPO / "rag" / "answering.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not imported & {"loader", "chunking", "retrieval", "rag.loader", "rag.chunking"}
    with pytest.raises(TypeError):
        answering.answer_stage.__wrapped__(None, object(), None)


def test_ac_ans_6_out_of_scope_probes(tmp_path):
    out = tmp_path / "out"
    assert run(SAMPLE_KB, FIXTURES / "probe_queries.json", out) == 0
    answers = load(out, "answers.json")
    assert len(answers) >= 6
    for a in answers:
        assert a["answer_label"] == vocab.INSUFFICIENT_CONTEXT, a
        assert a["answer"] == vocab.INSUFFICIENT_ANSWER_TEXT


def test_ac_ans_6_conflicting_kb(tmp_path):
    out = tmp_path / "out"
    assert run(FIXTURES / "conflict_kb", FIXTURES / "conflict_queries.json", out) == 0
    (a,) = load(out, "answers.json")
    assert a["answer_label"] == vocab.CONFLICTING_CONTEXT
    assert len(set(a["used_chunk_ids"])) >= 2
    assert "30 minutes" in a["answer"] and "60 minutes" in a["answer"]


def test_ac_ans_6_no_false_conflicts_on_sample(sample_out):
    assert all(a["answer_label"] != vocab.CONFLICTING_CONTEXT for a in load(sample_out, "answers.json"))


def test_negation_conflict(sample_index):
    from rag.stages import PRODUCER_SEAL, QueryRetrieval, RetrievedHit
    hits = tuple(RetrievedHit(rank=i + 1, chunk_id=f"chunk_{i + 1}", doc_title=t, section="",
                              score=1.0 - i / 10, chunk_text=txt, start_char=0, bm25=1, ngram=1)
                 for i, (t, txt) in enumerate([("A", "Demo profits can be withdrawn to a bank."),
                                               ("B", "Demo profits cannot be withdrawn to a bank."),
                                               ("C", "Unrelated text about fees.")]))
    terms = ("demo", "profit", "withdrawn")
    qr = QueryRetrieval(_seal=PRODUCER_SEAL, query_id="n", question="Can demo profits be withdrawn?",
                        hits=hits, query_terms=terms, term_idf={t: 1.5 for t in terms},
                        idf_ceiling=1.5, n_chunks=3)
    assert answering.answer_extractive(qr)["answer_label"] == vocab.CONFLICTING_CONTEXT


# --- AC-LLM-2 -----------------------------------------------------------------

class StubClient:
    provider, model, temperature = "stub", "stub-model", 0

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, prompt):
        self.prompts.append(prompt.text)
        return self.reply(prompt)


def _good_reply(prompt):
    line = next(l for l in prompt.user.splitlines() if "citation token:" in l)
    cit = line.split("citation token: ", 1)[1]
    text = next(l for l in prompt.user.splitlines() if l.startswith("text: "))[6:].splitlines()[0]
    first = text.split("\n")[0]
    return json.dumps({"answer_label": "grounded_answer", "answer": f"{first} {cit}", "citations": [cit]})


@pytest.fixture
def llm_run(tmp_path, monkeypatch):
    def go(reply):
        client = StubClient(reply)
        monkeypatch.setattr(llm, "make_client", lambda provider, model=None: client)
        out = tmp_path / "out"
        from conftest import SAMPLE_QUERIES
        code = run(SAMPLE_KB, SAMPLE_QUERIES, out, "--llm", "gemini")
        return code, out, client
    return go


def test_ac_llm_2_one_record_per_query(llm_run, sample_corpus):
    code, out, client = llm_run(_good_reply)
    assert code == 0
    records = [json.loads(l) for l in (out / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["query_id"] for r in records] == [q.query_id for q in sample_corpus[1]]
    required = {"stage", "query_id", "timestamp", "provider", "model", "prompt_hash",
                "input_artifacts", "output_artifact", "temperature", "fallback", "latency_ms"}
    for r, prompt in zip(records, client.prompts):
        assert required <= set(r)
        assert r["prompt_hash"] == "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert load(out, "run_manifest.json")["answer_generator"] == "llm"
    assert any(a["generator"] == "llm" for a in load(out, "answers.json"))
    from conftest import validate
    assert validate(out).ok


@pytest.mark.parametrize("reply", [
    lambda p: "not json",
    lambda p: json.dumps({"answer_label": "grounded_answer", "answer": "Made up 99 days. [X §chunk_1]",
                          "citations": ["[X §chunk_1]"]}),
    lambda p: json.dumps({"answer_label": "maybe", "answer": "", "citations": []}),
    lambda p: (_ for _ in ()).throw(llm.LLMError("HTTP 500")),
], ids=["not-json", "hallucinated", "bad-label", "api-error"])
def test_ac_llm_2_invalid_output_falls_back(llm_run, reply):
    code, out, _ = llm_run(reply)
    assert code == 0
    answers = load(out, "answers.json")
    assert all(a["generator"] == vocab.GENERATOR_LLM_FALLBACK for a in answers)
    records = [json.loads(l) for l in (out / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["fallback"] for r in records)
    from conftest import validate
    assert validate(out).ok


def test_llm_never_overrides_sufficiency_gate(sample_index):
    qr = retrieval.retrieve_query(sample_index, "x", "What is the capital of France?")
    h = qr.hits[0]
    cit = vocab.format_citation(h.doc_title, h.chunk_id)
    reply = json.dumps({"answer_label": "grounded_answer", "answer": f"{h.chunk_text} {cit}", "citations": [cit]})
    record, call = answering.answer_with_llm(qr, StubClient(lambda p: reply))
    assert record["answer_label"] == vocab.INSUFFICIENT_CONTEXT and call["fallback"]


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_load_dotenv", lambda path=None: None)  # ignore a local .env
    with pytest.raises(llm.LLMError):
        llm.make_client("gemini")
