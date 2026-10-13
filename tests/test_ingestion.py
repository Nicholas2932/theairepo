"""AC-ING-1..4 and AC-ROB-2."""

import json
import re
import shutil

from conftest import FIXTURES, SAMPLE_KB, SAMPLE_QUERIES, load, run
from rag import chunking, config, loader, vocab
from rag.textproc import sentence_spans

REQUIRED = ("chunk_id", "doc_title", "section", "text", "start_char", "end_char")


def test_ac_ing_1_headers_parsed_and_fields_present(sample_corpus, sample_out):
    docs, _ = sample_corpus
    for d in docs:
        raw = loader.read_normalised(next(p for p in SAMPLE_KB.iterdir()
                                          if p.name == d.source_file.split("/")[-1]))
        title = re.search(r"^Title:\s*(.+)$", raw, re.M).group(1).strip()
        section = re.search(r"^Section:\s*(.+)$", raw, re.M).group(1).strip()
        assert d.title == title and d.segments[0].section == section
        assert "Title:" not in d.body and not d.body.startswith("Section:")
    for c in load(sample_out, "chunks.json"):
        assert all(k in c for k in REQUIRED)
        assert isinstance(c["start_char"], int) and isinstance(c["end_char"], int)


def test_ac_ing_2_exact_slices_and_full_coverage(sample_corpus, sample_out):
    docs, _ = sample_corpus
    by_id = {d.doc_id: d for d in docs}
    chunks = load(sample_out, "chunks.json")
    for c in chunks:
        assert by_id[c["doc_id"]].body[c["start_char"]:c["end_char"]] == c["text"]
        assert c["text"] == c["text"].strip()
    for d in docs:
        sentences = [d.body[s:e] for seg in d.segments for s, e in sentence_spans(d.body, seg.start, seg.end)]
        mine = [c for c in chunks if c["doc_id"] == d.doc_id]
        joined = [c["text"][s:e] for c in mine for s, e in sentence_spans(c["text"])]
        assert joined == sentences   # every sentence exactly once, in order
        spans = sorted((c["start_char"], c["end_char"]) for c in mine)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # no overlap


def _strip(chunks):
    return [{k: v for k, v in c.items() if k != "source_file"} for c in chunks]


def test_ac_ing_3_crlf_and_bom(tmp_path, sample_out):
    kb = tmp_path / "kb"
    kb.mkdir()
    for p in SAMPLE_KB.glob("*.txt"):
        text = p.read_text(encoding="utf-8")
        (kb / p.name).write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
    out = tmp_path / "out"
    assert run(kb, SAMPLE_QUERIES, out) == 0
    assert load(out, "chunks.json") == load(sample_out, "chunks.json")


def test_ac_ing_4_rename_and_shuffle(tmp_path, sample_out):
    kb = tmp_path / "kb"
    (kb / "nested").mkdir(parents=True)
    files = sorted(SAMPLE_KB.glob("*.txt"))
    for i, p in enumerate(reversed(files)):
        target = (kb / "nested" if i % 2 else kb) / f"{chr(122 - i)}_{i}.txt"
        shutil.copy(p, target)
    out = tmp_path / "out"
    assert run(kb, SAMPLE_QUERIES, out) == 0
    assert _strip(load(out, "chunks.json")) == _strip(load(sample_out, "chunks.json"))
    assert load(out, "retrieval.json") == load(sample_out, "retrieval.json")
    assert load(out, "answers.json") == load(sample_out, "answers.json")


def test_ac_rob_2_edge_documents(tmp_path):
    docs, warnings = loader.load_documents(FIXTURES / "edge_kb")
    joined = "\n".join(warnings)
    assert "missing Title" in joined
    assert "no body text" in joined          # empty file skipped
    assert "duplicate title" in joined
    titles = [d.title for d in docs]
    assert "Some opening line without a title header." in titles
    multi = next(d for d in docs if d.source_file.endswith("multi.txt"))
    assert [s.section for s in multi.segments] == ["", "Fees", "Limits"]
    chunks = chunking.chunk_documents(docs, config.PRIMARY_CHUNKING)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        assert "Section:" not in c.text
        if c.doc_id == multi.doc_id:
            assert c.section == {"Intro": "", "Fees": "Fees", "Daily": "Limits"}[c.text.split()[0]]
    assert any("€" in c.text and "été" in c.text for c in chunks)

    out = tmp_path / "out"
    assert run(FIXTURES / "edge_kb", FIXTURES / "edge_queries.json", out) == 0
    e2 = next(r for r in load(out, "eval.json")["results"] if r["query_id"] == "e2")
    assert e2["retrieval_status"] == vocab.HIT and e2["matched_expected_title"] is None
    warnings = load(out, "run_manifest.json")["warnings"]
    assert any("duplicate title" in w for w in warnings)


def test_all_strategies_produce_exact_slices(sample_corpus):
    docs, _ = sample_corpus
    by_id = {d.doc_id: d for d in docs}
    for strategy in vocab.CHUNKING_STRATEGIES:
        for c in chunking.chunk_documents(docs, strategy):
            assert by_id[c.doc_id].body[c.start_char:c.end_char] == c.text
            assert c.text and c.text == c.text.strip()


def test_fixed_size_does_not_cut_words(sample_corpus):
    docs, _ = sample_corpus
    by_id = {d.doc_id: d for d in docs}
    for c in chunking.chunk_documents(docs, vocab.FIXED_SIZE):
        body = by_id[c.doc_id].body
        assert c.start_char == 0 or not body[c.start_char - 1].isalnum()
        assert c.end_char == len(body) or not body[c.end_char].isalnum()


def test_queries_validation(tmp_path):
    import pytest
    bad = tmp_path / "q.json"
    for payload in ([{"query_id": "a", "question": "x"}, {"query_id": "a", "question": "y"}],
                    [{"query_id": "a", "question": "  "}], [], {"nope": 1}):
        bad.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(loader.InputError):
            loader.load_queries(bad)
