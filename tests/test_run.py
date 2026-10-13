"""AC-RUN-1..4, AC-LLM-1, AC-ROB-1."""

import hashlib
import re

from conftest import (DETERMINISTIC, REPO, SAMPLE_KB, SAMPLE_QUERIES, load, run, validate)
from rag import config
from rag.stages import STAGE_ORDER, Stage


def test_ac_run_1_clean_run_writes_all_artifacts(sample_out):
    for name in DETERMINISTIC + ("run_manifest.json",):
        assert (sample_out / name).is_file(), name


def test_ac_run_1_from_missing_artifacts_dir(tmp_path):
    out = tmp_path / "does" / "not" / "exist"
    assert run(SAMPLE_KB, SAMPLE_QUERIES, out) == 0
    assert (out / "answers.json").is_file()


def test_ac_run_2_fresh_run_validates(sample_out):
    report = validate(sample_out)
    assert report.ok, report.render()


def test_ac_run_3_deterministic(sample_out, tmp_path):
    out2 = tmp_path / "second"
    assert run(SAMPLE_KB, SAMPLE_QUERIES, out2) == 0
    for name in DETERMINISTIC:
        h1 = hashlib.sha256((sample_out / name).read_bytes()).hexdigest()
        h2 = hashlib.sha256((out2 / name).read_bytes()).hexdigest()
        assert h1 == h2, name


def test_ac_run_4_final_state_and_stage_log(sample_out):
    m = load(sample_out, "run_manifest.json")
    assert m["status"] == "ok"
    assert m["final_state"] == Stage.RESULTS_FINALISED.name
    assert [e["stage"] for e in m["stage_log"]] == [s.name for s in STAGE_ORDER]


def test_failed_run_is_loud(tmp_path):
    out = tmp_path / "out"
    assert run(tmp_path / "no_kb", SAMPLE_QUERIES, out) != 0
    m = load(out, "run_manifest.json")
    assert m["status"] == "failed" and m["final_state"] != Stage.RESULTS_FINALISED.name
    assert not validate(out).ok


def test_stale_artifacts_removed_at_start(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / config.LLM_CALLS_FILE).write_text("{}\n", encoding="utf-8")
    assert run(SAMPLE_KB, SAMPLE_QUERIES, out) == 0
    assert not (out / config.LLM_CALLS_FILE).exists()


def test_ac_llm_1_no_llm_by_default(sample_out):
    assert not (sample_out / config.LLM_CALLS_FILE).exists()
    assert load(sample_out, "run_manifest.json")["answer_generator"] == "extractive"
    assert all(a["generator"] == "extractive" for a in load(sample_out, "answers.json"))
    raw = (REPO / "README.md").read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        readme = raw.decode("utf-16")
    elif raw[1:2] == b"\x00":                 # UTF-16LE without BOM
        readme = raw.decode("utf-16-le")
    else:
        readme = raw.decode("utf-8-sig")
    assert re.search(r"no LLM is used", readme, re.I)


def test_ac_rob_1_no_sample_literals_in_code(sample_corpus):
    docs, queries = sample_corpus
    literals = [d.title for d in docs] + [q.query_id for q in queries] + ["chunk_2"]
    files = list((REPO / "rag").glob("*.py")) + [REPO / n for n in
                                                  ("run_pipeline.py", "validate.py", "serve.py")]
    for f in files:
        text = f.read_text(encoding="utf-8")
        for lit in literals:
            pattern = r"[\"']" + re.escape(lit) + r"[\"']" if re.fullmatch(r"\w+", lit) else re.escape(lit)
            assert not re.search(pattern, text), f"{lit!r} hardcoded in {f.name}"
