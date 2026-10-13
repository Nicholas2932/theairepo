"""AC-VAL-1, AC-VAL-2 (each mutation must fail validation), AC-RUN-2."""

import json
import shutil
import subprocess
import sys

import pytest

from conftest import REPO, SAMPLE_KB, SAMPLE_QUERIES, validate
from rag import validation


def test_ac_val_1_all_rules_implemented(sample_out):
    report = validate(sample_out)
    assert [r.rule for r in report.results] == [f"V-{i:02d}" for i in range(1, 14)]
    assert all(r.status == validation.PASS for r in report.results)


def edit(name, fn):
    def go(d, kb):
        data = json.loads((d / name).read_text(encoding="utf-8"))
        fn(data)
        (d / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return go


def first_answer(fn):
    return edit("answers.json", lambda x: fn(x[0]))


def swap_kb(d, kb):
    p = next(kb.glob("*.txt"))
    p.write_text(p.read_text(encoding="utf-8") + "\nAn extra line.\n", encoding="utf-8")


MUTATIONS = {
    "delete-artifact": (lambda d, kb: (d / "retrieval.json").unlink(), "V-01"),
    "corrupt-json": (lambda d, kb: (d / "answers.json").write_text("[{", encoding="utf-8"), "V-02"),
    "drop-query": (edit("eval.json", lambda x: x["results"].pop()), "V-03"),
    "truncate-to-2": (edit("retrieval.json", lambda x: x[0].update(top_k=x[0]["top_k"][:2])), "V-04"),
    "score-as-string": (edit("retrieval.json", lambda x: x[0]["top_k"][0].update(score="0.9")), "V-05"),
    "unknown-label": (first_answer(lambda a: a.update(answer_label="probably")), "V-06"),
    "grounded-no-citations": (first_answer(lambda a: a.update(
        answer="Plain text.", citations=[], used_chunk_ids=[])), "V-07"),
    "non-retrieved-chunk": (first_answer(lambda a: a.update(
        answer=a["answer"].replace(a["used_chunk_ids"][0], "chunk_999"),
        citations=[c.replace(a["used_chunk_ids"][0], "chunk_999") for c in a["citations"]],
        used_chunk_ids=["chunk_999"])), "V-08"),
    "wrong-title": (first_answer(lambda a: a.update(
        answer=a["answer"].replace("[", "[Wrong ", 1),
        citations=["[Wrong " + a["citations"][0][1:]] + a["citations"][1:])), "V-08"),
    "used-ids-mismatch": (first_answer(lambda a: a.update(used_chunk_ids=["chunk_1", "chunk_2"])), "V-08"),
    "unknown-status": (edit("eval.json", lambda x: x["results"][0].update(retrieval_status="ok")), "V-09"),
    "remove-summary": (edit("eval.json", lambda x: x.pop("summary")), "V-10"),
    "summary-miscount": (edit("eval.json", lambda x: x["summary"].update(hits=0)), "V-10"),
    "chunk-text-altered": (edit("chunks.json", lambda x: x[0].update(text=x[0]["text"].upper())), "V-11"),
    "swap-kb": (swap_kb, "V-12"),
    "injected-number": (first_answer(lambda a: a.update(
        answer=a["answer"].replace(a["answer"].split()[0], "In 42 days,", 1))), "V-13"),
    "stale-llm-calls": (lambda d, kb: (d / "llm_calls.jsonl").write_text("{}\n", encoding="utf-8"), "V-01"),
    "failed-manifest": (edit("run_manifest.json", lambda x: x.update(status="failed")), "V-01"),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_ac_val_2_mutations_fail(name, sample_copy, tmp_path):
    kb = tmp_path / "kb"
    shutil.copytree(SAMPLE_KB, kb)
    mutate, rule = MUTATIONS[name]
    mutate(sample_copy, kb)
    report = validation.validate_artifacts(sample_copy, kb, SAMPLE_QUERIES)
    assert not report.ok
    assert rule in {r.rule for r in report.failures()}, report.render()


def test_ac_run_2_cli_exit_codes(sample_copy):
    cmd = [sys.executable, str(REPO / "validate.py"), "--artifacts", str(sample_copy)]
    assert subprocess.run(cmd, capture_output=True, cwd=REPO).returncode == 0
    (sample_copy / "eval.json").unlink()
    assert subprocess.run(cmd, capture_output=True, cwd=REPO).returncode != 0
