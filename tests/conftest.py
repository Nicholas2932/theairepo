"""Shared fixtures. Pipeline runs happen in-process into tmp directories, so
the repo's own artifacts/ is never touched by the test suite."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_KB = REPO / "kb"
SAMPLE_QUERIES = REPO / "queries.json"
sys.path.insert(0, str(REPO))

import run_pipeline  # noqa: E402
from rag import config, validation  # noqa: E402

DETERMINISTIC = ("chunks.json", "retrieval.json", "answers.json", "eval.json",
                 "grounding_check.json", "chunking_comparison.json")


def run(kb: Path, queries: Path, out: Path, *extra: str) -> int:
    return run_pipeline.main(["--kb", str(kb), "--queries", str(queries), "--out", str(out), *extra])


def load(out: Path, name: str):
    return json.loads((out / name).read_text(encoding="utf-8"))


def validate(out: Path, kb: Path = SAMPLE_KB, queries: Path = SAMPLE_QUERIES):
    return validation.validate_artifacts(out, kb, queries)


@pytest.fixture(scope="session")
def sample_out(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("sample") / "artifacts"
    assert run(SAMPLE_KB, SAMPLE_QUERIES, out) == 0
    return out


@pytest.fixture
def sample_copy(sample_out, tmp_path) -> Path:
    """A mutable copy of the sample artifacts."""
    dst = tmp_path / "artifacts"
    shutil.copytree(sample_out, dst)
    return dst


@pytest.fixture(scope="session")
def sample_corpus():
    from rag import loader
    docs, _ = loader.load_documents(SAMPLE_KB)
    return docs, loader.load_queries(SAMPLE_QUERIES)


@pytest.fixture(scope="session")
def sample_index(sample_corpus):
    from rag import chunking, retrieval
    return retrieval.build_index(chunking.chunk_documents(sample_corpus[0], config.PRIMARY_CHUNKING))
