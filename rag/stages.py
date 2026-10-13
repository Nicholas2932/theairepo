"""Pipeline state machine and typed stage hand-off objects (SPEC §5.2).

Enforcement:
* Pipeline.advance() only moves to the immediate successor stage.
* @stage_step functions check the pipeline is in their required stage before
  running and advance to their produced stage afterwards.
* Hand-off objects are "sealed": they can only be constructed by their
  producing module (which passes PRODUCER_SEAL). So answering cannot be called
  with retrieval results that did not come out of the retrieval code.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
from datetime import datetime, timezone
from typing import Any, Callable


class Stage(enum.Enum):
    INIT = 0
    DOCUMENTS_LOADED = 1
    DOCUMENTS_CHUNKED = 2
    INDEX_BUILT = 3
    RETRIEVAL_COMPLETE = 4
    ANSWERS_GENERATED = 5
    EVALUATION_COMPLETE = 6
    VALIDATION_COMPLETE = 7
    RESULTS_FINALISED = 8


STAGE_ORDER = tuple(Stage)


class StageOrderError(RuntimeError):
    """A stage was attempted out of order, or a hand-off object was forged."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Pipeline:
    def __init__(self) -> None:
        self._state = Stage.INIT
        self.log: list[dict] = [{"stage": Stage.INIT.name, "entered_at": utc_now()}]

    @property
    def state(self) -> Stage:
        return self._state

    def advance(self, next_stage: Stage) -> None:
        idx = STAGE_ORDER.index(self._state)
        expected = STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None
        if next_stage is not expected:
            allowed = expected.name if expected else "none (pipeline finished)"
            raise StageOrderError(
                f"cannot move from {self._state.name} to {next_stage.name}; "
                f"next allowed stage is {allowed}")
        self._state = next_stage
        self.log.append({"stage": next_stage.name, "entered_at": utc_now()})

    def require(self, stage: Stage, what: str = "operation") -> None:
        if self._state is not stage:
            raise StageOrderError(
                f"{what} requires stage {stage.name}, pipeline is at {self._state.name}")


def stage_step(requires: Stage, produces: Stage) -> Callable:
    """Decorate a stage function: fn(pipeline, *args) runs only at `requires`
    and moves the pipeline to `produces` when it returns successfully."""
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(pipeline: Pipeline, *args: Any, **kwargs: Any) -> Any:
            if not isinstance(pipeline, Pipeline):
                raise TypeError(f"{fn.__name__} must be called with a Pipeline")
            pipeline.require(requires, fn.__name__)
            result = fn(pipeline, *args, **kwargs)
            pipeline.advance(produces)
            return result
        wrapper.requires = requires  # type: ignore[attr-defined]
        wrapper.produces = produces  # type: ignore[attr-defined]
        return wrapper
    return decorate


# --- Sealed hand-off objects ----------------------------------------------

PRODUCER_SEAL = object()


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sealed:
    _seal: object = dataclasses.field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not PRODUCER_SEAL:
            raise StageOrderError(
                f"{type(self).__name__} can only be created by its producing stage")


@dataclasses.dataclass(frozen=True, kw_only=True)
class LoadedCorpus(Sealed):
    documents: tuple          # tuple[loader.Document]
    queries: tuple            # tuple[loader.Query]
    warnings: tuple
    fingerprint: str
    kb_dir: str
    queries_path: str
    num_files: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChunkSet(Sealed):
    chunks: tuple             # tuple[chunking.Chunk]
    strategy: str
    params: dict


@dataclasses.dataclass(frozen=True)
class RetrievedHit:
    rank: int
    chunk_id: str
    doc_title: str
    section: str
    score: float
    chunk_text: str
    start_char: int
    bm25: float
    ngram: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class QueryRetrieval(Sealed):
    """Everything the answer generator is allowed to see for one question:
    the question, its retrieved chunks, and corpus idf for the question's own
    terms (statistics only, no other KB text)."""
    query_id: str
    question: str
    hits: tuple               # tuple[RetrievedHit], rank order
    query_terms: tuple        # unique stemmed content terms, first-seen order
    term_idf: dict            # term -> idf (df = 0 terms included)
    idf_ceiling: float        # idf of a term occurring in exactly one chunk
    n_chunks: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class RetrievalResults(Sealed):
    per_query: tuple          # tuple[QueryRetrieval], queries.json order
    top_k: int
    num_chunks: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class AnswerSet(Sealed):
    answers: tuple            # tuple[dict], answers.json records
    generator_mode: str
    llm_calls: tuple
    warnings: tuple


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvaluationResults(Sealed):
    eval_doc: dict
    comparison: dict | None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ValidationOutcome(Sealed):
    grounding: dict
    report: Any               # validation.ValidationReport
