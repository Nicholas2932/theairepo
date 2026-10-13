"""AC-STG-1: every stage function raises StageOrderError from a wrong state."""

import pytest

import run_pipeline
from conftest import SAMPLE_KB, SAMPLE_QUERIES
from rag import answering, chunking, config, evaluation, loader, retrieval, validation
from rag.stages import (STAGE_ORDER, ChunkSet, LoadedCorpus, Pipeline, QueryRetrieval,
                        RetrievalResults, Stage, StageOrderError)

STAGE_FUNCS = [loader.load_stage, chunking.chunk_stage, retrieval.build_index_stage,
               retrieval.retrieve_stage, answering.answer_stage, evaluation.evaluate_stage,
               validation.validation_stage, run_pipeline.finalise_stage]


def pipeline_at(stage: Stage) -> Pipeline:
    p = Pipeline()
    for s in STAGE_ORDER[1:STAGE_ORDER.index(stage) + 1]:
        p.advance(s)
    return p


def test_stage_functions_form_the_chain():
    chain = [(f.requires, f.produces) for f in STAGE_FUNCS]
    assert chain == list(zip(STAGE_ORDER, STAGE_ORDER[1:]))


@pytest.mark.parametrize("fn", STAGE_FUNCS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("state", list(Stage), ids=lambda s: s.name)
def test_wrong_state_raises(fn, state):
    if state is fn.requires:
        pytest.skip("correct state")
    p = pipeline_at(state)
    with pytest.raises(StageOrderError):
        fn(p, *([None] * 6))   # the state check fires before arguments are used
    assert p.state is state  # nothing advanced


def test_generate_answers_at_index_built_raises():
    with pytest.raises(StageOrderError):
        answering.answer_stage(pipeline_at(Stage.INDEX_BUILT), None, None)


@pytest.mark.parametrize("bad", [Stage.INDEX_BUILT, Stage.INIT, Stage.RESULTS_FINALISED])
def test_advance_only_to_successor(bad):
    p = pipeline_at(Stage.DOCUMENTS_LOADED)
    with pytest.raises(StageOrderError):
        p.advance(bad)


def test_cannot_advance_past_final():
    with pytest.raises(StageOrderError):
        pipeline_at(Stage.RESULTS_FINALISED).advance(Stage.RESULTS_FINALISED)


@pytest.mark.parametrize("cls", [LoadedCorpus, ChunkSet, QueryRetrieval, RetrievalResults])
def test_handoff_objects_cannot_be_forged(cls):
    with pytest.raises(StageOrderError):
        cls(**{f: None for f in cls.__dataclass_fields__ if f != "_seal"})


def test_answering_rejects_non_retrieval_input():
    with pytest.raises(TypeError):
        answering.answer_extractive({"query_id": "x", "hits": []})


def test_happy_path_reaches_final_state(tmp_path):
    p = Pipeline()
    corpus = loader.load_stage(p, SAMPLE_KB, SAMPLE_QUERIES)
    cs = chunking.chunk_stage(p, corpus, config.PRIMARY_CHUNKING, None)
    idx = retrieval.build_index_stage(p, cs)
    res = retrieval.retrieve_stage(p, idx, corpus, config.TOP_K, None)
    answering.answer_stage(p, res, None)
    evaluation.evaluate_stage(p, res, corpus, cs, None, run_comparison=False)
    assert p.state is Stage.EVALUATION_COMPLETE
