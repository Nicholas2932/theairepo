"""Deterministic retrieval evaluation (SPEC §6.4, §6.5).

Operates on retrieval.json-shaped entries plus the queries' expected titles.
Imports nothing from the answering or LLM modules.
"""

from __future__ import annotations

from . import config, vocab
from .jsonio import ArtifactStore
from .stages import (ChunkSet, EvaluationResults, LoadedCorpus, PRODUCER_SEAL, RetrievalResults,
                     Stage, stage_step)
from .textproc import content_terms, norm_title


def _quote_list(titles) -> str:
    return ", ".join(f'"{t}"' for t in titles)


def _point_coverage(point: str, chunk_texts: list[str]) -> bool:
    terms = set(content_terms(point))
    if not terms:
        return True
    return any(len(terms & set(content_terms(t))) / len(terms) >= config.ANSWER_POINT_COVERAGE
               for t in chunk_texts)


def evaluate_query(entry: dict, query, kb_titles: set[str], k: int = config.EVAL_K) -> dict:
    hits = entry.get("top_k") or []
    top = hits[:k]
    retrieved_top: list[str] = []
    for h in top:
        if h["doc_title"] not in retrieved_top:
            retrieved_top.append(h["doc_title"])
    r_norm = [norm_title(t) for t in retrieved_top]

    expected: list[str] = []
    for t in query.expected_doc_titles:
        if norm_title(t) not in (norm_title(e) for e in expected):
            expected.append(t)
    e_norm = [norm_title(t) for t in expected]

    first_match_rank = next((h["rank"] for h in hits if norm_title(h["doc_title"]) in e_norm), None)
    found = [t for t, n in zip(expected, e_norm) if n in r_norm]
    missing = [t for t, n in zip(expected, e_norm) if n not in r_norm]

    if not expected:
        status, matched = vocab.HIT, None
        explanation = "No expected titles given; counted as a vacuous hit and excluded from the hit rate"
    elif not missing:
        status, matched = vocab.HIT, True
        if len(expected) == 1:
            rank = next(h["rank"] for h in top if norm_title(h["doc_title"]) == e_norm[0])
            explanation = f"Expected title found at rank {rank}"
        else:
            explanation = f"All {len(expected)} expected titles found in top {k}"
    elif found:
        status, matched = vocab.PARTIAL_HIT, True
        explanation = (f"{len(found)} of {len(expected)} expected titles found in top {k} "
                       f"(missing: {_quote_list(missing)})")
    else:
        status, matched = vocab.MISS, False
        parts = []
        for t in missing:
            if norm_title(t) not in kb_titles:
                parts.append(f'Expected title "{t}" not present in knowledge base')
            else:
                parts.append(f'Expected title "{t}" not in top {k} '
                             f"(retrieved: {_quote_list(retrieved_top)})")
        explanation = "; ".join(parts)

    texts = [h["chunk_text"] for h in top]
    return {
        "query_id": entry["query_id"],
        "expected_doc_titles": list(query.expected_doc_titles),
        "retrieved_doc_titles_top3": retrieved_top,
        "retrieval_status": status,
        "matched_expected_title": matched,
        "explanation": explanation,
        "first_match_rank": first_match_rank,
        "answer_points_covered_top3": [{"point": p, "covered": _point_coverage(p, texts)}
                                       for p in query.ground_truth_answer_points],
    }


def summarise(results: list[dict], k: int = config.EVAL_K) -> dict:
    total = len(results)
    answerable = [r for r in results if r["matched_expected_title"] is not None]
    hits = sum(1 for r in results if r["retrieval_status"] == vocab.HIT)
    partial = sum(1 for r in results if r["retrieval_status"] == vocab.PARTIAL_HIT)
    misses = sum(1 for r in results if r["retrieval_status"] == vocab.MISS)
    full_answerable = sum(1 for r in answerable if r["retrieval_status"] == vocab.HIT)
    matched = sum(1 for r in answerable if r["matched_expected_title"])
    n = len(answerable)
    mrr = (sum(1.0 / r["first_match_rank"] for r in answerable if r["first_match_rank"]) / n) if n else 0.0
    points = [p for r in results for p in r["answer_points_covered_top3"]]
    return {
        "top3_hit_rate": matched / n if n else 0.0,
        "top3_full_hit_rate": full_answerable / n if n else 0.0,
        "total_queries": total,
        "answerable_queries": n,
        "hits": hits,
        "partial_hits": partial,
        "misses": misses,
        "vacuous_hits": total - n,
        "mrr": mrr,
        "answer_point_recall_top3": (sum(1 for p in points if p["covered"]) / len(points)) if points else None,
        "k": k,
        "hit_rate_denominator": "answerable_queries",
    }


def evaluate_retrieval(entries: list[dict], queries, kb_titles: set[str],
                       k: int = config.EVAL_K) -> dict:
    by_id = {q.query_id: q for q in queries}
    results = [evaluate_query(e, by_id[e["query_id"]], kb_titles, k) for e in entries]
    return {"summary": summarise(results, k), "results": results}


@stage_step(requires=Stage.ANSWERS_GENERATED, produces=Stage.EVALUATION_COMPLETE)
def evaluate_stage(pipeline, results: RetrievalResults, corpus: LoadedCorpus, chunkset: ChunkSet,
                   store: ArtifactStore | None, run_comparison: bool = True) -> EvaluationResults:
    from .retrieval import retrieval_to_json   # retrieval-side serialiser only
    if not isinstance(results, RetrievalResults):
        raise TypeError("evaluate_stage needs RetrievalResults")
    kb_titles = {norm_title(d.title) for d in corpus.documents}
    eval_doc = evaluate_retrieval(retrieval_to_json(results), corpus.queries, kb_titles)
    comparison = None
    if run_comparison:
        from .comparison import compare_strategies
        comparison = compare_strategies(corpus, chunkset.strategy, results.top_k)
    if store is not None:
        store.write_json(config.EVAL_FILE, eval_doc)
        if comparison is not None:
            store.write_json(config.COMPARISON_FILE, comparison)
    return EvaluationResults(_seal=PRODUCER_SEAL, eval_doc=eval_doc, comparison=comparison)
