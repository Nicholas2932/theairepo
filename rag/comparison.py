"""Chunking-strategy comparison (SPEC §6.7).

Throw-away in-memory sub-runs (chunk -> index -> retrieve -> evaluate) with
the same retriever and top_k. Never touches the primary artifacts. The
explanation is templated from the numbers; no LLM involved.
"""

from __future__ import annotations

from . import config, vocab
from .chunking import chunk_documents
from .evaluation import evaluate_retrieval
from .retrieval import build_index, retrieval_to_json, retrieve_all
from .textproc import norm_title

_METRICS = (("top3_hit_rate", "top-3 title hit rate"),
            ("mrr", "MRR"),
            ("answer_point_recall_top3", "answer-point recall in the top 3"))


def run_strategy(corpus, strategy: str, top_k: int) -> dict:
    params = dict(config.CHUNKING_PARAMS[strategy])
    chunks = chunk_documents(corpus.documents, strategy, params)
    results = retrieve_all(build_index(chunks), corpus.queries, top_k)
    kb_titles = {norm_title(d.title) for d in corpus.documents}
    summary = evaluate_retrieval(retrieval_to_json(results), corpus.queries, kb_titles)["summary"]
    keep = ("top3_hit_rate", "top3_full_hit_rate", "hits", "partial_hits", "misses",
            "mrr", "answer_point_recall_top3")
    return {"name": strategy, "params": params, "num_chunks": len(chunks),
            "avg_chunk_chars": (sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0.0,
            "summary": {key: summary[key] for key in keep}}


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _explain(rows: list[dict], primary: str) -> str:
    lines = []
    for key, label in _METRICS:
        values = {r["name"]: r["summary"][key] for r in rows if r["summary"][key] is not None}
        if not values:
            lines.append(f"{label}: not measurable (no ground-truth data).")
            continue
        best = max(values.values())
        leaders = [n for n, v in values.items() if v == best]
        detail = ", ".join(f"{n} {_fmt(v)}" for n, v in values.items())
        if len(leaders) == len(values):
            lines.append(f"{label}: all strategies tie at {_fmt(best)}.")
        else:
            lines.append(f"{label}: {' and '.join(leaders)} lead(s) ({detail}).")
    by_size = sorted(rows, key=lambda r: (r["avg_chunk_chars"], r["name"]))
    small, large = by_size[0], by_size[-1]
    if small["name"] != large["name"]:
        lines.append(
            f"{small['name']} gives the smallest chunks ({small['num_chunks']} chunks, "
            f"avg {small['avg_chunk_chars']:.0f} chars): tighter citations, less context per chunk. "
            f"{large['name']} gives the largest ({large['num_chunks']} chunks, "
            f"avg {large['avg_chunk_chars']:.0f} chars): more context, coarser citations and more "
            "unrelated text competing in each chunk.")
    lines.append(f"Primary strategy for this run: {primary}.")
    return " ".join(lines)


def compare_strategies(corpus, primary: str, top_k: int) -> dict:
    rows = [run_strategy(corpus, s, top_k) for s in vocab.CHUNKING_STRATEGIES]
    n_docs, n_q = len(corpus.documents), len(corpus.queries)
    caveat = (f"Measured on {n_docs} documents and {n_q} queries. At this size differences are "
              "anecdotal, and title-level hit rate cannot separate strategies when each document "
              "fits in a handful of chunks; treat MRR and answer-point recall as tie-breakers only.")
    return {"primary_strategy": primary, "strategies": rows,
            "tradeoff_explanation": _explain(rows, primary), "caveat": caveat}
