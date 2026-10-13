"""Hybrid local retrieval: BM25 over stemmed terms + character n-gram TF-IDF
cosine, max-normalised and fused with fixed weights (SPEC §6.2).

Retrieval reads only query ids and questions; it never sees ground truth.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter

from . import config
from .chunking import chunk_ordinal
from .jsonio import ArtifactStore
from .stages import (ChunkSet, LoadedCorpus, PRODUCER_SEAL, QueryRetrieval, RetrievalResults,
                     RetrievedHit, Sealed, Stage, stage_step)
from .textproc import char_ngrams, content_terms, unique_terms


def bm25_idf(df: int, n: int) -> float:
    """Lucene-style idf: always positive, even on tiny corpora."""
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Index(Sealed):
    chunks: tuple
    term_freqs: tuple          # Counter per chunk
    doc_lens: tuple
    df: dict
    avgdl: float
    ngram_idf: dict
    ngram_vecs: tuple          # dict per chunk: ngram -> tf*idf
    ngram_norms: tuple
    ordinals: tuple

    @property
    def n(self) -> int:
        return len(self.chunks)


def index_text(chunk) -> str:
    """Title and section are indexed as a prefix; stored text is untouched."""
    return f"{chunk.doc_title} {chunk.section} {chunk.text}"


def build_index(chunks) -> Index:
    term_freqs, doc_lens, df = [], [], Counter()
    ngram_counts, ngram_df = [], Counter()
    for c in chunks:
        text = index_text(c)
        tf = Counter(content_terms(text))
        term_freqs.append(tf)
        doc_lens.append(sum(tf.values()))
        df.update(tf.keys())
        ng = char_ngrams(text)
        ngram_counts.append(ng)
        ngram_df.update(ng.keys())
    n = len(chunks)
    ngram_idf = {g: math.log((1 + n) / (1 + d)) + 1.0 for g, d in ngram_df.items()}
    vecs, norms = [], []
    for ng in ngram_counts:
        vec = {g: tf * ngram_idf[g] for g, tf in ng.items()}
        vecs.append(vec)
        norms.append(math.sqrt(sum(w * w for w in vec.values())))
    avgdl = (sum(doc_lens) / n) if n else 0.0
    return Index(_seal=PRODUCER_SEAL, chunks=tuple(chunks), term_freqs=tuple(term_freqs),
                 doc_lens=tuple(doc_lens), df=dict(df), avgdl=avgdl, ngram_idf=ngram_idf,
                 ngram_vecs=tuple(vecs), ngram_norms=tuple(norms),
                 ordinals=tuple(chunk_ordinal(c.chunk_id) for c in chunks))


def _bm25_scores(index: Index, q_terms: tuple[str, ...]) -> list[float]:
    n, k1, b = index.n, config.BM25_K1, config.BM25_B
    idfs = {t: bm25_idf(index.df.get(t, 0), n) for t in q_terms}
    scores = []
    for tf, dl in zip(index.term_freqs, index.doc_lens):
        s = 0.0
        norm = k1 * (1 - b + b * dl / index.avgdl) if index.avgdl else k1
        for t in q_terms:
            f = tf.get(t, 0)
            if f:
                s += idfs[t] * f * (k1 + 1) / (f + norm)
        scores.append(s)
    return scores


def _ngram_scores(index: Index, question: str) -> list[float]:
    q = char_ngrams(question)
    q_vec = {g: tf * index.ngram_idf[g] for g, tf in q.items() if g in index.ngram_idf}
    q_norm = math.sqrt(sum(w * w for w in q_vec.values()))
    scores = []
    for vec, norm in zip(index.ngram_vecs, index.ngram_norms):
        if not q_norm or not norm:
            scores.append(0.0)
            continue
        dot = sum(w * vec.get(g, 0.0) for g, w in q_vec.items())
        scores.append(dot / (q_norm * norm))
    return scores


def _max_normalise(values: list[float]) -> list[float]:
    top = max(values, default=0.0)
    return [v / top for v in values] if top > 0 else [0.0 for _ in values]


def retrieve_query(index: Index, query_id: str, question: str,
                   top_k: int = config.TOP_K) -> QueryRetrieval:
    if not isinstance(index, Index):
        raise TypeError("retrieve_query needs an Index")
    q_terms = unique_terms(question)
    bm25 = _max_normalise(_bm25_scores(index, q_terms))
    ngram = _max_normalise(_ngram_scores(index, question))
    w = config.FUSION_WEIGHTS
    fused = [w["bm25"] * b + w["ngram"] * g for b, g in zip(bm25, ngram)]

    order = sorted(range(index.n), key=lambda i: (
        -round(fused[i], config.SCORE_SORT_DP), index.chunks[i].start_char, index.ordinals[i]))
    k = min(max(top_k, config.MIN_RETRIEVED), index.n)
    hits = tuple(
        RetrievedHit(rank=r, chunk_id=index.chunks[i].chunk_id, doc_title=index.chunks[i].doc_title,
                     section=index.chunks[i].section, score=fused[i],
                     chunk_text=index.chunks[i].text, start_char=index.chunks[i].start_char,
                     bm25=bm25[i], ngram=ngram[i])
        for r, i in enumerate(order[:k], 1))
    n = index.n
    return QueryRetrieval(
        _seal=PRODUCER_SEAL, query_id=query_id, question=question, hits=hits,
        query_terms=q_terms, term_idf={t: bm25_idf(index.df.get(t, 0), n) for t in q_terms},
        idf_ceiling=bm25_idf(1, n), n_chunks=n)


def retrieve_all(index: Index, queries, top_k: int = config.TOP_K) -> RetrievalResults:
    per_query = tuple(retrieve_query(index, q.query_id, q.question, top_k) for q in queries)
    return RetrievalResults(_seal=PRODUCER_SEAL, per_query=per_query, top_k=top_k,
                            num_chunks=index.n)


def hit_to_json(h: RetrievedHit) -> dict:
    return {"rank": h.rank, "chunk_id": h.chunk_id, "doc_title": h.doc_title,
            "score": h.score, "chunk_text": h.chunk_text, "section": h.section,
            "score_components": {"bm25": h.bm25, "ngram": h.ngram}}


def retrieval_to_json(results: RetrievalResults) -> list[dict]:
    return [{"query_id": qr.query_id, "question": qr.question,
             "top_k": [hit_to_json(h) for h in qr.hits]}
            for qr in results.per_query]


@stage_step(requires=Stage.DOCUMENTS_CHUNKED, produces=Stage.INDEX_BUILT)
def build_index_stage(pipeline, chunkset: ChunkSet) -> Index:
    if not isinstance(chunkset, ChunkSet):
        raise TypeError("build_index_stage needs a ChunkSet")
    return build_index(chunkset.chunks)


@stage_step(requires=Stage.INDEX_BUILT, produces=Stage.RETRIEVAL_COMPLETE)
def retrieve_stage(pipeline, index: Index, corpus: LoadedCorpus, top_k: int,
                   store: ArtifactStore | None) -> RetrievalResults:
    results = retrieve_all(index, corpus.queries, top_k)
    if store is not None:
        store.write_json(config.RETRIEVAL_FILE, retrieval_to_json(results))
    return results
