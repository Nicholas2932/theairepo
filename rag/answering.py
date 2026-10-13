"""Citation-strict answer generation (SPEC §6.3).

Input is a QueryRetrieval only — the question, its retrieved chunks and idf
for the question's own terms. This module never sees the rest of the KB.

Pipeline per query:
  1. sufficiency gate      -> insufficient_context
  2. conflict check        -> conflicting_context
  3. extractive answer     -> grounded_answer (verbatim sentences + citations)
  4. self-check            -> never emit a record that breaks the rules
Optional LLM mode replaces step 3 with Gemini, boxed in by steps 1 and 4 plus
the grounding check, with the extractive answer as fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import config, grounding, vocab
from .jsonio import ArtifactStore
from .stages import (AnswerSet, PRODUCER_SEAL, QueryRetrieval, RetrievalResults, RetrievedHit,
                     Stage, stage_step, utc_now)
from .textproc import has_negation, number_units, sentence_spans, term_set


@dataclass(frozen=True)
class _Sentence:
    hit: RetrievedHit
    text: str
    start: int
    terms: frozenset
    rel: float          # idf-weighted share of question terms covered

    @property
    def order(self) -> tuple[int, int]:
        return (self.hit.rank, self.start)


def _coverage(terms: frozenset, qr: QueryRetrieval) -> tuple[float, frozenset]:
    total = sum(qr.term_idf[t] for t in qr.query_terms)
    matched = frozenset(t for t in qr.query_terms if t in terms)
    if total <= 0:
        return 0.0, matched
    return sum(qr.term_idf[t] for t in matched) / total, matched


def _sentences(qr: QueryRetrieval) -> list[_Sentence]:
    out = []
    for hit in qr.hits:
        for s, e in sentence_spans(hit.chunk_text):
            text = hit.chunk_text[s:e]
            terms = term_set(text)
            out.append(_Sentence(hit, text, s, terms, _coverage(terms, qr)[0]))
    return out


# --- Step 1: sufficiency gate ---------------------------------------------

def sufficiency(qr: QueryRetrieval) -> dict:
    """Deterministic gate: some retrieved chunk must cover >= TAU_COV of the
    question's idf mass AND contain at least one specific (high-idf) term."""
    if not qr.query_terms or not qr.hits:
        return {"sufficient": False, "best_coverage": 0.0, "reason": "no content terms or no hits"}
    strong = config.TAU_IDF_RATIO * qr.idf_ceiling
    best, best_ok = 0.0, False
    for hit in qr.hits:
        terms = term_set(f"{hit.doc_title} {hit.section} {hit.chunk_text}")
        cov, matched = _coverage(terms, qr)
        has_strong = any(qr.term_idf[t] >= strong for t in matched)
        best = max(best, cov)
        best_ok = best_ok or (cov >= config.TAU_COV and has_strong)
    return {"sufficient": best_ok, "best_coverage": best,
            "reason": "ok" if best_ok else "question terms not covered by retrieved chunks"}


# --- Step 2: conflict check -----------------------------------------------

def _non_numeric(terms: frozenset) -> frozenset:
    return frozenset(t for t in terms if not t.isdigit())


def _numeric_disagreement(a: str, b: str) -> bool:
    ua, ub = number_units(a), number_units(b)
    return any(ua[u] != ub[u] for u in ua.keys() & ub.keys())


def find_conflict(sentences: list[_Sentence]) -> tuple[_Sentence, _Sentence] | None:
    pool = [s for s in sentences
            if s.hit.rank <= config.CONFLICT_TOP_N and s.rel >= config.CONFLICT_REL_FLOOR]
    pool.sort(key=lambda s: s.order)
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a.hit.chunk_id == b.hit.chunk_id:
                continue
            ta, tb = _non_numeric(a.terms), _non_numeric(b.terms)
            if not ta or not tb:
                continue
            if len(ta & tb) / len(ta | tb) < config.CONFLICT_JACCARD:
                continue
            if _numeric_disagreement(a.text, b.text) or has_negation(a.text) != has_negation(b.text):
                return a, b
    return None


# --- Step 3: extractive answer --------------------------------------------

def _cite(s: _Sentence) -> str:
    return vocab.format_citation(s.hit.doc_title, s.hit.chunk_id)


def _record(qr: QueryRetrieval, label: str, answer: str, citations: list[str], generator: str) -> dict:
    used = [vocab.parse_citation(c)[1] for c in citations]
    return {"query_id": qr.query_id, "answer_label": label, "answer": answer,
            "citations": citations, "used_chunk_ids": used, "generator": generator}


def insufficient(qr: QueryRetrieval, generator: str = vocab.GENERATOR_EXTRACTIVE) -> dict:
    return _record(qr, vocab.INSUFFICIENT_CONTEXT, vocab.INSUFFICIENT_ANSWER_TEXT, [], generator)


def _compose(lead: str | None, picked: list[_Sentence]) -> tuple[str, list[str]]:
    parts, citations = ([lead] if lead else []), []
    for s in picked:
        c = _cite(s)
        parts.append(f"{s.text} {c}")
        if c not in citations:
            citations.append(c)
    return " ".join(parts), citations


def _select(sentences: list[_Sentence], k: int) -> list[_Sentence]:
    def score(s: _Sentence) -> float:
        return s.rel + config.RANK_BONUS * (k - s.hit.rank + 1) / k

    ranked = sorted((s for s in sentences if s.rel > 0), key=lambda s: (-score(s), s.order))
    if not ranked:
        return []
    best = score(ranked[0])
    picked, seen = [], set()
    for s in ranked:
        if len(picked) >= config.MAX_ANSWER_SENTENCES or score(s) < config.SENTENCE_KEEP_RATIO * best:
            break
        key = " ".join(s.text.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        picked.append(s)
    return sorted(picked, key=lambda s: s.order)


def answer_extractive(qr: QueryRetrieval, generator: str = vocab.GENERATOR_EXTRACTIVE) -> dict:
    if not isinstance(qr, QueryRetrieval):
        raise TypeError("answers can only be generated from a QueryRetrieval")
    if not sufficiency(qr)["sufficient"]:
        return insufficient(qr, generator)
    sentences = _sentences(qr)
    conflict = find_conflict(sentences)
    if conflict:
        text, citations = _compose(vocab.CONFLICT_LEAD_IN, sorted(conflict, key=lambda s: s.order))
        record = _record(qr, vocab.CONFLICTING_CONTEXT, text, citations, generator)
    else:
        picked = _select(sentences, len(qr.hits))
        if not picked:
            return insufficient(qr, generator)
        text, citations = _compose(None, picked)
        record = _record(qr, vocab.GROUNDED_ANSWER, text, citations, generator)
    if self_check(record, qr):          # step 4: never emit an invalid record
        return insufficient(qr, generator)
    return record


# --- Step 4: self-check -----------------------------------------------------

def _retrieved_map(qr: QueryRetrieval) -> dict[str, dict]:
    return {h.chunk_id: {"chunk_id": h.chunk_id, "doc_title": h.doc_title, "chunk_text": h.chunk_text}
            for h in qr.hits}


def self_check(record: dict, qr: QueryRetrieval, require_grounding: bool = True) -> list[str]:
    retrieved = _retrieved_map(qr)
    problems = [msg for _, msg in grounding.structural_problems(record, retrieved)]
    if require_grounding and not problems and record.get("answer_label") != vocab.INSUFFICIENT_CONTEXT:
        g = grounding.check_answer(record, list(retrieved.values()))
        problems.extend(g["reasons"])
    return problems


# --- Optional LLM path --------------------------------------------------------

def answer_with_llm(qr: QueryRetrieval, client) -> tuple[dict, dict]:
    """One LLM call per query. Returns (answer record, llm_calls.jsonl record)."""
    from . import llm  # imported lazily: extractive mode never loads the adapter

    gate = sufficiency(qr)
    t0 = time.perf_counter()
    output, prompt, error = llm.generate(client, qr)
    latency_ms = int(round((time.perf_counter() - t0) * 1000))

    reasons: list[str] = [error] if error else []
    record = None
    if output is not None:
        label = output.get("answer_label")
        citations = output.get("citations")
        if label == vocab.INSUFFICIENT_CONTEXT:
            # The model may always decline; the refusal text is our fixed template.
            record = insufficient(qr, vocab.GENERATOR_LLM)
        elif not gate["sufficient"]:
            reasons.append(f"sufficiency gate rejected the question ({gate['reason']})")
        elif label not in vocab.ANSWER_LABELS:
            reasons.append(f"unknown answer_label {label!r}")
        elif not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
            reasons.append("citations is not a list of strings")
        else:
            parsed = [vocab.parse_citation(c) for c in citations]
            if any(p is None for p in parsed):
                reasons.append("malformed citation in model output")
            else:
                candidate = {"query_id": qr.query_id, "answer_label": label,
                             "answer": str(output.get("answer", "")), "citations": citations,
                             "used_chunk_ids": [p[1] for p in parsed],
                             "generator": vocab.GENERATOR_LLM}
                reasons.extend(self_check(candidate, qr))
                if not reasons:
                    record = candidate

    fallback = record is None
    if fallback:
        record = answer_extractive(qr, vocab.GENERATOR_LLM_FALLBACK)
    call = {
        "stage": Stage.ANSWERS_GENERATED.name, "query_id": qr.query_id, "timestamp": utc_now(),
        "provider": client.provider, "model": client.model, "prompt_hash": prompt.sha256,
        "input_artifacts": [f"artifacts/{config.RETRIEVAL_FILE}"],
        "output_artifact": f"artifacts/{config.ANSWERS_FILE}",
        "temperature": client.temperature, "fallback": fallback, "latency_ms": latency_ms,
        "error": error, "fallback_reasons": reasons if fallback else [],
    }
    return record, call


# --- Stage ----------------------------------------------------------------------

def answer_query(qr: QueryRetrieval, client=None) -> tuple[dict, dict | None]:
    if client is None:
        return answer_extractive(qr), None
    return answer_with_llm(qr, client)


@stage_step(requires=Stage.RETRIEVAL_COMPLETE, produces=Stage.ANSWERS_GENERATED)
def answer_stage(pipeline, results: RetrievalResults, store: ArtifactStore | None,
                 client=None) -> AnswerSet:
    if not isinstance(results, RetrievalResults):
        raise TypeError("answer_stage needs RetrievalResults from the retrieval stage")
    answers, calls, warnings = [], [], []
    for qr in results.per_query:
        record, call = answer_query(qr, client)
        answers.append(record)
        if call is not None:
            calls.append(call)
            if call["fallback"]:
                warnings.append(f"query {qr.query_id}: LLM output rejected, extractive fallback used")
    if store is not None:
        store.write_json(config.ANSWERS_FILE, answers)
        if client is not None:
            store.write_jsonl(config.LLM_CALLS_FILE, calls)
    mode = vocab.GENERATOR_EXTRACTIVE if client is None else vocab.GENERATOR_LLM
    return AnswerSet(_seal=PRODUCER_SEAL, answers=tuple(answers), generator_mode=mode,
                     llm_calls=tuple(calls), warnings=tuple(warnings))
