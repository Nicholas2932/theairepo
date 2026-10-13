"""Answer grounding check (SPEC §6.6) and the structural answer rules shared
by the answer self-check (§6.3 step 4) and the validator (V-06..V-08).

Works on plain dicts in the artifact formats (answers.json records and
retrieval.json entries), so the same code runs in-pipeline and on disk.
"""

from __future__ import annotations

from . import config, vocab
from .textproc import content_terms, norm_for_match, numbers, sentence_spans

# Problem kinds returned by structural_problems (mapped to validator rules).
P_LABEL, P_COUNT, P_CITATION, P_USED_IDS = "label", "count", "citation", "used_ids"


def retrieved_by_id(top_k: list[dict]) -> dict[str, dict]:
    return {h.get("chunk_id"): h for h in top_k if isinstance(h, dict)}


def structural_problems(answer: dict, retrieved: dict[str, dict]) -> list[tuple[str, str]]:
    """Return (kind, message) problems for one answer record against that
    query's retrieved chunks ({chunk_id: hit dict})."""
    problems: list[tuple[str, str]] = []
    label = answer.get("answer_label")
    text = answer.get("answer")
    citations = answer.get("citations")
    used = answer.get("used_chunk_ids")

    if label not in vocab.ANSWER_LABELS:
        problems.append((P_LABEL, f"unknown answer_label {label!r}"))
    if not isinstance(text, str):
        problems.append((P_LABEL, "answer is not a string"))
        text = ""
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        problems.append((P_CITATION, "citations must be a list of strings"))
        citations = []
    if not isinstance(used, list) or not all(isinstance(u, str) for u in used):
        problems.append((P_USED_IDS, "used_chunk_ids must be a list of strings"))
        used = []

    if label in vocab.ANSWER_LABELS:
        need = vocab.MIN_CITATIONS[label]
        if label == vocab.INSUFFICIENT_CONTEXT and citations:
            problems.append((P_COUNT, "insufficient_context must have no citations"))
        elif len(citations) < need:
            problems.append((P_COUNT, f"{label} needs >= {need} citations, has {len(citations)}"))
        if label != vocab.INSUFFICIENT_CONTEXT and not text.strip():
            problems.append((P_LABEL, f"{label} has empty answer text"))

    parsed_ids: list[str] = []
    for c in citations:
        parsed = vocab.parse_citation(c)
        if parsed is None:
            problems.append((P_CITATION, f"malformed citation {c!r}"))
            continue
        title, cid = parsed
        parsed_ids.append(cid)
        hit = retrieved.get(cid)
        if hit is None:
            problems.append((P_CITATION, f"citation {c!r} refers to a chunk not retrieved for this query"))
        elif hit.get("doc_title") != title:
            problems.append((P_CITATION, f"citation {c!r} title does not match chunk title "
                                         f"{hit.get('doc_title')!r}"))
        if c not in text:
            problems.append((P_CITATION, f"citation {c!r} does not appear in the answer text"))
    if len(set(citations)) != len(citations):
        problems.append((P_CITATION, "duplicate citations"))
    if label == vocab.CONFLICTING_CONTEXT and len(set(parsed_ids)) < 2:
        problems.append((P_COUNT, "conflicting_context must cite >= 2 distinct chunks"))

    stray = [m.group(0) for m in vocab.CITATION_SCAN_RE.finditer(text) if m.group(0) not in citations]
    if stray:
        problems.append((P_CITATION, f"citation tokens in answer text missing from citations: {stray}"))

    if sorted(set(used)) != sorted(set(parsed_ids)) or len(set(used)) != len(used):
        problems.append((P_USED_IDS, f"used_chunk_ids {used} != chunk ids cited {parsed_ids}"))
    return problems


def claim_sentences(answer_text: str, citations: list[str]) -> list[str]:
    """Answer prose with citation tokens and the fixed conflict lead-in removed,
    split into sentences."""
    t = answer_text
    for c in citations:
        t = t.replace(c, " ")
    t = vocab.CITATION_SCAN_RE.sub(" ", t)
    stripped = t.lstrip()
    if stripped.startswith(vocab.CONFLICT_LEAD_IN):
        t = stripped[len(vocab.CONFLICT_LEAD_IN):]
    return [t[s:e] for s, e in sentence_spans(t)]


def check_answer(answer: dict, top_k: list[dict]) -> dict:
    """Grounding result for one answer against its query's retrieval entry."""
    label = answer.get("answer_label")
    result = {"query_id": answer.get("query_id"), "answer_label": label,
              "status": vocab.GROUNDING_PASS, "citations_checked": [], "sentences": [],
              "reasons": []}
    retrieved = retrieved_by_id(top_k)
    citations = answer.get("citations") if isinstance(answer.get("citations"), list) else []

    if label == vocab.INSUFFICIENT_CONTEXT:
        result["status"] = vocab.GROUNDING_NOT_APPLICABLE
        if citations:
            result["status"] = vocab.GROUNDING_FAIL
            result["reasons"].append("insufficient_context answer carries citations")
        return result

    cited_texts: list[str] = []
    for c in citations:
        parsed = vocab.parse_citation(c)
        cid = parsed[1] if parsed else None
        hit = retrieved.get(cid) if cid else None
        entry = {"citation": c, "chunk_id": cid, "in_retrieval": hit is not None,
                 "title_matches": bool(hit and parsed and hit.get("doc_title") == parsed[0])}
        result["citations_checked"].append(entry)
        if not (entry["in_retrieval"] and entry["title_matches"]):
            result["reasons"].append(f"citation {c!r} not a matching retrieved chunk")
        if hit is not None and isinstance(hit.get("chunk_text"), str):
            cited_texts.append(hit["chunk_text"])

    if not citations:
        result["reasons"].append("factual answer has no citations")

    cited_terms = set().union(*(set(content_terms(t)) for t in cited_texts)) if cited_texts else set()
    cited_numbers = set().union(*(numbers(t) for t in cited_texts)) if cited_texts else set()
    cited_norm = [norm_for_match(t) for t in cited_texts]

    text = answer.get("answer") if isinstance(answer.get("answer"), str) else ""
    claims = 0
    for sentence in claim_sentences(text, citations):
        terms = set(content_terms(sentence))
        nums = numbers(sentence)
        if not terms and not nums:
            continue
        claims += 1
        key = norm_for_match(sentence)
        verbatim = bool(key) and any(key in t for t in cited_norm)
        recall = 1.0 if verbatim else (len(terms & cited_terms) / len(terms) if terms else 1.0)
        numbers_ok = nums <= cited_numbers
        supported = (verbatim or recall >= config.GROUNDING_TOKEN_RECALL) and numbers_ok
        result["sentences"].append({"text": sentence, "token_recall": recall,
                                    "numbers_supported": numbers_ok, "supported": supported})
        if not supported:
            why = "unsupported numbers" if not numbers_ok else f"token recall {recall:.2f}"
            result["reasons"].append(f"sentence not supported by cited chunks ({why}): {sentence!r}")
    if claims == 0:
        result["reasons"].append("answer contains no claim text")

    if result["reasons"]:
        result["status"] = vocab.GROUNDING_FAIL
    return result


def check_all(answers: list[dict], retrieval_entries: list[dict]) -> dict:
    by_query = {e.get("query_id"): e.get("top_k") or [] for e in retrieval_entries}
    results = [check_answer(a, by_query.get(a.get("query_id"), [])) for a in answers]
    counts = {s: sum(1 for r in results if r["status"] == s) for s in vocab.GROUNDING_STATUS}
    applicable = counts[vocab.GROUNDING_PASS] + counts[vocab.GROUNDING_FAIL]
    summary = {"total": len(results), "passed": counts[vocab.GROUNDING_PASS],
               "failed": counts[vocab.GROUNDING_FAIL],
               "not_applicable": counts[vocab.GROUNDING_NOT_APPLICABLE],
               "pass_rate": (counts[vocab.GROUNDING_PASS] / applicable) if applicable else 1.0}
    return {"summary": summary, "results": results}
