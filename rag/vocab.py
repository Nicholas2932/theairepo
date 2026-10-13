"""Controlled vocabularies and the citation format (SPEC §7.10).

Single source of truth: the pipeline, the validator and the LLM adapter all
import from here. Nothing else may spell these strings out.
"""

from __future__ import annotations

import re

# --- Answer labels ---------------------------------------------------------
GROUNDED_ANSWER = "grounded_answer"
INSUFFICIENT_CONTEXT = "insufficient_context"
CONFLICTING_CONTEXT = "conflicting_context"
ANSWER_LABELS = (GROUNDED_ANSWER, INSUFFICIENT_CONTEXT, CONFLICTING_CONTEXT)

# Minimum number of citations per label (insufficient_context must have exactly 0).
MIN_CITATIONS = {GROUNDED_ANSWER: 1, CONFLICTING_CONTEXT: 2, INSUFFICIENT_CONTEXT: 0}

# --- Retrieval evaluation statuses ----------------------------------------
HIT = "hit"
PARTIAL_HIT = "partial_hit"
MISS = "miss"
RETRIEVAL_STATUS = (HIT, PARTIAL_HIT, MISS)

# --- Grounding check statuses (extra) --------------------------------------
GROUNDING_PASS = "pass"
GROUNDING_FAIL = "fail"
GROUNDING_NOT_APPLICABLE = "not_applicable"
GROUNDING_STATUS = (GROUNDING_PASS, GROUNDING_FAIL, GROUNDING_NOT_APPLICABLE)

# --- Answer generators (extra field on each answer) ------------------------
GENERATOR_EXTRACTIVE = "extractive"
GENERATOR_LLM = "llm"
GENERATOR_LLM_FALLBACK = "llm_fallback_extractive"
GENERATORS = (GENERATOR_EXTRACTIVE, GENERATOR_LLM, GENERATOR_LLM_FALLBACK)

# Run-level answer generator recorded in run_manifest.json.
ANSWER_GENERATOR_MODES = (GENERATOR_EXTRACTIVE, GENERATOR_LLM)

# --- Chunking strategies ---------------------------------------------------
SENTENCE_WINDOW = "sentence_window"
PARAGRAPH = "paragraph"
FIXED_SIZE = "fixed_size"
CHUNKING_STRATEGIES = (SENTENCE_WINDOW, PARAGRAPH, FIXED_SIZE)

# --- Fixed answer texts ----------------------------------------------------
INSUFFICIENT_ANSWER_TEXT = (
    "The knowledge base does not contain enough information to answer this question."
)
CONFLICT_LEAD_IN = "The retrieved sources disagree:"

# --- Citation format -------------------------------------------------------
CITATION_FORMAT = "[{doc_title} §{chunk_id}]"
# Anchored parse of a single citation token. The greedy title group means the
# split happens on the LAST " §", so titles containing "§" or "]" still parse.
CITATION_RE = re.compile(r"^\[(?P<title>.+) §(?P<chunk_id>[^\s\]§]+)\]$")
# Unanchored scan for citation-like tokens inside answer prose. Used only to
# strip/flag stray tokens; authoritative parsing uses CITATION_RE.
CITATION_SCAN_RE = re.compile(r"\[(?P<title>[^\[\]\n]+?) §(?P<chunk_id>[^\s\]§\[]+)\]")


def format_citation(doc_title: str, chunk_id: str) -> str:
    if "\n" in doc_title or not doc_title:
        raise ValueError(f"doc_title cannot be cited: {doc_title!r}")
    return CITATION_FORMAT.format(doc_title=doc_title, chunk_id=chunk_id)


def parse_citation(token: object) -> tuple[str, str] | None:
    """Return (doc_title, chunk_id) for a well-formed citation token, else None."""
    if not isinstance(token, str):
        return None
    m = CITATION_RE.match(token)
    if not m:
        return None
    return m.group("title"), m.group("chunk_id")
