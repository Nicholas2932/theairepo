"""All tunables in one place (SPEC §2, §6).

Thresholds are fixed for every run; they are never adjusted per query or per
fixture. Calibration notes sit next to each value.
"""

from __future__ import annotations

from pathlib import Path

from . import __version__, vocab

PIPELINE_VERSION = __version__

# --- Paths (defaults; every CLI can override) ------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_DIR = REPO_ROOT / "kb"
DEFAULT_QUERIES_PATH = REPO_ROOT / "queries.json"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"

KB_EXTENSIONS = (".txt", ".md")

# --- Artifact file names ---------------------------------------------------
CHUNKS_FILE = "chunks.json"
RETRIEVAL_FILE = "retrieval.json"
ANSWERS_FILE = "answers.json"
EVAL_FILE = "eval.json"
GROUNDING_FILE = "grounding_check.json"
COMPARISON_FILE = "chunking_comparison.json"
MANIFEST_FILE = "run_manifest.json"
LLM_CALLS_FILE = "llm_calls.jsonl"
ALL_ARTIFACTS = (
    CHUNKS_FILE, RETRIEVAL_FILE, ANSWERS_FILE, EVAL_FILE,
    GROUNDING_FILE, COMPARISON_FILE, MANIFEST_FILE, LLM_CALLS_FILE,
)
CORE_ARTIFACTS = (CHUNKS_FILE, RETRIEVAL_FILE, ANSWERS_FILE, EVAL_FILE)

# --- Output formatting -----------------------------------------------------
FLOAT_DP = 6

# --- Chunking --------------------------------------------------------------
PRIMARY_CHUNKING = vocab.SENTENCE_WINDOW
CHUNKING_PARAMS = {
    vocab.SENTENCE_WINDOW: {"max_chars": 300},
    vocab.PARAGRAPH: {"max_chars_hard": 1200},
    vocab.FIXED_SIZE: {"size": 200, "overlap": 40},
}

# --- Text processing -------------------------------------------------------
MIN_STEM_LEN = 3
NGRAM_N = 4
# How many tokens after a number we look for its unit word (conflict heuristic).
UNIT_LOOKAHEAD = 3

# --- Retrieval -------------------------------------------------------------
TOP_K = 5              # chunks stored per query; must be >= MIN_RETRIEVED
MIN_RETRIEVED = 3      # brief: "at least top 3"
EVAL_K = 3             # evaluation looks at the top 3 only
BM25_K1 = 1.5
BM25_B = 0.75
FUSION_WEIGHTS = {"bm25": 0.7, "ngram": 0.3}
SCORE_SORT_DP = 9      # rounding applied to the sort key to absorb float noise

# --- Answering: sufficiency gate (SPEC §6.3 step 1) ------------------------
# coverage(c) = idf mass of question terms present in chunk c / idf mass of all
# question terms. Unknown question terms (df = 0) carry the highest idf, so
# out-of-scope questions get low coverage.
# Calibration (placeholder sample KB + 8 out-of-scope probes): lowest in-scope
# best coverage 0.564, highest out-of-scope 0.383 (margin 1.47x >= 1.25x).
# Re-check against the real fixtures.
TAU_COV = 0.5
# At least one matched term must be "specific": idf >= ratio * idf of a term
# that occurs in exactly one chunk. Stops a ubiquitous word carrying an answer.
TAU_IDF_RATIO = 0.4

# --- Answering: conflict heuristic (SPEC §6.3 step 2) ----------------------
CONFLICT_TOP_N = 3
CONFLICT_REL_FLOOR = 0.25   # each sentence's own query coverage
CONFLICT_JACCARD = 0.8      # non-numeric content-term Jaccard

# --- Answering: extractive generator (SPEC §6.3 step 3) --------------------
MAX_ANSWER_SENTENCES = 2
SENTENCE_KEEP_RATIO = 0.6
RANK_BONUS = 0.05

# --- Evaluation / grounding ------------------------------------------------
ANSWER_POINT_COVERAGE = 0.8
GROUNDING_TOKEN_RECALL = 0.8

# --- Optional LLM (Gemini) -------------------------------------------------
LLM_PROVIDERS = ("gemini",)
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"   # override with --llm-model or GEMINI_MODEL
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
LLM_TEMPERATURE = 0
LLM_TIMEOUT_S = 60
LLM_MAX_RETRIES = 2

# --- API (stretch) ---------------------------------------------------------
API_HOST = "127.0.0.1"
API_DEFAULT_PORT = 8000
API_MAX_BODY_BYTES = 64 * 1024


def check_config() -> None:
    """Fail fast on an internally inconsistent configuration."""
    if TOP_K < MIN_RETRIEVED:
        raise ValueError("TOP_K must be >= MIN_RETRIEVED")
    if EVAL_K > TOP_K:
        raise ValueError("EVAL_K must be <= TOP_K")
    if PRIMARY_CHUNKING not in vocab.CHUNKING_STRATEGIES:
        raise ValueError("PRIMARY_CHUNKING not in vocab")
    if abs(sum(FUSION_WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError("FUSION_WEIGHTS must sum to 1")


def snapshot(strategy: str, top_k: int) -> dict:
    """Config recorded in run_manifest.json."""
    return {
        "chunking": strategy,
        "chunking_params": CHUNKING_PARAMS[strategy],
        "top_k": top_k,
        "eval_k": EVAL_K,
        "bm25": {"k1": BM25_K1, "b": BM25_B},
        "ngram_n": NGRAM_N,
        "weights": dict(FUSION_WEIGHTS),
        "sufficiency": {"tau_cov": TAU_COV, "tau_idf_ratio": TAU_IDF_RATIO},
        "conflict": {"top_n": CONFLICT_TOP_N, "rel_floor": CONFLICT_REL_FLOOR,
                     "jaccard": CONFLICT_JACCARD},
        "extractive": {"max_sentences": MAX_ANSWER_SENTENCES,
                       "keep_ratio": SENTENCE_KEEP_RATIO, "rank_bonus": RANK_BONUS},
    }
