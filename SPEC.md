# Mini-RAG Pipeline — Spec & Design Notes

**Status:** pre-implementation. Nothing is coded yet.
**Source:** the pasted build brief (sample KB, sample `queries.json`, vocabularies, stages, requirements).
**Purpose of this file:** pin down every ambiguity in the brief *before* code is written, so the build is a mechanical exercise and the acceptance criteria are checkable.

Conventions: **MUST / SHOULD / MAY** as in RFC 2119. Items marked ⚠ are interpretation calls where a hidden evaluator could disagree with me — see §10 for the ones I'd like you to confirm.

---

## 1. What we're building (one paragraph)

A local, deterministic, replayable pipeline: `kb/*.txt` + `queries.json` → parse → chunk → index → retrieve → answer (citation-strict) → evaluate retrieval → ground-check → validate → write artifacts. Retrieval, ranking, evaluation, grounding check and validation are plain code. An LLM is **optional** and, if used, is boxed in by the same validators. The evaluator will run it from a clean checkout, possibly against *different* KB files and queries with the same schema, so nothing may depend on sample filenames, doc order, query ids, or sample wording.

---

## 2. Decisions at a glance

| Area | Decision |
|---|---|
| Language / deps | Python 3.10+ (3.12 present on this machine), **stdlib only** for the core pipeline. `pytest` for dev tests only. |
| Entry points | `python run_pipeline.py`, `python validate.py`, optional `python serve.py` (stretch). `Makefile` is a thin wrapper — **`make` is not installed on this Windows box**, so Python scripts are the primary interface. |
| Chunking | Primary: `sentence_window` (pack whole sentences up to `max_chars`). Compared against `paragraph` and `fixed_size`. |
| Retrieval | Hybrid, fully local: BM25 over stemmed word tokens + character 4-gram TF-IDF cosine, fixed-weight fusion. Title/section are indexed as a prefix but never altered in stored chunk text. |
| Answering | Default: **deterministic extractive generator** (no LLM, no network, no key). Optional LLM adapter behind `--llm`, off by default, with validator + extractive fallback. |
| Sufficiency / conflict | Deterministic IDF-weighted query-coverage gate → `insufficient_context`; narrow numeric/negation contradiction heuristic → `conflicting_context`. |
| Evaluation | Title-set matching on the top-3 chunks; definitions of `hit` / `partial_hit` / `miss` fixed in §6.5. |
| Determinism | Same inputs ⇒ byte-identical artifacts (except `run_manifest.json` and `llm_calls.jsonl`, which carry timestamps). |
| Stage enforcement | Explicit linear state machine; stage functions refuse to run out of order; typed hand-off objects make "answer before retrieval" unrepresentable. |

---

## 3. Gaps and ambiguities in the brief → resolutions

| # | Gap in the brief | Resolution | Confirm? |
|---|---|---|---|
| G1 | `chunk_id` scheme is unspecified. The sample answer cites `chunk_2` for the *second* article, which could mean per-document index, global index, or "one chunk per doc". `used_chunk_ids` has no doc title, so per-document ids would be ambiguous. | `chunk_id` = `chunk_<n>`, **globally unique** per run, 1-based. Order is deterministic and **independent of filename/order**: docs sorted by `(casefold(title), sha256(body))`, chunks in body order. The sample's `chunk_2` is illustrative; no code or validator may assume specific ids. | ⚠ |
| G2 | `start_char`/`end_char` reference frame unspecified ("exactly as sourced from the document body"). | Offsets are **into the document body** (text after the leading header block), so `body[start_char:end_char] == text` always holds. Extra field `body_offset` gives the body's position in the normalised file. Sample `start_char: 0` is consistent with this. | ⚠ |
| G3 | Top-level shape of `eval.json` ("each record…, also an aggregate summary"). | `eval.json` is an **object** `{"summary": {...}, "results": [...]}`. `chunks.json`, `retrieval.json`, `answers.json` are bare **arrays**. `validate.py` should also tolerate a bare array for eval + a separate summary only if cheap, but the canonical form is the object. | ⚠ |
| G4 | `hit` / `partial_hit` / `miss` are never defined. | Defined in §6.5 over the *set* of expected titles vs. distinct titles in top-3. `partial_hit` only arises when ≥2 expected titles and some-but-not-all are found. | ⚠ |
| G5 | `retrieved_doc_titles_top3` — per chunk or per doc? The brief's example shows 2 titles for a top-3. | Distinct titles of the top-3 **chunks**, in rank order of first appearance (so length ≤ 3). | ⚠ |
| G6 | Queries with an empty/missing `expected_doc_titles` (an unanswerable probe is plausible in "equivalent fixtures"). | Counted in `total_queries`; status is **`hit` vacuously**, `matched_expected_title: null`, explanation says so. Summary also reports `answerable_queries` and computes `top3_hit_rate` over answerable queries only (denominator is stated in the summary). Answer side: should be `insufficient_context`. | ⚠ |
| G7 | How to decide `insufficient_context` / `conflicting_context` deterministically (the brief only names the labels). | §6.3: IDF-weighted coverage gate and a conservative contradiction heuristic. Thresholds live in `config.py`, calibrated once on sample + probe fixtures, never per query. | |
| G8 | What counts as a "factual answer" needing a citation. | Any answer with label `grounded_answer` or `conflicting_context`. `insufficient_context` answers are a fixed refusal template with **zero** citations and no claims. | |
| G9 | Value of *k*. Brief says "at least top 3". | `top_k = 5` stored; `min(3, n_chunks)` is the validator floor. Evaluation looks only at top **3**. "Retrieved chunks" for citation purposes = all `top_k` stored for that query. | |
| G10 | Score semantics, ties, zero matches. | Numeric float, rounded to 6 dp, higher = better, in `[0, 1]` after per-query normalisation. Ties broken by `start_char`, then `chunk_id`. Zero-score chunks may still appear to satisfy the "≥3" rule. | |
| G11 | Doc titles that contain `]` or `§`, or duplicate titles across files. | Citation parse regex is anchored on the **last** ` §` and the final `]`. Duplicate titles are legal (chunk ids stay unique); eval matches by normalised title, so a duplicate title can't distinguish docs — warn in manifest. | |
| G12 | Header parsing edge cases: missing `Section:`, missing `Title:`, extra `Section:` lines mid-document, BOM, CRLF. | §6.1. Missing Section → `""`; missing Title → first non-empty line, else filename stem, **with a warning** in manifest; mid-doc `Section:` starts a new section and is never inside chunk text; BOM stripped; CRLF → LF. | |
| G13 | `VALIDATION_COMPLETE` stage vs. standalone `validate.py`. | Both call the same `rag/validation.py`. The in-pipeline stage runs it (plus grounding check) and aborts finalisation on failure; `validate.py` is a thin CLI over it that works from artifacts on disk without re-running anything. | |
| G14 | Where does chunking comparison sit in the stage list? | Inside `EVALUATION_COMPLETE`, on throw-away in-memory sub-runs (chunk → index → retrieve → eval only). It never touches the primary artifacts. | |
| G15 | `llm_calls.jsonl` when no LLM is used. | File is **not created** (stale one is deleted at run start); README and `run_manifest.json` state `answer_generator: "extractive"`. If LLM mode is on, it is required and must have exactly one record per query. | |
| G16 | Stale artifacts from a previous fixture could make validation pass falsely. | Run start deletes the known artifact files. Manifest carries `kb_fingerprint` (hash of sorted doc hashes + queries hash); `validate.py` recomputes it and fails on mismatch. | |
| G17 | "Replayable" vs. timestamps in `llm_calls.jsonl`. | Determinism contract excludes only `run_manifest.json` and `llm_calls.jsonl`. | |
| G18 | Brief never says how the evaluator invokes the pipeline. | One documented command at the top of README (`python run_pipeline.py && python validate.py`), plus `Makefile` targets `run`/`validate`. Default paths resolve relative to the repo root, overridable by CLI flags. | |
| G19 | Sample KB is 4 tiny docs / 5 queries → chunking comparison and thresholds are statistically meaningless. | Comparison reports extra tie-breaker metrics (MRR, answer-point recall) and a mandatory honesty caveat in its explanation; thresholds are validated against probe fixtures, not just the sample. | |

---

## 4. Constraints

### 4.1 Hard constraints from the brief
1. Chunking, retrieval, evaluation implemented as runnable code in the repo; evaluation is deterministic code over retrieval output.
2. Answers grounded **only** in retrieved chunks; cite retrieved chunks only; never invent facts.
3. Allowed labels / statuses / citation format are controlled vocabularies **defined in code** and validated.
4. Pipeline stages enforced in code, strictly ordered; no answer generation before retrieval completes.
5. No hardcoded answers for sample queries; no dependence on filenames, doc order, or exact wording.
6. Must regenerate all artifacts from a clean checkout; static outputs are not acceptable.
7. No external proprietary data. Another engineer must be able to run it unmodified.
8. Chunk text must be preserved exactly as sourced (no stripping inside, no normalisation of stored text).
9. Retrieval returns ≥ 3 chunks per query; scores numeric.
10. If an LLM is used: one `llm_calls.jsonl` record per answer-generation call; if not, README says so.

### 4.2 Engineering constraints I'm adding
- **No network or API key needed** for the default path.
- **Windows-safe:** read/write all files as UTF-8 explicitly (`§` must survive), force `sys.stdout.reconfigure(encoding="utf-8")`, no shell-specific assumptions, `pathlib` everywhere.
- **No overfitting to the sample:** no query-id branching, no sample-specific stopwords or synonyms, no title literals.
- **Anti-staleness:** artifacts deleted at start; atomic write (temp file + rename).
- **Failure is loud:** any validation failure ⇒ non-zero exit and the final state is *not* `RESULTS_FINALISED`.
- **Single source of truth for vocabularies:** one `vocab.py` imported by pipeline, validator, and tests.

### 4.3 Explicitly out of scope
Embedding models / vector DBs / rerankers, a UI, auth, persistence of the index across runs, multi-language support.

---

## 5. Architecture

### 5.1 Repo layout

```
Deriv AI/
├─ README.md                  # how to run, design summary, "no LLM used" statement
├─ SPEC.md                    # this file
├─ kb/                        # sample fixtures (replaceable)
├─ queries.json               # sample fixture (replaceable)
├─ run_pipeline.py            # CLI: runs all stages
├─ validate.py                # CLI: validates artifacts on disk
├─ serve.py                   # stretch: POST /answer
├─ Makefile                   # run / validate / test wrappers
├─ requirements.txt           # empty (stdlib) — pytest listed under dev only
├─ rag/
│  ├─ config.py               # all tunables (k, BM25 params, weights, thresholds)
│  ├─ vocab.py                # controlled vocabularies + citation regex
│  ├─ stages.py               # state machine + typed stage outputs
│  ├─ loader.py               # kb/ + queries.json parsing, header parsing
│  ├─ chunking.py             # sentence_window / paragraph / fixed_size
│  ├─ textproc.py             # normalise, tokenise, stopwords, stemmer, n-grams
│  ├─ retrieval.py            # BM25 + n-gram index, fusion, ranking
│  ├─ answering.py            # extractive generator, sufficiency gate, conflict heuristic
│  ├─ llm.py                  # optional adapter + llm_calls.jsonl logging
│  ├─ evaluation.py           # deterministic retrieval eval
│  ├─ grounding.py            # answer grounding check
│  ├─ comparison.py           # chunking-strategy comparison
│  ├─ validation.py           # shared by pipeline stage and validate.py
│  └─ jsonio.py               # canonical JSON writer, atomic writes
├─ tests/                     # unit + fixture tests (see §9)
└─ artifacts/                 # generated; overwritten every run
```

### 5.2 Stage machine

```
INIT → DOCUMENTS_LOADED → DOCUMENTS_CHUNKED → INDEX_BUILT → RETRIEVAL_COMPLETE
     → ANSWERS_GENERATED → EVALUATION_COMPLETE → VALIDATION_COMPLETE → RESULTS_FINALISED
```

| Stage entered | Work done | Artifact written at this point |
|---|---|---|
| `DOCUMENTS_LOADED` | Parse docs + queries, headers, normalise | — |
| `DOCUMENTS_CHUNKED` | Chunk with primary strategy | `chunks.json` |
| `INDEX_BUILT` | Build BM25 + n-gram index (in memory) | — |
| `RETRIEVAL_COMPLETE` | Top-k per query | `retrieval.json` |
| `ANSWERS_GENERATED` | Gate → generate → per-answer self-check | `answers.json`, `llm_calls.jsonl` (LLM mode only) |
| `EVALUATION_COMPLETE` | Retrieval eval; chunking comparison sub-runs | `eval.json`, `chunking_comparison.json` |
| `VALIDATION_COMPLETE` | Grounding check; full artifact validation | `grounding_check.json` |
| `RESULTS_FINALISED` | Write manifest, print summary | `run_manifest.json` |

**Enforcement (MUST, in code):**
- `Pipeline.advance(next_stage)` raises `StageOrderError` unless `next_stage` is the immediate successor of the current stage. No skipping, no going back.
- Each stage function declares the stage it requires and the stage it produces; calling it in any other state raises.
- Stage hand-off uses typed objects (`LoadedCorpus` → `ChunkSet` → `Index` → `RetrievalResults` → `AnswerSet` …). `generate_answers()` takes `RetrievalResults` as a parameter, and that type is only constructible by the retrieval stage — "answers before retrieval" cannot be written without deliberately bypassing the type.
- The manifest records the stage log. On any exception the manifest is still written with `final_state` = last stage reached and `status: "failed"`.

---

## 6. Component specs

### 6.1 Ingestion & chunking

**Loading**
- Read every regular, non-hidden file under `--kb` (recursive), extensions `.txt`/`.md`, sorted by path for stable iteration. Decode UTF-8 (BOM tolerant), normalise `\r\n`/`\r` → `\n`.
- Header block = leading consecutive `Key: value` lines. Recognised keys, case-insensitive: `Title`, `Section`. Body begins after the header block and any blank lines that follow it; `body_offset` records where.
- A later `Section:` line at the start of a body line starts a new section; it is excluded from chunk text and chunks never span it.
- Skip empty/body-less files with a manifest warning. Zero usable documents ⇒ hard error (clear message).
- `doc_id` = `doc_<n>` assigned after sorting by `(casefold(title), sha256(body))` — **not** by filename. `source_file` is recorded for debugging only and is never used for logic.

**Chunk record** (extras beyond the required six are allowed and useful):

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | string | `chunk_<n>`, global, 1-based (G1) |
| `doc_title` | string | from `Title:` |
| `section` | string | section in effect at chunk start |
| `text` | string | **exact** substring of body |
| `start_char` | int | offset into body, inclusive |
| `end_char` | int | offset into body, exclusive; `body[start_char:end_char] == text` |
| `doc_id` | string | extra |
| `source_file` | string | extra, debug only |
| `chunk_index` | int | extra, 0-based within doc |
| `body_offset` | int | extra |
| `strategy` | string | extra |

**Strategies** (all deterministic, no stripping inside chunk text; boundaries are chosen so chunk text has no leading/trailing whitespace):

| Name | Rule | Default params |
|---|---|---|
| `sentence_window` *(primary)* | Split body into sentences on line breaks and `(?<=[.!?])\s+`; pack consecutive whole sentences until adding the next would exceed `max_chars`; ≥ 1 sentence per chunk; no overlap | `max_chars=300` |
| `paragraph` | Split on blank lines; a paragraph longer than `max_chars_hard` is split with `sentence_window` | `max_chars_hard=1200` |
| `fixed_size` | Windows of `size` chars with `overlap`, end snapped back to the nearest whitespace so words aren't cut | `size=200, overlap=40` |

Rationale for the primary: the sample docs are one-fact-per-line, so whole-paragraph chunks make every query "retrieve the doc" and teach us nothing; one-sentence chunks lose context ("If backup codes are unavailable…" depends on the previous line); a small sentence window keeps the citation unit tight while retaining a bit of context.

### 6.2 Retrieval

**Text processing (`textproc.py`)** — shared by index and query:
1. NFKC normalise, lowercase.
2. Tokenise with `[^\W_]+` (letters/digits). `e-wallet` → `e`, `wallet`; drop 1-char alphabetic tokens, keep digits.
3. Stopword list in code (standard English function words **plus** question/aux filler such as `can, could, will, would, does, did, how, long, what, tell, work, working, yesterday`). Negation words (`not, no, never, cannot`) are removed for retrieval but retained for the conflict heuristic.
4. Light, in-code stemmer (plural `s/es/ies`, `ing`, `ed`, `al`, `ion` suffix stripping with a minimum stem length) so `links/link`, `bills/bill`, `withdrawals/withdrawal`. No external NLP packages.

**Index text** for a chunk = `doc_title + " " + section + " " + text` (so "Password reset…" in the title helps a "reset link" query). Stored `chunk_text` is the unprefixed `text`.

**Scorers**
- *BM25* over stemmed tokens: Lucene-style IDF `ln(1 + (N − df + 0.5)/(df + 0.5))` (non-negative even on a 5-chunk corpus), `k1=1.5`, `b=0.75`.
- *Char 4-gram TF-IDF cosine* over the normalised index text — catches `withdraw`/`withdrawn` and typos that a stemmer misses.
- *Fusion:* each scorer's per-query scores divided by that scorer's max (→ `[0,1]`; all-zero stays zero), then `score = 0.7·bm25 + 0.3·ngram`. Weights live in `config.py`. Component scores kept in `retrieval.json` under `score_components` for debuggability.

**Ranking:** sort by `score` desc, ties by `start_char` asc then `chunk_id`. Return `top_k` (default 5, always ≥ min(3, n_chunks)). `rank` is 1-based.

### 6.3 Answer generation (citation-strict)

**Inputs:** only the `top_k` retrieved chunks for the query (plus the question). The generator never sees the rest of the KB.

**Step 1 — sufficiency gate** (deterministic)
- `Q` = stemmed content terms of the question (stopwords removed).
- For each retrieved chunk `c`: `coverage(c) = Σ idf(t) for t ∈ Q ∩ terms(c) / Σ idf(t) for t ∈ Q`, with `idf` from the index.
- Sufficient iff `max coverage ≥ τ_cov` **and** at least one matched term has `idf ≥ τ_idf` (so a lone ubiquitous word like "account" can't carry an answer).
- Not sufficient ⇒ `insufficient_context`, answer = fixed template ("The knowledge base does not contain enough information to answer this question."), `citations=[]`, `used_chunk_ids=[]`.
- `τ_cov`, `τ_idf` are calibrated once (acceptance criterion AC-ANS-6) and fixed in `config.py`.

**Step 2 — conflict check** (only if sufficient). Over the top-3 chunks, split into sentences; a pair is *conflicting* iff **all** hold:
1. both sentences individually pass a relevance floor w.r.t. the question;
2. non-numeric content-token Jaccard ≥ 0.8 (same claim, same subject);
3. they contain a number with the **same unit token** (`minutes`, `days`, `%`…) but different values, **or** exactly one contains a negation token.

Sample-KB sanity: "Bank withdrawals … 1 to 3 business days after approval" vs. "E-wallet withdrawals … 4 hours after approval" must **not** conflict (different units, low Jaccard). Result: label `conflicting_context`, answer states both sentences verbatim with a "sources disagree" lead-in, ≥ 2 citations, no attempt to pick a winner.

**Step 3 — extractive answer** (label `grounded_answer`)
- Sentence-split the retrieved chunks; score each sentence by IDF-weighted overlap with `Q` (plus a small bonus for the chunk's rank).
- Select the best sentence and any further sentence within 0.6× of its score, max `max_answer_sentences=2`, in source order.
- Answer = selected sentences **verbatim**, each followed by its citation `[doc_title §chunk_id]`. Mirrors the brief's example: `Bank withdrawals may take 1 to 3 business days after approval. [Cash withdrawal processing §chunk_2]`.
- `citations` = ordered, de-duplicated; `used_chunk_ids` = exactly the chunk ids appearing in `citations` (consistency rule).

**Step 4 — self-check** (every answer, either generator). Label ∈ vocab; for non-insufficient labels ≥ 1 citation (≥ 2 for conflicting); every citation parses, its chunk id is in this query's retrieved set, and its title equals that chunk's `doc_title`. Failure ⇒ replace with extractive output; if that fails too ⇒ `insufficient_context`. Never emit an invalid record.

**Optional LLM mode** (`--llm gemini`, off by default; Google Gemini REST API via stdlib `urllib`, key from `GEMINI_API_KEY`)
- Adapter interface `generate(question, chunks) -> {answer_label, answer, citations}` at temperature 0; prompt lists chunks with their literal citation tokens and demands JSON only.
- Output goes through Step 4 and the grounding check; invalid ⇒ extractive fallback for that query (record `fallback: true`).
- One `llm_calls.jsonl` line per query (schema §7.8). `prompt_hash` = SHA-256 of the exact prompt text.
- Provider, model id and key come from env/CLI (`GEMINI_API_KEY`, `GEMINI_MODEL` / `--llm-model`); default model id lives in `config.py`.

### 6.4 Evaluation inputs
`expected_doc_titles` and `ground_truth_answer_points` are read from `queries.json`; both are optional (missing ⇒ `[]`). Title comparison uses normalised titles (NFKC, casefold, collapse whitespace). Evaluation operates on `retrieval.json` data only — no access to answers.

### 6.5 Deterministic retrieval evaluation

Let `E` = set of normalised expected titles, `R3` = distinct normalised titles of the top-3 chunks.

| Condition | `retrieval_status` | `matched_expected_title` |
|---|---|---|
| `E` non-empty and `E ⊆ R3` | `hit` | `true` |
| `E ∩ R3` non-empty but `E ⊄ R3` | `partial_hit` | `true` |
| `E` non-empty and `E ∩ R3 = ∅` | `miss` | `false` |
| `E` empty | `hit` (vacuous) | `null` |

Explanation is a fixed template filled with data, e.g. `Expected title found at rank 1`, `1 of 2 expected titles found in top 3 (missing: "X")`, `Expected title "X" not in top 3 (retrieved: "A", "B")`, `Expected title "X" not present in knowledge base`.

Extra per-query diagnostics (do **not** affect status): `first_match_rank`, `answer_points_covered_top3` (a point is covered if ≥ 80 % of its stemmed content tokens appear in some top-3 chunk text).

Summary: `top3_hit_rate` = queries with `matched_expected_title == true` ÷ answerable queries (hit **or** partial); also `top3_full_hit_rate` (hits only), `mrr`, `answer_point_recall_top3`, plus the brief's required `total_queries`, `hits`, `partial_hits`, `misses`.

### 6.6 Grounding check (SHOULD)
Per answer: parse citations → each must exist in `retrieval.json` for **that** `query_id`, with matching title. Strip citation tokens, split into sentences, and for each sentence require: (a) normalised substring of a cited chunk, **or** content-token recall ≥ 0.8 against the union of cited chunks; and (b) every number in the sentence appears in a cited chunk. `insufficient_context` ⇒ `not_applicable`. Statuses: `pass | fail | not_applicable`.

### 6.7 Chunking comparison (SHOULD)
Run chunk → index → retrieve → evaluate for each of `sentence_window`, `paragraph`, `fixed_size` with the same retriever and `top_k`. Report per strategy: chunk count, mean chunk chars, eval summary, MRR, answer-point recall@3. The `tradeoff_explanation` is **templated from the numbers** (which strategy leads on which metric; tie handling) and always ends with a caveat that a 4-doc/5-query sample can't separate strategies on title-hit-rate. No LLM involved.

### 6.8 Validation (MUST)
`validate.py` / `rag/validation.py` — read-only over disk; exit 0 iff all rules pass; prints a rule-by-rule report.

| Rule | Check |
|---|---|
| V-01 | Required artifacts exist (`chunks`, `retrieval`, `answers`, `eval`; `grounding_check` and `chunking_comparison` required if the manifest says they ran; `llm_calls.jsonl` required iff `answer_generator` is an LLM) |
| V-02 | Every JSON file parses; top-level types match §7 |
| V-03 | Query-id set in retrieval = answers = eval = `queries.json` (no missing, no extra, no duplicates) |
| V-04 | Each retrieval entry has ≥ min(3, n_chunks) chunks; ranks 1..n contiguous |
| V-05 | `score` is a finite JSON number (not string/null/bool) |
| V-06 | `answer_label` ∈ vocab |
| V-07 | `grounded_answer` / `conflicting_context` have ≥ 1 / ≥ 2 citations; `insufficient_context` has 0 |
| V-08 | Every citation matches the citation regex and refers to a chunk in **that query's** retrieval, with matching title; `used_chunk_ids` == ids parsed from citations |
| V-09 | `retrieval_status` ∈ vocab |
| V-10 | `eval.summary` present with required keys; counts add up to `total_queries`; `top3_hit_rate` consistent with counts |
| V-11 | Chunks: `body[start:end] == text` when `kb/` is available; chunk ids unique; required fields present |
| V-12 | `kb_fingerprint` in manifest matches current `kb/` + `queries.json` |
| V-13 | If `grounding_check.json` exists, it covers every answer and uses vocab statuses |

### 6.9 Minimal API (STRETCH)
`python serve.py --port 8000` — stdlib `http.server`, binds `127.0.0.1`. At startup runs stages up to `INDEX_BUILT`; per request runs retrieval + answering with the **same** functions as the batch pipeline.

- `POST /answer` body `{"question": "<non-empty string>"}` → `200 {"answer_label", "answer", "citations"}` (exactly these three keys).
- `400` on invalid JSON / missing or empty `question`; `404`/`405` otherwise. `GET /health` → `{"status":"ok"}`.

---

## 7. Schemas

### 7.1 Input `queries.json`
```json
[ { "query_id": "string (unique, required)",
    "question": "string (non-empty, required)",
    "expected_doc_titles": ["string"],            // optional → []
    "ground_truth_answer_points": ["string"] } ]  // optional → []
```
Unknown extra keys are ignored. Duplicate `query_id` or empty `question` ⇒ hard error. Output order = input order.

### 7.2 `artifacts/chunks.json` (array)
```json
{ "chunk_id": "chunk_1", "doc_title": "Cash withdrawal processing", "section": "Payments",
  "text": "Withdrawal requests submitted before 17:00 UTC are usually reviewed on the same business day.",
  "start_char": 0, "end_char": 96,
  "doc_id": "doc_1", "source_file": "kb/article_02.txt", "chunk_index": 0,
  "body_offset": 41, "strategy": "sentence_window" }
```

### 7.3 `artifacts/retrieval.json` (array, one per query)
```json
{ "query_id": "Q1", "question": "…",
  "top_k": [ { "rank": 1, "chunk_id": "chunk_2", "doc_title": "Cash withdrawal processing",
               "score": 0.912345, "chunk_text": "…",
               "section": "Payments",
               "score_components": { "bm25": 1.0, "ngram": 0.71 } } ] }
```

### 7.4 `artifacts/answers.json` (array, one per query)
```json
{ "query_id": "Q1", "answer_label": "grounded_answer",
  "answer": "Bank withdrawals may take 1 to 3 business days after approval. [Cash withdrawal processing §chunk_2]",
  "citations": ["[Cash withdrawal processing §chunk_2]"],
  "used_chunk_ids": ["chunk_2"],
  "generator": "extractive" }
```
`generator` ∈ `extractive | llm | llm_fallback_extractive` (extra field).

### 7.5 `artifacts/eval.json`
```json
{ "summary": { "top3_hit_rate": 1.0, "top3_full_hit_rate": 1.0, "total_queries": 5,
               "answerable_queries": 5, "hits": 5, "partial_hits": 0, "misses": 0,
               "mrr": 1.0, "answer_point_recall_top3": 1.0, "k": 3 },
  "results": [ { "query_id": "Q1", "expected_doc_titles": ["Cash withdrawal processing"],
                 "retrieved_doc_titles_top3": ["Cash withdrawal processing", "Demo account behaviour"],
                 "retrieval_status": "hit", "matched_expected_title": true,
                 "explanation": "Expected title found at rank 1",
                 "first_match_rank": 1,
                 "answer_points_covered_top3": [ { "point": "…", "covered": true } ] } ] }
```

### 7.6 `artifacts/grounding_check.json`
```json
{ "summary": { "total": 5, "passed": 5, "failed": 0, "not_applicable": 0, "pass_rate": 1.0 },
  "results": [ { "query_id": "Q1", "answer_label": "grounded_answer", "status": "pass",
                 "citations_checked": [ { "citation": "[…§chunk_2]", "chunk_id": "chunk_2",
                                          "in_retrieval": true, "title_matches": true } ],
                 "sentences": [ { "text": "…", "token_recall": 1.0, "numbers_supported": true,
                                  "supported": true } ],
                 "reasons": [] } ] }
```

### 7.7 `artifacts/chunking_comparison.json`
```json
{ "primary_strategy": "sentence_window",
  "strategies": [ { "name": "paragraph", "params": {"max_chars_hard": 1200},
                    "num_chunks": 4, "avg_chunk_chars": 280.5,
                    "summary": { "top3_hit_rate": 1.0, "hits": 5, "partial_hits": 0, "misses": 0,
                                 "mrr": 1.0, "answer_point_recall_top3": 1.0 } } ],
  "tradeoff_explanation": "templated text …",
  "caveat": "…" }
```

### 7.8 `llm_calls.jsonl` (one JSON object per line, LLM mode only)
```json
{ "stage": "ANSWERS_GENERATED", "query_id": "Q1", "timestamp": "2026-10-13T09:00:00Z",
  "provider": "gemini", "model": "<model id>", "prompt_hash": "sha256:…",
  "input_artifacts": ["artifacts/retrieval.json"], "output_artifact": "artifacts/answers.json",
  "temperature": 0, "fallback": false, "latency_ms": 812 }
```

### 7.9 `artifacts/run_manifest.json` (non-deterministic by design)
```json
{ "pipeline_version": "1.0.0", "status": "ok", "final_state": "RESULTS_FINALISED",
  "stage_log": [ { "stage": "INIT", "entered_at": "…" } ],
  "inputs": { "kb_dir": "kb", "num_documents": 4, "queries_path": "queries.json",
              "num_queries": 5, "kb_fingerprint": "sha256:…" },
  "config": { "chunking": "sentence_window", "top_k": 5, "weights": {"bm25": 0.7, "ngram": 0.3} },
  "answer_generator": "extractive",
  "warnings": [] }
```

### 7.10 Controlled vocabularies (`vocab.py`)
```python
ANSWER_LABELS    = ("grounded_answer", "insufficient_context", "conflicting_context")
RETRIEVAL_STATUS = ("hit", "partial_hit", "miss")
GROUNDING_STATUS = ("pass", "fail", "not_applicable")          # extra, mine
CITATION_FORMAT  = "[{doc_title} §{chunk_id}]"
CITATION_RE      = re.compile(r"^\[(?P<title>.+) §(?P<chunk_id>[^\s\]§]+)\]$")
```

---

## 8. Determinism rules

- Sort every dict/set before iteration that affects output; never rely on `hash()` or filesystem order.
- Round floats to 6 dp **at write time**; fixed summation order.
- Canonical JSON: `indent=2`, `ensure_ascii=False`, UTF-8, trailing newline, fixed key order (dataclass order).
- No timestamps, absolute paths, or environment info in deterministic artifacts (`source_file` is repo-relative POSIX style).
- Test: run twice, `sha256` of each deterministic artifact must match; run with `kb/` files renamed and shuffled, chunk **text**, ids and retrieval ranking must match.

---

## 9. Acceptance criteria

### 9.1 MUST

| ID | Criterion |
|---|---|
| AC-RUN-1 | From a clean checkout with no `artifacts/`, `python run_pipeline.py` exits 0 and writes `chunks`, `retrieval`, `answers`, `eval` (+ `grounding_check`, `chunking_comparison`, `run_manifest`) |
| AC-RUN-2 | `python validate.py` exits 0 on a fresh run and non-zero on each mutation in AC-VAL-2 |
| AC-RUN-3 | Two consecutive runs produce byte-identical deterministic artifacts (§8) |
| AC-RUN-4 | Pipeline ends with `final_state = RESULTS_FINALISED`; stage log shows all 9 stages in order |
| AC-STG-1 | Calling any stage function from a wrong state raises `StageOrderError` (unit test per stage, incl. "generate answers at INDEX_BUILT") |
| AC-ING-1 | Title and Section parsed from every sample doc; every chunk has the six required fields |
| AC-ING-2 | `body[start_char:end_char] == text` for every chunk; chunk texts concatenated cover every non-header sentence exactly once (primary strategy) |
| AC-ING-3 | CRLF + BOM variant of the KB yields the same chunk texts and ids |
| AC-ING-4 | Renaming/shuffling KB files does not change chunk ids, texts or retrieval ranking |
| AC-RET-1 | Every query has ≥ min(3, n_chunks) results; ranks contiguous; scores finite floats; order is non-increasing in score |
| AC-RET-2 | On the sample fixtures, the expected doc title is in the top-3 for **all 5** queries (`top3_hit_rate == 1.0`) |
| AC-RET-3 | Retrieval never reads `queries.json` ground truth |
| AC-ANS-1 | Labels only from vocab; `grounded_answer` ⇒ ≥ 1 citation; `conflicting_context` ⇒ ≥ 2; `insufficient_context` ⇒ 0 citations |
| AC-ANS-2 | Every citation is in that query's retrieved set and its title matches the chunk |
| AC-ANS-3 | `used_chunk_ids` equals the set of ids parsed from `citations` |
| AC-ANS-4 | Sample Q1–Q5 each yield `grounded_answer` whose answer text covers the corresponding ground-truth point (token recall ≥ 0.8) |
| AC-ANS-5 | Answers are produced only from `RetrievalResults`; the answering module has no access to the full KB |
| AC-ANS-6 | Probe set (≥ 6 out-of-scope questions, e.g. "What is the capital of France?", "How do I change my email address?") ⇒ all `insufficient_context`; a conflicting-KB probe (two docs: reset link expires after 30 vs 60 minutes) ⇒ `conflicting_context`; sample KB produces **no** false conflicts |
| AC-EVAL-1 | `eval.json` has one record per query with all brief fields; statuses in vocab; `summary` has `top3_hit_rate`, `total_queries`, `hits`, `partial_hits`, `misses` and counts sum to `total_queries` |
| AC-EVAL-2 | Hand-built retrieval fixtures exercise each row of the §6.5 table (hit, partial, miss, vacuous, title-not-in-KB) with expected explanations |
| AC-EVAL-3 | Evaluation code imports nothing from the LLM or answering modules |
| AC-VAL-1 | `validate.py` implements V-01…V-13 |
| AC-VAL-2 | Mutation tests (each must fail validation): delete an artifact; corrupt JSON; drop a query; truncate a query to 2 results; score as string; unknown label; grounded answer without citations; citation to non-retrieved chunk; wrong title in citation; unknown retrieval status; remove eval summary; swap `kb/` content after run (fingerprint) |
| AC-LLM-1 | README states whether an LLM is used; with no LLM, `llm_calls.jsonl` is absent and `answer_generator == "extractive"` |
| AC-LLM-2 | (LLM mode) exactly one record per query with all required fields; `prompt_hash` is SHA-256 of the sent prompt; invalid model output triggers fallback, never an invalid artifact |
| AC-ROB-1 | No sample-specific literals in `rag/` (grep test for sample titles, `Q1`…`Q5`, "chunk_2") |
| AC-ROB-2 | Handles: missing `Section:`, missing `Title:` (warning), multiple `Section:` lines, empty file (warning), unicode in text, duplicate titles (warning), query without expected titles |

### 9.2 SHOULD

| ID | Criterion |
|---|---|
| AC-GRD-1 | `grounding_check.json` passes all 5 sample answers; a hand-tampered answer (injected number / unsupported sentence) is flagged `fail` |
| AC-CMP-1 | `chunking_comparison.json` covers ≥ 2 strategies (target 3), has per-strategy summaries and a non-empty explanation + caveat |
| AC-CMP-2 | Primary-strategy numbers in the comparison equal the numbers in `eval.json` |

### 9.3 STRETCH

| ID | Criterion |
|---|---|
| AC-API-1 | `POST /answer` with the Q1 question returns `200` and exactly `answer_label`, `answer`, `citations`; answer equals the batch pipeline's answer for the same question |
| AC-API-2 | Invalid body ⇒ `400`, server stays up |

---

## 10. Open questions for you

None of these block starting; each has a default I'll use unless you say otherwise. Ordered by how much a hidden evaluator could care.

1. **`eval.json` shape (G3).** Object `{summary, results}` vs. a bare array of records with the summary in a separate file (e.g. `eval_summary.json`). *Default: object.*
2. **`hit` vs. `partial_hit` meaning (G4).** Mine: set-based over expected titles. Alternative: `hit` = expected title at rank 1, `partial_hit` = rank 2–3. *Default: set-based, since the brief's explanation example is "found at rank 1" independent of status.*
3. **Chunk id scheme and offsets (G1, G2).** Global `chunk_<n>`; body-relative offsets. *Default as stated.*
4. **LLM or not.** Default is no LLM — deterministic, zero setup, and it satisfies "retrieval/eval not delegated to the LLM" trivially. *Resolved: optional Gemini adapter built (`--llm gemini`), off by default.*
5. **Unanswerable queries in the fixtures (G6).** Vacuous `hit` and excluded from the hit-rate denominator vs. counted as `miss`. *Default: vacuous hit, excluded.*
6. **Commit generated `artifacts/`?** I'd commit them (reviewers can read outputs without running) and let each run overwrite. Say if you'd rather `.gitignore` them.
7. **Python floor.** Assuming ≥ 3.10; say if the evaluator environment is older.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Thresholds (`τ_cov`, `τ_idf`) overfit a 4-doc corpus | Calibrate against probe fixtures (AC-ANS-6), require a margin (lowest in-scope coverage ≥ 1.25× highest out-of-scope), record the numbers in `config.py` comments |
| Hidden fixtures with larger/longer docs or multi-section files | Chunkers are length-driven, not sample-shaped; AC-ROB-2 covers header variants; sentence splitter tolerant of missing punctuation (line breaks are boundaries) |
| Semantic gaps ("an hour" vs. "30 minutes") | Handled at the retrieval level by title prefix + overlapping stems; the answer returns the KB fact verbatim rather than reasoning about the user's number — documented limitation |
| Conflict heuristic misses subtle contradictions or fires falsely | Deliberately narrow (high Jaccard + same unit); precision favoured over recall; false-positive guard test on the sample |
| Windows encoding (`§`, BOM, CRLF) | Explicit UTF-8 everywhere; AC-ING-3; stdout reconfigured |
| `make` absent on Windows | Python entry points are primary |
| Evaluator expects different file/field shapes | Extras are additive only; required fields are exactly the brief's; §10 items 1–3 flagged |

---

## 12. Suggested build order

1. `vocab.py`, `config.py`, `jsonio.py`, `stages.py` (+ stage-order tests)
2. `loader.py`, `chunking.py` → `chunks.json` (AC-ING-*)
3. `textproc.py`, `retrieval.py` → `retrieval.json` (AC-RET-*)
4. `evaluation.py` → `eval.json` (AC-EVAL-*)
5. `answering.py` (gate → extractive → conflict) → `answers.json` (AC-ANS-*)
6. `validation.py` + `validate.py` + mutation tests (AC-VAL-*)
7. `grounding.py`, `comparison.py`
8. Optional: `llm.py`, `serve.py`
9. README (run command, design summary, "no LLM used" statement), Makefile, probe fixtures, determinism test
