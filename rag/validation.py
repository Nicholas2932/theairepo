"""Artifact validation, rules V-01..V-13 (SPEC §6.8).

Read-only over the artifacts on disk. Used by the pipeline's
VALIDATION_COMPLETE stage (before the manifest exists, so the run context is
passed in) and by validate.py (which reads run_manifest.json).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from . import config, grounding, loader, vocab
from .jsonio import ArtifactStore, read_json, read_jsonl
from .stages import (AnswerSet, PRODUCER_SEAL, RetrievalResults, Stage, ValidationOutcome,
                     stage_step)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
MAX_MESSAGES = 12

RULES = {
    "V-01": "Required artifacts exist",
    "V-02": "JSON parses and top-level types are correct",
    "V-03": "Query ids consistent across queries, retrieval, answers, eval",
    "V-04": "Each query retrieves >= min(3, n_chunks) chunks with contiguous ranks",
    "V-05": "Retrieval scores are finite numbers",
    "V-06": "Answer labels are in the controlled vocabulary",
    "V-07": "Citation counts match the answer label",
    "V-08": "Citations are well-formed, retrieved for that query, title-matched; used_chunk_ids consistent",
    "V-09": "Retrieval statuses are in the controlled vocabulary",
    "V-10": "Evaluation summary present and consistent",
    "V-11": "Chunks are complete, unique and exact slices of their documents",
    "V-12": "Input fingerprint matches the current kb/ and queries.json",
    "V-13": "Grounding check covers every answer and all answers are grounded",
}

_MISSING = object()
_INVALID = object()

_CHUNK_FIELDS = {"chunk_id": str, "doc_title": str, "section": str, "text": str,
                 "start_char": int, "end_char": int}
_EVAL_FIELDS = ("query_id", "expected_doc_titles", "retrieved_doc_titles_top3",
                "retrieval_status", "matched_expected_title", "explanation")
_SUMMARY_FIELDS = ("top3_hit_rate", "total_queries", "hits", "partial_hits", "misses")


@dataclass
class RuleResult:
    rule: str
    title: str
    status: str
    messages: list = field(default_factory=list)


@dataclass
class ValidationReport:
    artifacts_dir: str
    results: list

    @property
    def ok(self) -> bool:
        return all(r.status != FAIL for r in self.results)

    def failures(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == FAIL]

    def render(self) -> str:
        lines = [f"Validation report for {self.artifacts_dir}"]
        for r in self.results:
            lines.append(f"  [{r.status}] {r.rule} {r.title}")
            for m in r.messages[:MAX_MESSAGES]:
                lines.append(f"         - {m}")
            if len(r.messages) > MAX_MESSAGES:
                lines.append(f"         ... {len(r.messages) - MAX_MESSAGES} more")
        passed = sum(1 for r in self.results if r.status == PASS)
        lines.append(f"Result: {'PASS' if self.ok else 'FAIL'} "
                     f"({passed} passed, {len(self.failures())} failed, "
                     f"{sum(1 for r in self.results if r.status == SKIP)} skipped)")
        return "\n".join(lines)


class ValidationFailed(Exception):
    def __init__(self, report: ValidationReport):
        super().__init__(f"artifact validation failed: "
                         f"{', '.join(r.rule for r in report.failures())}")
        self.report = report


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class _Validator:
    def __init__(self, artifacts_dir: Path, kb_dir: Path, queries_path: Path,
                 manifest_override: dict | None):
        self.dir = Path(artifacts_dir)
        self.kb_dir = Path(kb_dir)
        self.queries_path = Path(queries_path)
        self.in_pipeline = manifest_override is not None
        self.raw: dict[str, object] = {}
        self.load_errors: dict[str, str] = {}
        for name in config.ALL_ARTIFACTS:
            p = self.dir / name
            if not p.is_file():
                self.raw[name] = _MISSING
                continue
            try:
                self.raw[name] = read_jsonl(p) if name.endswith(".jsonl") else read_json(p)
            except Exception as exc:  # noqa: BLE001 - any parse failure is a finding
                self.raw[name] = _INVALID
                self.load_errors[name] = str(exc)
        self.manifest = manifest_override if self.in_pipeline else self._typed(config.MANIFEST_FILE, dict)
        self.chunks = self._list_of_dicts(config.CHUNKS_FILE)
        self.retrieval = self._list_of_dicts(config.RETRIEVAL_FILE)
        self.answers = self._list_of_dicts(config.ANSWERS_FILE)
        ev = self._typed(config.EVAL_FILE, dict)
        self.eval_summary = ev.get("summary") if ev and isinstance(ev.get("summary"), dict) else None
        self.eval_results = (ev.get("results") if ev and isinstance(ev.get("results"), list)
                             and all(isinstance(r, dict) for r in ev["results"]) else None)
        self.grounding = self._typed(config.GROUNDING_FILE, dict)
        self.llm_calls = self._list_of_dicts(config.LLM_CALLS_FILE)
        self.retrieval_by_q = ({e.get("query_id"): e for e in self.retrieval}
                               if self.retrieval is not None else None)
        try:
            self.queries = loader.load_queries(self.queries_path)
            self.queries_error = None
        except loader.InputError as exc:
            self.queries, self.queries_error = None, str(exc)

    # -- helpers --
    def _typed(self, name: str, typ):
        v = self.raw.get(name)
        return v if isinstance(v, typ) else None

    def _list_of_dicts(self, name: str):
        v = self._typed(name, list)
        return v if v is not None and all(isinstance(x, dict) for x in v) else None

    def generator_mode(self):
        return (self.manifest or {}).get("answer_generator")

    # -- rules --
    def v01(self):
        msgs = []
        required = list(config.CORE_ARTIFACTS)
        if not self.in_pipeline:
            required.append(config.MANIFEST_FILE)
        written = (self.manifest or {}).get("artifacts_written") or []
        required += [n for n in written if n in config.ALL_ARTIFACTS and n not in required]
        mode = self.generator_mode()
        if mode == vocab.GENERATOR_LLM and config.LLM_CALLS_FILE not in required:
            required.append(config.LLM_CALLS_FILE)
        for name in required:
            if self.raw.get(name) is _MISSING:
                msgs.append(f"missing {name}")
        if self.manifest is not None:
            if mode not in vocab.ANSWER_GENERATOR_MODES:
                msgs.append(f"manifest answer_generator {mode!r} not in {vocab.ANSWER_GENERATOR_MODES}")
            elif mode == vocab.GENERATOR_EXTRACTIVE and self.raw.get(config.LLM_CALLS_FILE) is not _MISSING:
                msgs.append(f"{config.LLM_CALLS_FILE} present but no LLM was used (stale file?)")
            if not self.in_pipeline:
                if self.manifest.get("status") != "ok":
                    msgs.append(f"manifest status is {self.manifest.get('status')!r}: "
                                f"{self.manifest.get('error', 'run did not complete')}")
                if self.manifest.get("final_state") != Stage.RESULTS_FINALISED.name:
                    msgs.append(f"manifest final_state is {self.manifest.get('final_state')!r}")
        return msgs

    def v02(self):
        msgs = [f"{name}: does not parse ({err})" for name, err in self.load_errors.items()]
        expectations = {
            config.CHUNKS_FILE: (self.chunks, "array of objects"),
            config.RETRIEVAL_FILE: (self.retrieval, "array of objects"),
            config.ANSWERS_FILE: (self.answers, "array of objects"),
            config.GROUNDING_FILE: (self.grounding, "object"),
            config.COMPARISON_FILE: (self._typed(config.COMPARISON_FILE, dict), "object"),
            config.LLM_CALLS_FILE: (self.llm_calls, "JSON lines of objects"),
        }
        if not self.in_pipeline:
            expectations[config.MANIFEST_FILE] = (self.manifest, "object")
        for name, (value, desc) in expectations.items():
            if self.raw.get(name) not in (_MISSING, _INVALID) and value is None:
                msgs.append(f"{name}: top level must be {desc}")
        if self.raw.get(config.EVAL_FILE) not in (_MISSING, _INVALID):
            if self.eval_summary is None:
                msgs.append(f"{config.EVAL_FILE}: needs a 'summary' object")
            if self.eval_results is None:
                msgs.append(f"{config.EVAL_FILE}: needs a 'results' array of objects")
        if self.grounding is not None and not isinstance(self.grounding.get("results"), list):
            msgs.append(f"{config.GROUNDING_FILE}: needs a 'results' array")
        return msgs

    def v03(self):
        if self.queries is None:
            return [f"cannot read queries: {self.queries_error}"]
        expected = [q.query_id for q in self.queries]
        msgs = []
        sources = {config.RETRIEVAL_FILE: self.retrieval, config.ANSWERS_FILE: self.answers,
                   config.EVAL_FILE: self.eval_results}
        if self.generator_mode() == vocab.GENERATOR_LLM:
            sources[config.LLM_CALLS_FILE] = self.llm_calls
        for name, records in sources.items():
            if records is None:
                msgs.append(f"{name}: unavailable")
                continue
            ids = [r.get("query_id") for r in records]
            dupes = sorted({str(i) for i in ids if ids.count(i) > 1})
            missing = [i for i in expected if i not in ids]
            extra = [str(i) for i in ids if i not in expected]
            if dupes:
                msgs.append(f"{name}: duplicate query ids {dupes}")
            if missing:
                msgs.append(f"{name}: missing query ids {missing}")
            if extra:
                msgs.append(f"{name}: unknown query ids {extra}")
            if not (dupes or missing or extra) and ids != expected:
                msgs.append(f"{name}: query order differs from queries.json")
        return msgs

    def v04(self):
        if self.retrieval is None:
            return None
        n_chunks = len(self.chunks) if self.chunks is not None else None
        need = min(config.MIN_RETRIEVED, n_chunks) if n_chunks is not None else config.MIN_RETRIEVED
        by_id = {c.get("chunk_id"): c for c in self.chunks} if self.chunks is not None else None
        msgs = []
        for e in self.retrieval:
            qid = e.get("query_id")
            top = e.get("top_k")
            if not isinstance(top, list) or not all(isinstance(h, dict) for h in top):
                msgs.append(f"{qid}: top_k must be an array of objects")
                continue
            if len(top) < need:
                msgs.append(f"{qid}: {len(top)} chunks retrieved, need >= {need}")
            if [h.get("rank") for h in top] != list(range(1, len(top) + 1)):
                msgs.append(f"{qid}: ranks are not 1..{len(top)}")
            scores = [h.get("score") for h in top]
            if all(_is_number(s) for s in scores) and any(a < b for a, b in zip(scores, scores[1:])):
                msgs.append(f"{qid}: scores are not in non-increasing rank order")
            ids = [h.get("chunk_id") for h in top]
            if len(set(ids)) != len(ids):
                msgs.append(f"{qid}: duplicate chunks in top_k")
            for h in top:
                for key in ("chunk_id", "doc_title", "chunk_text"):
                    if not isinstance(h.get(key), str):
                        msgs.append(f"{qid}: rank {h.get('rank')} missing {key}")
                if by_id is not None and isinstance(h.get("chunk_id"), str):
                    c = by_id.get(h["chunk_id"])
                    if c is None:
                        msgs.append(f"{qid}: {h['chunk_id']} not in {config.CHUNKS_FILE}")
                    elif c.get("doc_title") != h.get("doc_title") or c.get("text") != h.get("chunk_text"):
                        msgs.append(f"{qid}: {h['chunk_id']} title/text differs from {config.CHUNKS_FILE}")
        return msgs

    def v05(self):
        if self.retrieval is None:
            return None
        msgs = []
        for e in self.retrieval:
            for h in e.get("top_k") or []:
                if isinstance(h, dict) and not _is_number(h.get("score")):
                    msgs.append(f"{e.get('query_id')}: rank {h.get('rank')} score "
                                f"{h.get('score')!r} is not a finite number")
        return msgs

    def _answer_problems(self) -> dict[str, list[str]] | None:
        if self.answers is None:
            return None
        out = {k: [] for k in (grounding.P_LABEL, grounding.P_COUNT, grounding.P_CITATION,
                               grounding.P_USED_IDS)}
        for a in self.answers:
            qid = a.get("query_id")
            entry = (self.retrieval_by_q or {}).get(qid) or {}
            top = entry.get("top_k") if isinstance(entry.get("top_k"), list) else []
            for kind, msg in grounding.structural_problems(a, grounding.retrieved_by_id(top)):
                out[kind].append(f"{qid}: {msg}")
            gen = a.get("generator")
            if gen is not None and gen not in vocab.GENERATORS:
                out[grounding.P_LABEL].append(f"{qid}: unknown generator {gen!r}")
        return out

    def v06(self):
        p = self._answer_problems()
        return None if p is None else p[grounding.P_LABEL]

    def v07(self):
        p = self._answer_problems()
        return None if p is None else p[grounding.P_COUNT]

    def v08(self):
        if self.retrieval is None:
            return None
        p = self._answer_problems()
        return None if p is None else p[grounding.P_CITATION] + p[grounding.P_USED_IDS]

    def v09(self):
        if self.eval_results is None:
            return None
        msgs = []
        for r in self.eval_results:
            qid = r.get("query_id")
            missing = [f for f in _EVAL_FIELDS if f not in r]
            if missing:
                msgs.append(f"{qid}: missing fields {missing}")
            status = r.get("retrieval_status")
            if status not in vocab.RETRIEVAL_STATUS:
                msgs.append(f"{qid}: unknown retrieval_status {status!r}")
            matched = r.get("matched_expected_title")
            ok = {vocab.HIT: (True, None), vocab.PARTIAL_HIT: (True,), vocab.MISS: (False,)}
            if status in ok and matched not in ok[status]:
                msgs.append(f"{qid}: matched_expected_title {matched!r} inconsistent with {status}")
            top3 = r.get("retrieved_doc_titles_top3")
            if not isinstance(top3, list) or len(top3) > config.EVAL_K:
                msgs.append(f"{qid}: retrieved_doc_titles_top3 must be a list of <= {config.EVAL_K}")
        return msgs

    def v10(self):
        if self.eval_summary is None or self.eval_results is None:
            return ["eval summary/results unavailable"]
        s, res = self.eval_summary, self.eval_results
        msgs = [f"summary missing {k}" for k in _SUMMARY_FIELDS if k not in s]
        if msgs:
            return msgs
        counts = {st: sum(1 for r in res if r.get("retrieval_status") == st) for st in vocab.RETRIEVAL_STATUS}
        if s["total_queries"] != len(res):
            msgs.append(f"total_queries {s['total_queries']} != {len(res)} results")
        if s["hits"] + s["partial_hits"] + s["misses"] != s["total_queries"]:
            msgs.append("hits + partial_hits + misses != total_queries")
        for key, st in (("hits", vocab.HIT), ("partial_hits", vocab.PARTIAL_HIT), ("misses", vocab.MISS)):
            if s[key] != counts[st]:
                msgs.append(f"summary {key}={s[key]} but results contain {counts[st]}")
        answerable = [r for r in res if r.get("matched_expected_title") is not None]
        expected_rate = (sum(1 for r in answerable if r.get("matched_expected_title")) / len(answerable)
                         if answerable else 0.0)
        if not _is_number(s["top3_hit_rate"]) or abs(s["top3_hit_rate"] - expected_rate) > 1e-6:
            msgs.append(f"top3_hit_rate {s['top3_hit_rate']!r} inconsistent with results "
                        f"({expected_rate:.6f})")
        return msgs

    def v11(self):
        if self.chunks is None:
            return None
        msgs = []
        ids = [c.get("chunk_id") for c in self.chunks]
        if len(set(ids)) != len(ids):
            msgs.append("duplicate chunk ids")
        try:
            docs, _ = loader.load_documents(self.kb_dir)
            by_doc = {d.doc_id: d for d in docs}
        except loader.InputError as exc:
            by_doc = None
            msgs.append(f"cannot load knowledge base to verify offsets: {exc}")
        for c in self.chunks:
            cid = c.get("chunk_id")
            bad = [k for k, t in _CHUNK_FIELDS.items()
                   if not isinstance(c.get(k), t) or isinstance(c.get(k), bool)]
            if bad:
                msgs.append(f"{cid}: missing/invalid fields {bad}")
                continue
            if not c["text"] or c["start_char"] < 0 or c["end_char"] - c["start_char"] != len(c["text"]):
                msgs.append(f"{cid}: offsets do not match text length")
            if by_doc is not None:
                doc = by_doc.get(c.get("doc_id"))
                if doc is None:
                    msgs.append(f"{cid}: doc_id {c.get('doc_id')!r} not found in knowledge base")
                elif doc.body[c["start_char"]:c["end_char"]] != c["text"]:
                    msgs.append(f"{cid}: body[start_char:end_char] != text")
                elif doc.title != c["doc_title"]:
                    msgs.append(f"{cid}: doc_title differs from document title")
        return msgs

    def v12(self):
        if self.manifest is None:
            return ["manifest unavailable"]
        recorded = (self.manifest.get("kb_fingerprint")
                    or (self.manifest.get("inputs") or {}).get("kb_fingerprint"))
        try:
            current = loader.compute_fingerprint(self.kb_dir, self.queries_path)
        except (loader.InputError, OSError) as exc:
            return [f"cannot fingerprint inputs: {exc}"]
        if recorded != current:
            return [f"artifacts were produced from different inputs "
                    f"(manifest {recorded}, current {current})"]
        return []

    def v13(self):
        if self.answers is None or self.retrieval is None:
            return None
        recomputed = grounding.check_all(self.answers, self.retrieval)
        msgs = [f"{r['query_id']}: {reason}" for r in recomputed["results"]
                if r["status"] == vocab.GROUNDING_FAIL for reason in r["reasons"]]
        if self.raw.get(config.GROUNDING_FILE) is _MISSING:
            return msgs
        results = (self.grounding or {}).get("results")
        if not isinstance(results, list):
            return msgs + [f"{config.GROUNDING_FILE} unreadable"]
        on_disk = {r.get("query_id"): r.get("status") for r in results if isinstance(r, dict)}
        for r in recomputed["results"]:
            st = on_disk.get(r["query_id"], _MISSING)
            if st is _MISSING:
                msgs.append(f"{r['query_id']}: not covered by {config.GROUNDING_FILE}")
            elif st not in vocab.GROUNDING_STATUS:
                msgs.append(f"{r['query_id']}: unknown grounding status {st!r}")
            elif st != r["status"]:
                msgs.append(f"{r['query_id']}: {config.GROUNDING_FILE} says {st}, recomputed {r['status']}")
        return msgs

    def run(self) -> ValidationReport:
        results = []
        for rule, title in RULES.items():
            msgs = getattr(self, "v" + rule[2:])()
            if msgs is None:
                results.append(RuleResult(rule, title, SKIP, ["prerequisite artifact unavailable"]))
            else:
                results.append(RuleResult(rule, title, FAIL if msgs else PASS, msgs))
        # A skipped rule is only acceptable when an earlier rule already failed.
        if any(r.status == SKIP for r in results) and not any(r.status == FAIL for r in results):
            for r in results:
                if r.status == SKIP:
                    r.status = FAIL
        return ValidationReport(self.dir.as_posix(), results)


def validate_artifacts(artifacts_dir: Path, kb_dir: Path, queries_path: Path,
                       manifest_override: dict | None = None) -> ValidationReport:
    return _Validator(artifacts_dir, kb_dir, queries_path, manifest_override).run()


@stage_step(requires=Stage.EVALUATION_COMPLETE, produces=Stage.VALIDATION_COMPLETE)
def validation_stage(pipeline, results: RetrievalResults, answer_set: AnswerSet,
                     store: ArtifactStore, kb_dir: Path, queries_path: Path,
                     run_context: dict) -> ValidationOutcome:
    """Grounding check, then full validation of what is on disk. Raises
    ValidationFailed, so the pipeline never reaches RESULTS_FINALISED."""
    from .retrieval import retrieval_to_json
    if not isinstance(answer_set, AnswerSet) or not isinstance(results, RetrievalResults):
        raise TypeError("validation_stage needs RetrievalResults and an AnswerSet")
    g = grounding.check_all(list(answer_set.answers), retrieval_to_json(results))
    store.write_json(config.GROUNDING_FILE, g)
    context = dict(run_context, artifacts_written=list(store.written))
    report = validate_artifacts(store.dir, kb_dir, queries_path, manifest_override=context)
    if not report.ok:
        raise ValidationFailed(report)
    return ValidationOutcome(_seal=PRODUCER_SEAL, grounding=g, report=report)
