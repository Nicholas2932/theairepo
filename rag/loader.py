"""Load kb/ documents and queries.json (SPEC §6.1, §7.1).

Nothing here depends on filenames or file order: documents are ordered and
numbered by (casefolded title, sha256 of body). source_file is kept for
debugging only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import config
from .stages import LoadedCorpus, PRODUCER_SEAL, Stage, stage_step

_HEADER_RE = re.compile(r"^[ \t]*(?P<key>[A-Za-z][A-Za-z ]*?)[ \t]*:[ \t]*(?P<value>.*?)[ \t]*$")
_SECTION_LINE_RE = re.compile(r"^[ \t]*section[ \t]*:[ \t]*(?P<value>.*?)[ \t]*$", re.IGNORECASE)
_HEADER_KEYS = ("title", "section")


class InputError(Exception):
    """Unusable input (missing/invalid KB or queries)."""


@dataclass(frozen=True)
class Segment:
    section: str
    start: int   # offsets into the document body
    end: int


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str
    body_offset: int
    source_file: str
    sha256: str
    segments: tuple


@dataclass(frozen=True)
class Query:
    query_id: str
    question: str
    expected_doc_titles: tuple
    ground_truth_answer_points: tuple


# --- KB --------------------------------------------------------------------

def discover_kb_files(kb_dir: Path) -> list[Path]:
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        raise InputError(f"knowledge base directory not found: {kb_dir}")
    files = []
    for p in kb_dir.rglob("*"):
        rel = p.relative_to(kb_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if p.is_file() and p.suffix.lower() in config.KB_EXTENSIONS:
            files.append(p)
    return sorted(files, key=lambda p: p.relative_to(kb_dir).as_posix())


def read_normalised(path: Path) -> str:
    """UTF-8 (BOM tolerant), CRLF/CR -> LF. No other change."""
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError(f"{path}: not valid UTF-8 ({exc})") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _lines_with_offsets(text: str) -> list[tuple[int, str]]:
    out, pos = [], 0
    for line in text.split("\n"):
        out.append((pos, line))
        pos += len(line) + 1
    return out


def parse_document(text: str, source_file: str) -> tuple[dict, list[str]]:
    """Split header block from body. Returns (fields, warnings)."""
    warnings: list[str] = []
    headers: dict[str, str] = {}
    lines = _lines_with_offsets(text)
    i = 0
    while i < len(lines) and not lines[i][1].strip():
        i += 1
    while i < len(lines):
        m = _HEADER_RE.match(lines[i][1])
        if not m:
            break
        key = m.group("key").strip().casefold()
        if key not in _HEADER_KEYS or key in headers:
            break
        headers[key] = " ".join(m.group("value").split())
        i += 1
    while i < len(lines) and not lines[i][1].strip():
        i += 1
    body_offset = lines[i][0] if i < len(lines) else len(text)
    body = text[body_offset:]

    title = headers.get("title", "")
    if not title:
        first = next((ln.strip() for _, ln in _lines_with_offsets(body) if ln.strip()), "")
        if first:
            title = " ".join(first.split())
            warnings.append(f"{source_file}: missing Title header; using first line as title")
        else:
            title = Path(source_file).stem
            warnings.append(f"{source_file}: missing Title header; using filename as title")

    # Body-level "Section:" lines start a new segment and are never chunk text.
    segments: list[Segment] = []
    current_section = headers.get("section", "")
    seg_start = 0
    for line_start, line in _lines_with_offsets(body):
        m = _SECTION_LINE_RE.match(line)
        if m:
            segments.append(Segment(current_section, seg_start, line_start))
            current_section = " ".join(m.group("value").split())
            seg_start = min(line_start + len(line) + 1, len(body))
    segments.append(Segment(current_section, seg_start, len(body)))
    segments = [s for s in segments if body[s.start:s.end].strip()]

    return {"title": title, "body": body, "body_offset": body_offset,
            "segments": tuple(segments)}, warnings


def load_documents(kb_dir: Path) -> tuple[tuple[Document, ...], list[str]]:
    kb_dir = Path(kb_dir)
    warnings: list[str] = []
    parsed = []
    for path in discover_kb_files(kb_dir):
        source_file = f"{kb_dir.name}/{path.relative_to(kb_dir).as_posix()}"
        fields, w = parse_document(read_normalised(path), source_file)
        warnings.extend(w)
        if not fields["segments"]:
            warnings.append(f"{source_file}: no body text; skipped")
            continue
        sha = hashlib.sha256(fields["body"].encode("utf-8")).hexdigest()
        parsed.append((fields, source_file, sha))
    if not parsed:
        raise InputError(f"no usable documents ({', '.join(config.KB_EXTENSIONS)}) in {kb_dir}")

    parsed.sort(key=lambda p: (p[0]["title"].casefold(), p[2], p[1]))
    docs = tuple(
        Document(doc_id=f"doc_{n}", title=f["title"], body=f["body"],
                 body_offset=f["body_offset"], source_file=src, sha256=sha,
                 segments=f["segments"])
        for n, (f, src, sha) in enumerate(parsed, 1))

    seen: dict[str, str] = {}
    for d in docs:
        key = d.title.casefold()
        if key in seen:
            warnings.append(f"duplicate title {d.title!r} ({seen[key]}, {d.source_file}); "
                            "evaluation cannot distinguish these documents")
        else:
            seen[key] = d.source_file
    return docs, warnings


# --- Queries -----------------------------------------------------------------

def _string_list(value: object, field: str, qid: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise InputError(f"query {qid}: {field} must be a list of strings")
    return tuple(v for v in value if v.strip())


def load_queries(path: Path) -> tuple[Query, ...]:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"queries file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputError(f"{path}: invalid JSON ({exc})") from exc
    if isinstance(data, dict) and isinstance(data.get("queries"), list):
        data = data["queries"]
    if not isinstance(data, list):
        raise InputError(f"{path}: expected a JSON array of queries")
    if not data:
        raise InputError(f"{path}: contains no queries")

    queries, seen = [], set()
    for n, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise InputError(f"{path}: entry {n} is not an object")
        qid = item.get("query_id")
        if isinstance(qid, int) and not isinstance(qid, bool):
            qid = str(qid)
        if not isinstance(qid, str) or not qid.strip():
            raise InputError(f"{path}: entry {n} has no query_id")
        if qid in seen:
            raise InputError(f"{path}: duplicate query_id {qid!r}")
        seen.add(qid)
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise InputError(f"{path}: query {qid} has an empty question")
        queries.append(Query(
            query_id=qid, question=question,
            expected_doc_titles=_string_list(item.get("expected_doc_titles"), "expected_doc_titles", qid),
            ground_truth_answer_points=_string_list(
                item.get("ground_truth_answer_points"), "ground_truth_answer_points", qid)))
    return tuple(queries)


# --- Fingerprint ---------------------------------------------------------------

def compute_fingerprint(kb_dir: Path, queries_path: Path) -> str:
    """Content hash of the inputs, independent of file names and order."""
    doc_hashes = sorted(hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in discover_kb_files(Path(kb_dir)))
    q_hash = hashlib.sha256(Path(queries_path).read_bytes()).hexdigest()
    h = hashlib.sha256()
    h.update("\n".join(doc_hashes).encode())
    h.update(b"\nqueries:")
    h.update(q_hash.encode())
    return "sha256:" + h.hexdigest()


def display_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(config.REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@stage_step(requires=Stage.INIT, produces=Stage.DOCUMENTS_LOADED)
def load_stage(pipeline, kb_dir: Path, queries_path: Path) -> LoadedCorpus:
    docs, warnings = load_documents(kb_dir)
    queries = load_queries(queries_path)
    for q in queries:
        if not q.expected_doc_titles:
            warnings.append(f"query {q.query_id}: no expected_doc_titles (vacuous hit in evaluation)")
    return LoadedCorpus(
        _seal=PRODUCER_SEAL, documents=docs, queries=queries, warnings=tuple(warnings),
        fingerprint=compute_fingerprint(kb_dir, queries_path),
        kb_dir=display_path(kb_dir), queries_path=display_path(queries_path),
        num_files=len(discover_kb_files(kb_dir)))
