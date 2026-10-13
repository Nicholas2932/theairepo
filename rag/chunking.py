"""Chunking strategies (SPEC §6.1).

Every chunk's text is an exact slice of its document body:
body[start_char:end_char] == text. Boundaries are chosen so chunks carry no
leading/trailing whitespace; nothing inside a chunk is altered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config, vocab
from .jsonio import ArtifactStore
from .stages import ChunkSet, LoadedCorpus, PRODUCER_SEAL, Stage, stage_step
from .textproc import sentence_spans, trim_span

_PARA_BREAK_RE = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_title: str
    section: str
    text: str
    start_char: int
    end_char: int
    doc_id: str
    source_file: str
    chunk_index: int
    body_offset: int
    strategy: str


def chunk_ordinal(chunk_id: str) -> int:
    return int(chunk_id.rsplit("_", 1)[1])


# --- Span builders (all offsets are into the document body) ---------------

def _sentence_window(body: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur: list[int] | None = None
    for s, e in sentence_spans(body, start, end):
        if cur is None:
            cur = [s, e]
        elif e - cur[0] <= max_chars:
            cur[1] = e
        else:
            out.append((cur[0], cur[1]))
            cur = [s, e]
    if cur is not None:
        out.append((cur[0], cur[1]))
    return out


def _paragraph(body: str, start: int, end: int, max_chars_hard: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = start
    bounds = [(m.start(), m.end()) for m in _PARA_BREAK_RE.finditer(body, start, end)]
    for p_end, next_start in bounds + [(end, end)]:
        s, e = trim_span(body, pos, p_end)
        if s < e:
            if e - s > max_chars_hard:
                out.extend(_sentence_window(body, s, e, max_chars_hard))
            else:
                out.append((s, e))
        pos = next_start
    return out


def _fixed_size(body: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = start
    last_end = -1
    while True:
        while pos < end and body[pos].isspace():
            pos += 1
        if pos >= end:
            break
        w_end = min(pos + size, end)
        if w_end < end and not body[w_end].isspace():
            cut = next((i for i in range(w_end - 1, pos, -1) if body[i].isspace()), None)
            if cut is not None:
                w_end = cut
        s, e = trim_span(body, pos, w_end)
        if s < e and e > last_end:
            out.append((s, e))
            last_end = e
        if w_end >= end:
            break
        nxt = max(w_end - overlap, pos + 1)
        if not body[nxt - 1].isspace():          # mid-word: move to next word start
            while nxt < w_end and not body[nxt].isspace():
                nxt += 1
        pos = nxt
    return out


def _spans(strategy: str, body: str, start: int, end: int, params: dict) -> list[tuple[int, int]]:
    if strategy == vocab.SENTENCE_WINDOW:
        return _sentence_window(body, start, end, params["max_chars"])
    if strategy == vocab.PARAGRAPH:
        return _paragraph(body, start, end, params["max_chars_hard"])
    if strategy == vocab.FIXED_SIZE:
        return _fixed_size(body, start, end, params["size"], params["overlap"])
    raise ValueError(f"unknown chunking strategy {strategy!r}")


def chunk_documents(documents, strategy: str, params: dict | None = None) -> tuple[Chunk, ...]:
    """Chunk documents (already in canonical order). chunk ids are global."""
    params = params if params is not None else config.CHUNKING_PARAMS[strategy]
    chunks: list[Chunk] = []
    for doc in documents:
        index = 0
        for seg in doc.segments:
            for s, e in _spans(strategy, doc.body, seg.start, seg.end, params):
                chunks.append(Chunk(
                    chunk_id=f"chunk_{len(chunks) + 1}", doc_title=doc.title,
                    section=seg.section, text=doc.body[s:e], start_char=s, end_char=e,
                    doc_id=doc.doc_id, source_file=doc.source_file, chunk_index=index,
                    body_offset=doc.body_offset, strategy=strategy))
                index += 1
    return tuple(chunks)


@stage_step(requires=Stage.DOCUMENTS_LOADED, produces=Stage.DOCUMENTS_CHUNKED)
def chunk_stage(pipeline, corpus: LoadedCorpus, strategy: str,
                store: ArtifactStore | None) -> ChunkSet:
    if not isinstance(corpus, LoadedCorpus):
        raise TypeError("chunk_stage needs a LoadedCorpus")
    params = dict(config.CHUNKING_PARAMS[strategy])
    chunks = chunk_documents(corpus.documents, strategy, params)
    if not chunks:
        raise ValueError("chunking produced no chunks")
    if store is not None:
        store.write_json(config.CHUNKS_FILE, list(chunks))
    return ChunkSet(_seal=PRODUCER_SEAL, chunks=chunks, strategy=strategy, params=params)
