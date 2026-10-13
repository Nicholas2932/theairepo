"""Shared text processing: normalisation, tokenising, stopwords, stemming,
sentence spans, numbers (SPEC §6.2). Used identically by index and query
sides so their vocabularies always line up.

Stored chunk text is never modified by anything here; these functions only
produce derived views for matching.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from . import config

_TOKEN_RE = re.compile(r"[^\W_]+")
_NUMBER_RE = re.compile(r"\d+(?:[.,:]\d+)*")
_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:,\d{3})+$")
_UNIT_SCAN_RE = re.compile(r"(?P<num>\d+(?:[.,:]\d+)*)|(?P<pct>%)|(?P<word>[^\W\d_]+)")
_SENT_BOUNDARY_RE = re.compile(r"\n+|(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_CONTRACTIONS = (
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"n't\b"), " not"),
)

NEGATIONS = frozenset({"not", "no", "never", "cannot"})

# Standard English function words plus question/auxiliary filler. Generic by
# design: nothing here is specific to any knowledge base.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by did do does doing down during each few
for from further had has have having he her here hers herself him himself his
i if in into is it its itself just me more most my myself nor of off on once
only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through
to too under until up very was we were what when where which while who whom why
with you your yours yourself yourselves will would can could shall may might
must also how long tell work working yesterday today please hi hello thanks
thank im ive id ill youre isn aren wasn weren doesn don didn hasn haven hadn
wouldn shouldn couldn mustn s t d ll m re ve get got getting still anymore
something anything someone anyone way ok okay really want wants need needs
know let lets
""".split())


def normalise(text: str) -> str:
    """NFKC, casefold, unify apostrophes, expand negative contractions."""
    t = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).casefold()
    for pattern, replacement in _CONTRACTIONS:
        t = pattern.sub(replacement, t)
    return t


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalise(text))


def _is_content(tok: str) -> bool:
    if tok in STOPWORDS or tok in NEGATIONS:
        return False
    if len(tok) == 1 and not tok.isdigit():
        return False
    return True


def stem(word: str) -> str:
    """Light deterministic suffix stripper: plurals, then one derivational /
    inflectional suffix, then a trailing 'e' and a doubled final consonant."""
    if not word.isalpha() or len(word) <= config.MIN_STEM_LEN:
        return word
    w = word
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith(("xes", "zes", "ches", "shes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suffix in ("ation", "ion", "ing", "ed", "ly", "al"):
        if w.endswith(suffix) and len(w) - len(suffix) >= config.MIN_STEM_LEN:
            w = w[: -len(suffix)]
            break
    if len(w) > config.MIN_STEM_LEN and w.endswith("e"):
        w = w[:-1]
    if len(w) > config.MIN_STEM_LEN and w[-1] == w[-2] and w[-1] not in "lsz":
        w = w[:-1]
    return w


def content_tokens(text: str) -> list[str]:
    """Unstemmed content tokens (stopwords, negations, 1-char letters removed)."""
    return [t for t in tokens(text) if _is_content(t)]


def content_terms(text: str) -> list[str]:
    """Stemmed content terms, in text order, with repeats."""
    return [stem(t) for t in content_tokens(text)]


def term_set(text: str) -> frozenset[str]:
    return frozenset(content_terms(text))


def unique_terms(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(content_terms(text)))


def has_negation(text: str) -> bool:
    return any(t in NEGATIONS for t in tokens(text))


def char_ngrams(text: str, n: int = config.NGRAM_N) -> Counter:
    """Character n-grams over the space-joined content tokens, word-padded."""
    toks = content_tokens(text)
    if not toks:
        return Counter()
    s = " " + " ".join(toks) + " "
    if len(s) < n:
        return Counter([s])
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def canon_number(value: str) -> str:
    return value.replace(",", "") if _THOUSANDS_RE.match(value) else value


def numbers(text: str) -> frozenset[str]:
    return frozenset(canon_number(m) for m in _NUMBER_RE.findall(normalise(text)))


def number_units(text: str) -> dict[str, frozenset[str]]:
    """Map unit -> set of values. A number's unit is '%' if it follows
    immediately, else the first content word within UNIT_LOOKAHEAD tokens
    (stemmed), else ''."""
    items = [(m.lastgroup, m.group()) for m in _UNIT_SCAN_RE.finditer(normalise(text))]
    out: dict[str, set[str]] = {}
    for i, (kind, value) in enumerate(items):
        if kind != "num":
            continue
        unit = ""
        for kind2, value2 in items[i + 1: i + 1 + config.UNIT_LOOKAHEAD]:
            if kind2 == "pct":
                unit = "%"
                break
            if kind2 == "word" and _is_content(value2):
                unit = stem(value2)
                break
        out.setdefault(unit, set()).add(canon_number(value))
    return {k: frozenset(v) for k, v in out.items()}


def collapse_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def norm_title(title: str) -> str:
    """Title comparison key: NFKC, casefold, whitespace collapsed."""
    return collapse_ws(unicodedata.normalize("NFKC", title).casefold())


def norm_for_match(text: str) -> str:
    """Loose form for substring checks: normalised, whitespace collapsed,
    trailing sentence punctuation dropped."""
    return collapse_ws(normalise(text)).rstrip(" .!?;:,")


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Sentence spans (absolute offsets into `text`) within [start, end).
    Boundaries: line breaks, and whitespace after . ! ?. Spans are trimmed."""
    end = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    pos = start
    for m in _SENT_BOUNDARY_RE.finditer(text, start, end):
        s, e = trim_span(text, pos, m.start())
        if s < e:
            spans.append((s, e))
        pos = m.end()
    s, e = trim_span(text, pos, end)
    if s < e:
        spans.append((s, e))
    return spans


def split_sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]
