"""Optional LLM answer generator: Google Gemini over its REST API, stdlib only
(SPEC §6.3 "Optional LLM mode"). Off by default.

The model only ever sees the question and that query's retrieved chunks.
Whatever it returns is untrusted: answering.py runs it through the same
self-check and grounding check as the extractive generator and falls back to
the extractive answer if it fails. Retrieval, ranking and evaluation never
touch this module.

Configuration (environment):
  GEMINI_API_KEY (or GOOGLE_API_KEY)  required in LLM mode
  GEMINI_MODEL                        optional, defaults to config.GEMINI_DEFAULT_MODEL
Values may also come from a git-ignored `.env` file at the repo root (see
`.env.example`); real environment variables take precedence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from . import config, vocab
from .stages import QueryRetrieval

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_RETRYABLE = {429, 500, 502, 503, 504}

SYSTEM_PROMPT = f"""You answer questions using ONLY the context chunks supplied in the user message.

Rules:
1. Use only facts stated in the chunks. Never add outside knowledge, numbers, advice or assumptions.
2. End every sentence of your answer with the citation token of the chunk that supports it, copied character-for-character from the context (for example "[Some title §chunk_N]"). Prefer the chunk's own wording.
3. Cite only citation tokens listed in the context.
4. If the chunks do not contain the information needed, set answer_label to "{vocab.INSUFFICIENT_CONTEXT}", answer to "" and citations to [].
5. If chunks contradict each other on the point asked, set answer_label to "{vocab.CONFLICTING_CONTEXT}", state each version with its own citation, and do not choose between them.
6. Otherwise set answer_label to "{vocab.GROUNDED_ANSWER}".
7. "citations" must list every citation token used in the answer, in order of first use, without duplicates.

Respond with a single JSON object only:
{{"answer_label": "...", "answer": "...", "citations": ["..."]}}"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer_label": {"type": "STRING", "enum": list(vocab.ANSWER_LABELS)},
        "answer": {"type": "STRING"},
        "citations": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["answer_label", "answer", "citations"],
}


class LLMError(Exception):
    """LLM unavailable, misconfigured, or returned unusable output."""


@dataclass
class Prompt:
    system: str
    user: str

    @property
    def text(self) -> str:
        return f"{self.system}\n\n---\n\n{self.user}"

    @property
    def sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def build_prompt(qr: QueryRetrieval) -> Prompt:
    if not isinstance(qr, QueryRetrieval):
        raise TypeError("build_prompt needs a QueryRetrieval")
    parts = [f"Question: {qr.question}", "", "Context chunks:"]
    for h in qr.hits:
        parts += ["", f"[{h.rank}] citation token: {vocab.format_citation(h.doc_title, h.chunk_id)}",
                  f"section: {h.section}", f"text: {h.chunk_text}"]
    return Prompt(system=SYSTEM_PROMPT, user="\n".join(parts))


@dataclass
class GeminiClient:
    model: str
    api_key: str = field(repr=False)
    timeout_s: float = config.LLM_TIMEOUT_S
    max_retries: int = config.LLM_MAX_RETRIES
    provider: str = "gemini"
    temperature: float = config.LLM_TEMPERATURE

    def complete(self, prompt: Prompt) -> str:
        url = (f"{config.GEMINI_API_BASE}/models/"
               f"{urllib.parse.quote(self.model, safe='')}:generateContent")
        body = {
            "systemInstruction": {"parts": [{"text": prompt.system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt.user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }
        data = json.dumps(body).encode("utf-8")
        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return _extract_text(payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:300].decode("utf-8", "replace")
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in _RETRYABLE:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(2 ** attempt)
        raise LLMError(f"Gemini request failed: {last_error}")


def _extract_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        reason = (payload.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise LLMError(f"Gemini returned no answer ({reason})")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise LLMError(f"Gemini returned empty content "
                       f"(finishReason={candidates[0].get('finishReason')})")
    return text


def parse_output(text: str) -> dict:
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"model output is not JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMError("model output is not a JSON object")
    return obj


def generate(client: GeminiClient, qr: QueryRetrieval) -> tuple[dict | None, Prompt, str | None]:
    """Adapter interface: (question, chunks) -> {answer_label, answer, citations}.
    Returns (parsed output or None, prompt, error message or None)."""
    prompt = build_prompt(qr)
    try:
        return parse_output(client.complete(prompt)), prompt, None
    except LLMError as exc:
        return None, prompt, str(exc)


def _load_dotenv(path: str | None = None) -> None:
    """Minimal stdlib `.env` reader: KEY=VALUE lines, never overrides the real env."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v:
                os.environ.setdefault(k, v)


def make_client(provider: str, model: str | None = None) -> GeminiClient:
    if provider != "gemini":
        raise LLMError(f"unsupported LLM provider {provider!r}; supported: {config.LLM_PROVIDERS}")
    _load_dotenv()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise LLMError("LLM mode needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment")
    return GeminiClient(model=model or os.environ.get("GEMINI_MODEL") or config.GEMINI_DEFAULT_MODEL,
                        api_key=key)
