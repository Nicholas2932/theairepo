"""Canonical JSON output and atomic writes (SPEC §8)."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from . import config


def to_jsonable(obj: Any) -> Any:
    """Convert to plain JSON types; round floats to FLOAT_DP; reject NaN/inf."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float in output: {obj!r}")
        value = round(obj, config.FLOAT_DP)
        return 0.0 if value == 0 else value  # normalise -0.0
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name))
                for f in dataclasses.fields(obj) if not f.name.startswith("_")}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(obj)]
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, dumps(obj))


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    lines = [json.dumps(to_jsonable(r), ensure_ascii=False, allow_nan=False) for r in records]
    _atomic_write_text(path, "".join(line + "\n" for line in lines))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list:
    records = []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {n}: {exc}") from exc
    return records


class ArtifactStore:
    """Owns the artifacts directory: cleans stale files, writes atomically,
    remembers what this run wrote (recorded in the manifest)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.written: list[str] = []

    def path(self, name: str) -> Path:
        return self.dir / name

    def clean(self) -> list[str]:
        """Delete every known artifact (and leftover temp files) so a stale
        file from an earlier run can never satisfy validation."""
        removed = []
        if self.dir.exists():
            for name in config.ALL_ARTIFACTS:
                p = self.dir / name
                if p.exists():
                    p.unlink()
                    removed.append(name)
            for p in sorted(self.dir.glob("*.tmp")):
                p.unlink()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.written = []
        return removed

    def write_json(self, name: str, obj: Any) -> None:
        write_json(self.dir / name, obj)
        self._mark(name)

    def write_jsonl(self, name: str, records: Iterable[Any]) -> None:
        write_jsonl(self.dir / name, records)
        self._mark(name)

    def _mark(self, name: str) -> None:
        if name not in self.written:
            self.written.append(name)
