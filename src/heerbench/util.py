from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def write_jsonl(path: Path, records) -> None:
    atomic_text(path, "".join(canonical(x) + "\n" for x in records))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def project_root() -> Path:
    for p in (Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents):
        if (p / "data/heerbench_v05/MANIFEST.sha256").exists():
            return p
    raise FileNotFoundError("Run inside the repository, or set an explicit dataset path in configuration.")


def software_fingerprint() -> str:
    source = Path(__file__).resolve().parent
    return digest({p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(source.rglob("*.py"))})


def estimated_tokens(text: str) -> int:
    """Budget estimate only. This is not a model-specific tokenizer."""
    return max(1, (len(text.encode("utf-8")) + 2) // 3)


def token_upper_bound(text: str) -> int:
    """Conservative UTF-8 byte bound for common byte-fallback tokenizers.

    Includes no claim of exact vendor accounting. Add message framing separately.
    A provider context error remains a visible failure; text is never truncated.
    """
    return len(text.encode("utf-8"))

