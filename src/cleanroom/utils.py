"""Small shared helpers: subprocess runs, filesystem, text."""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


class CleanroomError(RuntimeError):
    """Raised for any expected pipeline failure (bad input, tool error, bad LLM output)."""


def run(cmd: list[str], *, capture: bool = False, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command; raise CleanroomError with stderr on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
    except FileNotFoundError as e:
        raise CleanroomError(f"Required tool not found: {cmd[0]}. Is it installed?") from e
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[-2000:]
        raise CleanroomError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{err}")
    return result


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:max_len] or "item"


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def sanitize_drawtext(text: str, max_len: int = 120) -> str:
    """Make text safe for ffmpeg drawtext (escape special chars)."""
    text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]
