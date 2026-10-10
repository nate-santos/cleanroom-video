"""Minimal Ollama native API client (stdlib only -- no new dependencies).

Ollama's `/api/chat` differs from the OpenAI shape in two ways that matter
here:
- images go in a plain `images` array as raw base64 (no `data:` URL prefix),
- the model is routed by name, e.g. `"model": "qwen2.5vl:7b"`.

For Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`) use the
`local` provider instead; this module is for the native API.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .utils import CleanroomError

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 600


def resolve_base_url(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> str:
    """POST /api/chat non-streaming with JSON mode; return assistant content."""
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": messages,
    }
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        raise CleanroomError(
            f"Ollama request failed (HTTP {e.code}): {detail}. "
            f"Is Ollama running at {base_url} with '{model}' pulled? "
            f"Try: ollama pull {model}"
        ) from e
    except (urllib.error.URLError, TimeoutError) as e:
        reason = getattr(e, "reason", e)
        raise CleanroomError(
            f"Could not reach Ollama at {base_url} ({reason}). "
            f"Is Ollama running? If it is, pull the model first: ollama pull {model}"
        ) from e
    try:
        content = body["message"]["content"]
    except (KeyError, TypeError) as e:
        raise CleanroomError(f"Unexpected Ollama response shape: {str(body)[:300]}") from e
    if not content or not content.strip():
        raise CleanroomError(f"Ollama returned an empty response for model '{model}'.")
    return content.strip()
