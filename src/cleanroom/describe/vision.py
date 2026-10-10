"""Vision LLM clients for stage 1 (the describer).

A VisionClient takes sampled frames + prompts and returns the raw spec JSON
string for one scene. Two implementations ship:

- MockVisionClient: deterministic, offline. Powers `cleanroom demo` and tests.
- OpenAIVisionClient: any OpenAI-compatible chat-completions endpoint with a
  vision-capable model (OpenAI, or a local server via OPENAI_BASE_URL).
- LocalVisionClient: same wire format, but skips `response_format=json_object`
  (many local servers -- Ollama, llama-server -- don't implement it) and
  tolerates a missing API key. Pair with OPENAI_BASE_URL=http://localhost:11434/v1
  and an Ollama vision model like qwen2.5vl:7b for fully offline description.
- OllamaVisionClient: Ollama's *native* `/api/chat` API (stdlib-only HTTP, no
  new dependencies). Model routing is by name and defaults to qwen2.5vl:7b;
  point OLLAMA_BASE_URL at any Ollama server (default http://localhost:11434).

To add a provider (Anthropic, Gemini, ...): implement `describe_scene`,
register it in `make_vision_client`, document it in README.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Protocol

from ..config import Settings
from ..utils import CleanroomError
from .extractor import FrameRef


class VisionClient(Protocol):
    def describe_scene(
        self,
        *,
        scene_id: str,
        start_s: float,
        end_s: float,
        frames: list[FrameRef],
        system: str,
        user: str,
    ) -> str:
        """Return the raw JSON spec string for one scene."""
        ...


class MockVisionClient:
    """Deterministic stand-in: builds a plausible spec from scene bounds.

    Never looks at the frames -- fine for pipeline testing, useless for real
    description (that's the point of a mock).
    """

    name = "mock"

    def describe_scene(self, *, scene_id, start_s, end_s, frames, system, user) -> str:
        mid = (start_s + end_s) / 2
        spec = {
            "id": scene_id,
            "start_s": start_s,
            "end_s": end_s,
            "setting": f"mock location for {scene_id}",
            "time_of_day": "day",
            "lighting": "even studio-like lighting",
            "color_palette": ["gray", "blue"],
            "mood": "neutral",
            "shots": [
                {
                    "id": f"{scene_id}-shot-1",
                    "start_s": start_s,
                    "end_s": round(mid, 2),
                    "camera": {"movement": "static", "angle": "eye-level", "framing": "wide shot"},
                    "subjects": ["mock subject"],
                    "action": "Placeholder action described by the mock vision client.",
                    "lighting": "even",
                    "mood": "neutral",
                    "transition_in": "hard cut" if start_s > 0 else "fade in",
                },
                {
                    "id": f"{scene_id}-shot-2",
                    "start_s": round(mid, 2),
                    "end_s": end_s,
                    "camera": {"movement": "slow pan right", "angle": "eye-level", "framing": "medium shot"},
                    "subjects": ["mock subject"],
                    "action": "Placeholder second shot described by the mock vision client.",
                    "lighting": "even",
                    "mood": "neutral",
                    "transition_in": "hard cut",
                },
            ],
            "audio": {"music": "", "sfx": [], "ambience": "silence"},
        }
        return json.dumps(spec)


class OpenAIVisionClient:
    """Vision via OpenAI chat completions (or any compatible base URL)."""

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        json_mode: bool = True,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise CleanroomError(
                "The 'openai' package is not installed. Run: pip install 'cleanroom-video[openai]'"
            ) from e
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )
        self._model = model
        self._json_mode = json_mode

    def describe_scene(self, *, scene_id, start_s, end_s, frames, system, user) -> str:
        content: list[dict] = [{"type": "text", "text": user}]
        for fr in frames:
            b64 = base64.b64encode(Path(fr.path).read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                }
            )
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": 8192,
        }
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise CleanroomError(f"Vision API call failed for scene {scene_id}: {e}") from e
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise CleanroomError(f"Vision API returned empty response for scene {scene_id}")
        return text


class LocalVisionClient(OpenAIVisionClient):
    """Vision via a local OpenAI-compatible server (Ollama, llama-server).

    Identical wire format, but no `response_format` (often unimplemented
    locally) and no API key required -- local servers ignore it.
    """

    name = "local"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        super().__init__(
            model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "sk-local",
            base_url=base_url,
            json_mode=False,
        )


class OllamaVisionClient:
    """Vision via Ollama's native `/api/chat` (stdlib-only, no new deps).

    Images are sent as raw base64 in the `images` array; the model is routed
    by name and defaults to qwen2.5vl:7b. Set OLLAMA_BASE_URL to reach a
    remote Ollama server; defaults to http://localhost:11434.
    """

    name = "ollama"
    default_model = "qwen2.5vl:7b"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        from ..ollama import resolve_base_url

        self._model = model or self.default_model
        self._base_url = resolve_base_url(base_url)

    def describe_scene(self, *, scene_id, start_s, end_s, frames, system, user) -> str:
        from ..ollama import chat as ollama_chat

        images = [
            base64.b64encode(Path(fr.path).read_bytes()).decode("ascii") for fr in frames
        ]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user, "images": images},
        ]
        try:
            return ollama_chat(self._base_url, self._model, messages)
        except CleanroomError as e:
            raise CleanroomError(f"Ollama vision failed for scene {scene_id}: {e}") from e


def make_vision_client(settings: Settings) -> VisionClient:
    provider = settings.vision_provider
    if provider == "mock":
        return MockVisionClient()
    if provider == "openai":
        return OpenAIVisionClient(
            model=settings.vision_model,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
    if provider == "local":
        return LocalVisionClient(
            model=settings.vision_model,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
    if provider == "ollama":
        # 'gpt-4o' is the generic Settings default and meaningless for Ollama:
        # route to Qwen2.5-VL unless the user explicitly set a model.
        model = None if settings.vision_model == "gpt-4o" else settings.vision_model
        return OllamaVisionClient(model=model)
    raise CleanroomError(
        f"Unknown vision provider '{provider}'. Choices: mock, openai, local, ollama. "
        "See describe/vision.py to add one."
    )
