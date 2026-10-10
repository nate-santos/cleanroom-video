"""LLM #2, part 1: turn each shot spec into a text-to-video prompt.

This is where the second LLM first touches the pipeline. It does NOT see the
source video -- only the spec. Its output (ShotPrompt) is what gets handed to
a video generation model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from ..config import Settings
from ..models.spec import SceneSpec, ShotSpec, VideoSpec
from ..utils import CleanroomError
from . import prompts


@dataclass
class ShotPrompt:
    shot_id: str
    prompt: str
    negative_prompt: str = ""
    duration_s: float = 3.0
    aspect_ratio: str = "16:9"


class TextClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the raw completion string."""
        ...


class MockTextClient:
    """Deterministic template-based planner for offline runs and tests."""

    name = "mock"

    def complete(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "prompt": "MOCK PLANNER: prompt derived from spec template (see manifest).",
                "negative_prompt": "watermark, text artifacts",
            }
        )


class OpenAITextClient:
    """Text completions via OpenAI (or compatible base URL)."""

    name = "openai"

    def __init__(self, model: str, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise CleanroomError(
                "The 'openai' package is not installed. Run: pip install 'cleanroom-video[openai]'"
            ) from e
        self._client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )
        self._model = model

    def complete(self, system: str, user: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
        except Exception as e:
            raise CleanroomError(f"Text API call failed: {e}") from e
        return (resp.choices[0].message.content or "").strip()


def make_text_client(settings: Settings) -> TextClient:
    if settings.text_provider == "mock":
        return MockTextClient()
    if settings.text_provider == "openai":
        return OpenAITextClient(model=settings.text_model)
    raise CleanroomError(
        f"Unknown text provider '{settings.text_provider}'. Choices: mock, openai."
    )


def _mock_shot_prompt(shot: ShotSpec, scene: SceneSpec, spec: VideoSpec) -> ShotPrompt:
    """Deterministic prompt built straight from the spec fields (no LLM)."""
    bits = []
    cam = shot.camera
    if cam.framing or cam.angle or cam.movement:
        bits.append(f"{cam.framing} {cam.angle}, {cam.movement}".strip(" ,"))
    if shot.subjects:
        bits.append("subjects: " + ", ".join(shot.subjects))
    if shot.subject_appearance:
        bits.append(shot.subject_appearance)
    if shot.action:
        bits.append(shot.action)
    if scene.setting:
        bits.append(f"setting: {scene.setting}")
    if shot.lighting:
        bits.append(f"lighting: {shot.lighting}")
    if shot.color_palette:
        bits.append("palette: " + ", ".join(shot.color_palette))
    if shot.mood or scene.mood:
        bits.append(f"mood: {shot.mood or scene.mood}")
    if spec.global_style_notes:
        bits.append(spec.global_style_notes)
    return ShotPrompt(
        shot_id=shot.id,
        prompt=". ".join(b for b in bits if b) + ".",
        negative_prompt="watermark, text artifacts, extra limbs, distorted faces",
        duration_s=round(shot.duration_s or 3.0, 2),
        aspect_ratio=spec.aspect_ratio,
    )


def plan_shot(
    shot: ShotSpec,
    scene: SceneSpec,
    spec: VideoSpec,
    client: TextClient,
) -> ShotPrompt:
    """Use LLM #2 to translate one shot spec into a generation prompt."""
    if isinstance(client, MockTextClient):
        return _mock_shot_prompt(shot, scene, spec)
    system = prompts.planner_system_prompt()
    user = prompts.shot_user_prompt(
        scene_setting=scene.setting,
        scene_mood=scene.mood,
        global_style=spec.global_style_notes,
        shot_json=shot.model_dump_json(indent=2),
    )
    raw = client.complete(system, user)
    try:
        data = json.loads(raw)
        prompt, negative = data["prompt"], data.get("negative_prompt", "")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise CleanroomError(
            f"Prompt planner returned unusable output for shot {shot.id}: {e}\nRaw: {raw[:500]}"
        ) from e
    return ShotPrompt(
        shot_id=shot.id,
        prompt=prompt,
        negative_prompt=negative,
        duration_s=round(shot.duration_s or 3.0, 2),
        aspect_ratio=spec.aspect_ratio,
    )
