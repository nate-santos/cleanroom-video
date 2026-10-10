"""Configuration: tunable detail levels, providers, and environment settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DetailLevel(str, Enum):
    """How much detail LLM #1 (the describer) puts into the spec.

    This is the main tunable knob: higher levels sample more frames and
    demand more fields per shot. Cost (frames sent to the vision model)
    grows roughly with the frame budget.
    """

    BRIEF = "brief"
    STANDARD = "standard"
    DEEP = "deep"
    FORENSIC = "forensic"

    def __str__(self) -> str:  # typer displays the value nicely
        return self.value


@dataclass(frozen=True)
class DetailProfile:
    name: str
    sample_fps: float          # frames per second extracted for the vision model
    max_frames_per_scene: int  # per-scene cap (uniformly subsampled if exceeded)
    max_frames_total: int      # whole-video cap, safety bound on API cost
    scene_threshold: float     # ffmpeg scene-cut sensitivity (lower = more cuts)
    required_fields: tuple[str, ...]  # JSON fields LLM #1 must fill per shot
    instruction: str           # verbosity guidance injected into the prompt


DETAIL_PROFILES: dict[DetailLevel, DetailProfile] = {
    DetailLevel.BRIEF: DetailProfile(
        name="brief",
        sample_fps=0.5,
        max_frames_per_scene=8,
        max_frames_total=48,
        scene_threshold=0.4,
        required_fields=("setting", "subjects", "action", "camera_movement", "dialogue_summary"),
        instruction=(
            "Give a compact, high-level description: one or two sentences per shot. "
            "Capture what is happening, who/what is present, and how the camera moves. "
            "Skip fine visual detail, exact colors, and background minutiae."
        ),
    ),
    DetailLevel.STANDARD: DetailProfile(
        name="standard",
        sample_fps=1.0,
        max_frames_per_scene=16,
        max_frames_total=96,
        scene_threshold=0.4,
        required_fields=(
            "setting", "time_of_day", "subjects", "action", "dialogue",
            "camera_movement", "camera_angle", "framing", "lighting", "mood",
        ),
        instruction=(
            "Give a thorough shot-by-shot description: several sentences per shot. "
            "Cover setting, subjects, action, dialogue, camera work, lighting, and mood. "
            "Mention notable colors and composition, but don't catalog every background object."
        ),
    ),
    DetailLevel.DEEP: DetailProfile(
        name="deep",
        sample_fps=2.0,
        max_frames_per_scene=32,
        max_frames_total=192,
        scene_threshold=0.35,
        required_fields=(
            "setting", "time_of_day", "subjects", "subject_appearance", "action",
            "dialogue", "onscreen_text", "camera_movement", "camera_angle", "framing",
            "lens_look", "lighting", "color_palette", "mood", "audio",
        ),
        instruction=(
            "Describe forensically but readably: a full paragraph per shot. "
            "Cover subject appearance (clothing, age, distinguishing features), precise action "
            "choreography, camera movement/angle/framing/lens character, lighting direction and "
            "quality, dominant colors, on-screen text, and the audio bed. Quote short dialogue lines."
        ),
    ),
    DetailLevel.FORENSIC: DetailProfile(
        name="forensic",
        sample_fps=4.0,
        max_frames_per_scene=64,
        max_frames_total=384,
        scene_threshold=0.3,
        required_fields=(
            "setting", "time_of_day", "subjects", "subject_appearance", "action",
            "action_beats", "dialogue", "onscreen_text", "camera_movement", "camera_angle",
            "framing", "lens_look", "depth_of_field", "lighting", "light_direction",
            "color_palette", "wardrobe", "props", "mood", "audio", "transitions",
        ),
        instruction=(
            "Maximum-fidelity description for frame-accurate recreation. Catalog everything: "
            "beat-by-beat action, wardrobe and props, exact camera path, depth of field, light "
            "direction and color temperature, palette as named colors, every line of dialogue and "
            "pixel of on-screen text, shot transitions. Leave nothing visual undescribed."
        ),
    ),
}


@dataclass
class Settings:
    """Runtime settings, resolved from the environment (see .env.example)."""

    vision_provider: str = "mock"   # mock | openai
    vision_model: str = "gpt-4o"
    text_provider: str = "mock"     # mock | openai
    text_model: str = "gpt-4o"
    video_provider: str = "mock"    # mock (placeholder clips via ffmpeg)
    transcriber: str = "mock"       # mock | faster-whisper
    workdir: Path = field(default_factory=lambda: Path("assets/work"))

    @classmethod
    def from_env(cls) -> "Settings":
        def get(key: str, default: str) -> str:
            return os.environ.get(key, default).strip() or default

        return cls(
            vision_provider=get("CLEANROOM_VISION_PROVIDER", "mock").lower(),
            vision_model=get("CLEANROOM_VISION_MODEL", "gpt-4o"),
            text_provider=get("CLEANROOM_TEXT_PROVIDER", "mock").lower(),
            text_model=get("CLEANROOM_TEXT_MODEL", "gpt-4o"),
            video_provider=get("CLEANROOM_VIDEO_PROVIDER", "mock").lower(),
            transcriber=get("CLEANROOM_TRANSCRIBER", "mock").lower(),
            workdir=Path(get("CLEANROOM_WORKDIR", "assets/work")),
        )
