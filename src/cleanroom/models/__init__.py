"""Spec schema: the contract between LLM #1 (describer) and LLM #2 (re-creator)."""
from .spec import (
    AudioSpec,
    CameraSpec,
    DialogueLine,
    SceneSpec,
    ShotSpec,
    VideoSpec,
)

__all__ = [
    "AudioSpec",
    "CameraSpec",
    "DialogueLine",
    "SceneSpec",
    "ShotSpec",
    "VideoSpec",
]
