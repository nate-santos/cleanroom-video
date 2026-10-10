"""Audio transcription (optional input to the describer).

The transcript is handed to LLM #1 alongside the frames so dialogue
attribution is grounded in audio, not lip-reading guesses. Transcription is
best-effort: if it fails, description proceeds without it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import Settings
from ..utils import CleanroomError


@dataclass
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str


class Transcriber(Protocol):
    def transcribe(self, audio_wav: Path) -> list[TranscriptSegment]: ...


class MockTranscriber:
    """No-op transcriber for offline runs."""

    name = "mock"

    def transcribe(self, audio_wav: Path) -> list[TranscriptSegment]:
        return []


class FasterWhisperTranscriber:
    """Local transcription via faster-whisper (lazy import, optional dep)."""

    name = "faster-whisper"

    def __init__(self, model_size: str = "base"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise CleanroomError(
                "faster-whisper is not installed. Run: pip install 'cleanroom-video[whisper]'"
            ) from e
        self._model = WhisperModel(model_size, compute_type="int8")

    def transcribe(self, audio_wav: Path) -> list[TranscriptSegment]:
        segments, _ = self._model.transcribe(str(audio_wav), beam_size=5)
        return [
            TranscriptSegment(start_s=s.start, end_s=s.end, text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]


def make_transcriber(settings: Settings) -> Transcriber:
    if settings.transcriber == "mock":
        return MockTranscriber()
    if settings.transcriber == "faster-whisper":
        return FasterWhisperTranscriber()
    raise CleanroomError(f"Unknown transcriber '{settings.transcriber}'. Choices: mock, faster-whisper.")


def transcript_for_range(
    segments: list[TranscriptSegment], start_s: float, end_s: float
) -> str:
    """Render transcript lines overlapping [start_s, end_s] as plain text."""
    lines = [
        f"[{s.start_s:.1f}s] {s.text}"
        for s in segments
        if s.end_s > start_s and s.start_s < end_s
    ]
    return "\n".join(lines)
