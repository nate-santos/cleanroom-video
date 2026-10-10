"""Video generators: render one clip per ShotPrompt.

A VideoGenerator takes a ShotPrompt and produces a video file. The mock
implementation renders a labeled placeholder clip with ffmpeg so the whole
pipeline runs offline. Real providers (Veo, Runway, Sora, ...) plug in here:

    class VeoGenerator:
        name = "veo"
        def generate(self, request, out_path): ...

then register in `make_video_generator` and document in README/ROADMAP.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import Settings
from ..utils import CleanroomError, ensure_dir, run, sanitize_drawtext, short_hash
from .promptgen import ShotPrompt

_ASPECT_SIZES = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (960, 960),
}


@dataclass
class GenerateRequest:
    prompt: ShotPrompt
    width: int = 1280
    height: int = 720
    fps: int = 24

    @classmethod
    def from_shot_prompt(cls, prompt: ShotPrompt, fps: int = 24) -> "GenerateRequest":
        w, h = _ASPECT_SIZES.get(prompt.aspect_ratio, (1280, 720))
        return cls(prompt=prompt, width=w, height=h, fps=fps)


class VideoGenerator(Protocol):
    name: str

    def generate(self, request: GenerateRequest, out_path: Path) -> Path:
        """Render the clip; return the path to the finished file."""
        ...


class MockVideoGenerator:
    """Renders a labeled color clip with ffmpeg -- proves the pipeline works
    end-to-end with no API keys. Obviously not a real recreation."""

    name = "mock"

    _COLORS = ["0x1a1a2e", "0x16213e", "0x0f3460", "0x533483", "0x1b4d3e", "0x5c3d2e"]

    def generate(self, request: GenerateRequest, out_path: Path) -> Path:
        ensure_dir(out_path.parent)
        color = self._COLORS[int(short_hash(request.prompt.shot_id), 16) % len(self._COLORS)]
        duration = max(0.5, request.prompt.duration_s)
        label = sanitize_drawtext(f"{request.prompt.shot_id} | {request.prompt.prompt}", 100)
        run(
            [
                "ffmpeg", "-hide_banner", "-y",
                "-f", "lavfi",
                "-i", f"color=c={color}:s={request.width}x{request.height}:d={duration}:r={request.fps}",
                "-vf",
                f"drawtext=text='{label}':fontsize=22:fontcolor=white:"
                f"x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.5",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out_path),
            ],
            capture=True,
        )
        return out_path


# --- Real-provider stubs. Each raises until implemented (see ROADMAP phase 1). ---

class _UnimplementedGenerator:
    name = "unimplemented"

    def generate(self, request: GenerateRequest, out_path: Path) -> Path:
        raise CleanroomError(
            f"The '{self.name}' video provider is not implemented yet. "
            "See ROADMAP.md phase 1 for the integration plan; "
            "use CLEANROOM_VIDEO_PROVIDER=mock until then."
        )


class VeoGenerator(_UnimplementedGenerator):
    """Google Veo (via Vertex AI or Gemini API)."""

    name = "veo"


class RunwayGenerator(_UnimplementedGenerator):
    """Runway Gen-3/4 API."""

    name = "runway"


class SoraGenerator(_UnimplementedGenerator):
    """OpenAI Sora API."""

    name = "sora"


def make_video_generator(settings: Settings) -> VideoGenerator:
    provider = settings.video_provider
    if provider == "mock":
        return MockVideoGenerator()
    if provider == "veo":
        return VeoGenerator()
    if provider == "runway":
        return RunwayGenerator()
    if provider == "sora":
        return SoraGenerator()
    raise CleanroomError(
        f"Unknown video provider '{provider}'. "
        "Choices: mock, veo, runway, sora. See recreate/generators.py to add one."
    )
