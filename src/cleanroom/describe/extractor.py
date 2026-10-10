"""Video intake: duration probing, scene-cut detection, frame sampling, audio extraction.

Everything here is plain ffmpeg/ffprobe -- no ML dependencies. Scene cuts
come from ffmpeg's `select='gt(scene,...)'` filter, which is fast and has no
extra install cost. (A learned detector like PySceneDetect can replace
`detect_scenes` later; see ROADMAP.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..utils import CleanroomError, ensure_dir, run


@dataclass
class FrameRef:
    timestamp: float  # seconds into the source video
    path: Path


def probe_duration(path: Path) -> float:
    """Return the video duration in seconds via ffprobe."""
    result = run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise CleanroomError(f"Could not read duration of {path}") from e


def detect_scenes(path: Path, threshold: float, duration: float) -> list[tuple[float, float]]:
    """Split the video into scenes via ffmpeg's scene-change filter.

    Returns a list of (start_s, end_s) boundaries, always starting at 0.0
    and ending at `duration`.
    """
    result = run(
        [
            "ffmpeg", "-hide_banner", "-i", str(path),
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-",
        ],
        capture=True,
    )
    cuts = sorted(
        {float(m.group(1)) for m in re.finditer(r"pts_time:([\d.]+)", result.stderr or "")}
    )
    # Merge cuts closer than 1s apart and drop near-zero cuts.
    bounds = [0.0]
    for t in cuts:
        if t > 0.5 and t - bounds[-1] > 1.0 and t < duration - 0.5:
            bounds.append(t)
    bounds.append(duration)
    return list(zip(bounds[:-1], bounds[1:]))


def extract_frames(path: Path, fps: float, outdir: Path) -> list[FrameRef]:
    """Sample frames uniformly at `fps`; returns timestamped FrameRefs."""
    ensure_dir(outdir)
    run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(path),
            "-vf", f"fps={fps}",
            "-qscale:v", "3",
            str(outdir / "frame_%06d.jpg"),
        ],
        capture=True,
    )
    files = sorted(outdir.glob("frame_*.jpg"))
    return [FrameRef(timestamp=i / fps, path=f) for i, f in enumerate(files)]


def extract_audio(path: Path, out_wav: Path) -> Path:
    """Extract mono 16kHz WAV for transcription; raises if no audio stream."""
    ensure_dir(out_wav.parent)
    run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(path),
            "-vn", "-ac", "1", "-ar", "16000",
            str(out_wav),
        ],
        capture=True,
    )
    return out_wav
