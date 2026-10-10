"""Assemble per-shot clips into the final video."""
from __future__ import annotations

from pathlib import Path

from ..utils import CleanroomError, ensure_dir, run


def assemble_clips(clips: list[Path], out_path: Path) -> Path:
    """Concatenate clips in order with the ffmpeg concat demuxer (stream copy).

    Clips must share codec/container params -- guaranteed when they come from
    a single generator run. Crossfades/transitions per spec are ROADMAP phase 2.
    """
    if not clips:
        raise CleanroomError("No clips to assemble.")
    missing = [c for c in clips if not c.exists()]
    if missing:
        raise CleanroomError(f"Missing clip files: {missing}")

    ensure_dir(out_path.parent)
    list_file = out_path.parent / "concat_list.txt"
    list_file.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    run(
        [
            "ffmpeg", "-hide_banner", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path),
        ],
        capture=True,
    )
    return out_path
