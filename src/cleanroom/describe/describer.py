"""Stage 1 orchestration: video -> VideoSpec.

Pipeline per run:
  1. probe duration, detect scene cuts, sample frames, extract + transcribe audio
  2. assign frames to scenes, subsample to the detail profile's frame budget
  3. call the vision LLM once per scene -> parse JSON -> SceneSpec
  4. assemble VideoSpec, write spec.json + spec.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ..config import DETAIL_PROFILES, DetailLevel, Settings
from ..models.spec import SceneSpec, VideoSpec
from ..utils import CleanroomError, ensure_dir, extract_json
from . import prompts
from .extractor import FrameRef, detect_scenes, extract_audio, extract_frames, probe_duration
from .transcribe import make_transcriber, transcript_for_range
from .vision import VisionClient, make_vision_client


@dataclass
class DescribeResult:
    spec: VideoSpec
    spec_path: Path
    markdown_path: Path
    workdir: Path
    frame_count: int


def _subsample(frames: list[FrameRef], cap: int) -> list[FrameRef]:
    """Uniformly subsample to at most `cap` frames, keeping first/last."""
    if len(frames) <= cap:
        return frames
    if cap <= 2:
        return [frames[0], frames[-1]][:cap]
    step = (len(frames) - 1) / (cap - 1)
    idx = {round(i * step) for i in range(cap)}
    return [f for i, f in enumerate(frames) if i in idx]


def _parse_scene(raw: str, scene_id: str, workdir: Path) -> SceneSpec:
    try:
        data = json.loads(extract_json(raw))
    except (json.JSONDecodeError, CleanroomError) as e:
        debug = workdir / f"{scene_id}.raw.txt"
        debug.write_text(raw)
        raise CleanroomError(
            f"LLM returned invalid JSON for scene {scene_id}: {e}. "
            f"Raw output saved to {debug}"
        ) from e
    try:
        return SceneSpec.model_validate(data)
    except ValidationError as e:
        debug = workdir / f"{scene_id}.raw.txt"
        debug.write_text(raw)
        raise CleanroomError(
            f"LLM output for scene {scene_id} failed spec validation: {e}. "
            f"Raw output saved to {debug}"
        ) from e


def describe_video(
    input_path: Path,
    *,
    detail: DetailLevel = DetailLevel.STANDARD,
    settings: Settings | None = None,
    out_path: Path | None = None,
    vision: VisionClient | None = None,
    with_audio: bool = True,
) -> DescribeResult:
    settings = settings or Settings.from_env()
    profile = DETAIL_PROFILES[detail]
    vision = vision or make_vision_client(settings)

    if not input_path.exists():
        raise CleanroomError(f"Input not found: {input_path}")

    workdir = ensure_dir(settings.workdir / f"describe_{input_path.stem}_{detail.value}")
    duration = probe_duration(input_path)
    scenes = detect_scenes(input_path, profile.scene_threshold, duration)
    frames = extract_frames(input_path, profile.sample_fps, workdir / "frames")

    transcript_segments: list = []
    if with_audio:
        try:
            wav = extract_audio(input_path, workdir / "audio.wav")
            transcript_segments = make_transcriber(settings).transcribe(wav)
        except CleanroomError as e:
            # Transcription is best-effort; description proceeds without it.
            # Keep the warning to one line -- the full ffmpeg stderr is noise here.
            first_line = str(e).strip().splitlines()[0][:160]
            print(f"Warning: audio/transcription skipped ({first_line})")

    total = 0
    scene_specs: list[SceneSpec] = []
    for i, (start_s, end_s) in enumerate(scenes):
        scene_id = f"scene-{i + 1:02d}"
        scene_frames = [f for f in frames if start_s <= f.timestamp < end_s or f.timestamp == start_s]
        scene_frames = _subsample(scene_frames, profile.max_frames_per_scene)
        total += len(scene_frames)
        if total > profile.max_frames_total:
            raise CleanroomError(
                f"Frame budget exceeded for detail={detail.value} "
                f"({profile.max_frames_total} frames). Lower --detail or raise the cap in config.py."
            )
        system = prompts.describer_system_prompt(profile, detail)
        user = prompts.scene_user_prompt(
            scene_id, start_s, end_s, len(scene_frames),
            transcript=transcript_for_range(transcript_segments, start_s, end_s),
        )
        raw = vision.describe_scene(
            scene_id=scene_id, start_s=start_s, end_s=end_s,
            frames=scene_frames, system=system, user=user,
        )
        scene = _parse_scene(raw, scene_id, workdir)
        # Trust the pipeline's timestamps over the LLM's.
        scene.id, scene.start_s, scene.end_s = scene_id, start_s, end_s
        scene_specs.append(scene)

    spec = VideoSpec(
        title=input_path.stem,
        detail_level=detail.value,
        total_duration_s=round(duration, 2),
        scenes=scene_specs,
    )
    out_path = out_path or Path(f"{input_path.stem}.spec.json")
    out_path.write_text(spec.model_dump_json(indent=2))
    md_path = out_path.with_suffix(".md")
    md_path.write_text(spec.to_markdown())
    return DescribeResult(
        spec=spec, spec_path=out_path, markdown_path=md_path,
        workdir=workdir, frame_count=total,
    )
