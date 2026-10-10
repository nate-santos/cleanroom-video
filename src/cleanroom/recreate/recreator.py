"""Stage 2 orchestration: VideoSpec -> recreated video.

Per shot: spec --[LLM #2: plan_shot]--> ShotPrompt --[generator]--> clip.
Then clips are concatenated and a manifest is written recording every prompt
used, so a recreation is reproducible and auditable from the spec alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..models.spec import VideoSpec
from ..utils import CleanroomError, ensure_dir, slugify
from .assembler import assemble_clips
from .generators import GenerateRequest, VideoGenerator, make_video_generator
from .promptgen import TextClient, make_text_client, plan_shot


@dataclass
class RecreateResult:
    output_path: Path
    manifest_path: Path
    clip_paths: list[Path]
    workdir: Path


def recreate_from_spec(
    spec_path: Path,
    *,
    settings: Settings | None = None,
    out_path: Path | None = None,
    text_client: TextClient | None = None,
    generator: VideoGenerator | None = None,
) -> RecreateResult:
    settings = settings or Settings.from_env()
    text_client = text_client or make_text_client(settings)
    generator = generator or make_video_generator(settings)

    if not spec_path.exists():
        raise CleanroomError(f"Spec not found: {spec_path}")
    spec = VideoSpec.model_validate_json(spec_path.read_text())
    if not spec.scenes:
        raise CleanroomError(f"Spec {spec_path} contains no scenes.")

    workdir = ensure_dir(settings.workdir / f"recreate_{spec_path.stem}")
    clips_dir = ensure_dir(workdir / "clips")

    manifest = {
        "spec": spec_path.name,
        "detail_level": spec.detail_level,
        "text_provider": settings.text_provider,
        "video_provider": getattr(generator, "name", "?"),
        "shots": [],
    }
    clip_paths: list[Path] = []
    for scene in spec.scenes:
        for shot in scene.shots:
            shot_prompt = plan_shot(shot, scene, spec, text_client)
            clip_path = clips_dir / f"{slugify(shot.id)}.mp4"
            generator.generate(GenerateRequest.from_shot_prompt(shot_prompt), clip_path)
            clip_paths.append(clip_path)
            manifest["shots"].append(
                {
                    "shot_id": shot.id,
                    "prompt": shot_prompt.prompt,
                    "negative_prompt": shot_prompt.negative_prompt,
                    "duration_s": shot_prompt.duration_s,
                    "clip": clip_path.name,
                }
            )

    out_path = out_path or Path(f"{spec_path.stem}.recreation.mp4")
    assemble_clips(clip_paths, out_path)
    manifest["output"] = str(out_path)
    manifest_path = workdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return RecreateResult(
        output_path=out_path,
        manifest_path=manifest_path,
        clip_paths=clip_paths,
        workdir=workdir,
    )
