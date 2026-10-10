"""Pipeline pieces with mocks. ffmpeg-dependent tests skip if ffmpeg is missing."""
import shutil

import pytest

from cleanroom.config import Settings
from cleanroom.describe.vision import MockVisionClient
from cleanroom.models.spec import CameraSpec, SceneSpec, ShotSpec, VideoSpec
from cleanroom.recreate.generators import (
    GenerateRequest,
    MockVideoGenerator,
    make_video_generator,
)
from cleanroom.recreate.promptgen import MockTextClient, plan_shot

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _spec():
    shot = ShotSpec(
        id="scene-01-shot-1", start_s=0.0, end_s=3.0,
        camera=CameraSpec(movement="slow dolly in", angle="eye-level", framing="medium shot"),
        subjects=["a cat"], action="The cat stretches.",
        lighting="warm window light", color_palette=["amber", "brown"],
    )
    scene = SceneSpec(id="scene-01", start_s=0.0, end_s=3.0,
                      setting="living room", mood="cozy", shots=[shot])
    return VideoSpec(title="t", detail_level="standard", total_duration_s=3.0, scenes=[scene])


def test_mock_vision_returns_valid_scene():
    client = MockVisionClient()
    raw = client.describe_scene(
        scene_id="scene-01", start_s=0.0, end_s=5.0, frames=[],
        system="x", user="y",
    )
    scene = SceneSpec.model_validate_json(raw)
    assert scene.id == "scene-01"
    assert len(scene.shots) == 2


def test_mock_prompt_planner_uses_spec_fields():
    spec = _spec()
    shot = spec.scenes[0].shots[0]
    sp = plan_shot(shot, spec.scenes[0], spec, MockTextClient())
    assert sp.shot_id == shot.id
    assert "cat" in sp.prompt
    assert "dolly" in sp.prompt
    assert sp.duration_s == pytest.approx(3.0)


@needs_ffmpeg
def test_mock_generator_produces_playable_clip(tmp_path):
    spec = _spec()
    shot = spec.scenes[0].shots[0]
    sp = plan_shot(shot, spec.scenes[0], spec, MockTextClient())
    out = tmp_path / "clip.mp4"
    MockVideoGenerator().generate(GenerateRequest.from_shot_prompt(sp), out)
    assert out.exists() and out.stat().st_size > 0


@needs_ffmpeg
def test_assembler_concatenates(tmp_path):
    from cleanroom.recreate.assembler import assemble_clips
    spec = _spec()
    gen = MockVideoGenerator()
    clips = []
    for i in range(2):
        shot = spec.scenes[0].shots[0]
        sp = plan_shot(shot, spec.scenes[0], spec, MockTextClient())
        p = tmp_path / f"c{i}.mp4"
        gen.generate(GenerateRequest.from_shot_prompt(sp), p)
        clips.append(p)
    out = tmp_path / "final.mp4"
    assemble_clips(clips, out)
    assert out.exists() and out.stat().st_size > 0


def test_unknown_providers_raise():
    from cleanroom.utils import CleanroomError
    s = Settings(video_provider="nope")
    with pytest.raises(CleanroomError):
        make_video_generator(s)
