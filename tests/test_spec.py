"""Spec schema: round-trips, validation, markdown rendering."""
import pytest
from pydantic import ValidationError

from cleanroom.models.spec import CameraSpec, SceneSpec, ShotSpec, VideoSpec


def _shot(i=1):
    return ShotSpec(
        id=f"scene-01-shot-{i}",
        start_s=0.0, end_s=2.5,
        camera=CameraSpec(movement="static", angle="eye-level", framing="wide shot"),
        subjects=["a red bicycle"],
        action="The bicycle leans against a wall.",
    )


def _spec():
    return VideoSpec(
        title="test",
        detail_level="standard",
        total_duration_s=5.0,
        scenes=[
            SceneSpec(id="scene-01", start_s=0.0, end_s=5.0,
                      setting="alley", shots=[_shot(1), _shot(2)])
        ],
    )


def test_round_trip_json():
    spec = _spec()
    restored = VideoSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    assert restored.shot_count == 2


def test_missing_required_field_fails():
    with pytest.raises(ValidationError):
        ShotSpec(start_s=0.0, end_s=1.0)  # id is required


def test_duration_properties():
    shot = _shot()
    assert shot.duration_s == pytest.approx(2.5)
    assert _spec().scenes[0].duration_s == pytest.approx(5.0)


def test_markdown_mentions_shots_and_subjects():
    md = _spec().to_markdown()
    assert "scene-01-shot-1" in md
    assert "red bicycle" in md
    assert "static" in md
