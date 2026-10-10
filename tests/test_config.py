"""Detail profiles: ordering, completeness, env defaults."""
from cleanroom.config import DETAIL_PROFILES, DetailLevel, Settings

ORDER = [DetailLevel.BRIEF, DetailLevel.STANDARD, DetailLevel.DEEP, DetailLevel.FORENSIC]


def test_all_levels_have_profiles():
    assert set(DETAIL_PROFILES) == set(DetailLevel)


def test_richer_levels_sample_more():
    fps = [DETAIL_PROFILES[d].sample_fps for d in ORDER]
    assert fps == sorted(fps), "sample_fps should grow with detail"
    caps = [DETAIL_PROFILES[d].max_frames_per_scene for d in ORDER]
    assert caps == sorted(caps), "frame caps should grow with detail"
    fields = [len(DETAIL_PROFILES[d].required_fields) for d in ORDER]
    assert fields == sorted(fields), "required fields should grow with detail"


def test_env_defaults_are_mock():
    s = Settings.from_env()
    assert s.vision_provider == "mock"
    assert s.video_provider == "mock"
    assert s.text_provider == "mock"


def test_env_override(monkeypatch):
    monkeypatch.setenv("CLEANROOM_VISION_PROVIDER", "openai")
    monkeypatch.setenv("CLEANROOM_VIDEO_PROVIDER", "veo")
    s = Settings.from_env()
    assert s.vision_provider == "openai"
    assert s.video_provider == "veo"
