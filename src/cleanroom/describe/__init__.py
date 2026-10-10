"""Stage 1: source video -> detailed spec (LLM #1, the describer)."""
from .describer import describe_video
from .extractor import FrameRef, detect_scenes, extract_audio, extract_frames, probe_duration

__all__ = [
    "describe_video",
    "FrameRef",
    "detect_scenes",
    "extract_audio",
    "extract_frames",
    "probe_duration",
]
