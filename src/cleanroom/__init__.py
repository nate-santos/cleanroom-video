"""cleanroom-video: two-LLM clean-room recreation of video.

Pipeline:
    source video --[LLM #1: describer]--> spec.json --[LLM #2: re-creator]--> recreated video

The spec file is the *only* thing that crosses from stage 1 to stage 2.
LLM #2 never sees the source video -- that is the clean-room property.
"""

__version__ = "0.1.0"
