"""Stage 2: spec -> recreated video (LLM #2, the re-creator).

LLM #2 never sees the source video -- only the spec file. That boundary is
the entire clean-room premise, and it is enforced structurally: `recreate`
accepts a spec path, never a video path.
"""
from .recreator import recreate_from_spec

__all__ = ["recreate_from_spec"]
