"""Prompt templates for LLM #1 (the describer).

These are the highest-leverage text in the project: the quality of every
spec -- and therefore every recreation -- starts here. Tune per detail level
in config.DetailProfile; tune wording here.
"""
from __future__ import annotations

from ..config import DetailLevel, DetailProfile

# Maps detail-level field names onto the spec schema so the prompt can demand
# exactly the fields that level promises.
_FIELD_DESCRIPTIONS = {
    "setting": '"setting": where the shot takes place (interior/exterior, location type)',
    "time_of_day": '"time_of_day": apparent time of day and weather if visible',
    "subjects": '"subjects": list of people/objects/animals in frame',
    "subject_appearance": '"subject_appearance": clothing, age, distinguishing features',
    "action": '"action": what happens in the shot, in order',
    "action_beats": '"action_beats": list of beat-by-beat micro-actions with rough timing',
    "dialogue": '"dialogue": list of {"speaker": "...", "line": "..."}; quote short lines verbatim',
    "dialogue_summary": '"dialogue_summary": one-line summary of what is said (no verbatim quotes)',
    "onscreen_text": '"onscreen_text": every readable word visible on screen (titles, signs, captions)',
    "camera_movement": '"camera": {"movement": "..."} -- e.g. static, pan left, dolly-in, handheld, crane up',
    "camera_angle": '"camera": {"angle": "..."} -- e.g. eye-level, low angle, overhead, dutch',
    "framing": '"camera": {"framing": "..."} -- e.g. extreme close-up, medium shot, wide establishing',
    "lens_look": '"camera": {"lens_look": "..."} -- depth of field, focal-length feel, anamorphic artifacts',
    "depth_of_field": '"depth_of_field": what is in focus vs blurred',
    "lighting": '"lighting": quality of light (soft/harsh, warm/cool, practical vs motivated)',
    "light_direction": '"light_direction": where key light comes from',
    "color_palette": '"color_palette": dominant colors as named colors',
    "wardrobe": '"wardrobe": notable clothing details',
    "props": '"props": notable objects and set dressing',
    "mood": '"mood": emotional tone of the shot',
    "audio": '"audio_notes": music, sfx, ambience audible during the shot',
    "transitions": '"transition_in": how this shot begins (hard cut, dissolve, fade, whip pan...)',
}


def describer_system_prompt(profile: DetailProfile, detail: DetailLevel) -> str:
    fields = "\n".join(f"  - {_FIELD_DESCRIPTIONS[f]}" for f in profile.required_fields)
    return f"""You are a forensic video analyst writing a shot-by-shot recreation spec.
You will receive still frames sampled from ONE scene of a video, in chronological
order, each labeled with its timestamp.

Your job is to describe the scene so completely that a skilled filmmaker who
never saw the original could recreate it as closely as possible from your
description alone. Describe only what is visible/audible. Do not guess at plot,
do not editorialize, do not invent details you cannot see.

Detail level: {detail.value.upper()}
{profile.instruction}

Output ONLY a single JSON object (no markdown fences, no commentary) with this shape:
{{
  "id": "<scene id, copied from the request>",
  "start_s": <scene start in seconds, copied from the request>,
  "end_s": <scene end in seconds, copied from the request>,
  "setting": "...",
  "time_of_day": "...",
  "lighting": "...",
  "color_palette": ["...", "..."],
  "mood": "...",
  "shots": [
    {{
      "id": "<scene-id>-shot-N",
      "start_s": <float>, "end_s": <float>,
      ... shot fields, one JSON object per continuous camera take ...
    }}
  ],
  "audio": {{"music": "...", "sfx": ["..."], "ambience": "..."}}
}}

Shot-level fields you MUST include:
{fields}

Rules:
- One shot object per continuous camera take. A cut = a new shot, even inside the scene.
- Timestamps are approximate; keep shots ordered and non-overlapping within the scene bounds.
- Keep dialogue quotes short (single lines). Never reproduce long passages of text verbatim.
- If a field truly has no content (no dialogue, no on-screen text), use "" or [].
- Valid JSON only. Numbers as numbers, not strings."""


def scene_user_prompt(
    scene_id: str,
    start_s: float,
    end_s: float,
    n_frames: int,
    transcript: str = "",
) -> str:
    prompt = (
        f"Scene {scene_id}: {start_s:.1f}s to {end_s:.1f}s. "
        f"{n_frames} frames follow in chronological order. "
        "Write the JSON spec for this scene now."
    )
    if transcript:
        prompt += (
            "\n\nAudio transcript for this time range (may help with dialogue attribution):\n"
            f"{transcript}"
        )
    return prompt
