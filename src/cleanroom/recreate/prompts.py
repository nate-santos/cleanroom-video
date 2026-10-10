"""Prompt templates for LLM #2 (the re-creator's prompt planner).

LLM #2's job is translation, not imagination: turn each shot's spec into a
self-contained text-to-video prompt that a generation model can render. The
prompts must carry every visual fact the spec captured, because the video
model sees nothing else.
"""
from __future__ import annotations


def planner_system_prompt() -> str:
    return """You are a prompt engineer for AI video generation models (Veo, Sora, Runway, Pika).
You receive a shot specification from a video recreation spec and must translate
it into a single self-contained text-to-video prompt.

Rules:
- Write ONE vivid paragraph describing exactly what the camera sees: subject(s),
  appearance, action in order, setting, lighting, color palette, mood, camera
  movement/angle/framing. The video model sees nothing but your prompt, so every
  visual fact in the spec must survive the translation.
- Camera language must be literal and model-friendly: "static wide shot",
  "slow dolly in", "handheld pan left", "eye-level medium close-up".
- Include dialogue only if it must be lip-synced or captioned; otherwise fold
  spoken content into the scene description ("a woman says ...") sparingly.
- Never invent subjects, actions, or style not present in the spec.
- Keep it under 150 words. No camera-brand jargon, no aspect-ratio tags.

Output ONLY a JSON object (no markdown fences, no commentary):
{
  "prompt": "<the generation prompt>",
  "negative_prompt": "<things to avoid, e.g. text artifacts, extra limbs, watermark>"
}"""


def shot_user_prompt(
    scene_setting: str,
    scene_mood: str,
    global_style: str,
    shot_json: str,
) -> str:
    context = []
    if global_style:
        context.append(f"Global style for the whole video: {global_style}")
    if scene_setting:
        context.append(f"Scene setting: {scene_setting}")
    if scene_mood:
        context.append(f"Scene mood: {scene_mood}")
    context_block = "\n".join(context)
    return (
        f"{context_block}\n\nShot spec (JSON):\n{shot_json}\n\n"
        "Write the text-to-video prompt JSON for this shot."
        if context_block
        else f"Shot spec (JSON):\n{shot_json}\n\nWrite the text-to-video prompt JSON for this shot."
    )
