"""Pydantic schema for the video spec.

The spec is the ONLY artifact that crosses the clean-room boundary:
LLM #1 writes it from the source video, LLM #2 reads it to recreate the
video. It must be fully self-contained -- no references to the source file,
no frame paths, nothing the re-creator could use to "peek".
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DialogueLine(BaseModel):
    speaker: str = ""  # e.g. "woman in red jacket", "narrator (v.o.)"
    line: str = ""


class CameraSpec(BaseModel):
    movement: str = ""   # e.g. "slow dolly-in", "static tripod", "handheld pan left"
    angle: str = ""      # e.g. "eye-level", "low angle", "overhead"
    framing: str = ""    # e.g. "medium close-up", "wide establishing"
    lens_look: str = ""  # e.g. "shallow depth of field, 35mm look"


class AudioSpec(BaseModel):
    dialogue: list[DialogueLine] = Field(default_factory=list)
    music: str = ""      # e.g. "tense synth pulse, ~120bpm, minor key"
    sfx: list[str] = Field(default_factory=list)
    ambience: str = ""   # e.g. "busy cafe murmur, clinking cups"


class ShotSpec(BaseModel):
    """One continuous camera take."""

    id: str
    start_s: float
    end_s: float
    camera: CameraSpec = Field(default_factory=CameraSpec)
    subjects: list[str] = Field(default_factory=list)
    subject_appearance: str = ""
    action: str = ""
    action_beats: list[str] = Field(default_factory=list)  # beat-by-beat at forensic detail
    dialogue: list[DialogueLine] = Field(default_factory=list)
    onscreen_text: list[str] = Field(default_factory=list)
    lighting: str = ""
    light_direction: str = ""
    color_palette: list[str] = Field(default_factory=list)
    wardrobe: str = ""
    props: list[str] = Field(default_factory=list)
    mood: str = ""
    transition_in: str = ""   # e.g. "hard cut", "dissolve", "whip pan"
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class SceneSpec(BaseModel):
    """A group of shots in one continuous time/place."""

    id: str
    start_s: float
    end_s: float
    setting: str = ""
    time_of_day: str = ""
    lighting: str = ""
    color_palette: list[str] = Field(default_factory=list)
    mood: str = ""
    shots: list[ShotSpec] = Field(default_factory=list)
    audio: AudioSpec = Field(default_factory=AudioSpec)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class VideoSpec(BaseModel):
    title: str = "untitled"
    detail_level: str = "standard"
    total_duration_s: float = 0.0
    aspect_ratio: str = "16:9"  # 16:9 | 9:16 | 1:1
    fps_hint: float | None = None
    global_style_notes: str = ""  # overall look: filmic, grade, era, texture
    scenes: list[SceneSpec] = Field(default_factory=list)

    @property
    def shot_count(self) -> int:
        return sum(len(s.shots) for s in self.scenes)

    def to_markdown(self) -> str:
        """Human-readable rendering of the spec."""
        lines = [
            f"# {self.title}",
            "",
            f"_Detail level: {self.detail_level} | Duration: {self.total_duration_s:.1f}s | "
            f"{self.aspect_ratio} | {len(self.scenes)} scenes, {self.shot_count} shots_",
            "",
        ]
        if self.global_style_notes:
            lines += ["## Global style", "", self.global_style_notes, ""]
        for scene in self.scenes:
            lines += [
                f"## Scene {scene.id} — {scene.setting} "
                f"({scene.start_s:.1f}s–{scene.end_s:.1f}s)",
                "",
            ]
            if scene.mood:
                lines.append(f"_Mood: {scene.mood}_")
                lines.append("")
            if scene.color_palette:
                lines.append(f"_Palette: {', '.join(scene.color_palette)}_")
                lines.append("")
            for shot in scene.shots:
                cam = shot.camera
                lines += [
                    f"### Shot {shot.id} ({shot.start_s:.1f}s–{shot.end_s:.1f}s)",
                    "",
                    f"- **Camera:** {cam.movement}; {cam.angle}; {cam.framing}"
                    + (f"; {cam.lens_look}" if cam.lens_look else ""),
                    f"- **Subjects:** {', '.join(shot.subjects) or '—'}",
                    f"- **Action:** {shot.action or '—'}",
                ]
                if shot.subject_appearance:
                    lines.append(f"- **Appearance:** {shot.subject_appearance}")
                if shot.lighting:
                    lines.append(f"- **Lighting:** {shot.lighting}")
                if shot.color_palette:
                    lines.append(f"- **Palette:** {', '.join(shot.color_palette)}")
                for dl in shot.dialogue:
                    who = f"**{dl.speaker}**" if dl.speaker else "Dialogue"
                    lines.append(f"- {who}: “{dl.line}”")
                if shot.onscreen_text:
                    lines.append(f"- **On-screen text:** {'; '.join(shot.onscreen_text)}")
                if shot.transition_in:
                    lines.append(f"- **Transition in:** {shot.transition_in}")
                if shot.notes:
                    lines.append(f"- **Notes:** {shot.notes}")
                lines.append("")
            if scene.audio.music or scene.audio.ambience or scene.audio.sfx:
                lines.append("**Audio:**")
                if scene.audio.music:
                    lines.append(f"- Music: {scene.audio.music}")
                if scene.audio.ambience:
                    lines.append(f"- Ambience: {scene.audio.ambience}")
                for sfx in scene.audio.sfx:
                    lines.append(f"- SFX: {sfx}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"
