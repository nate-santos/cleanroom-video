"""CLI: cleanroom describe | recreate | run | demo"""
from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from .config import DetailLevel, Settings
from .describe import describe_video
from .recreate import recreate_from_spec
from .utils import CleanroomError, run as run_cmd

app = typer.Typer(
    help=(
        "Two-LLM clean-room video recreation.\n\n"
        "Stage 1 (describe): source video -> spec.json via a vision LLM.\n"
        "Stage 2 (recreate): spec.json -> recreated video via a second LLM.\n"
        "The spec is the only thing that crosses between stages."
    )
)


def _settings() -> Settings:
    return Settings.from_env()


@app.command()
def describe(
    input: Path = typer.Argument(..., help="Source video file."),
    detail: DetailLevel = typer.Option(
        DetailLevel.STANDARD, "--detail", "-d",
        help="How much detail LLM #1 puts into the spec.",
    ),
    out: Path = typer.Option(None, "--out", "-o", help="Where to write spec.json."),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip audio extraction/transcription."),
):
    """Stage 1: describe a video into a detailed spec file."""
    try:
        result = describe_video(
            input, detail=detail, settings=_settings(),
            out_path=out, with_audio=not no_audio,
        )
    except CleanroomError as e:
        rprint(f"[red]Describe failed:[/red] {e}")
        raise typer.Exit(1)
    rprint(f"[green]Spec written:[/green] {result.spec_path}")
    rprint(f"[green]Readable version:[/green] {result.markdown_path}")
    rprint(
        f"{len(result.spec.scenes)} scenes, {result.spec.shot_count} shots, "
        f"{result.frame_count} frames analyzed (detail={detail.value})."
    )


@app.command()
def recreate(
    spec: Path = typer.Argument(..., help="Spec JSON from `cleanroom describe`."),
    out: Path = typer.Option(None, "--out", "-o", help="Output video path."),
):
    """Stage 2: recreate a video from a spec file (never sees the source)."""
    try:
        result = recreate_from_spec(spec, settings=_settings(), out_path=out)
    except CleanroomError as e:
        rprint(f"[red]Recreate failed:[/red] {e}")
        raise typer.Exit(1)
    rprint(f"[green]Recreation written:[/green] {result.output_path}")
    rprint(f"[dim]Manifest (prompts used): {result.manifest_path}[/dim]")


@app.command()
def run(
    input: Path = typer.Argument(..., help="Source video file."),
    detail: DetailLevel = typer.Option(DetailLevel.STANDARD, "--detail", "-d"),
    out: Path = typer.Option(None, "--out", "-o", help="Output video path."),
):
    """Full pipeline: describe the video, then recreate it from the spec."""
    settings = _settings()
    try:
        desc = describe_video(input, detail=detail, settings=settings)
        rprint(
            f"{len(desc.spec.scenes)} scenes, {desc.spec.shot_count} shots, "
            f"{desc.frame_count} frames analyzed (detail={detail.value})."
        )
        rec = recreate_from_spec(desc.spec_path, settings=settings, out_path=out)
    except CleanroomError as e:
        rprint(f"[red]Run failed:[/red] {e}")
        raise typer.Exit(1)
    rprint(f"[green]Spec:[/green] {desc.spec_path}")
    rprint(f"[green]Recreation:[/green] {rec.output_path}")


@app.command()
def demo(
    out: Path = typer.Option(Path("assets/demo_recreation.mp4"), "--out", "-o"),
):
    """End-to-end smoke test with zero API keys.

    Generates a synthetic source video, describes it with the mock vision
    client, and recreates it with the mock generator. Proves the plumbing works.
    """
    from .config import Settings as S
    settings = S(
        vision_provider="mock", text_provider="mock",
        video_provider="mock", transcriber="mock",
    )
    workdir = settings.workdir / "demo"
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "demo_source.mp4"
    if not source.exists():
        rprint("[dim]Generating synthetic source video...[/dim]")
        try:
            run_cmd(
                [
                    "ffmpeg", "-hide_banner", "-y",
                    "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=12:r=24",
                    "-f", "lavfi", "-i", "sine=frequency=440:d=12",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest",
                    str(source),
                ],
                capture=True,
            )
        except CleanroomError as e:
            rprint(f"[red]Demo failed:[/red] {e}")
            raise typer.Exit(1)
    try:
        desc = describe_video(source, detail=DetailLevel.STANDARD, settings=settings,
                              out_path=workdir / "demo.spec.json", with_audio=False)
        rec = recreate_from_spec(desc.spec_path, settings=settings, out_path=out)
    except CleanroomError as e:
        rprint(f"[red]Demo failed:[/red] {e}")
        raise typer.Exit(1)
    rprint(f"[green]Demo complete.[/green] Recreation: {rec.output_path}")
    rprint(f"[dim]Spec: {desc.spec_path}[/dim]")


@app.command(name="render-md")
def render_md(
    spec: Path = typer.Argument(..., help="Spec JSON file."),
):
    """Print the human-readable Markdown rendering of a spec."""
    from .models.spec import VideoSpec
    try:
        s = VideoSpec.model_validate_json(spec.read_text())
    except Exception as e:
        rprint(f"[red]Could not read spec:[/red] {e}")
        raise typer.Exit(1)
    print(s.to_markdown())


if __name__ == "__main__":
    app()
