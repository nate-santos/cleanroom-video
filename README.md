# cleanroom-video

Two LLMs recreate a video, clean-room style:

1. **LLM #1 — the describer** watches the source video (sampled frames + audio transcript) and writes a **tremendously detailed spec**: scenes, shots, camera work, subjects, action, dialogue, lighting, palette, mood, audio.
2. **LLM #2 — the re-creator** reads *only the spec* and translates each shot into a text-to-video prompt, which a video generation model renders into clips. Clips are concatenated into the final video.

The spec file (`spec.json`) is the **only thing that crosses between stages**. LLM #2 never sees the source video — that boundary is enforced structurally (`cleanroom recreate` takes a spec path, never a video path). That's the whole clean-room premise: description in, recreation out, no copying.

```
source.mp4
   │  ffmpeg: scene cuts, frame sampling, audio
   ▼
┌─────────────┐   frames + transcript    ┌──────────────┐
│  LLM #1     │ ───────────────────────▶ │  spec.json   │
│ (describer) │                          │  (+ spec.md) │
└─────────────┘                          └──────┬───────┘
                                                │  spec only — no video
                                                ▼
                                         ┌──────────────┐   per-shot    ┌───────────┐
                                         │  LLM #2      │ ──prompts──▶ │  video    │──▶ clips
                                         │ (prompt      │              │  generator│
                                         │  planner)    │              │ (Veo/...) │
                                         └──────────────┘              └─────┬─────┘
                                                                           ▼
                                                                     recreation.mp4
```

## Quickstart

```bash
pip install -e ".[dev]"        # or: pip install -e ".[all]" for openai + whisper too
cp .env.example .env           # optional; everything defaults to mock

# Zero-API-key smoke test: synthetic video -> mock spec -> mock recreation
cleanroom demo

# Real usage (needs a vision-capable model; set CLEANROOM_VISION_PROVIDER=openai)
cleanroom describe myvideo.mp4 --detail deep -o myspec.json
cleanroom render-md myspec.json          # read what the describer saw
cleanroom recreate myspec.json -o myrecreation.mp4
cleanroom run myvideo.mp4 --detail standard -o out.mp4   # both stages at once
```

## The detail knob

`--detail` controls how hard LLM #1 looks. It sets frame sampling rate, per-scene/total frame budgets, scene-cut sensitivity, and which fields the spec must contain:

| level | fps | frames/scene | total cap | spec depth |
|---|---|---|---|---|
| `brief` | 0.5 | 8 | 48 | setting, subjects, action, camera move, dialogue summary |
| `standard` | 1.0 | 16 | 96 | + time of day, dialogue, angle/framing, lighting, mood |
| `deep` | 2.0 | 32 | 192 | + appearance, on-screen text, lens look, palette, audio |
| `forensic` | 4.0 | 64 | 384 | + beat-by-beat action, wardrobe, props, light direction, transitions |

Higher detail = better recreation fidelity = more vision-API cost. `brief` is for triage; `forensic` is for the shots that matter.

## Running fully offline (no API keys)

All LLM calls in this project are plain OpenAI-compatible HTTP. Point them at
a local server and the whole pipeline runs offline — no accounts, no keys.

**Option A — Ollama (easiest).** Install from https://ollama.com, then:

```bash
ollama pull qwen2.5vl:7b      # vision model for LLM #1 (the describer)
ollama pull qwen2.5:7b         # text model for LLM #2 (the prompt planner)
```

Qwen2.5-VL is open-weight (Apache 2.0) with native video understanding; the
7B Q4 quant is ~5GB, so it fits comfortably on an 8GB GPU. (`qwen2.5vl:3b` is
the lighter/faster alternative.)

```bash
export CLEANROOM_VISION_PROVIDER=local
export CLEANROOM_VISION_MODEL=qwen2.5vl:7b
export CLEANROOM_TEXT_PROVIDER=local
export CLEANROOM_TEXT_MODEL=qwen2.5:7b
export OPENAI_BASE_URL=http://localhost:11434/v1
cleanroom demo   # then real videos, fully offline
```

**Option B — llama-server.** For more control (quant choice, GPU backends):

```bash
llama-server -hf <Qwen2.5-VL-7B-Instruct-GGUF-repo> --port 8080
# then OPENAI_BASE_URL=http://localhost:8080/v1 with provider=local
```

Why this works with no code changes: the describer already reduces video to
sampled JPEG frames, so any multi-image vision model can do the job — native
video input is a bonus, not a requirement. The `local` provider is identical
to `openai` on the wire except it skips `response_format=json_object` (many
local servers don't implement it) and doesn't need an API key; JSON parsing
tolerates the markdown fences local models like to add.

Honest trade-offs: a local 7B sees less fine detail than a frontier vision
model — compensate with `--detail deep` (more frames, more fields). The one
piece that stays cloud-bound for now is stage-2 *rendering*: local video
generation (Wan, CogVideoX) is on the ROADMAP, not implemented.

## Configuration

All via environment (see `.env.example`):

| var | default | options |
|---|---|---|
| `CLEANROOM_VISION_PROVIDER` | `mock` | `mock`, `openai`, `local` (Ollama/llama-server via `OPENAI_BASE_URL`, no key needed) |
| `CLEANROOM_VISION_MODEL` | `gpt-4o` | any vision-capable chat model (`qwen2.5vl:7b` for local) |
| `CLEANROOM_TEXT_PROVIDER` | `mock` | `mock`, `openai`, `local` |
| `CLEANROOM_VIDEO_PROVIDER` | `mock` | `mock` (ffmpeg placeholder clips); `veo`/`runway`/`sora` are stubbed for phase 1 |
| `CLEANROOM_TRANSCRIBER` | `mock` | `mock`, `faster-whisper` (local) |
| `CLEANROOM_WORKDIR` | `assets/work` | scratch dir for frames, clips, manifests |

## Project layout

```
src/cleanroom/
├── cli.py                  # typer CLI: describe / recreate / run / demo / render-md
├── config.py               # DetailLevel profiles + Settings.from_env()
├── utils.py
├── models/spec.py          # THE contract: VideoSpec/SceneSpec/ShotSpec (+ to_markdown)
├── describe/               # stage 1 — LLM #1
│   ├── extractor.py        # ffmpeg: duration, scene cuts, frame sampling, audio
│   ├── prompts.py          # describer prompt templates (highest-leverage text here)
│   ├── vision.py           # VisionClient protocol + mock/openai impls + factory
│   ├── transcribe.py       # Transcriber protocol + mock/faster-whisper
│   └── describer.py        # orchestration: video -> VideoSpec
└── recreate/               # stage 2 — LLM #2
    ├── prompts.py          # prompt-planner templates
    ├── promptgen.py        # TextClient protocol; spec shot -> ShotPrompt
    ├── generators.py       # VideoGenerator protocol; mock impl; veo/runway/sora stubs
    ├── assembler.py        # concat clips -> final mp4
    └── recreator.py        # orchestration: spec -> clips -> video + manifest.json
tests/                      # pytest; ffmpeg tests skip gracefully if missing
examples/spec.example.json  # a realistic hand-written spec to develop against
```

## How to extend (the long-term improvement guide)

This repo is built to be iterated on. The seams are protocols + factories:

- **Better describer prompts** → edit `describe/prompts.py` and the `DetailProfile.instruction` strings in `config.py`. This is the cheapest fidelity win.
- **New vision model** (Anthropic, Gemini, local VLM) → implement `describe_scene` in `describe/vision.py`, add a branch in `make_vision_client`.
- **New text model for LLM #2** → implement `complete` in `recreate/promptgen.py`, add a branch in `make_text_client`.
- **Real video generation** → implement `generate()` on the stub in `recreate/generators.py` (Veo/Runway/Sora), register in `make_video_generator`. Each generator gets the full `ShotPrompt` (prompt, negative, duration, aspect).
- **Better scene detection** → swap `detect_scenes` in `describe/extractor.py` (PySceneDetect is the obvious upgrade).
- **Smarter assembly** → `recreate/assembler.py` is where per-spec transitions (dissolves, fades) and audio beds belong.
- **Fidelity evaluation** → the natural phase-2 loop: describe the *recreation*, diff the two specs, score. See ROADMAP.

Every `recreate` run writes `manifest.json` (spec → prompts → clips mapping) into the workdir, so any recreation is reproducible and auditable from its spec alone.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Note on use

Built as a technical exercise in description-driven media synthesis. The describer deliberately keeps dialogue quotes short and describes rather than transcribes — recreate your own footage, or footage you have the rights to work with.
