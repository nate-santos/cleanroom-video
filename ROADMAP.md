# ROADMAP

Living plan for improving cleanroom-video long-term. Phases are roughly ordered;
each should land as a small, reviewable change.

## Phase 0 — Scaffold ✅
- [x] Repo structure, spec schema, CLI (`describe` / `recreate` / `run` / `demo`)
- [x] Tunable detail levels (brief → forensic)
- [x] Mock vision/text/video providers so the pipeline runs with zero keys
- [x] ffmpeg scene-cut detection, frame sampling, audio extraction hooks
- [x] Manifest per recreation run (prompts + clips, reproducible from spec)

## Phase 1 — Real providers
- [ ] Implement one real video generator (`recreate/generators.py` stub: Veo, Runway, or Sora)
  - Handle per-shot duration limits (most models cap at 5–10s; split long shots)
  - Handle async generation (poll/pickup, don't block the CLI awkwardly)
  - Reference/start-frame support: last frame of previous clip as first frame of next (huge continuity win)
- [ ] Implement a real vision provider end-to-end against a long video; measure cost per detail level
- [ ] Wire `faster-whisper` transcription into the describer prompt and verify dialogue attribution improves
- [ ] API key handling: confirm `.env` flow, document per-provider setup in README

## Phase 2 — Fidelity loop (the interesting part)
- [ ] **Round-trip evaluation**: describe the *recreation* with LLM #1, diff the two specs (structure-aware diff on scenes/shots/fields), produce a fidelity score per shot
- [ ] Iterate: feed low-scoring shots' diffs back to LLM #2 as "fix notes", regenerate only those shots
- [ ] Track scores across detail levels → empirically answer "how much detail is enough?"
- [ ] Human eval harness: side-by-side viewer for original vs recreation with per-shot notes

## Phase 3 — Assembly & audio
- [ ] Per-spec transitions in `assembler.py` (dissolves, fades, whip pans per `transition_in`)
- [ ] Audio bed: dialogue via TTS aligned to shot timestamps, music/SFX via a generative audio model, ambience mix
- [ ] Aspect-ratio / fps normalization across heterogeneous generator outputs

## Phase 4 — Input robustness
- [ ] Better scene detection (PySceneDetect or learned shot-boundary model)
- [ ] Keyframe-aware sampling: sample denser around cuts and motion, sparser in static stretches (cheaper than flat fps at the same fidelity)
- [ ] Long-video support: chunked description with cross-chunk consistency (character bible carried between scenes)
- [ ] Subtitle/OCR track: feed burned-in subtitles and on-screen text via OCR rather than relying on the VLM

## Phase 5 — Experience
- [ ] Web UI: upload → watch spec build live → tweak per-shot prompts → regenerate shots → compare
- [ ] Spec editor: hand-fix a shot's spec and re-run only that shot (fast iteration loop)
- [ ] Cost estimator: `cleanroom estimate video.mp4 --detail deep` before spending

## Open questions
- Which video model gives the best prompt-adherence per dollar for 3–8s shots? (Needs phase-1 bake-off.)
- How should LLM #2 handle shots longer than the generator's max clip length — split prompts or time-stretched single prompt?
- Character consistency across shots/scenes is the known hard problem; options: reference images from the spec, LoRA/custom models, or casting to the best available approximation.
- Legal framing: fine for own/rights-held footage; keep the describer's short-quote behavior for anything else.
