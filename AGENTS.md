# AGENTS.md - Codex Guide for Koshi-Flux

## Repo Identity

This is the active GitHub repo for the Flux motion work:

- Local root: `/Users/radek/Documents/GIthub/DEFORUM-AI/Deofrum2026/Koshi-Flux`
- Remote: `https://github.com/koshimazaki/Koshi-Flux.git`
- Main package: `flux-motion`
- Import namespace: `flux_motion`

The parent `Deofrum2026/` folder is a working umbrella, not this repo's Git root. Do not edit sibling repos such as `LTX-RESEARCH/`, `LTX-2/`, `aimedia_hf/`, or `flux2-main/` unless the user explicitly asks.

## Project Purpose

Koshi-Flux is a motion pipeline for Black Forest Labs FLUX models, especially FLUX.2 [klein]. It brings Deforum-style animation control into FLUX image models with:

- Video-to-video motion using optical flow, warping, temporal blending, and color matching.
- FLUX.1 and FLUX.2 latent motion engines.
- Audio-reactive schedule extraction and mapping.
- Preset scripts for RunPod experiments and BFL-facing demos.

When describing output publicly, keep the accuracy guardrail clear: FLUX/Klein is the image model layer; motion/video guidance comes from the pipeline, video input, schedules, or downstream video models.

## Repository Map

```text
Koshi-Flux/
├── flux/                    # Python package: flux_motion
│   ├── src/flux_motion/
│   │   ├── flux1/           # FLUX.1 native pipeline and motion engine
│   │   ├── flux2/           # FLUX.2/Klein pipeline, config, LoRA helpers
│   │   ├── shared/          # Base engines, transforms, optical flow helpers
│   │   ├── feedback/        # Feedback sampler, color matching, processors
│   │   ├── audio/           # Audio feature extraction and schedule generation
│   │   ├── bridge/          # Parameter adapter bridge
│   │   └── api/             # FastAPI surface for generation
│   ├── tests/               # Unit/API tests
│   ├── examples/            # Small package examples
│   └── scripts/             # RunPod and smoke-test scripts
├── presets/
│   ├── hybrid-v2v/          # Recommended BFL denoise + VAE/video workflows
│   ├── native/              # Pure BFL SDK scripts
│   └── diffusers/           # Fast standalone diffusers experiments
├── scheduling/              # Parseq-like scheduling, audio, adapters
├── bfl_demo/                # Standalone BFL demo prototype
├── app/                     # Unified FLUX/LTX app prototype
└── core/                    # Older Deforum core/setup package
```

## Quick Commands

Run from the repo root unless noted:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e './flux[dev,video,audio]'
cd flux && pytest tests -v
```

Focused checks:

```bash
cd flux && pytest tests/test_motion_engine.py tests/test_parameter_adapter.py -v
cd flux && pytest tests/test_audio.py -v
cd flux && ruff check src tests && black --check src tests && mypy src
```

Useful smoke scripts:

```bash
cd flux && python examples/parameter_parsing.py
cd flux && python examples/audio_to_animation.py
cd flux && python scripts/run_comprehensive_tests.py --quick
```

GPU generation and BFL SDK scripts usually need RunPod or another CUDA machine. Validate imports and arguments locally first, then run one complete remote command.

## Flux SDK Strategy

Prefer the native BFL SDK path for serious FLUX/Klein work:

- Use `flux2.sampling`, `flux2.util`, and `flux_motion.flux2.Flux2Pipeline` where available.
- Use `presets/hybrid-v2v/` as the default lane for practical V2V: diffusers VAE where useful, BFL native denoise for the generation loop.
- Use `presets/native/` for pure BFL SDK experiments and BFL-facing proof.
- Use `presets/diffusers/` or root-level quick scripts only for fast iteration, isolated experiments, or when the user asks for diffusers specifically.

Do not silently replace a native BFL workflow with diffusers. If diffusers is used for speed or compatibility, say so in settings, README notes, and final summaries.

If the SDK source is needed, inspect it from the local sibling `../flux2-main/` or the installed package, but keep code changes inside this repo unless the user explicitly asks to patch the SDK.

## Audio-Reactive Workflow

Audio-reactive work in this repo should connect audio features to motion schedules, not only post-process decoded frames.

Primary locations:

- `flux/src/flux_motion/audio/extractor.py` - audio feature extraction.
- `flux/src/flux_motion/audio/mapping_config.py` - feature-to-parameter mapping.
- `flux/src/flux_motion/audio/schedule_generator.py` - Deforum/Parseq-style schedule output.
- `scheduling/audio/` - Parseq-like audio timeseries and stem helpers.
- `presets/native/klein_v2v_audio.py` - audio-reactive Klein V2V preset.

Default mapping idea:

- Low/bass/kick drives zoom, strength, or structure-heavy motion.
- Mid/snare drives translation or temporal contrast.
- High/hat drives angle, jitter, texture, or detail modulation.
- Amplitude/energy controls overall motion strength.

For Comfy/Koshi-node work, treat this repo as the Flux motion reference. Do not assume Comfy node names or schemas without inspecting the relevant Comfy repo first.

## Generation Settings

Every generation script must save a JSON settings file next to the video output. Prefer `presets/klein_utils.py` and its `GenerationContext` helper for presets.

Required fields for new scripts:

```json
{
  "timestamp": "2026-06-03T12:00:00Z",
  "model": "black-forest-labs/FLUX.2-klein-4b",
  "pipeline": "hybrid-v2v",
  "prompt": "prompt text",
  "negative_prompt": "",
  "steps": 4,
  "guidance_scale": 3.5,
  "width": 768,
  "height": 512,
  "num_frames": 60,
  "fps": 24,
  "seed": 12345,
  "input_video": "",
  "audio": "",
  "motion_params": {},
  "extra": {}
}
```

Log useful generation results in the umbrella `../generations/README.md` when outputs are part of an experiment series.

## RunPod Practices

GPU time is expensive. Write complete, self-contained commands and fail fast before loading models.

- Prefer one-line remote commands for SSH/RunPod.
- Validate paths, prompt, frame count, output directory, and settings JSON path before model load.
- Use `@torch.no_grad()` for inference.
- Use `torch.float16` or `torch.bfloat16` on GPU.
- Enable offload on lower VRAM cards.
- Call `torch.cuda.empty_cache()` between long generations.
- Log VRAM usage for serious tests.

Example remote shape:

```bash
ssh root@$RUNPOD_IP "cd /workspace/Koshi-Flux && pip install -e './flux[video,audio]' -q && python presets/native/klein_v2v_audio.py -i input.mp4 -a music.wav -o outputs/v2v_audio.mp4 -p 'audio reactive botanical motion'"
```

## Python Standards

- Python: 3.10+ for `flux-motion`; keep compatibility with 3.10, 3.11, and 3.12.
- Formatting: `black`, line length 100.
- Linting: `ruff` with E, F, W, I, N, B, C4.
- Types: add annotations for new public functions and complex helpers.
- Docstrings: Google-style for public APIs and non-trivial helpers.
- Imports: stdlib, third-party, local.
- Errors: validate early and include shape/path/model context.
- Security: never log tokens, HF keys, private SSH details, or API secrets.

PyTorch rules:

- Use `@torch.no_grad()` around inference paths.
- Validate tensor rank and channel count before transforms.
- Preserve dtype/device unless there is a deliberate conversion.
- Include expected and actual shapes in exceptions.
- Keep CPU smoke tests possible for motion transforms and schedule parsing.

## Motion Architecture Notes

Core package paths:

- `flux/src/flux_motion/shared/base_engine.py`
- `flux/src/flux_motion/shared/parameter_adapter.py`
- `flux/src/flux_motion/shared/transforms.py`
- `flux/src/flux_motion/flux1/motion_engine.py`
- `flux/src/flux_motion/flux2/motion_engine.py`

Tensor conventions:

- 4D image latent: `(B, C, H, W)`
- 5D sequence latent: `(B, T, C, H, W)`
- FLUX.1: 16 channels
- FLUX.2/Klein: 128 channels

Common motion parameters include `zoom`, `angle`, `translation_x`, `translation_y`, strength/blend values, and schedule strings compatible with Deforum-style keyframes.

## Editing Rules

- Read real files before changing code; this repo has older scripts with mixed patterns.
- Keep presets unless the user explicitly asks to replace them. Prefer adding `_v2`, `_audio`, or a clearly named new preset over overwriting a working one.
- Keep generated outputs out of commits unless requested.
- Do not commit cache folders such as `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, model weights, or video outputs.
- Keep changes scoped to this repo and the task at hand.

## Completion Checklist

Before saying work is done:

- Confirm you worked inside `/Users/radek/Documents/GIthub/DEFORUM-AI/Deofrum2026/Koshi-Flux`.
- Run relevant tests, lint, typecheck, or a smoke command.
- For GPU work, report what ran locally vs remotely.
- For generations, confirm both video and JSON settings exist.
- Store important decisions or new project facts in Open-Brain when they affect future sessions.
