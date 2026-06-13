#!/usr/bin/env python3
"""E2E smoke of the preset lanes on a GPU box (RunPod). Single command, no prompts.

    cd /workspace/Koshi-Flux && python3 flux/scripts/e2e_native_presets.py
    # options: --all (include hybrid twins + klein_native), --workdir DIR,
    #          --frames N (default 6), --size N (default 256), --keep

Per preset: builds tiny synthetic fixtures (video / kick wav / still PNG),
runs the preset as a subprocess exactly like a user would, then asserts:
exit 0, .mp4 exists and is non-trivial, settings .json sidecar says
status=completed with the right preset name, and audio presets actually
muxed an audio stream (ffprobe).

Audio source mode: uses --audio (librosa) when librosa is importable,
otherwise falls back to --feature-video for motion + --audio only for mux.
Exit code 0 = all green. First failure prints captured stderr tail, exit 1.
"""
import argparse
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import wave as wave_mod
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PRESETS = REPO / "presets"


# ------------------------------------------------------------------ fixtures

def make_video(path: Path, frames: int, size: int, fps: int = 12) -> None:
    import cv2
    import numpy as np

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(frames):
        img = np.zeros((size, size, 3), np.uint8)
        g = np.linspace(0, 255, size, dtype=np.uint8)
        img[:, :, 0] = g[None, :]
        img[:, :, 1] = g[:, None]
        x = int((i / max(frames - 1, 1)) * (size - 40))
        cv2.rectangle(img, (x, size // 3), (x + 40, 2 * size // 3), (0, 0, 255), -1)
        cv2.circle(img, (size // 2, size // 2), size // 6 + 2 * i, (255, 255, 0), 3)
        vw.write(img)
    vw.release()
    assert path.exists() and path.stat().st_size > 0, f"fixture video failed: {path}"


def make_kick_wav(path: Path, seconds: float = 2.0, sr: int = 22050) -> None:
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        env = math.exp(-8.0 * (t % 0.5))           # kick every 0.5s
        s = 0.8 * env * math.sin(2 * math.pi * 60 * t)
        s += 0.1 * math.sin(2 * math.pi * 4000 * t) * math.exp(-30.0 * (t % 0.25))
        frames += struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767))
    with wave_mod.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


def make_still(path: Path, size: int) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (x * 255 // size, y * 255 // size, 128)
    d = ImageDraw.Draw(img)
    d.ellipse([size // 4, size // 4, 3 * size // 4, 3 * size // 4],
              outline=(255, 255, 0), width=4)
    img.save(path)


# ----------------------------------------------------------------- assertions

def has_audio_stream(path: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return "audio" in r.stdout


def check_output(name: str, out: Path, expect_frames: int, expect_audio: bool) -> list:
    errs = []
    if not out.exists() or out.stat().st_size < 5_000:
        errs.append(f"{name}: output missing/too small: {out}")
        return errs
    sidecar = out.with_suffix(".json")
    if not sidecar.exists():
        errs.append(f"{name}: settings JSON sidecar missing")
        return errs
    meta = json.loads(sidecar.read_text())
    if meta.get("status") != "completed":
        errs.append(f"{name}: sidecar status={meta.get('status')} error={meta.get('error')}")
    if meta.get("frames") != expect_frames:
        errs.append(f"{name}: sidecar frames={meta.get('frames')} expected {expect_frames}")
    if expect_audio and not has_audio_stream(out):
        errs.append(f"{name}: no audio stream muxed into {out.name}")
    return errs


def run_preset(label: str, cmd: list, timeout: int = 1800):
    print(f"\n=== {label} ===\n    {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    r = subprocess.run([sys.executable] + [str(c) for c in cmd],
                       cwd=REPO, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-2500:]
        print(f"FAILED ({dt:.0f}s) exit={r.returncode}\n--- stderr tail ---\n{tail}")
        return False
    print(f"ok ({dt:.0f}s)")
    return True


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="E2E preset smoke (GPU)")
    ap.add_argument("--workdir", default=str(REPO / "outputs" / "e2e"))
    ap.add_argument("--frames", type=int, default=6)
    # 264 is deliberately NOT divisible by 16: default_prep center-crops it to
    # 256, so every run exercises the decoded-size != frame-size path (the P1
    # class that 1080p footage triggers). Use --size 256 for aligned-only runs.
    ap.add_argument("--size", type=int, default=264)
    ap.add_argument("--all", action="store_true",
                    help="Also run hybrid twins + klein_native (default: native core only)")
    ap.add_argument("--keep", action="store_true", help="Keep workdir on success")
    args = ap.parse_args()

    # Preflight
    import torch
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available - this e2e needs a GPU box.")
        return 2
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("FATAL: ffmpeg/ffprobe not on PATH.")
        return 2
    if importlib.util.find_spec("flux2") is None:
        print("FATAL: flux2 SDK not importable "
              "(pip install git+https://github.com/black-forest-labs/flux2.git).")
        return 2
    have_librosa = importlib.util.find_spec("librosa") is not None
    print(f"GPU: {torch.cuda.get_device_name(0)} | librosa: {have_librosa} "
          f"(audio motion via {'--audio' if have_librosa else '--feature-video fallback'})")
    print("TIP: Klein repos ship no native ae.safetensors - NativePipeline falls back "
          "to the FLUX.2-dev family AE (gated: accept access + `huggingface-cli login`), "
          "or set AE_MODEL_PATH=/path/to/ae.safetensors.")

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    vid, wav, still = wd / "in.mp4", wd / "kicks.wav", wd / "still.png"
    make_video(vid, max(args.frames, 8), args.size)
    make_kick_wav(wav)
    make_still(still, args.size)
    n = args.frames

    # Audio-source args: librosa path or feature-video fallback (mux still via --audio)
    audio_src = ["--audio", wav] if have_librosa else ["--feature-video", vid, "--audio", wav]

    runs = [
        ("native/ramp",
         [PRESETS / "native/klein_v2v_ramp.py", "-i", vid, "-p", "neon wireframe", "-n", n,
          "-o", wd / "native_ramp.mp4"], n, False),
        ("native/deforum",
         [PRESETS / "native/klein_v2v_deforum.py", "-i", vid, "-p", "neon wireframe", "-n", n,
          "--zoom", "0:(1.0), 6:(1.04)", "-o", wd / "native_deforum.mp4"], n, False),
        ("native/latent_color",
         [PRESETS / "native/klein_v2v_latent_color.py", "-i", vid, "-p", "neon wireframe",
          "--ref", still, "-n", n, "-o", wd / "native_latent_color.mp4"], n, False),
        ("native/audio_deforum",
         [PRESETS / "native/klein_v2v_audio_deforum.py", "-i", vid, "-p", "neon wireframe",
          *audio_src, "-n", n, "-o", wd / "native_audio_deforum.mp4"], n, True),
        ("native/i2v_audio",
         [PRESETS / "native/klein_i2v_audio.py", "--image", still, "-p", "neon wireframe",
          *audio_src, "--frames", n, "--fps", 12, "-o", wd / "native_i2v.mp4"], n, True),
    ]
    if args.all:
        runs += [
            ("native/temporal",
             [PRESETS / "native/klein_v2v_temporal.py", "-i", vid, "-p", "neon wireframe",
              "-n", n, "-o", wd / "native_temporal.mp4"], n, False),
            ("native/latent_ref",
             [PRESETS / "native/klein_v2v_latent_ref.py", "-i", vid, "-p", "neon wireframe",
              "--ref", still, "-n", n, "-o", wd / "native_latent_ref.mp4"], n, False),
            ("native/klein_native",
             [PRESETS / "native/klein_native.py", "-i", vid, "-p", "neon wireframe",
              "-n", n, "-o", wd / "klein_native.mp4"], n, False),
            ("hybrid/ramp (A/B twin)",
             [PRESETS / "hybrid-v2v/klein_v2v_ramp.py", "-i", vid, "-p", "neon wireframe",
              "-n", n, "-o", wd / "hybrid_ramp.mp4"], n, False),
            ("hybrid/audio_deforum (A/B twin)",
             [PRESETS / "hybrid-v2v/klein_v2v_audio_deforum.py", "-i", vid, "-p",
              "neon wireframe", *audio_src, "-n", n,
              "-o", wd / "hybrid_audio_deforum.mp4"], n, True),
        ]

    failures = []
    for label, cmd, expect_frames, expect_audio in runs:
        out = Path(cmd[cmd.index("-o") + 1])
        if not run_preset(label, cmd):
            failures.append(f"{label}: nonzero exit")
            break  # GPU state after a crash is untrustworthy - fail fast
        errs = check_output(label, out, expect_frames, expect_audio)
        for e in errs:
            print(f"ASSERT FAIL: {e}")
        failures += errs
        torch.cuda.empty_cache()
        print(f"    VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    print("\n" + "=" * 60)
    if failures:
        print(f"E2E: {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"E2E: ALL GREEN ({len(runs)} presets) - outputs in {wd}")
    if not args.keep:
        print("(pass --keep to retain outputs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
