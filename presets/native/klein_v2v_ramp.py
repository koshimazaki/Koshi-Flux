#!/usr/bin/env python3
"""Klein V2V Ramp (NATIVE BFL SDK) - strength ramps from start to end over frames.

Native twin of hybrid-v2v/klein_v2v_ramp.py: identical CLI, defaults, and frame
loop - the only variable is the pipeline (pure BFL AutoEncoder + DiT instead of
diffusers VAE + BFL DiT), so the two outputs A/B-compare the VAE/latent path.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from klein_utils import (
    load_video, match_color_lab, blend, clear_cuda, tqdm, Image,
    GenerationContext  # ENFORCED: Always save settings JSON
)
from native_utils import NativePipeline

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/native_v2v_ramp.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--strength-start", type=float, default=0.20, help="Start strength (0.20 = 80%% video)")
parser.add_argument("--strength-end", type=float, default=0.35, help="End strength (0.35 = 65%% video)")
parser.add_argument("--ramp-mode", choices=["linear", "sine"], default="linear", help="Ramp mode")
parser.add_argument("--prev-blend", type=float, default=0.3)
parser.add_argument("--ref", type=str)
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model", default="flux.2-klein-4b",
                    choices=["flux.2-klein-4b", "flux.2-klein-9b"])
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--guidance", type=float, default=1.0)
args = parser.parse_args()


def get_strength(frame_idx, total_frames, start, end, mode="linear"):
    """Calculate strength at frame index."""
    t = frame_idx / max(total_frames - 1, 1)
    if mode == "linear":
        return start + (end - start) * t
    elif mode == "sine":
        return start + (end - start) * (0.5 + 0.5 * math.sin(2 * math.pi * t - math.pi / 2))
    return start


frames, fps = load_video(args.input, max_frames=args.max_frames)
ref_img = Image.open(args.ref).convert("RGB").resize(frames[0].size, Image.LANCZOS) if args.ref else None
total = len(frames)

with GenerationContext(args.output) as gen:
    gen.update(
        preset="native_v2v_ramp",
        pipeline="native-bfl",
        input=args.input,
        prompt=args.prompt,
        strength_start=args.strength_start,
        strength_end=args.strength_end,
        ramp_mode=args.ramp_mode,
        prev_blend=args.prev_blend,
        ref=args.ref,
        seed=args.seed,
        model=args.model,
        steps=args.steps,
        guidance=args.guidance,
    )
    gen.fps = fps

    pipe = NativePipeline(model_name=args.model)
    output, prev_gen, anchor = [], None, None

    for i, frame in enumerate(tqdm(frames, desc="NativeRamp")):
        strength = get_strength(i, total, args.strength_start, args.strength_end, args.ramp_mode)
        # prev_gen is decoded at default_prep size (/16 crop, pixel cap) - resize
        # to frame.size or the pixel blend crashes on 1080p/non-/16 inputs.
        source = blend(ref_img, frame, 0.3) if (i == 0 and ref_img) else (blend(prev_gen.resize(frame.size), frame, args.prev_blend) if i > 0 else frame)
        img = pipe.generate(source, args.prompt, strength=strength,
                            num_steps=args.steps, guidance=args.guidance, seed=args.seed)
        anchor = img if i == 0 else anchor
        img = img if i == 0 else match_color_lab(img, anchor)
        output.append(img)
        prev_gen = img
        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
