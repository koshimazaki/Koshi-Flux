#!/usr/bin/env python3
"""Klein V2V Ramp - Strength ramps from start to end over frames."""
import argparse
import math
from klein_utils import (
    load_video, match_color_lab, blend, get_pipeline, generate, clear_cuda, tqdm, Image,
    GenerationContext
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/v2v_ramp.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--strength-start", type=float, default=0.20, help="Start strength (0.20 = 80%% video)")
parser.add_argument("--strength-end", type=float, default=0.35, help="End strength (0.35 = 65%% video)")
parser.add_argument("--ramp-mode", choices=["linear", "sine"], default="linear", help="Ramp mode")
parser.add_argument("--prev-blend", type=float, default=0.3)
parser.add_argument("--ref", type=str)
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
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
        preset="v2v_ramp",
        input=args.input,
        prompt=args.prompt,
        strength_start=args.strength_start,
        strength_end=args.strength_end,
        ramp_mode=args.ramp_mode,
        prev_blend=args.prev_blend,
        ref=args.ref,
        seed=args.seed,
        model="flux.2-klein-4b",
        steps=4,
    )
    gen.fps = fps

    pipe = get_pipeline()
    output, prev_gen, anchor = [], None, None

    for i, frame in enumerate(tqdm(frames, desc="Ramp")):
        strength = get_strength(i, total, args.strength_start, args.strength_end, args.ramp_mode)
        source = blend(ref_img, frame, 0.3) if (i == 0 and ref_img) else (blend(prev_gen, frame, args.prev_blend) if i > 0 else frame)
        img = generate(pipe, source, args.prompt, strength, args.seed)
        anchor = img if i == 0 else anchor
        img = img if i == 0 else match_color_lab(img, anchor)
        output.append(img)
        prev_gen = img
        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
