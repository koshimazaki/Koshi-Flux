#!/usr/bin/env python3
"""Klein V2V Temporal - Smooth transitions via prev_gen + curr_input blending with enforced JSON."""
import argparse
from klein_utils import (
    load_video, match_color_lab, blend, get_pipeline, generate, clear_cuda, tqdm, Image,
    GenerationContext  # ENFORCED: Always save settings JSON
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/v2v_temporal.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--strength", "-s", type=float, default=0.5)
parser.add_argument("--prev-blend", type=float, default=0.3)
parser.add_argument("--ref", type=str)
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

frames, fps = load_video(args.input, max_frames=args.max_frames)
ref_img = Image.open(args.ref).convert("RGB").resize(frames[0].size, Image.LANCZOS) if args.ref else None

# ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
with GenerationContext(args.output) as gen:
    gen.update(
        preset="v2v_temporal",
        input=args.input,
        prompt=args.prompt,
        strength=args.strength,
        prev_blend=args.prev_blend,
        ref=args.ref,
        seed=args.seed,
        model="flux.2-klein-4b",
        steps=4,
    )
    gen.fps = fps

    pipe = get_pipeline()
    output, prev_gen, anchor = [], None, None

    for i, frame in enumerate(tqdm(frames, desc="Temporal")):
        source = blend(ref_img, frame, 0.3) if (i == 0 and ref_img) else (blend(prev_gen, frame, args.prev_blend) if i > 0 else frame)
        img = generate(pipe, source, args.prompt, args.strength, args.seed)
        anchor = img if i == 0 else anchor
        img = img if i == 0 else match_color_lab(img, anchor)
        output.append(img)
        prev_gen = img
        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
