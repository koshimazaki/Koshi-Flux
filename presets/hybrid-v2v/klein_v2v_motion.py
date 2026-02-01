#!/usr/bin/env python3
"""Klein V2V Motion - Optical flow transfer with enforced JSON metadata."""
import argparse
from klein_utils import (
    load_video, match_color_lab, optical_flow, warp, get_pipeline, generate, clear_cuda, tqdm,
    GenerationContext  # ENFORCED: Always save settings JSON
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/v2v_motion.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--strength", "-s", type=float, default=0.3)
parser.add_argument("--init-from-input", action="store_true")
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

frames, fps = load_video(args.input, max_frames=args.max_frames)

# ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
with GenerationContext(args.output) as gen:
    gen.update(
        preset="v2v_motion",
        input=args.input,
        prompt=args.prompt,
        strength=args.strength,
        init_from_input=args.init_from_input,
        seed=args.seed,
        model="flux.2-klein-4b",
        steps=4,
    )
    gen.fps = fps

    pipe = get_pipeline()
    output, prev_input, prev_gen = [], frames[0], None

    for i, frame in enumerate(tqdm(frames, desc="Motion")):
        if i == 0:
            img = generate(pipe, frame, args.prompt, 0.7 if args.init_from_input else 0.95, args.seed)
        else:
            warped = warp(prev_gen, optical_flow(prev_input, frame))
            img = generate(pipe, warped, args.prompt, args.strength, args.seed)
            img = match_color_lab(img, output[0])
        output.append(img)
        prev_input, prev_gen = frame, img
        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
