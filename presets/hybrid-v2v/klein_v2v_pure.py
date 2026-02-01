#!/usr/bin/env python3
"""Klein V2V Pure - True latent-guided img2img with enforced JSON metadata."""
import argparse
from klein_utils import (
    load_video, match_color_lab, get_pipeline, generate, clear_cuda, tqdm,
    GenerationContext  # ENFORCED: Always save settings JSON
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/v2v_pure.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--strength", "-s", type=float, default=0.65)
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

frames, fps = load_video(args.input, max_frames=args.max_frames)

# ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
with GenerationContext(args.output) as gen:
    gen.update(
        preset="v2v_pure",
        input=args.input,
        prompt=args.prompt,
        strength=args.strength,
        seed=args.seed,
        model="flux.2-klein-4b",
        steps=4,
    )
    gen.fps = fps

    pipe = get_pipeline()
    output, anchor = [], None

    for i, frame in enumerate(tqdm(frames, desc="V2V")):
        img = generate(pipe, frame, args.prompt, args.strength, args.seed)
        anchor = img if i == 0 else anchor
        output.append(img if i == 0 else match_color_lab(img, anchor))
        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
