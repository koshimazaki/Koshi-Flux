#!/usr/bin/env python3
"""Klein V2V Ultimate - Motion + Temporal + Init Image with enforced JSON metadata."""
import argparse
from klein_utils import (
    load_video, match_color_lab, blend, optical_flow, warp, get_pipeline, generate, clear_cuda, tqdm, Image,
    GenerationContext  # ENFORCED: Always save settings JSON
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/v2v_ultimate.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--init", type=str)
parser.add_argument("--init-blend", type=float, default=0.3)
parser.add_argument("--strength", "-s", type=float, default=0.5)
parser.add_argument("--flow-blend", type=float, default=0.5)
parser.add_argument("--temporal-blend", type=float, default=0.3)
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

frames, fps = load_video(args.input, max_frames=args.max_frames)
init_img = Image.open(args.init).convert("RGB").resize(frames[0].size, Image.LANCZOS) if args.init else None

# ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
with GenerationContext(args.output) as gen:
    gen.update(
        preset="v2v_ultimate",
        input=args.input,
        prompt=args.prompt,
        init=args.init,
        init_blend=args.init_blend,
        strength=args.strength,
        flow_blend=args.flow_blend,
        temporal_blend=args.temporal_blend,
        seed=args.seed,
        model="flux.2-klein-4b",
        steps=4,
    )
    gen.fps = fps

    pipe = get_pipeline()
    output, prev_input, prev_gen, anchor = [], frames[0], None, None

    for i, frame in enumerate(tqdm(frames, desc="Ultimate")):
        if i == 0:
            source = blend(init_img, frame, args.init_blend) if init_img else frame
            img = generate(pipe, source, args.prompt, 0.7, args.seed)
            anchor = img
        else:
            warped = warp(prev_gen, optical_flow(prev_input, frame))
            motion_guided = blend(warped, frame, args.flow_blend)
            source = blend(prev_gen, motion_guided, args.temporal_blend) if args.temporal_blend > 0 else motion_guided
            img = match_color_lab(generate(pipe, source, args.prompt, args.strength, args.seed), anchor)
        output.append(img)
        prev_input, prev_gen = frame, img
        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
