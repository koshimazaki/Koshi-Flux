#!/usr/bin/env python3
"""Klein V2V Temporal (NATIVE BFL SDK) - smooth transitions via prev_gen + curr_input blending.

Native twin of hybrid-v2v/klein_v2v_temporal.py: identical CLI, defaults, and
frame loop - only the pipeline differs (pure BFL AutoEncoder + DiT instead of
diffusers VAE + BFL DiT), so outputs A/B-compare the VAE/latent path.
"""
import argparse
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
parser.add_argument("--output", "-o", default="outputs/native_v2v_temporal.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--strength", "-s", type=float, default=0.5)
parser.add_argument("--prev-blend", type=float, default=0.3)
parser.add_argument("--ref", type=str)
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model", default="flux.2-klein-4b",
                    choices=["flux.2-klein-4b", "flux.2-klein-9b"])
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--guidance", type=float, default=1.0)
args = parser.parse_args()

frames, fps = load_video(args.input, max_frames=args.max_frames)
ref_img = Image.open(args.ref).convert("RGB").resize(frames[0].size, Image.LANCZOS) if args.ref else None

# ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
with GenerationContext(args.output) as gen:
    gen.update(
        preset="native_v2v_temporal",
        pipeline="native-bfl",
        input=args.input,
        prompt=args.prompt,
        strength=args.strength,
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

    for i, frame in enumerate(tqdm(frames, desc="NativeTemporal")):
        # prev_gen is decoded at default_prep size (/16 crop, pixel cap) - resize
        # to frame.size or the pixel blend crashes on 1080p/non-/16 inputs.
        source = blend(ref_img, frame, 0.3) if (i == 0 and ref_img) else (blend(prev_gen.resize(frame.size), frame, args.prev_blend) if i > 0 else frame)
        img = pipe.generate(source, args.prompt, strength=args.strength,
                            num_steps=args.steps, guidance=args.guidance, seed=args.seed)
        anchor = img if i == 0 else anchor
        img = img if i == 0 else match_color_lab(img, anchor)
        output.append(img)
        prev_gen = img
        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
