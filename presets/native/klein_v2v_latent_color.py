#!/usr/bin/env python3
"""Klein V2V Latent Reference + Latent Color Matching (NATIVE BFL SDK).

Native successor of hybrid-v2v/klein_v2v_latent_ref_LAB.py, renamed honestly:
despite the old "_LAB" name, NO LAB color space is involved - it matches color
statistics directly in the 128-channel latent (channels 32-63, color/lighting)
precisely to AVOID the washed-out colors that pixel LAB matching causes.

Structure channels (0-31) and texture channels (64-127) are preserved.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from klein_utils import (
    load_video, blend, match_color_latent, clear_cuda, tqdm, Image,
    GenerationContext
)
from native_utils import NativePipeline

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/native_v2v_latent_color.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--ref", type=str, required=True, help="Reference image for style")
parser.add_argument("--ref-blend", type=float, default=0.3, help="Latent blend: 0.3 = 30%% ref")
parser.add_argument("--strength", "-s", type=float, default=0.25, help="Generation strength")
parser.add_argument("--prev-blend", type=float, default=0.3, help="Temporal blend with prev gen")
parser.add_argument("--color-blend", type=float, default=0.7, help="Latent color match strength")
parser.add_argument("--color-channels", type=str, default="32,64", help="Color channel range")
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model", default="flux.2-klein-4b",
                    choices=["flux.2-klein-4b", "flux.2-klein-9b"])
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--guidance", type=float, default=1.0)
args = parser.parse_args()


def generate_with_latent_ref_and_color(
    pipe, frame, ref_latent, anchor_latent, prompt,
    ref_blend, strength, steps, guidance, seed, color_blend, color_channels
):
    """Generate with reference blending AND latent color matching."""
    frame_latent = pipe.encode(frame)

    # Step 1: Blend with reference in latent space (semantic style transfer)
    blended_latent = ref_blend * ref_latent + (1 - ref_blend) * frame_latent

    # Step 2: Latent color matching to anchor (maintains color consistency)
    if anchor_latent is not None and color_blend > 0:
        blended_latent = match_color_latent(
            blended_latent, anchor_latent,
            color_channels=color_channels,
            blend=color_blend
        )

    # Step 3: Generate from color-matched blended latent
    img, out_latent = pipe.generate_from_latent(
        blended_latent, prompt, strength=strength,
        num_steps=steps, guidance=guidance, seed=seed,
    )
    return img, out_latent


# Parse color channels argument
color_ch = tuple(int(x) for x in args.color_channels.split(","))

frames, fps = load_video(args.input, max_frames=args.max_frames)
ref_img = Image.open(args.ref).convert("RGB").resize(frames[0].size, Image.LANCZOS)

with GenerationContext(args.output) as gen:
    gen.update(
        preset="native_v2v_latent_color",
        pipeline="native-bfl",
        input=args.input,
        prompt=args.prompt,
        ref=args.ref,
        ref_blend=args.ref_blend,
        strength=args.strength,
        prev_blend=args.prev_blend,
        color_blend=args.color_blend,
        color_channels=args.color_channels,
        seed=args.seed,
        model=args.model,
        steps=args.steps,
        guidance=args.guidance,
    )
    gen.fps = fps

    pipe = NativePipeline(model_name=args.model)

    # Encode reference to latent ONCE
    ref_latent = pipe.encode(ref_img)

    output = []
    prev_gen = None
    anchor_latent = None  # First frame's output latent for color consistency

    for i, frame in enumerate(tqdm(frames, desc="NativeLatentColor")):
        # Temporal blend with previous generation (pixel space; resize: decoded
        # size can differ from frame size on 1080p/non-/16 inputs)
        if i > 0 and prev_gen:
            frame = blend(prev_gen.resize(frame.size), frame, args.prev_blend)

        # Generate with latent-space reference + color matching
        img, out_latent = generate_with_latent_ref_and_color(
            pipe, frame, ref_latent, anchor_latent, args.prompt,
            args.ref_blend, args.strength, args.steps, args.guidance,
            args.seed, args.color_blend, color_ch
        )

        # First frame becomes color anchor
        if i == 0:
            anchor_latent = out_latent.clone()

        output.append(img)
        prev_gen = img

        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
