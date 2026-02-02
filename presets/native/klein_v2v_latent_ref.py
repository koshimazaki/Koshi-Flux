#!/usr/bin/env python3
"""Klein V2V Latent Reference - Blend reference in latent space for stronger style influence.

Unlike pixel blending, latent blending merges style at the semantic level.
"""
import argparse
import torch
from klein_utils import (
    load_video, match_color_lab, blend, get_pipeline, clear_cuda, tqdm, Image,
    GenerationContext
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/v2v_latent_ref.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--ref", type=str, required=True, help="Reference image for style")
parser.add_argument("--ref-blend", type=float, default=0.3, help="Latent blend: 0.3 = 30%% ref, 70%% video")
parser.add_argument("--strength", "-s", type=float, default=0.25, help="Generation strength")
parser.add_argument("--prev-blend", type=float, default=0.3, help="Temporal blend with previous gen")
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def generate_with_latent_ref(pipe, frame, ref_latent, prompt, ref_blend, strength, seed):
    """Generate with reference blended in latent space."""
    frame_latent = pipe._encode_to_latent(frame)

    # Blend in latent space (semantic blending, not pixel)
    blended_latent = ref_blend * ref_latent + (1 - ref_blend) * frame_latent

    # Generate from blended latent with strength
    img, _ = pipe._generate_motion_frame(
        prev_latent=blended_latent, prompt=prompt, motion_params={},
        width=frame.width, height=frame.height,
        num_inference_steps=4, guidance_scale=1.0,
        strength=strength, seed=seed
    )
    return img


frames, fps = load_video(args.input, max_frames=args.max_frames)
ref_img = Image.open(args.ref).convert("RGB").resize(frames[0].size, Image.LANCZOS)

with GenerationContext(args.output) as gen:
    gen.update(
        preset="v2v_latent_ref",
        input=args.input,
        prompt=args.prompt,
        ref=args.ref,
        ref_blend=args.ref_blend,
        strength=args.strength,
        prev_blend=args.prev_blend,
        seed=args.seed,
        model="flux.2-klein-4b",
        steps=4,
    )
    gen.fps = fps

    pipe = get_pipeline()

    # Encode reference to latent ONCE (reused for all frames)
    ref_latent = pipe._encode_to_latent(ref_img)

    output, prev_gen, anchor = [], None, None

    for i, frame in enumerate(tqdm(frames, desc="LatentRef")):
        # Temporal blend with previous generation
        if i > 0 and prev_gen:
            frame = blend(prev_gen, frame, args.prev_blend)

        # Generate with latent-space reference blending
        img = generate_with_latent_ref(pipe, frame, ref_latent, args.prompt,
                                        args.ref_blend, args.strength, args.seed)

        # Color consistency
        anchor = img if i == 0 else anchor
        img = img if i == 0 else match_color_lab(img, anchor)

        output.append(img)
        prev_gen = img

        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
