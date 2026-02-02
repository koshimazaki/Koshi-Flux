#!/usr/bin/env python3
"""Klein V2V Latent Reference with Latent-Space Color Matching.

Same as v2v_latent_ref but uses latent-space color matching instead of pixel LAB.
This should prevent the "fainted" (washed out) colors that pixel LAB causes.

Latent color matching only adjusts channels 32-63 (color/lighting) while
preserving structure channels (0-31) and texture channels (64-127).
"""
import argparse
import torch
from klein_utils import (
    load_video, blend, get_pipeline, clear_cuda, tqdm, Image,
    GenerationContext
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/v2v_latent_ref_LAB.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--ref", type=str, required=True, help="Reference image for style")
parser.add_argument("--ref-blend", type=float, default=0.3, help="Latent blend: 0.3 = 30%% ref")
parser.add_argument("--strength", "-s", type=float, default=0.25, help="Generation strength")
parser.add_argument("--prev-blend", type=float, default=0.3, help="Temporal blend with prev gen")
parser.add_argument("--color-blend", type=float, default=0.7, help="Latent color match strength")
parser.add_argument("--color-channels", type=str, default="32,64", help="Color channel range")
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def match_color_latent(
    current_latent: torch.Tensor,
    reference_latent: torch.Tensor,
    color_channels: tuple = (32, 64),
    blend: float = 0.7,
) -> torch.Tensor:
    """Match color statistics in 128-channel latent space.

    Unlike pixel LAB which shifts entire histogram, this only adjusts
    mean/std of color channels while preserving structure.

    FLUX.2 channel semantics (128 channels):
        0-31:   Structure (preserve - don't touch)
        32-47:  Color palette
        48-63:  Lighting/atmosphere
        64-95:  Texture/detail (preserve)
        96-127: Context/transitions (preserve)

    Args:
        current_latent: Current frame latent (B, 128, H, W)
        reference_latent: Anchor frame latent (B, 128, H, W)
        color_channels: Channel range for color matching (start, end)
        blend: 0.0 = no match, 1.0 = full match (0.7 recommended)

    Returns:
        Color-matched latent with same structure
    """
    if blend <= 0:
        return current_latent

    result = current_latent.clone()
    start, end = color_channels

    # Extract color channel slices
    curr_color = current_latent[:, start:end]
    ref_color = reference_latent[:, start:end]

    # Compute per-channel statistics
    curr_mean = curr_color.mean(dim=(2, 3), keepdim=True)
    curr_std = curr_color.std(dim=(2, 3), keepdim=True) + 1e-6
    ref_mean = ref_color.mean(dim=(2, 3), keepdim=True)
    ref_std = ref_color.std(dim=(2, 3), keepdim=True) + 1e-6

    # Normalize to zero mean, unit variance, then rescale to reference
    normalized = (curr_color - curr_mean) / curr_std
    matched = normalized * ref_std + ref_mean

    # Blend matched with original (allows partial matching)
    result[:, start:end] = curr_color * (1 - blend) + matched * blend

    return result


def generate_with_latent_ref_and_color(
    pipe, frame, ref_latent, anchor_latent, prompt,
    ref_blend, strength, seed, color_blend, color_channels
):
    """Generate with reference blending AND latent color matching."""
    frame_latent = pipe._encode_to_latent(frame)

    # Step 1: Blend with reference in latent space (semantic style transfer)
    blended_latent = ref_blend * ref_latent + (1 - ref_blend) * frame_latent

    # Step 2: Apply latent color matching to anchor (maintains color consistency)
    if anchor_latent is not None and color_blend > 0:
        blended_latent = match_color_latent(
            blended_latent, anchor_latent,
            color_channels=color_channels,
            blend=color_blend
        )

    # Step 3: Generate from color-matched blended latent
    img, out_latent = pipe._generate_motion_frame(
        prev_latent=blended_latent, prompt=prompt, motion_params={},
        width=frame.width, height=frame.height,
        num_inference_steps=4, guidance_scale=1.0,
        strength=strength, seed=seed
    )
    return img, out_latent


# Parse color channels argument
color_ch = tuple(int(x) for x in args.color_channels.split(","))

frames, fps = load_video(args.input, max_frames=args.max_frames)
ref_img = Image.open(args.ref).convert("RGB").resize(frames[0].size, Image.LANCZOS)

with GenerationContext(args.output) as gen:
    gen.update(
        preset="v2v_latent_ref_LAB",
        input=args.input,
        prompt=args.prompt,
        ref=args.ref,
        ref_blend=args.ref_blend,
        strength=args.strength,
        prev_blend=args.prev_blend,
        color_blend=args.color_blend,
        color_channels=args.color_channels,
        seed=args.seed,
        model="flux.2-klein-4b",
        steps=4,
    )
    gen.fps = fps

    pipe = get_pipeline()

    # Encode reference to latent ONCE
    ref_latent = pipe._encode_to_latent(ref_img)

    output = []
    prev_gen = None
    anchor_latent = None  # First frame's latent for color consistency

    for i, frame in enumerate(tqdm(frames, desc="LatentRefLAB")):
        # Temporal blend with previous generation (pixel space)
        if i > 0 and prev_gen:
            frame = blend(prev_gen, frame, args.prev_blend)

        # Generate with latent-space reference + color matching
        img, out_latent = generate_with_latent_ref_and_color(
            pipe, frame, ref_latent, anchor_latent, args.prompt,
            args.ref_blend, args.strength, args.seed,
            args.color_blend, color_ch
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
