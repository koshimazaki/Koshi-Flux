#!/usr/bin/env python3
"""Klein V2V Latent Reference (NATIVE BFL SDK) - blend reference in latent space.

Native twin of hybrid-v2v/klein_v2v_latent_ref.py: identical CLI, defaults, and
loop - only the pipeline differs (pure BFL AutoEncoder + DiT). Latent blending
happens in the native 128-channel latent space before token packing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from klein_utils import (
    load_video, match_color_lab, blend, clear_cuda, tqdm, Image,
    GenerationContext
)
from native_utils import NativePipeline

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", default="outputs/native_v2v_latent_ref.mp4")
parser.add_argument("--prompt", "-p", required=True)
parser.add_argument("--ref", type=str, required=True, help="Reference image for style")
parser.add_argument("--ref-blend", type=float, default=0.3, help="Latent blend: 0.3 = 30%% ref, 70%% video")
parser.add_argument("--strength", "-s", type=float, default=0.25, help="Generation strength")
parser.add_argument("--prev-blend", type=float, default=0.3, help="Temporal blend with previous gen")
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model", default="flux.2-klein-4b",
                    choices=["flux.2-klein-4b", "flux.2-klein-9b"])
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--guidance", type=float, default=1.0)
args = parser.parse_args()


def generate_with_latent_ref(pipe, frame, ref_latent, prompt, ref_blend, strength,
                             steps, guidance, seed):
    """Generate with reference blended in native latent space."""
    frame_latent = pipe.encode(frame)

    # Blend in latent space (semantic blending, not pixel)
    blended_latent = ref_blend * ref_latent + (1 - ref_blend) * frame_latent

    # Generate from blended latent with strength
    img, _ = pipe.generate_from_latent(
        blended_latent, prompt, strength=strength,
        num_steps=steps, guidance=guidance, seed=seed,
    )
    return img


frames, fps = load_video(args.input, max_frames=args.max_frames)
ref_img = Image.open(args.ref).convert("RGB").resize(frames[0].size, Image.LANCZOS)

with GenerationContext(args.output) as gen:
    gen.update(
        preset="native_v2v_latent_ref",
        pipeline="native-bfl",
        input=args.input,
        prompt=args.prompt,
        ref=args.ref,
        ref_blend=args.ref_blend,
        strength=args.strength,
        prev_blend=args.prev_blend,
        seed=args.seed,
        model=args.model,
        steps=args.steps,
        guidance=args.guidance,
    )
    gen.fps = fps

    pipe = NativePipeline(model_name=args.model)

    # Encode reference to latent ONCE (reused for all frames)
    ref_latent = pipe.encode(ref_img)

    output, prev_gen, anchor = [], None, None

    for i, frame in enumerate(tqdm(frames, desc="NativeLatentRef")):
        # Temporal blend with previous generation (resize: decoded size can
        # differ from frame size on 1080p/non-/16 inputs)
        if i > 0 and prev_gen:
            frame = blend(prev_gen.resize(frame.size), frame, args.prev_blend)

        # Generate with latent-space reference blending
        img = generate_with_latent_ref(pipe, frame, ref_latent, args.prompt,
                                       args.ref_blend, args.strength,
                                       args.steps, args.guidance, args.seed)

        # Color consistency
        anchor = img if i == 0 else anchor
        img = img if i == 0 else match_color_lab(img, anchor)

        output.append(img)
        prev_gen = img

        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
