#!/usr/bin/env python3
"""
Klein V2V Deforum - Video (Light) mode + Motion Engine schedules.

Combines:
- Simple pixel ops (optical flow, warp, LAB match) from Video (Light)
- Motion engine transforms (zoom, rotate, translate) in latent space
- BFL native denoising

Usage:
    python klein_v2v_deforum.py -i input.mp4 -p "oil painting" -o output.mp4

    # With motion schedules (Deforum format)
    python klein_v2v_deforum.py -i input.mp4 -p "cyberpunk city" \\
        --zoom "0:(1.0), 60:(1.05)" \\
        --angle "0:(0), 30:(5), 60:(0)" \\
        -o output.mp4
"""
import argparse
from klein_utils import (
    load_video, match_color_lab, optical_flow, warp,
    get_pipeline, clear_cuda,
    GenerationContext  # ENFORCED: Always save settings JSON
)
from tqdm import tqdm
from flux_motion.shared import FluxParameterAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Klein V2V with Deforum motion schedules")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="outputs/v2v_deforum.mp4")
    parser.add_argument("--prompt", "-p", required=True, help="Style prompt")
    parser.add_argument("--strength", "-s", type=float, default=0.3, help="Denoise strength")
    parser.add_argument("--init-from-input", action="store_true", help="Use lower strength for frame 0")
    parser.add_argument("--max-frames", "-n", type=int, help="Limit frames")
    parser.add_argument("--seed", type=int, default=42)

    # Motion schedules (Deforum format)
    parser.add_argument("--zoom", type=str, default="0:(1.0)", help="Zoom schedule")
    parser.add_argument("--angle", type=str, default="0:(0)", help="Rotation schedule (degrees)")
    parser.add_argument("--translation-x", type=str, default="0:(0)", help="X translation schedule")
    parser.add_argument("--translation-y", type=str, default="0:(0)", help="Y translation schedule")
    parser.add_argument("--translation-z", type=str, default="0:(0)", help="Z translation schedule (depth)")

    return parser.parse_args()


def generate_with_motion(pipe, frame, prompt, strength, seed, motion_params, width, height):
    """Generate frame with motion transforms applied in latent space."""
    latent = pipe._encode_to_latent(frame)
    img, new_latent = pipe._generate_motion_frame(
        prev_latent=latent,
        prompt=prompt,
        motion_params=motion_params,
        width=width,
        height=height,
        num_inference_steps=4,
        guidance_scale=1.0,
        strength=strength,
        seed=seed
    )
    return img


def main():
    args = parse_args()

    frames, fps = load_video(args.input, max_frames=args.max_frames)
    num_frames = len(frames)
    width, height = frames[0].size

    # ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
    with GenerationContext(args.output) as gen:
        gen.update(
            preset="v2v_deforum",
            input=args.input,
            prompt=args.prompt,
            strength=args.strength,
            init_from_input=args.init_from_input,
            zoom=args.zoom,
            angle=args.angle,
            translation_x=args.translation_x,
            translation_y=args.translation_y,
            translation_z=args.translation_z,
            seed=args.seed,
            model="flux.2-klein-4b",
            steps=4,
        )
        gen.fps = fps

        # Parse motion schedules
        param_adapter = FluxParameterAdapter()
        motion_params = {
            "zoom": args.zoom,
            "angle": args.angle,
            "translation_x": args.translation_x,
            "translation_y": args.translation_y,
            "translation_z": args.translation_z,
            "prompts": {0: args.prompt},
        }
        motion_frames = param_adapter.convert_deforum_params(motion_params, num_frames)

        pipe = get_pipeline()
        output = []
        prev_input = frames[0]
        prev_gen = None

        for i, (frame, motion_frame) in enumerate(tqdm(
            zip(frames, motion_frames), total=num_frames, desc="V2V Deforum"
        )):
            motion_dict = motion_frame.to_dict()

            if i == 0:
                # First frame - higher strength for style transfer
                init_strength = 0.7 if args.init_from_input else 0.95
                img = generate_with_motion(
                    pipe, frame, args.prompt, init_strength, args.seed,
                    motion_dict, width, height
                )
            else:
                # Video (Light) pixel ops: flow + warp
                warped = warp(prev_gen, optical_flow(prev_input, frame))

                # Generate with motion engine transforms
                img = generate_with_motion(
                    pipe, warped, args.prompt, args.strength, args.seed,
                    motion_dict, width, height
                )

                # LAB color match to anchor
                img = match_color_lab(img, output[0])

            output.append(img)
            prev_input = frame
            prev_gen = img

            if i % 20 == 0:
                clear_cuda()

        gen.frames = output
        gen.save_video()


if __name__ == "__main__":
    main()
