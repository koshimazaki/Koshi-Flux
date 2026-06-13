#!/usr/bin/env python3
"""Klein V2V Deforum (NATIVE BFL SDK) - Video (Light) mode + motion engine schedules.

Native twin of hybrid-v2v/klein_v2v_deforum.py: identical CLI, defaults, and
loop - only the pipeline differs (pure BFL AutoEncoder + DiT). Motion schedules
are applied in the 128-channel latent space by the same Flux2MotionEngine the
hybrid pipeline uses, so the A/B isolates the VAE/latent path.

Usage:
    python klein_v2v_deforum.py -i input.mp4 -p "oil painting" -o output.mp4

    # With motion schedules (Deforum format)
    python klein_v2v_deforum.py -i input.mp4 -p "cyberpunk city" \\
        --zoom "0:(1.0), 60:(1.05)" \\
        --angle "0:(0), 30:(5), 60:(0)" \\
        -o output.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from klein_utils import (
    load_video, match_color_lab, optical_flow, warp,
    clear_cuda,
    GenerationContext  # ENFORCED: Always save settings JSON
)
from native_utils import NativePipeline
from tqdm import tqdm
from flux_motion.shared import FluxParameterAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Klein V2V Deforum schedules (native BFL)")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="outputs/native_v2v_deforum.mp4")
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

    parser.add_argument("--model", default="flux.2-klein-4b",
                        choices=["flux.2-klein-4b", "flux.2-klein-9b"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    frames, fps = load_video(args.input, max_frames=args.max_frames)
    num_frames = len(frames)

    # ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
    with GenerationContext(args.output) as gen:
        gen.update(
            preset="native_v2v_deforum",
            pipeline="native-bfl",
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
            model=args.model,
            steps=args.steps,
            guidance=args.guidance,
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

        pipe = NativePipeline(model_name=args.model)
        output = []
        prev_input = frames[0]
        prev_gen = None

        for i, (frame, motion_frame) in enumerate(tqdm(
            zip(frames, motion_frames), total=num_frames, desc="NativeDeforum"
        )):
            motion_dict = motion_frame.to_dict()

            if i == 0:
                # First frame - higher strength for style transfer
                init_strength = 0.7 if args.init_from_input else 0.95
                z = pipe.encode(frame)
                img, _ = pipe.generate_from_latent(
                    z, args.prompt, strength=init_strength,
                    num_steps=args.steps, guidance=args.guidance,
                    seed=args.seed, motion_params=motion_dict,
                )
            else:
                # Video (Light) pixel ops: flow + warp. Resize prev_gen to the
                # flow's resolution - decoded size differs on 1080p/non-/16
                # inputs and warping a smaller image through a full-size flow
                # grid produces distorted feedback.
                warped = warp(prev_gen.resize(frame.size), optical_flow(prev_input, frame))

                # Generate with motion engine transforms (latent space)
                z = pipe.encode(warped)
                img, _ = pipe.generate_from_latent(
                    z, args.prompt, strength=args.strength,
                    num_steps=args.steps, guidance=args.guidance,
                    seed=args.seed, motion_params=motion_dict,
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
