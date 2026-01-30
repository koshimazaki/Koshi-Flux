#!/usr/bin/env python3
"""
Klein Hybrid Deforum - Full pipeline with motion engine + FeedbackProcessor.

Combines:
1. Input video for structure guidance (like V2V)
2. Motion engine transforms (zoom, rotate, translate schedules)
3. FeedbackProcessor with burn/blur detection
4. Noise coherence for temporal consistency

Usage:
    python klein_hybrid_deforum.py -i input.mp4 -p "oil painting style" -o output.mp4

    # With motion schedules (Deforum format)
    python klein_hybrid_deforum.py -i input.mp4 -p "cyberpunk city" \\
        --zoom "0:(1.0), 60:(1.05)" \\
        --angle "0:(0), 30:(5), 60:(0)" \\
        -o output.mp4

    # With burn/blur detection
    python klein_hybrid_deforum.py -i input.mp4 -p "watercolor" \\
        --detect-issues --adaptive-strength -o output.mp4
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from klein_utils import (
    load_video, save_video, save_metadata,
    optical_flow, warp, get_pipeline, clear_cuda
)

from flux_motion.feedback import FeedbackProcessor, FeedbackConfig
from flux_motion.shared import FluxParameterAdapter
from flux_motion.flux2.config import AdaptiveCorrectionConfig
from flux_motion.shared.noise_coherence import WarpedNoiseManager, NoiseCoherenceConfig

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Klein Hybrid Deforum - V2V with motion engine + full feedback pipeline"
    )
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="outputs/hybrid_deforum.mp4")
    parser.add_argument("--prompt", "-p", required=True, help="Style prompt")
    parser.add_argument("--max-frames", "-n", type=int, help="Limit frames")

    parser.add_argument("--strength", "-s", type=float, default=0.35,
                        help="Denoise strength (0.2-0.5 recommended)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--video-blend", type=float, default=0.5,
                        help="Input video influence (0=ignore, 1=pure V2V)")
    parser.add_argument("--flow-blend", type=float, default=0.4,
                        help="Optical flow warp influence")
    parser.add_argument("--temporal-blend", type=float, default=0.2,
                        help="Previous generation influence")

    parser.add_argument("--zoom", type=str, default="0:(1.0)",
                        help="Zoom schedule, e.g., '0:(1.0), 60:(1.05)'")
    parser.add_argument("--angle", type=str, default="0:(0)",
                        help="Rotation schedule, e.g., '0:(0), 30:(5)'")
    parser.add_argument("--translation-x", type=str, default="0:(0)")
    parser.add_argument("--translation-y", type=str, default="0:(0)")
    parser.add_argument("--translation-z", type=str, default="0:(0)")

    parser.add_argument("--sharpen", type=float, default=0.15)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--color-mode", type=str, default="LAB",
                        choices=["LAB", "RGB", "HSV", "None"])

    parser.add_argument("--detect-issues", action="store_true",
                        help="Enable burn/blur detection")
    parser.add_argument("--adaptive-strength", action="store_true",
                        help="Reduce strength on high-motion frames")
    parser.add_argument("--burn-threshold", type=float, default=0.1)
    parser.add_argument("--blur-threshold", type=float, default=0.1)

    parser.add_argument("--warp-noise", action="store_true", default=True)
    parser.add_argument("--noise-blend", type=float, default=0.7)

    return parser.parse_args()


def blend_images(img1: Image.Image, img2: Image.Image, alpha: float) -> Image.Image:
    """Blend: result = alpha * img1 + (1-alpha) * img2"""
    if alpha <= 0:
        return img2
    if alpha >= 1:
        return img1
    arr = alpha * np.array(img1).astype(np.float32) + (1 - alpha) * np.array(img2).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    frames, fps = load_video(args.input, max_frames=args.max_frames)
    num_frames = len(frames)
    width, height = frames[0].size
    tqdm.write(f"Hybrid Deforum: {num_frames} frames @ {width}x{height}")
    tqdm.write(f"  Strength: {args.strength}, Video blend: {args.video_blend}")
    tqdm.write(f"  Motion: zoom={args.zoom}, angle={args.angle}")

    pipe = get_pipeline()

    feedback_processor = FeedbackProcessor()
    feedback_config = FeedbackConfig(
        color_mode=args.color_mode,
        sharpen_amount=args.sharpen,
        noise_amount=args.noise,
        noise_type="perlin",
        contrast_boost=args.contrast,
    )

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

    correction_config = AdaptiveCorrectionConfig(
        adaptive_strength=args.adaptive_strength,
        burn_detection=args.detect_issues,
        blur_detection=args.detect_issues,
        burn_threshold=args.burn_threshold,
        blur_threshold=args.blur_threshold,
    )

    noise_config = NoiseCoherenceConfig(
        warp_noise=args.warp_noise,
        warp_blend=args.noise_blend,
        use_slerp=True,
        slerp_strength=0.2,
    )
    noise_manager = WarpedNoiseManager(config=noise_config, seed=args.seed)

    output_frames = []
    prev_latent = None
    prev_gen = None
    prev_input = None
    anchor_frame = None
    detection_history = []

    mode_str = "detection" if args.detect_issues else "standard"
    tqdm.write(f"\nGenerating with {mode_str} mode...")

    for i, (input_frame, motion_frame) in enumerate(tqdm(
        zip(frames, motion_frames), total=num_frames, desc="Hybrid"
    )):
        motion_dict = motion_frame.to_dict()
        frame_seed = args.seed + i

        effective_strength = args.strength
        if args.adaptive_strength and i > 0:
            effective_strength = correction_config.compute_adaptive_strength(motion_dict)

        if i == 0:
            source = input_frame
            latent = pipe._encode_to_latent(source)
            img, latent = pipe._generate_motion_frame(
                prev_latent=latent,
                prompt=args.prompt,
                motion_params={},
                width=width,
                height=height,
                num_inference_steps=4,
                guidance_scale=1.0,
                strength=0.7,
                seed=frame_seed,
            )
            anchor_frame = img

        else:
            flow = optical_flow(prev_input, input_frame)
            warped_prev = warp(prev_gen, flow)

            video_guided = blend_images(input_frame, warped_prev, args.video_blend)
            temporal_blended = blend_images(prev_gen, video_guided, args.temporal_blend)

            blended_latent = pipe._encode_to_latent(temporal_blended)

            transformed_latent = pipe.motion_engine.apply_motion(
                blended_latent, motion_dict
            )

            transformed_image = pipe._decode_latent(transformed_latent)

            image_np = np.array(transformed_image)
            anchor_np = np.array(anchor_frame)
            prev_np = np.array(prev_gen) if prev_gen else None

            if args.detect_issues:
                processed_np, detection = feedback_processor.process_with_detection(
                    image_np, anchor_np, prev_np,
                    config=feedback_config,
                    burn_threshold=args.burn_threshold,
                    blur_threshold=args.blur_threshold,
                    auto_correct=True,
                )
                detection_history.append(detection)
                if detection.needs_burn_correction or detection.needs_blur_correction:
                    tqdm.write(f"  Frame {i}: {detection}")
            else:
                processed_np = feedback_processor.process(image_np, anchor_np, feedback_config)

            processed_image = Image.fromarray(processed_np)
            processed_latent = pipe._encode_to_latent(processed_image)

            img, latent = pipe._generate_motion_frame(
                prev_latent=processed_latent,
                prompt=args.prompt,
                motion_params={},
                width=width,
                height=height,
                num_inference_steps=4,
                guidance_scale=1.0,
                strength=effective_strength,
                seed=frame_seed,
                noise_manager=noise_manager,
            )

        output_frames.append(img)
        prev_latent = latent
        prev_gen = img
        prev_input = input_frame

        if i % 20 == 0:
            clear_cuda()

    save_video(output_frames, args.output, fps)

    meta = {
        "preset": "hybrid_deforum",
        "input": args.input,
        "prompt": args.prompt,
        "strength": args.strength,
        "video_blend": args.video_blend,
        "flow_blend": args.flow_blend,
        "temporal_blend": args.temporal_blend,
        "zoom": args.zoom,
        "angle": args.angle,
        "translation_x": args.translation_x,
        "translation_y": args.translation_y,
        "sharpen": args.sharpen,
        "noise": args.noise,
        "color_mode": args.color_mode,
        "detect_issues": args.detect_issues,
        "adaptive_strength": args.adaptive_strength,
        "seed": args.seed,
        "frames": num_frames,
        "fps": fps,
    }

    if detection_history:
        burn_frames = sum(1 for d in detection_history if d.needs_burn_correction)
        blur_frames = sum(1 for d in detection_history if d.needs_blur_correction)
        meta["detection_summary"] = {
            "burn_corrections": burn_frames,
            "blur_corrections": blur_frames,
            "avg_burn_score": float(np.mean([d.burn_score for d in detection_history])),
            "avg_blur_score": float(np.mean([d.blur_score for d in detection_history])),
        }
        tqdm.write(f"\nDetection: {burn_frames} burn, {blur_frames} blur corrections")

    save_metadata(args.output, **meta)
    tqdm.write(f"Done: {args.output}")


if __name__ == "__main__":
    main()
