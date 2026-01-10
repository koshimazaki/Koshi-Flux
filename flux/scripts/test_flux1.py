#!/usr/bin/env python3
"""
FLUX.1 Deforum Pipeline Test
Works on RTX 5090 (32GB VRAM)
"""

import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

workspace = os.environ.get("WORKSPACE", "/workspace")
sys.path.insert(0, f"{workspace}/flux/src")
sys.path.insert(0, f"{workspace}/Deforum2026/core/src")


def test_single_image():
    """Generate a single test image with FLUX.1."""
    logger.info("=" * 50)
    logger.info("FLUX.1 Single Image Test")
    logger.info("=" * 50)

    from deforum_flux import create_flux1_pipeline

    pipe = create_flux1_pipeline(device="cuda", offload=True)

    prompt = "a mystical forest with glowing mushrooms, ethereal lighting"
    logger.info(f"Prompt: {prompt}")
    logger.info("Generating (first run downloads ~32GB models)...")

    image = pipe.generate_single_frame(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=42,
    )

    output_path = f"{workspace}/outputs/flux1_test.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

    logger.info(f"Saved: {output_path}")
    logger.info("SUCCESS!")
    return True


def test_animation():
    """Generate a short animation with FLUX.1."""
    logger.info("=" * 50)
    logger.info("FLUX.1 Animation Test")
    logger.info("=" * 50)

    from deforum_flux import create_flux1_pipeline

    pipe = create_flux1_pipeline(device="cuda", offload=True)

    prompts = {0: "a serene mountain landscape at sunrise"}
    motion_params = {
        "zoom": "0:(1.0), 30:(1.03)",
        "angle": "0:(0), 15:(2), 30:(0)",
    }

    logger.info("Generating 30 frame animation...")

    output_path = pipe.generate_animation(
        prompts=prompts,
        motion_params=motion_params,
        num_frames=30,
        width=1024,
        height=1024,
        fps=24,
        output_path=f"{workspace}/outputs/flux1_animation.mp4",
        seed=42,
    )

    logger.info(f"Saved: {output_path}")
    logger.info("SUCCESS!")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test FLUX.1 Pipeline")
    parser.add_argument("--animation", action="store_true", help="Also test animation")
    args = parser.parse_args()

    logger.info("\nFLUX.1 Deforum Pipeline Test\n")

    if not test_single_image():
        sys.exit(1)

    if args.animation:
        if not test_animation():
            sys.exit(1)

    logger.info("\nALL TESTS PASSED!")


if __name__ == "__main__":
    main()
