#!/usr/bin/env python3
"""
FLUX.2 Deforum Pipeline Test Script
Run on RunPod with RTX 5090

Usage:
    python test_flux2.py
    python test_flux2.py --single  # Single image only
    python test_flux2.py --frames 30  # Custom frame count
"""

import argparse
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add paths
workspace = os.environ.get("WORKSPACE", "/workspace")
sys.path.insert(0, f"{workspace}/flux2-main/src")
sys.path.insert(0, f"{workspace}/Deforum2026/flux/src")


def test_imports():
    """Test all required imports."""
    logger.info("Testing imports...")

    import torch
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    import transformers
    logger.info(f"  Transformers: {transformers.__version__}")

    try:
        from flux2.util import load_flow_model, load_ae, load_mistral_small_embedder
        from flux2.sampling import get_schedule, denoise
        logger.info("  flux2: OK")
    except ImportError as e:
        logger.error(f"  flux2: FAILED - {e}")
        return False

    try:
        from flux_motion.flux2 import Flux2Pipeline, Flux2MotionEngine
        from flux_motion import create_flux2_pipeline
        logger.info("  flux_motion: OK")
    except ImportError as e:
        logger.error(f"  flux_motion: FAILED - {e}")
        return False

    logger.info("All imports OK!\n")
    return True


def test_single_image(prompt: str = "a mystical forest with glowing mushrooms"):
    """Generate a single test image."""
    logger.info("=" * 50)
    logger.info("Single Image Generation Test")
    logger.info("=" * 50)

    from flux_motion.flux2 import Flux2Pipeline

    pipe = Flux2Pipeline(
        model_name="flux.2-dev",
        device="cuda",
        offload=True,
    )

    logger.info(f"Prompt: {prompt}")
    logger.info("Generating...")

    image = pipe.generate_single_frame(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=42,
    )

    output_path = Path(workspace) / "outputs" / "test_single.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    logger.info(f"Saved: {output_path}")
    return True


def test_animation(num_frames: int = 24):
    """Generate a short test animation."""
    logger.info("=" * 50)
    logger.info("Animation Generation Test")
    logger.info("=" * 50)

    from flux_motion import create_flux2_pipeline

    pipe = create_flux2_pipeline(device="cuda", offload=True)

    prompts = {0: "a serene mountain landscape at sunrise"}
    motion_params = {
        "zoom": f"0:(1.0), {num_frames}:(1.05)",
        "angle": f"0:(0), {num_frames//2}:(3), {num_frames}:(0)",
    }

    logger.info(f"Frames: {num_frames}")
    logger.info("Generating...")

    output_path = pipe.generate_animation(
        prompts=prompts,
        motion_params=motion_params,
        num_frames=num_frames,
        width=1024,
        height=1024,
        fps=24,
        output_path=f"{workspace}/outputs/test_animation.mp4",
        seed=42,
    )

    logger.info(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test FLUX.2 Deforum Pipeline")
    parser.add_argument("--single", action="store_true", help="Single image test only")
    parser.add_argument("--frames", type=int, default=24, help="Number of frames")
    parser.add_argument("--skip-imports", action="store_true", help="Skip import test")
    args = parser.parse_args()

    logger.info("\nFLUX.2 Deforum Pipeline Test\n")

    if not args.skip_imports:
        if not test_imports():
            sys.exit(1)

    if not test_single_image():
        sys.exit(1)

    if not args.single:
        if not test_animation(args.frames):
            sys.exit(1)

    logger.info("\nALL TESTS PASSED!")


if __name__ == "__main__":
    main()
