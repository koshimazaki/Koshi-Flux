#!/usr/bin/env python3
"""Path A: bind a LoRA on Klein via a *diffusers* FLUX.2/Klein pipeline.

This is the diffusers lane for LoRA. It loads a diffusers Klein/FLUX.2 pipeline
through :func:`klein_utils.get_diffusers_klein_pipeline` and binds the LoRA with
diffusers' PEFT integration (``KleinLoRAManager(backend="diffusers")``), then
runs a single text-to-image generation as a smoke test.

STATUS: diffusers does not yet ship a FLUX.2/Klein DiT pipeline (only the Klein
VAE is loadable via ``AutoencoderKL``; the DiT runs through the native BFL
``flux2`` SDK). Until such a class lands upstream this script fails fast with a
clear, actionable error naming the missing dependency. For a working LoRA today,
use the native lane instead::

    python presets/native/klein_v2v_audio.py -i in.mp4 -a music.wav \
        -o out.mp4 -p "cyber flowers" --lora cyber_flowers.safetensors \
        --lora-backend native

Usage (once diffusers supports Klein):
    python presets/diffusers/klein_lora_diffusers.py \
        --lora cyber_flowers.safetensors --lora-strength 0.8 \
        -p "a field of cyber flowers" -o outputs/klein_lora_diffusers.png
"""
import argparse
import sys
from pathlib import Path

# Add parent dir for klein_utils import (done lazily in main()).
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-id",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HuggingFace repo for the diffusers Klein pipeline",
    )
    p.add_argument("--prompt", "-p", required=True, help="Generation prompt")
    p.add_argument("--lora", required=True, help="LoRA path or HF repo to bind")
    p.add_argument("--lora-strength", type=float, default=0.8)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=4, help="Klein default is 4")
    p.add_argument("--guidance", type=float, default=1.0, help="Klein default 1.0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", "-o", default="outputs/klein_lora_diffusers.png")
    return p.parse_args()


def main():
    args = parse_args()

    # Imported here so --help works even if diffusers/cv2 are unavailable, and so
    # the guarded NotImplementedError/ImportError surfaces at run time.
    import torch
    from klein_utils import GenerationContext, get_diffusers_klein_pipeline

    # Raises a clear, actionable error if diffusers has no Klein pipeline yet.
    pipe = get_diffusers_klein_pipeline(
        repo_id=args.repo_id,
        lora=args.lora,
        lora_strength=args.lora_strength,
        device=args.device,
    )

    out_path = Path(args.output)
    with GenerationContext(str(out_path)) as gen:
        gen.update(
            preset="klein_lora_diffusers",
            pipeline="diffusers",
            model=args.repo_id,
            prompt=args.prompt,
            lora=args.lora,
            lora_strength=args.lora_strength,
            lora_backend="diffusers",
            lora_applied=getattr(pipe, "_lora_manager", None) is not None,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        result = pipe(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )
        image = result.images[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        # Single still: store it as a 1-frame "video" record for the JSON.
        gen.frames = [image]
        print(f"[klein_lora_diffusers] saved {out_path}")


if __name__ == "__main__":
    main()
