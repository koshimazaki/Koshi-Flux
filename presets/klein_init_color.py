#!/usr/bin/env python3
"""Klein V2V with Init Image Color Reference + Enforced JSON metadata

Pipeline concept:
- Colors from init image (FLUX native output)
- Shape/motion from video
- Style from prompt

Usage:
    python klein_init_color.py

Requires: diffusers (main branch), torch, opencv-python, pillow, tqdm
"""
from diffusers import Flux2KleinPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

from klein_utils import GenerationContext  # ENFORCED: Always save settings JSON

# === CONFIG ===
INIT_IMAGE = "/workspace/input/native1.png"  # Color reference (FLUX output)
INPUT_VIDEO = "/workspace/Candy-Grey/1024/1_1024.mp4"
OUTPUT_VIDEO = "/workspace/outputs/candy/1_initcolor_1024.mp4"

PROMPT = "a vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal."

SIZE = 1024
STEPS = 4
SEED = 42
MAX_FRAMES = None  # None = all frames

# Temporal settings (warp85 preset)
EMA_ALPHA = 0.85      # Smoothing between frames
WARP_BLEND = 0.85     # Optical flow warp strength
COLOR_MATCH = 0.9     # How much to match init image colors (0-1)
VIDEO_BLEND = 0.3     # How much original video shows through (0=none, 1=full)


def match_lab(source, reference):
    """Match source colors to reference using LAB space."""
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        src_mean, src_std = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-6
        ref_mean, ref_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-6
        src_lab[:, :, c] = (src_lab[:, :, c] - src_mean) * (ref_std / src_std) + ref_mean
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def warp(prev, curr, blend):
    """Warp previous frame to current using optical flow."""
    prev = prev.astype(np.uint8)
    curr = curr.astype(np.uint8)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 21, 3, 7, 1.5, 0
    )
    h, w = prev.shape[:2]
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)
    flow_map += flow
    warped = cv2.remap(prev, flow_map[:, :, 0], flow_map[:, :, 1], cv2.INTER_LINEAR)
    return cv2.addWeighted(warped, blend, curr, 1 - blend, 0)


def main():
    """Run V2V generation with init image color reference."""
    # Load video frames first to get fps
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    input_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
        input_frames.append(frame)
        if MAX_FRAMES and len(input_frames) >= MAX_FRAMES:
            break
    cap.release()

    # ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
    with GenerationContext(OUTPUT_VIDEO) as gen:
        gen.update(
            preset="init_color",
            input_video=INPUT_VIDEO,
            init_image=INIT_IMAGE,
            prompt=PROMPT,
            model="flux.2-klein-4b",
            size=SIZE,
            steps=STEPS,
            seed=SEED,
            ema_alpha=EMA_ALPHA,
            warp_blend=WARP_BLEND,
            color_match=COLOR_MATCH,
            video_blend=VIDEO_BLEND,
        )
        gen.fps = fps

        # Load pipeline
        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4b",
            torch_dtype=torch.bfloat16
        ).to("cuda")

        # Load init image for color reference
        init_img = cv2.cvtColor(cv2.imread(INIT_IMAGE), cv2.COLOR_BGR2RGB)
        init_img = cv2.resize(init_img, (SIZE, SIZE))

        # Generate
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        output_frames = []
        ema_frame = None

        for i, inp in enumerate(tqdm(input_frames, desc="InitColor")):
            # Warp previous output to follow video motion
            if ema_frame is not None:
                ema_uint8 = np.clip(ema_frame, 0, 255).astype(np.uint8)
                init = warp(ema_uint8, inp, WARP_BLEND)
            else:
                init = inp

            # Generate with Klein
            result = pipe(
                prompt=PROMPT,
                image=Image.fromarray(init),
                num_inference_steps=STEPS,
                height=SIZE,
                width=SIZE,
                generator=generator,
            ).images[0]

            out = np.array(result)

            # Update EMA for temporal smoothing
            if ema_frame is None:
                ema_frame = out.astype(np.float32)
            else:
                ema_frame = EMA_ALPHA * out.astype(np.float32) + (1 - EMA_ALPHA) * ema_frame

            # Match colors to init image
            out_matched = match_lab(out, init_img)
            out = cv2.addWeighted(out_matched, COLOR_MATCH, out, 1 - COLOR_MATCH, 0)

            # Blend with original video to preserve shape
            if VIDEO_BLEND > 0:
                out = cv2.addWeighted(out, 1 - VIDEO_BLEND, inp, VIDEO_BLEND, 0)

            output_frames.append(out)

            if i % 50 == 0:
                torch.cuda.empty_cache()

        # Save video
        os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (SIZE, SIZE))
        for f in output_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()

        # Convert numpy arrays to PIL for GenerationContext
        gen.frames = [Image.fromarray(f) for f in output_frames]

    return OUTPUT_VIDEO


if __name__ == "__main__":
    main()
