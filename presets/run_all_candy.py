#!/usr/bin/env python3
"""Batch runner for Candy-Grey videos with multiple configs + enforced JSON metadata.

Runs all 3 videos with 2 settings each (6 total):
- creature: pure generation (video_blend=0)
- blend40: 40% video shows through

Usage:
    python run_all_candy.py
"""
from diffusers import Flux2KleinPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

from klein_utils import GenerationContext  # ENFORCED: Always save settings JSON

PROMPT = "a vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal."
SIZE = 1024
STEPS = 4
SEED = 42

# Temporal settings (warp85 preset)
EMA_ALPHA = 0.85
WARP_BLEND = 0.85
POST_LAB = 0.9

VIDEOS = [
    ("/workspace/Candy-Grey/1024/1_1024.mp4", "1"),
    ("/workspace/Candy-Grey/1024/2_1024.mp4", "2"),
    ("/workspace/Candy-Grey/1024/3_1024.mp4", "3"),
]

CONFIGS = [
    ("creature", 0.0),   # Pure generation
    ("blend40", 0.4),    # 40% video shows through
]


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


def process_video(pipe, input_path, output_path, video_blend, cfg_name):
    """Process single video with given settings."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    input_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
        input_frames.append(frame)
    cap.release()

    # ENFORCED: GenerationContext guarantees JSON is saved (even on crash)
    with GenerationContext(output_path) as gen:
        gen.update(
            preset=cfg_name,
            input=input_path,
            prompt=PROMPT,
            model="flux.2-klein-4b",
            size=SIZE,
            steps=STEPS,
            seed=SEED,
            ema_alpha=EMA_ALPHA,
            warp_blend=WARP_BLEND,
            post_lab=POST_LAB,
            video_blend=video_blend,
        )
        gen.fps = fps

        generator = torch.Generator(device="cuda").manual_seed(SEED)
        output_frames = []
        ema_frame = None
        anchor = None

        for i, inp in enumerate(tqdm(input_frames, desc=os.path.basename(output_path))):
            if ema_frame is not None:
                ema_uint8 = np.clip(ema_frame, 0, 255).astype(np.uint8)
                init = warp(ema_uint8, inp, WARP_BLEND)
            else:
                init = inp

            result = pipe(
                prompt=PROMPT,
                image=Image.fromarray(init),
                num_inference_steps=STEPS,
                height=SIZE,
                width=SIZE,
                generator=generator,
            ).images[0]

            out = np.array(result)

            if anchor is None:
                anchor = out.copy()

            if ema_frame is None:
                ema_frame = out.astype(np.float32)
            else:
                ema_frame = EMA_ALPHA * out.astype(np.float32) + (1 - EMA_ALPHA) * ema_frame

            out = cv2.addWeighted(match_lab(out, anchor), POST_LAB, out, 1 - POST_LAB, 0)

            if video_blend > 0:
                out = cv2.addWeighted(out, 1 - video_blend, inp, video_blend, 0)

            output_frames.append(out)

            if i % 50 == 0:
                torch.cuda.empty_cache()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (SIZE, SIZE))
        for f in output_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()

        gen.frames = [Image.fromarray(f) for f in output_frames]


def main():
    """Run batch processing."""
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4b", torch_dtype=torch.bfloat16
    ).to("cuda")

    os.makedirs("/workspace/outputs/candy", exist_ok=True)

    for vid_path, vid_name in VIDEOS:
        for cfg_name, video_blend in CONFIGS:
            output_path = f"/workspace/outputs/candy/{vid_name}_{cfg_name}_1024.mp4"
            process_video(pipe, vid_path, output_path, video_blend, cfg_name)


if __name__ == "__main__":
    main()
