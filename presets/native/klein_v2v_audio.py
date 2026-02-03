#!/usr/bin/env python3
"""Klein V2V Audio Reactive - Sync video generation to audio.

Maps audio features to generation parameters:
- Kick/Bass → Zoom (pulse on beat)
- Snare/Mid → Strength (style intensity on snare hits)
- Kick threshold → Prompt toggle (scene change)

Uses flux_motion.audio.extractor for robust audio analysis.

Usage:
    python klein_v2v_audio.py -i video.mp4 -a audio.wav -o output.mp4 -p "prompt"
"""
import argparse
import sys
import numpy as np
from pathlib import Path

# Add flux_motion to path
SCRIPT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "flux/src"))

from flux_motion.audio.extractor import AudioFeatureExtractor

from klein_utils import (
    load_video, blend, get_pipeline, clear_cuda, tqdm, Image,
    GenerationContext, match_color_latent
)
import torch
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True, help="Input video")
parser.add_argument("--audio", "-a", required=True, help="Audio file (wav/mp3)")
parser.add_argument("--output", "-o", default="outputs/v2v_audio.mp4")
parser.add_argument("--prompt", "-p", required=True, help="Primary prompt")
parser.add_argument("--prompt2", type=str, help="Secondary prompt (toggle on kicks)")
parser.add_argument("--prompt-threshold", type=float, default=0.7, help="Kick threshold to switch prompt")
parser.add_argument("--strength-min", type=float, default=0.20, help="Base strength")
parser.add_argument("--strength-max", type=float, default=0.40, help="Max strength on snare")
parser.add_argument("--zoom-min", type=float, default=1.0, help="Base zoom")
parser.add_argument("--zoom-max", type=float, default=1.08, help="Max zoom on kick")
parser.add_argument("--prev-blend", type=float, default=0.3)
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def apply_zoom(img: Image.Image, zoom: float) -> Image.Image:
    """Apply center zoom to image."""
    if abs(zoom - 1.0) < 0.001:
        return img
    arr = np.array(img)
    h, w = arr.shape[:2]
    new_h, new_w = int(h / zoom), int(w / zoom)
    start_y, start_x = (h - new_h) // 2, (w - new_w) // 2
    cropped = arr[start_y:start_y+new_h, start_x:start_x+new_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(zoomed)


frames, fps = load_video(args.input, max_frames=args.max_frames)
num_frames = len(frames)

# Extract audio features using flux_motion extractor
extractor = AudioFeatureExtractor(normalize=True, smooth_window=3)
audio = extractor.extract(args.audio, fps=fps, duration=num_frames/fps)

# Map features: bass → kick/zoom, mid → snare/strength
kick = audio.bass[:num_frames]
snare = audio.mid[:num_frames]

with GenerationContext(args.output) as gen:
    gen.update(
        preset="v2v_audio",
        input=args.input,
        audio=args.audio,
        prompt=args.prompt,
        prompt2=args.prompt2,
        prompt_threshold=args.prompt_threshold,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        zoom_min=args.zoom_min,
        zoom_max=args.zoom_max,
        prev_blend=args.prev_blend,
        tempo=audio.tempo,
        seed=args.seed,
        model="flux.2-klein-4b",
        steps=4,
    )
    gen.fps = fps

    pipe = get_pipeline()
    output = []
    prev_gen = None
    anchor_latent = None

    for i, frame in enumerate(tqdm(frames, desc="AudioReactive")):
        frame_kick = kick[i]
        frame_snare = snare[i]

        zoom = args.zoom_min + frame_kick * (args.zoom_max - args.zoom_min)
        strength = args.strength_min + frame_snare * (args.strength_max - args.strength_min)

        # Prompt toggle on kick hits
        if args.prompt2 and frame_kick > args.prompt_threshold:
            current_prompt = args.prompt2
        else:
            current_prompt = args.prompt

        frame_zoomed = apply_zoom(frame, zoom)

        if i > 0 and prev_gen:
            frame_zoomed = blend(prev_gen, frame_zoomed, args.prev_blend)

        latent = pipe._encode_to_latent(frame_zoomed)

        if anchor_latent is not None:
            latent = match_color_latent(latent, anchor_latent, (32, 64), 0.7)

        img, out_latent = pipe._generate_motion_frame(
            prev_latent=latent, prompt=current_prompt, motion_params={},
            width=frame.width, height=frame.height,
            num_inference_steps=4, guidance_scale=1.0,
            strength=strength, seed=args.seed
        )

        if i == 0:
            anchor_latent = out_latent.clone()

        output.append(img)
        prev_gen = img

        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
