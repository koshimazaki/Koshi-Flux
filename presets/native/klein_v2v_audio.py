#!/usr/bin/env python3
"""Klein V2V Audio Reactive - Sync video generation to audio.

Maps audio features to generation parameters:
- Kick/Bass → Zoom (pulse on beat)
- Snare → Strength (style intensity on snare hits)

Usage:
    python klein_v2v_audio.py -i video.mp4 -a audio.wav -o output.mp4 -p "prompt"
"""
import argparse
import numpy as np
from pathlib import Path

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

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
parser.add_argument("--smooth", type=float, default=0.3, help="Audio smoothing")
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def load_audio_features(audio_path: str, num_frames: int, fps: float) -> dict:
    """Extract kick and snare energy per frame from audio."""
    if not HAS_LIBROSA:
        return {
            'kick': np.random.rand(num_frames) * 0.5,
            'snare': np.random.rand(num_frames) * 0.5,
        }

    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr
    frame_times = np.linspace(0, duration, num_frames)

    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.times_like(D, sr=sr)

    kick_mask = (freqs >= 60) & (freqs <= 150)
    snare_mask = (freqs >= 150) & (freqs <= 400)

    kick_energy = D[kick_mask, :].mean(axis=0)
    snare_energy = D[snare_mask, :].mean(axis=0)

    kick_energy = (kick_energy - kick_energy.min()) / (kick_energy.max() - kick_energy.min() + 1e-6)
    snare_energy = (snare_energy - snare_energy.min()) / (snare_energy.max() - snare_energy.min() + 1e-6)

    kick_interp = np.interp(frame_times, times, kick_energy)
    snare_interp = np.interp(frame_times, times, snare_energy)

    return {'kick': kick_interp, 'snare': snare_interp}


def smooth_signal(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average smoothing."""
    if alpha <= 0:
        return signal
    smoothed = np.zeros_like(signal)
    smoothed[0] = signal[0]
    for i in range(1, len(signal)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * signal[i]
    return smoothed


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

audio_features = load_audio_features(args.audio, num_frames, fps)
kick = smooth_signal(audio_features['kick'], args.smooth)
snare = smooth_signal(audio_features['snare'], args.smooth)

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
        smooth=args.smooth,
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
