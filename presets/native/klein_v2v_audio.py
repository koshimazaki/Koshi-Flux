#!/usr/bin/env python3
"""Klein V2V Audio Reactive - Sync video generation to audio.

Maps audio features to generation parameters:
- Kick/Bass → Zoom (pulse on beat)
- Snare/Mid → Strength (style intensity on snare hits)
- Kick threshold → Prompt toggle (scene change)

Uses flux_motion.audio.bridge for dashboard JSON, audio-file, and MP4 feature sources.

Usage:
    python klein_v2v_audio.py -i video.mp4 -a audio.wav -o output.mp4 -p "prompt"
    python klein_v2v_audio.py -i video.mp4 --analysis-json-file analysis.json -o output.mp4 -p "prompt"
    python klein_v2v_audio.py -i video.mp4 --feature-video motion_source.mp4 -o output.mp4 -p "prompt"
"""
import argparse
import sys
import numpy as np
from pathlib import Path

# Add flux_motion to path
SCRIPT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "flux/src"))
sys.path.insert(0, str(SCRIPT_DIR / "presets"))

from flux_motion.audio import (  # noqa: E402
    build_audio_motion_schedule,
    tracks_from_analysis_file,
    tracks_from_analysis_json,
    tracks_from_audio_file,
    tracks_from_video,
)

from klein_utils import (  # noqa: E402
    load_video,
    blend,
    get_pipeline,
    clear_cuda,
    tqdm,
    Image,
    GenerationContext,
    match_color_latent,
)
import cv2  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True, help="Input video")
parser.add_argument("--audio", "-a", help="Audio file (wav/mp3)")
parser.add_argument(
    "--analysis-json-file", help="BFL dashboard or Fill-Nodes analysis JSON file"
)
parser.add_argument(
    "--analysis-json", help="Inline BFL dashboard or Fill-Nodes analysis JSON"
)
parser.add_argument(
    "--feature-video", help="MP4/video file to analyze for motion features"
)
parser.add_argument("--output", "-o", default="outputs/v2v_audio.mp4")
parser.add_argument("--prompt", "-p", required=True, help="Primary prompt")
parser.add_argument("--prompt2", type=str, help="Secondary prompt (toggle on kicks)")
parser.add_argument(
    "--prompt-threshold",
    type=float,
    default=0.7,
    help="Kick threshold to switch prompt",
)
parser.add_argument("--strength-min", type=float, default=0.20, help="Base strength")
parser.add_argument(
    "--strength-max", type=float, default=0.40, help="Max strength on snare"
)
parser.add_argument("--zoom-min", type=float, default=1.0, help="Base zoom")
parser.add_argument("--zoom-max", type=float, default=1.08, help="Max zoom on kick")
parser.add_argument("--driver", choices=["auto", "waveform", "markers"], default="auto")
parser.add_argument("--smoothing", type=float, default=0.2)
parser.add_argument("--translation-gain", type=float, default=0.0)
parser.add_argument("--angle-gain", type=float, default=0.0)
parser.add_argument(
    "--latent-motion",
    action="store_true",
    help="Also pass motion params to Flux latent engine",
)
parser.add_argument(
    "--no-pixel-zoom", action="store_true", help="Disable pixel-space zoom pre-warp"
)
parser.add_argument("--prev-blend", type=float, default=0.3)
parser.add_argument("--max-frames", "-n", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lora", help="LoRA path or HF repo to apply (e.g. cyber-flowers)")
parser.add_argument(
    "--lora-strength", type=float, default=0.8, help="LoRA strength (0.0-2.0)"
)
args = parser.parse_args()


def apply_zoom(img: Image.Image, zoom: float) -> Image.Image:
    """Apply center zoom to image."""
    if abs(zoom - 1.0) < 0.001:
        return img
    arr = np.array(img)
    h, w = arr.shape[:2]
    new_h, new_w = int(h / zoom), int(w / zoom)
    start_y, start_x = (h - new_h) // 2, (w - new_w) // 2
    cropped = arr[start_y : start_y + new_h, start_x : start_x + new_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(zoomed)


def load_feature_tracks(args, fps: float, num_frames: int):
    """Load the selected dashboard/audio/video feature source."""
    duration = num_frames / fps if fps else None
    if args.analysis_json_file:
        return tracks_from_analysis_file(args.analysis_json_file), "analysis_json_file"
    if args.analysis_json:
        return tracks_from_analysis_json(args.analysis_json), "analysis_json"
    if args.feature_video:
        return (
            tracks_from_video(args.feature_video, max_frames=num_frames),
            "feature_video",
        )
    if args.audio:
        return (
            tracks_from_audio_file(args.audio, fps=fps, duration=duration),
            "audio_file",
        )
    raise ValueError(
        "Provide one feature source: --audio, --analysis-json-file, --analysis-json, "
        "or --feature-video."
    )


frames, fps = load_video(args.input, max_frames=args.max_frames)
num_frames = len(frames)

tracks, feature_source = load_feature_tracks(args, fps, num_frames)
motion = build_audio_motion_schedule(
    tracks,
    num_frames=num_frames,
    fps=fps,
    feature=args.driver,
    base_zoom=args.zoom_min,
    zoom_gain=args.zoom_max - args.zoom_min,
    angle_gain=args.angle_gain,
    translation_gain=args.translation_gain,
    base_strength=args.strength_min,
    strength_gain=args.strength_max - args.strength_min,
    smoothing=args.smoothing,
)
kick = motion.bands["low"]

with GenerationContext(args.output) as gen:
    gen.update(
        preset="v2v_audio",
        input=args.input,
        audio=args.audio,
        analysis_json_file=args.analysis_json_file,
        feature_video=args.feature_video,
        feature_source=feature_source,
        audio_motion=motion.to_settings(),
        prompt=args.prompt,
        prompt2=args.prompt2,
        prompt_threshold=args.prompt_threshold,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        zoom_min=args.zoom_min,
        zoom_max=args.zoom_max,
        driver=args.driver,
        resolved_driver=motion.driver,
        smoothing=args.smoothing,
        translation_gain=args.translation_gain,
        angle_gain=args.angle_gain,
        latent_motion=args.latent_motion,
        pixel_zoom=not args.no_pixel_zoom,
        prev_blend=args.prev_blend,
        tempo=tracks.meta.get("tempo"),
        seed=args.seed,
        lora=args.lora,
        lora_strength=args.lora_strength,
        model="flux.2-klein-4b",
        steps=4,
    )
    gen.fps = fps
    gen.audio = args.audio

    pipe = get_pipeline(lora=args.lora, lora_strength=args.lora_strength)
    output = []
    prev_gen = None
    anchor_latent = None

    for i, frame in enumerate(tqdm(frames, desc="AudioReactive")):
        frame_kick = kick[i]
        motion_frame = motion.motion_frames[i]
        zoom = motion_frame.zoom
        strength = motion_frame.strength

        # Prompt toggle on kick hits
        if args.prompt2 and frame_kick > args.prompt_threshold:
            current_prompt = args.prompt2
        else:
            current_prompt = args.prompt

        frame_zoomed = frame if args.no_pixel_zoom else apply_zoom(frame, zoom)

        if i > 0 and prev_gen:
            frame_zoomed = blend(prev_gen, frame_zoomed, args.prev_blend)

        latent = pipe._encode_to_latent(frame_zoomed)

        if anchor_latent is not None:
            latent = match_color_latent(latent, anchor_latent, (32, 64), 0.7)

        img, out_latent = pipe._generate_motion_frame(
            prev_latent=latent,
            prompt=current_prompt,
            motion_params=motion_frame.to_dict() if args.latent_motion else {},
            width=frame.width,
            height=frame.height,
            num_inference_steps=4,
            guidance_scale=1.0,
            strength=strength,
            seed=args.seed,
        )

        if i == 0:
            anchor_latent = out_latent.clone()

        output.append(img)
        prev_gen = img

        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
