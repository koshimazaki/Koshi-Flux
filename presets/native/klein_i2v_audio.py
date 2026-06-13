#!/usr/bin/env python3
"""Klein I2V Audio (NATIVE BFL SDK) - animate a single still image from audio alone.

No driving video: classic Deforum feedback in latent space. The keyframe is
encoded ONCE, then each frame is the previous frame's output latent, warped by
audio-driven motion (Flux2MotionEngine) and re-denoised at audio-driven
strength. Color stays anchored to frame 0 via latent color matching.

This completes the consistency thesis: one Klein keyframe (+ LoRA identity)
animated to a track, no input footage needed.

Per-frame seed increments (seed+i) - a fixed seed in a feedback loop burns
static noise patterns into the animation.

Usage:
    python klein_i2v_audio.py --image still.png -a track.wav -p "prompt" -o out.mp4
    python klein_i2v_audio.py --image still.png --analysis-json-file dash.json \\
        --frames 240 --fps 24 -p "prompt"

    # With LoRA + scene change on kicks
    python klein_i2v_audio.py --image flower.png -a track.wav -p "cyber flower" \\
        --prompt2 "blooming neon flower" --lora cyber-flowers.safetensors
"""
import argparse
import sys
from pathlib import Path

# Add flux_motion + presets to path (same bootstrap as klein_v2v_audio)
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "flux/src"))
sys.path.insert(0, str(SCRIPT_DIR / "core/src"))
sys.path.insert(0, str(SCRIPT_DIR / "presets"))

from flux_motion.audio import (  # noqa: E402
    build_audio_motion_schedule,
    tracks_from_analysis_file,
    tracks_from_analysis_json,
    tracks_from_audio_file,
    tracks_from_video,
)

from klein_utils import (  # noqa: E402
    apply_lora,
    clear_cuda,
    tqdm,
    Image,
    GenerationContext,
    match_color_latent,
)
from native_utils import NativePipeline  # noqa: E402

parser = argparse.ArgumentParser(description="Animate a still image from audio (native BFL)")
parser.add_argument("--image", "-i", required=True, help="Input still image (the keyframe)")
parser.add_argument("--audio", "-a", help="Audio file (wav/mp3) - also muxed onto output")
parser.add_argument(
    "--analysis-json-file", help="BFL dashboard or Fill-Nodes analysis JSON file"
)
parser.add_argument(
    "--analysis-json", help="Inline BFL dashboard or Fill-Nodes analysis JSON"
)
parser.add_argument(
    "--feature-video", help="MP4/video file to analyze for motion features"
)
parser.add_argument("--output", "-o", default="outputs/i2v_audio.mp4")
parser.add_argument("--prompt", "-p", required=True, help="Primary prompt")
parser.add_argument("--prompt2", type=str, help="Secondary prompt (toggle on kicks)")
parser.add_argument(
    "--prompt-threshold", type=float, default=0.7, help="Kick threshold to switch prompt"
)
parser.add_argument("--frames", "-n", type=int, default=120, help="Number of frames to generate")
parser.add_argument("--fps", type=float, default=24.0, help="Output FPS (audio analysis grid)")
# Audio -> motion mapping (same knobs/defaults as the V2V audio presets)
parser.add_argument("--strength-min", type=float, default=0.20, help="Base strength")
parser.add_argument("--strength-max", type=float, default=0.40, help="Max strength on snare")
parser.add_argument("--zoom-min", type=float, default=1.0, help="Base zoom")
parser.add_argument("--zoom-max", type=float, default=1.08, help="Max zoom on kick")
parser.add_argument("--driver", choices=["auto", "waveform", "markers"], default="auto")
parser.add_argument("--smoothing", type=float, default=0.2)
parser.add_argument("--translation-gain", type=float, default=0.0)
parser.add_argument("--angle-gain", type=float, default=0.0)
parser.add_argument(
    "--init-strength", type=float, default=0.7,
    help="Frame-0 strength (how much the keyframe is restyled before animating)",
)
parser.add_argument(
    "--color-blend", type=float, default=0.7,
    help="Latent color match strength vs frame-0 anchor (0 = off)",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model", default="flux.2-klein-4b",
                    choices=["flux.2-klein-4b", "flux.2-klein-9b"])
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--guidance", type=float, default=1.0)
parser.add_argument("--lora", help="LoRA .safetensors to merge into the native DiT")
parser.add_argument("--lora-strength", type=float, default=0.8, help="LoRA strength (0.0-2.0)")
args = parser.parse_args()


def load_feature_tracks(args, fps: float, num_frames: int):
    """Load the selected dashboard/audio/video feature source."""
    duration = num_frames / fps if fps else None
    if args.analysis_json_file:
        return tracks_from_analysis_file(args.analysis_json_file), "analysis_json_file"
    if args.analysis_json:
        return tracks_from_analysis_json(args.analysis_json), "analysis_json"
    if args.feature_video:
        return tracks_from_video(args.feature_video, max_frames=num_frames), "feature_video"
    if args.audio:
        return tracks_from_audio_file(args.audio, fps=fps, duration=duration), "audio_file"
    raise ValueError(
        "Provide one feature source: --audio, --analysis-json-file, --analysis-json, "
        "or --feature-video."
    )


num_frames = args.frames
fps = args.fps

keyframe = Image.open(args.image).convert("RGB")

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
        preset="i2v_audio",
        pipeline="native-bfl",
        image=args.image,
        audio=args.audio,
        analysis_json_file=args.analysis_json_file,
        feature_video=args.feature_video,
        feature_source=feature_source,
        audio_motion=motion.to_settings(),
        prompt=args.prompt,
        prompt2=args.prompt2,
        prompt_threshold=args.prompt_threshold,
        num_frames=num_frames,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        zoom_min=args.zoom_min,
        zoom_max=args.zoom_max,
        driver=args.driver,
        resolved_driver=motion.driver,
        smoothing=args.smoothing,
        translation_gain=args.translation_gain,
        angle_gain=args.angle_gain,
        init_strength=args.init_strength,
        color_blend=args.color_blend,
        seed=args.seed,
        seed_mode="increment",
        lora=args.lora,
        lora_strength=args.lora_strength,
        tempo=tracks.meta.get("tempo"),
        model=args.model,
        steps=args.steps,
        guidance=args.guidance,
    )
    gen.fps = fps
    gen.audio = args.audio

    pipe = NativePipeline(model_name=args.model)

    # Optional LoRA - merges into the native DiT weights (single-applied)
    lora_manager = apply_lora(pipe, args.lora, args.lora_strength, lora_backend="native") \
        if args.lora else None
    gen.set("lora_applied", lora_manager is not None)
    if lora_manager is not None:
        loaded = lora_manager.list_loaded()
        if loaded:
            gen.set("lora_modules_matched", loaded[0].matched)
            gen.set("lora_modules_total", loaded[0].total)

    output = []
    latent = None
    anchor_latent = None

    for i in tqdm(range(num_frames), desc="I2VAudio"):
        motion_frame = motion.motion_frames[i]
        motion_dict = motion_frame.to_dict()

        # Prompt toggle on kick hits
        if args.prompt2 and kick[i] > args.prompt_threshold:
            current_prompt = args.prompt2
        else:
            current_prompt = args.prompt

        if i == 0:
            # Encode the keyframe ONCE; restyle it as the anchor frame
            z = pipe.encode(keyframe)
            img, latent = pipe.generate_from_latent(
                z, current_prompt, strength=args.init_strength,
                num_steps=args.steps, guidance=args.guidance,
                seed=args.seed, motion_params=motion_dict,
            )
            anchor_latent = latent.clone()
        else:
            # Latent feedback: previous output latent -> color anchor ->
            # audio-driven motion warp -> re-denoise at audio-driven strength
            if anchor_latent is not None and args.color_blend > 0:
                latent = match_color_latent(latent, anchor_latent, (32, 64), args.color_blend)

            img, latent = pipe.generate_from_latent(
                latent, current_prompt, strength=motion_frame.strength,
                num_steps=args.steps, guidance=args.guidance,
                seed=args.seed + i, motion_params=motion_dict,
            )

        output.append(img)

        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
