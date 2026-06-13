#!/usr/bin/env python3
"""Klein V2V Audio Deforum (NATIVE BFL SDK) - audio-driven Deforum motion in latent space.

Native twin of hybrid-v2v/klein_v2v_audio_deforum.py: identical CLI and loop,
pure BFL pipeline (load_ae BatchNorm VAE + flux2.sampling denoise). Motion
schedules come from the audio bridge and are applied in the 128-channel latent
space by Flux2MotionEngine - the same engine the hybrid lane uses.

NATIVE-FIRST: this is the preferred lane; the hybrid twin exists for A/B
comparison of the VAE path.

Usage:
    python klein_v2v_audio_deforum.py -i video.mp4 -a track.wav -p "prompt" -o out.mp4
    python klein_v2v_audio_deforum.py -i video.mp4 --analysis-json-file dash.json -p "prompt"

    # With native LoRA + secondary prompt on kicks
    python klein_v2v_audio_deforum.py -i video.mp4 -a track.wav -p "calm forest" \\
        --prompt2 "exploding neon forest" --lora cyber-flowers.safetensors
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
    load_video,
    blend,
    optical_flow,
    warp,
    apply_lora,
    clear_cuda,
    tqdm,
    GenerationContext,
    match_color_latent,
)
from native_utils import NativePipeline  # noqa: E402

parser = argparse.ArgumentParser(description="Audio-driven Deforum motion, native BFL SDK")
parser.add_argument("--input", "-i", required=True, help="Input (driving) video")
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
parser.add_argument("--output", "-o", default="outputs/native_v2v_audio_deforum.mp4")
parser.add_argument("--prompt", "-p", required=True, help="Primary prompt")
parser.add_argument("--prompt2", type=str, help="Secondary prompt (toggle on kicks)")
parser.add_argument(
    "--prompt-threshold", type=float, default=0.7, help="Kick threshold to switch prompt"
)
# Audio -> motion mapping (same knobs/defaults as the hybrid twin)
parser.add_argument("--strength-min", type=float, default=0.20, help="Base strength")
parser.add_argument("--strength-max", type=float, default=0.40, help="Max strength on snare")
parser.add_argument("--zoom-min", type=float, default=1.0, help="Base zoom")
parser.add_argument("--zoom-max", type=float, default=1.08, help="Max zoom on kick")
parser.add_argument("--driver", choices=["auto", "waveform", "markers"], default="auto")
parser.add_argument("--smoothing", type=float, default=0.2)
parser.add_argument("--translation-gain", type=float, default=0.0)
parser.add_argument("--angle-gain", type=float, default=0.0)
# Deforum-loop knobs (same semantics as the hybrid twin)
parser.add_argument(
    "--video-blend", type=float, default=0.0,
    help="Blend current input frame back into the flow-warped feedback "
         "(0.0 = pure Deforum feedback, 0.3-0.5 = follow video more)",
)
parser.add_argument(
    "--init-from-input", action="store_true",
    help="Frame 0 keeps more of the input (strength 0.7 instead of 0.95)",
)
parser.add_argument(
    "--color-blend", type=float, default=0.7,
    help="Latent color match strength vs frame-0 anchor (0 = off)",
)
parser.add_argument("--max-frames", "-n", type=int)
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
        preset="native_v2v_audio_deforum",
        pipeline="native-bfl",
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
        video_blend=args.video_blend,
        init_from_input=args.init_from_input,
        color_blend=args.color_blend,
        seed=args.seed,
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
    prev_input = frames[0]
    prev_gen = None
    anchor_latent = None

    for i, frame in enumerate(tqdm(frames, desc="NativeAudioDeforum")):
        motion_frame = motion.motion_frames[i]
        motion_dict = motion_frame.to_dict()

        # Prompt toggle on kick hits
        if args.prompt2 and kick[i] > args.prompt_threshold:
            current_prompt = args.prompt2
        else:
            current_prompt = args.prompt

        if i == 0:
            init_strength = 0.7 if args.init_from_input else 0.95
            z = pipe.encode(frame)
            img, out_latent = pipe.generate_from_latent(
                z, current_prompt, strength=init_strength,
                num_steps=args.steps, guidance=args.guidance,
                seed=args.seed, motion_params=motion_dict,
            )
            anchor_latent = out_latent.clone()
        else:
            # Deforum feedback: warp prev_gen along the input video's motion.
            # Resize prev_gen to the flow's resolution - the native decode size
            # differs from frame size on 1080p/non-/16 inputs.
            flow = optical_flow(prev_input, frame)
            warped = warp(prev_gen.resize(frame.size), flow)

            # Optionally pull back toward the actual input frame
            source = blend(frame, warped, args.video_blend) if args.video_blend > 0 else warped

            z = pipe.encode(source)

            # Color consistency vs frame-0 anchor (latent space, ch 32-63)
            if anchor_latent is not None and args.color_blend > 0:
                z = match_color_latent(z, anchor_latent, (32, 64), args.color_blend)

            # Audio-driven motion applied in latent space; strength from amplitude
            img, out_latent = pipe.generate_from_latent(
                z, current_prompt, strength=motion_frame.strength,
                num_steps=args.steps, guidance=args.guidance,
                seed=args.seed, motion_params=motion_dict,
            )

        output.append(img)
        prev_input = frame
        prev_gen = img

        if i % 20 == 0:
            clear_cuda()

    gen.frames = output
    gen.save_video()
