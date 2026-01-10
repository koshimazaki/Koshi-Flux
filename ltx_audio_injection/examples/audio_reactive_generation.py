#!/usr/bin/env python3
"""
Audio-Reactive Video Generation with LTX-Video

This script demonstrates how to use the LTX Audio Injection module
to generate videos that react to audio input through deep cross-attention
integration.

Features:
- Audio-conditioned video generation
- Beat-synchronized visual effects
- Multiple audio encoder backends
- Configurable injection modes

Usage:
    python audio_reactive_generation.py \
        --prompt "A cosmic nebula pulsing with energy" \
        --audio path/to/music.mp3 \
        --output output_video.mp4 \
        --num_frames 121 \
        --audio_scale 1.0

Requirements:
    - LTX-Video (pip install ltx-video)
    - PyTorch 2.0+
    - torchaudio
    - transformers (optional, for CLAP/Wav2Vec2 encoders)
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate audio-reactive video with LTX-Video"
    )

    # Required arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file (mp3, wav, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_video.mp4",
        help="Output video path",
    )

    # Video parameters
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--fps", type=float, default=24.0, help="Frame rate")

    # Audio parameters
    parser.add_argument(
        "--audio_scale",
        type=float,
        default=1.0,
        help="Scale factor for audio conditioning (0.0-2.0)",
    )
    parser.add_argument(
        "--audio_encoder",
        type=str,
        default="spectrogram",
        choices=["spectrogram", "clap", "wav2vec2"],
        help="Audio encoder backend",
    )
    parser.add_argument(
        "--audio_injection_mode",
        type=str,
        default="cross_attention",
        choices=["cross_attention", "add", "gate"],
        help="How to inject audio features",
    )

    # Generation parameters
    parser.add_argument(
        "--num_inference_steps", type=int, default=20, help="Denoising steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=4.5, help="CFG scale"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, distorted",
        help="Negative prompt",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="Lightricks/LTX-Video",
        help="Path to LTX-Video model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision",
    )

    # Advanced options
    parser.add_argument(
        "--use_beat_features",
        action="store_true",
        help="Enable beat detection for audio encoding",
    )
    parser.add_argument(
        "--audio_injection_layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices for audio injection (e.g., '0,1,2,3')",
    )

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.float16)


def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: float = 24.0,
    audio_path: Optional[str] = None,
):
    """
    Save video frames to file, optionally with audio.

    Args:
        frames: Video frames tensor (batch, channels, frames, height, width)
        output_path: Output file path
        fps: Frame rate
        audio_path: Optional audio file to mux with video
    """
    import torchvision.io as io

    # Ensure frames are in correct format (T, H, W, C) for video writer
    if frames.dim() == 5:
        frames = frames[0]  # Remove batch dimension

    # Convert from (C, T, H, W) to (T, H, W, C)
    frames = frames.permute(1, 2, 3, 0)

    # Convert to uint8
    if frames.dtype != torch.uint8:
        frames = (frames * 255).clamp(0, 255).to(torch.uint8)

    # Save video
    temp_path = output_path + ".temp.mp4"
    io.write_video(temp_path, frames.cpu(), fps=fps)

    # Mux with audio if provided
    if audio_path is not None:
        try:
            import subprocess

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_path,
                    "-i",
                    audio_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
            os.remove(temp_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: ffmpeg not available, saving video without audio")
            os.rename(temp_path, output_path)
    else:
        os.rename(temp_path, output_path)


def main():
    args = parse_args()

    print(f"Audio-Reactive Video Generation")
    print(f"=" * 50)
    print(f"Prompt: {args.prompt}")
    print(f"Audio: {args.audio}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} @ {args.fps} fps")
    print(f"Audio scale: {args.audio_scale}")
    print(f"Audio encoder: {args.audio_encoder}")
    print(f"Injection mode: {args.audio_injection_mode}")
    print(f"=" * 50)

    # Import modules (after argument parsing for faster --help)
    from ltx_audio_injection import (
        AudioEncoder,
        AudioEncoderConfig,
        LTXAudioVideoPipeline,
    )
    from ltx_audio_injection.utils import load_audio, compute_audio_energy

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    else:
        generator = None

    # Get dtype
    dtype = get_dtype(args.dtype)

    # Parse audio injection layers
    audio_injection_layers = None
    if args.audio_injection_layers:
        audio_injection_layers = [
            int(x) for x in args.audio_injection_layers.split(",")
        ]

    # Configure audio encoder
    audio_config = AudioEncoderConfig(
        encoder_type=args.audio_encoder,
        use_beat_features=args.use_beat_features,
        frames_per_second=args.fps,
    )

    print(f"\nLoading pipeline from {args.model_path}...")

    # Load pipeline
    # Note: This uses a simplified loader. In production, use from_pretrained_ltx
    try:
        pipeline = LTXAudioVideoPipeline.from_pretrained_ltx(
            args.model_path,
            audio_encoder_config=audio_config,
            audio_injection_mode=args.audio_injection_mode,
            audio_scale=args.audio_scale,
            torch_dtype=dtype,
        )
    except Exception as e:
        print(f"Note: Full pipeline loading requires LTX-Video installation.")
        print(f"Error: {e}")
        print(f"\nDemonstrating standalone audio encoding instead...")

        # Demonstrate audio encoding
        audio_encoder = AudioEncoder(audio_config)

        print(f"\nLoading audio from {args.audio}...")
        waveform, sr = load_audio(args.audio, target_sample_rate=16000)
        print(f"Audio loaded: {waveform.shape[-1] / sr:.2f} seconds")

        print(f"\nEncoding audio for {args.num_frames} video frames...")
        audio_embeddings = audio_encoder(
            waveform,
            num_video_frames=args.num_frames,
            sample_rate=sr,
        )
        print(f"Audio embeddings shape: {audio_embeddings.shape}")

        # Compute audio energy for visualization
        energy = compute_audio_energy(waveform)
        print(f"Audio energy shape: {energy.shape}")
        print(f"Max energy: {energy.max():.4f}, Mean energy: {energy.mean():.4f}")

        print(f"\nAudio encoding complete!")
        print(f"These embeddings would be passed to the transformer for")
        print(f"audio-conditioned video generation via cross-attention.")

        return

    pipeline = pipeline.to(args.device)

    # Load and preprocess audio
    print(f"\nLoading audio from {args.audio}...")
    waveform, sr = load_audio(args.audio, target_sample_rate=16000)
    print(f"Audio loaded: {waveform.shape[-1] / sr:.2f} seconds")

    # Generate video
    print(f"\nGenerating audio-reactive video...")
    output = pipeline(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.fps,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        audio=waveform,
        audio_sample_rate=16000,
        audio_scale=args.audio_scale,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        return_audio_features=True,
    )

    # Save video
    print(f"\nSaving video to {args.output}...")
    save_video(
        output.images,
        args.output,
        fps=args.fps,
        audio_path=args.audio,
    )

    print(f"\nDone! Video saved to {args.output}")

    # Print audio feature info if available
    if output.audio_features is not None:
        print(f"\nAudio features extracted:")
        for key, value in output.audio_features.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}")


if __name__ == "__main__":
    main()
