#!/usr/bin/env python3
"""Command-line interface for audio feature extraction and schedule generation.

Usage:
    # Extract features and generate schedule
    python -m koshi_flux.audio.cli music.mp3 -o schedule.json

    # Use a specific mapping preset
    python -m koshi_flux.audio.cli music.mp3 -m bass_pulse -o schedule.json

    # List available presets
    python -m koshi_flux.audio.cli --list-presets

    # Generate with custom settings
    python -m koshi_flux.audio.cli music.mp3 -m spectrum --fps 30 --duration 60
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Check for librosa before proceeding
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio features and generate Deforum/Parseq schedules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - extract features and generate schedule
  python -m koshi_flux.audio.cli music.mp3 -o schedule.json

  # Use a specific mapping preset
  python -m koshi_flux.audio.cli music.mp3 -m bass_pulse -o schedule.json

  # Generate with custom FPS and duration
  python -m koshi_flux.audio.cli music.mp3 --fps 30 --duration 60 -o schedule.json

  # Save features for later use
  python -m koshi_flux.audio.cli music.mp3 --save-features features.json

  # Generate Deforum keyframe strings
  python -m koshi_flux.audio.cli music.mp3 -m spectrum --format deforum

Available mapping presets:
  ambient       - Subtle, flowing movement for atmospheric content
  bass_pulse    - Strong zoom pulses on bass hits
  beat_rotation - Rotation synchronized to beats
  spectrum      - Maps bass/mid/high to different motion axes
  immersive_3d  - Full 3D rotation and depth movement
  cinematic     - Slow, smooth movements for cinematic content
  intense       - Aggressive movement for high-energy content
"""
    )

    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to audio file (mp3, wav, flac, etc.)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output path for schedule JSON"
    )

    parser.add_argument(
        "-m", "--mapping",
        default="bass_pulse",
        help="Mapping preset name (default: bass_pulse)"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Video frame rate (default: 24)"
    )

    parser.add_argument(
        "--duration",
        type=float,
        help="Duration in seconds (default: full audio)"
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0)"
    )

    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=1,
        help="Frames between keyframes (default: 1, every frame)"
    )

    parser.add_argument(
        "--prompt",
        default="",
        help="Default prompt for all frames"
    )

    parser.add_argument(
        "--format",
        choices=["parseq", "deforum", "both"],
        default="parseq",
        help="Output format (default: parseq)"
    )

    parser.add_argument(
        "--save-features",
        help="Save extracted features to JSON file"
    )

    parser.add_argument(
        "--load-features",
        help="Load features from JSON instead of extracting"
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available mapping presets"
    )

    parser.add_argument(
        "--show-mapping",
        help="Show details of a specific mapping preset"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Handle special commands
    if args.list_presets:
        list_presets()
        return 0

    if args.show_mapping:
        show_mapping(args.show_mapping)
        return 0

    # Check for required audio file
    if not args.audio_file and not args.load_features:
        parser.error("Audio file is required (or use --load-features)")

    # Check librosa availability
    if not LIBROSA_AVAILABLE and not args.load_features:
        print("ERROR: librosa is required for audio feature extraction.")
        print("Install with: pip install librosa audioread soundfile")
        return 1

    # Import here to avoid errors if librosa not installed
    from .extractor import AudioFeatureExtractor, AudioFeatures
    from .schedule_generator import ScheduleGenerator
    from .mapping_config import DEFAULT_MAPPINGS

    # Validate mapping
    if args.mapping not in DEFAULT_MAPPINGS:
        print(f"ERROR: Unknown mapping preset: {args.mapping}")
        print(f"Available presets: {', '.join(DEFAULT_MAPPINGS.keys())}")
        return 1

    # Extract or load features
    if args.load_features:
        print(f"Loading features from {args.load_features}...")
        features = AudioFeatures.load(args.load_features)
    else:
        print(f"Extracting features from {args.audio_file}...")
        extractor = AudioFeatureExtractor()
        features = extractor.extract(
            args.audio_file,
            fps=args.fps,
            duration=args.duration,
            start_time=args.start,
        )

    # Print feature summary
    print(f"\nAudio Analysis:")
    print(f"  Duration: {features.duration:.2f}s")
    print(f"  Tempo: {features.tempo:.1f} BPM")
    print(f"  Frames: {features.num_frames} @ {features.fps}fps")
    print(f"  Beats detected: {len(features.beats)}")

    # Save features if requested
    if args.save_features:
        features.save(args.save_features)
        print(f"\nSaved features to {args.save_features}")

    # Generate schedule
    print(f"\nGenerating schedule with '{args.mapping}' mapping...")
    generator = ScheduleGenerator()
    schedule = generator.generate(
        features,
        mapping=args.mapping,
        keyframe_interval=args.keyframe_interval,
        prompt=args.prompt,
        audio_path=args.audio_file,
    )

    print(f"  Generated {len(schedule.keyframes)} keyframes")

    # Output based on format
    if args.format in ("parseq", "both"):
        output_path = args.output or "schedule.json"
        schedule.save(output_path)
        print(f"\nSaved Parseq schedule to {output_path}")

    if args.format in ("deforum", "both"):
        deforum_strings = schedule.to_deforum_strings()
        print("\n" + "="*60)
        print("DEFORUM KEYFRAME STRINGS")
        print("="*60)
        print("Copy these into your Deforum settings:\n")

        for param, value in deforum_strings.items():
            # Truncate for display if very long
            display_value = value
            if len(value) > 200:
                # Show first and last parts
                display_value = value[:100] + " ... " + value[-100:]
            print(f"{param}:")
            print(f"  {display_value}\n")

        # Also save to file if both format
        if args.format == "both" and args.output:
            deforum_path = Path(args.output).with_suffix(".koshi.json")
            with open(deforum_path, 'w') as f:
                json.dump(deforum_strings, f, indent=2)
            print(f"Saved Deforum strings to {deforum_path}")

    if args.verbose:
        print("\n" + "="*60)
        print("FEATURE STATISTICS")
        print("="*60)
        for feature_name in features.list_features():
            values = features.get_feature(feature_name)
            print(f"{feature_name:20s}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")

    print("\nDone!")
    return 0


def list_presets():
    """List all available mapping presets."""
    from .mapping_config import DEFAULT_MAPPINGS

    print("\nAvailable Mapping Presets:")
    print("=" * 60)

    for name, config in DEFAULT_MAPPINGS.items():
        print(f"\n{name}")
        print(f"  {config.description}")
        print(f"  Mappings:")
        for m in config.mappings:
            print(f"    {m.feature} -> {m.parameter} [{m.min_value}, {m.max_value}]")


def show_mapping(name: str):
    """Show details of a specific mapping preset."""
    from .mapping_config import DEFAULT_MAPPINGS

    if name not in DEFAULT_MAPPINGS:
        print(f"ERROR: Unknown preset: {name}")
        print(f"Available: {', '.join(DEFAULT_MAPPINGS.keys())}")
        return

    config = DEFAULT_MAPPINGS[name]

    print(f"\nMapping Preset: {config.name}")
    print("=" * 60)
    print(f"Description: {config.description}")
    print(f"Global Smoothing: {config.global_smoothing}")
    print(f"\nMappings:")

    for m in config.mappings:
        print(f"\n  {m.feature} -> {m.parameter}")
        print(f"    Range: [{m.min_value}, {m.max_value}]")
        print(f"    Curve: {m.curve.value}")
        if m.smoothing > 0:
            print(f"    Smoothing: {m.smoothing}")
        if m.threshold > 0:
            print(f"    Threshold: {m.threshold}")
        if m.invert:
            print(f"    Inverted: True")

    print(f"\nDefault Values:")
    for param, value in config.defaults.items():
        print(f"  {param}: {value}")


if __name__ == "__main__":
    sys.exit(main())
