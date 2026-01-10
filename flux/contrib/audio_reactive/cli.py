#!/usr/bin/env python3
"""
Command-line interface for audio reactive animation generation.

Usage
-----
::

    # Generate schedule from audio
    python -m audio_reactive music.mp3 -o schedule.json

    # Use specific preset
    python -m audio_reactive music.mp3 -m bass_pulse -o schedule.json

    # List available presets
    python -m audio_reactive --list-presets

    # Show Deforum strings
    python -m audio_reactive music.mp3 --format deforum
"""

import argparse
import json
import sys
from pathlib import Path


def main(args=None):
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="audio_reactive",
        description="Generate audio-reactive animation schedules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m audio_reactive music.mp3 -o schedule.json

  # Use specific preset
  python -m audio_reactive music.mp3 -m spectrum -o schedule.json

  # Custom settings
  python -m audio_reactive music.mp3 --fps 30 --duration 60 -o out.json

  # Show Deforum strings
  python -m audio_reactive music.mp3 --format deforum

Presets:
  ambient       Subtle, flowing movement
  bass_pulse    Zoom pulses on bass hits
  beat_rotation Rotation synced to beats
  spectrum      Bass/mid/high to different axes
  immersive_3d  Full 3D rotation and depth
  cinematic     Slow, smooth movements
  intense       Aggressive high-energy
"""
    )

    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to audio file (mp3, wav, flac, etc.)"
    )

    parser.add_argument(
        "-o", "--output",
        default="schedule.json",
        help="Output path (default: schedule.json)"
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
        help="Frames between keyframes (default: 1)"
    )

    parser.add_argument(
        "--format",
        choices=["parseq", "deforum", "both"],
        default="parseq",
        help="Output format (default: parseq)"
    )

    parser.add_argument(
        "--save-features",
        metavar="PATH",
        help="Save extracted features to JSON"
    )

    parser.add_argument(
        "--load-features",
        metavar="PATH",
        help="Load features from JSON instead of extracting"
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit"
    )

    parser.add_argument(
        "--show-preset",
        metavar="NAME",
        help="Show details of a preset and exit"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit"
    )

    parsed = parser.parse_args(args)

    # Handle info commands
    if parsed.version:
        from . import __version__
        print(f"audio_reactive {__version__}")
        return 0

    if parsed.list_presets:
        return cmd_list_presets()

    if parsed.show_preset:
        return cmd_show_preset(parsed.show_preset)

    # Require audio file or features
    if not parsed.audio_file and not parsed.load_features:
        parser.error("Audio file required (or use --load-features)")

    return cmd_generate(parsed)


def cmd_list_presets():
    """List available presets."""
    from .presets import PRESETS

    print("\nAvailable Mapping Presets")
    print("=" * 50)

    for name, config in sorted(PRESETS.items()):
        print(f"\n{name}")
        print(f"  {config.description}")
        for m in config.mappings:
            print(f"    {m.feature} → {m.parameter} [{m.min_value}, {m.max_value}]")

    return 0


def cmd_show_preset(name: str):
    """Show preset details."""
    from .presets import PRESETS, describe_preset

    if name not in PRESETS:
        print(f"Error: Unknown preset '{name}'")
        print(f"Available: {', '.join(sorted(PRESETS.keys()))}")
        return 1

    print(describe_preset(name))
    return 0


def cmd_generate(args):
    """Generate schedule."""
    from .extractor import LIBROSA_AVAILABLE

    # Check librosa
    if not LIBROSA_AVAILABLE and not args.load_features:
        print("Error: librosa is required for audio extraction.")
        print("Install: pip install librosa audioread soundfile")
        return 1

    from .features import AudioFeatures
    from .extractor import AudioFeatureExtractor
    from .generator import ScheduleGenerator
    from .presets import PRESETS

    # Validate preset
    if args.mapping not in PRESETS:
        print(f"Error: Unknown preset '{args.mapping}'")
        print(f"Available: {', '.join(sorted(PRESETS.keys()))}")
        return 1

    print(f"\n{'='*50}")
    print("Audio Reactive Schedule Generator")
    print(f"{'='*50}")

    # Load or extract features
    if args.load_features:
        print(f"\nLoading features from: {args.load_features}")
        features = AudioFeatures.load(args.load_features)
    else:
        print(f"\nExtracting features from: {args.audio_file}")
        extractor = AudioFeatureExtractor()
        features = extractor.extract(
            args.audio_file,
            fps=args.fps,
            duration=args.duration,
            start_time=args.start,
        )

    # Show summary
    print(f"\nAudio Analysis:")
    print(f"  Duration: {features.duration:.2f}s")
    print(f"  Tempo:    {features.tempo:.1f} BPM")
    print(f"  Frames:   {features.num_frames} @ {features.fps}fps")
    print(f"  Beats:    {len(features.beats)} detected")

    # Save features if requested
    if args.save_features:
        features.save(args.save_features)
        print(f"\nSaved features to: {args.save_features}")

    # Generate schedule
    print(f"\nGenerating schedule with '{args.mapping}' preset...")
    generator = ScheduleGenerator()
    schedule = generator.generate(
        features,
        mapping=args.mapping,
        keyframe_interval=args.keyframe_interval,
        audio_path=args.audio_file,
    )

    print(f"  Generated {len(schedule.keyframes)} keyframes")

    # Output
    if args.format in ("parseq", "both"):
        schedule.save(args.output)
        print(f"\nSaved Parseq schedule: {args.output}")

    if args.format in ("deforum", "both"):
        strings = schedule.to_deforum_strings()

        print(f"\n{'='*50}")
        print("Deforum Keyframe Strings")
        print(f"{'='*50}\n")

        for param, value in strings.items():
            # Truncate for display
            if len(value) > 100:
                display = value[:50] + " ... " + value[-50:]
            else:
                display = value
            print(f"{param}:")
            print(f"  {display}\n")

        if args.format == "both":
            deforum_path = Path(args.output).with_suffix(".deforum.json")
            with open(deforum_path, 'w') as f:
                json.dump(strings, f, indent=2)
            print(f"Saved Deforum strings: {deforum_path}")

    # Verbose stats
    if args.verbose:
        print(f"\n{'='*50}")
        print("Feature Statistics")
        print(f"{'='*50}\n")

        for name in features.list_features():
            arr = features.get_feature(name)
            print(
                f"{name:20s}: "
                f"min={arr.min():.3f}, max={arr.max():.3f}, "
                f"mean={arr.mean():.3f}"
            )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
