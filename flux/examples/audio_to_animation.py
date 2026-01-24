#!/usr/bin/env python3
"""
Audio-Driven Animation Example for Koshi FLUX

This example demonstrates the complete workflow:
1. Extract audio features from a music file
2. Map features to animation parameters
3. Generate Parseq-compatible schedule JSON
4. Get Deforum keyframe strings

Requirements:
    pip install deforum-flux[audio]
    # or: pip install librosa audioread soundfile

Usage:
    python audio_to_animation.py path/to/music.mp3 -o schedule.json
    python audio_to_animation.py music.mp3 -m bass_pulse --fps 30
    python audio_to_animation.py music.mp3 --list-presets
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Deforum/Parseq schedules from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio_file", nargs="?", help="Path to audio file")
    parser.add_argument("-o", "--output", default="schedule.json", help="Output JSON path")
    parser.add_argument("-m", "--mapping", default="bass_pulse", help="Mapping preset")
    parser.add_argument("--fps", type=float, default=24.0, help="Video FPS")
    parser.add_argument("--duration", type=float, help="Duration in seconds")
    parser.add_argument("--list-presets", action="store_true", help="List presets")
    parser.add_argument("--show-deforum", action="store_true", help="Show Deforum strings")

    args = parser.parse_args()

    # Try to import audio module
    try:
        from koshi_flux.audio import (
            AudioFeatureExtractor,
            ScheduleGenerator,
            DEFAULT_MAPPINGS,
        )
    except ImportError:
        print("ERROR: Audio module requires librosa.")
        print("Install with: pip install deforum-flux[audio]")
        print("Or: pip install librosa audioread soundfile")
        return 1

    # List presets
    if args.list_presets:
        print("\nAvailable Mapping Presets:")
        print("=" * 50)
        for name, config in DEFAULT_MAPPINGS.items():
            print(f"\n{name}")
            print(f"  {config.description}")
            for m in config.mappings:
                print(f"    {m.feature} -> {m.parameter} [{m.min_value}, {m.max_value}]")
        return 0

    # Require audio file
    if not args.audio_file:
        parser.error("Audio file is required")

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return 1

    # Validate mapping
    if args.mapping not in DEFAULT_MAPPINGS:
        print(f"ERROR: Unknown mapping: {args.mapping}")
        print(f"Available: {', '.join(DEFAULT_MAPPINGS.keys())}")
        return 1

    print(f"\n{'='*60}")
    print("AUDIO-DRIVEN ANIMATION GENERATOR")
    print(f"{'='*60}")

    # Step 1: Extract features
    print(f"\n[1/3] Extracting features from: {audio_path.name}")
    extractor = AudioFeatureExtractor()
    features = extractor.extract(
        audio_path,
        fps=args.fps,
        duration=args.duration,
    )

    print(f"      Duration: {features.duration:.2f}s")
    print(f"      Tempo: {features.tempo:.1f} BPM")
    print(f"      Frames: {features.num_frames} @ {features.fps}fps")
    print(f"      Beats: {len(features.beats)}")

    # Step 2: Generate schedule
    print(f"\n[2/3] Generating schedule with '{args.mapping}' mapping")
    generator = ScheduleGenerator()
    schedule = generator.generate(
        features,
        mapping=args.mapping,
        audio_path=str(audio_path),
    )
    print(f"      Generated {len(schedule.keyframes)} keyframes")

    # Step 3: Save output
    print(f"\n[3/3] Saving to: {args.output}")
    schedule.save(args.output)

    # Show Deforum strings if requested
    if args.show_deforum:
        deforum_strings = schedule.to_deforum_strings()
        print(f"\n{'='*60}")
        print("DEFORUM KEYFRAME STRINGS")
        print(f"{'='*60}")
        for param, value in deforum_strings.items():
            # Truncate for display
            if len(value) > 100:
                display = value[:50] + " ... " + value[-50:]
            else:
                display = value
            print(f"\n{param}:")
            print(f"  {display}")

    print(f"\n{'='*60}")
    print("DONE! Schedule saved to:", args.output)
    print(f"{'='*60}")

    # Usage tips
    print("\nNext steps:")
    print("  1. Open schedule.json in Parseq or a text editor")
    print("  2. Copy keyframe data into your Deforum settings")
    print("  3. Or use schedule.to_deforum_strings() in Python")
    print("\nTo try different mappings:")
    print(f"  python {sys.argv[0]} {audio_path} -m spectrum")
    print(f"  python {sys.argv[0]} --list-presets")

    return 0


if __name__ == "__main__":
    sys.exit(main())
