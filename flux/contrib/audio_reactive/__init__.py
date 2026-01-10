"""
Audio Reactive Animation Module for Deforum/Parseq
===================================================

This module provides audio feature extraction and animation schedule generation
for creating audio-reactive animations with Deforum and Parseq.

Architecture
------------
::

    audio_reactive/
    ├── __init__.py          # Public API
    ├── types.py             # Core data types and enums
    ├── features.py          # AudioFeatures dataclass
    ├── extractor.py         # AudioFeatureExtractor
    ├── curves.py            # Easing curve functions
    ├── mapping.py           # Mapping configuration
    ├── presets.py           # Built-in mapping presets
    ├── generator.py         # Schedule generation
    ├── schedule.py          # Parseq/Deforum output formats
    └── cli.py               # Command-line interface

Quick Start
-----------
::

    from audio_reactive import extract_features, generate_schedule

    # One-liner
    schedule = generate_schedule("music.mp3", mapping="bass_pulse")
    schedule.save("schedule.json")

    # Or step-by-step
    features = extract_features("music.mp3", fps=24)
    schedule = generate_schedule(features, mapping="spectrum")
    deforum_strings = schedule.to_deforum_strings()

Available Presets
-----------------
- ``ambient``: Subtle, flowing movement
- ``bass_pulse``: Zoom pulses on bass hits
- ``beat_rotation``: Rotation synced to beats
- ``spectrum``: Bass/mid/high to different axes
- ``immersive_3d``: Full 3D rotation and depth
- ``cinematic``: Slow, smooth movements
- ``intense``: Aggressive high-energy movement

Requirements
------------
- numpy (required)
- librosa (required for audio extraction)
- audioread, soundfile (recommended for format support)

Install with::

    pip install numpy librosa audioread soundfile

"""

__version__ = "0.2.0"
__author__ = "Deforum Team"

# Core types
from .types import (
    FeatureType,
    ParameterType,
    CurveType,
    FEATURE_NAMES,
    PARAMETER_NAMES,
)

# Data structures
from .features import AudioFeatures
from .mapping import FeatureMapping, MappingConfig
from .schedule import ParseqKeyframe, ParseqSchedule

# Processing
from .curves import apply_curve, apply_smoothing
from .extractor import AudioFeatureExtractor, LIBROSA_AVAILABLE
from .generator import ScheduleGenerator

# Presets
from .presets import (
    PRESETS,
    get_preset,
    list_presets,
)

# Convenience functions
from .api import (
    extract_features,
    generate_schedule,
    create_mapping,
)

__all__ = [
    # Version
    "__version__",
    # Types
    "FeatureType",
    "ParameterType",
    "CurveType",
    "FEATURE_NAMES",
    "PARAMETER_NAMES",
    # Data structures
    "AudioFeatures",
    "FeatureMapping",
    "MappingConfig",
    "ParseqKeyframe",
    "ParseqSchedule",
    # Processing
    "apply_curve",
    "apply_smoothing",
    "AudioFeatureExtractor",
    "ScheduleGenerator",
    "LIBROSA_AVAILABLE",
    # Presets
    "PRESETS",
    "get_preset",
    "list_presets",
    # Convenience
    "extract_features",
    "generate_schedule",
    "create_mapping",
]
