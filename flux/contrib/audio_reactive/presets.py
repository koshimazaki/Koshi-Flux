"""
Built-in mapping presets for common use cases.

This module provides ready-to-use mapping configurations for various
styles of audio-reactive animation.

Available Presets
-----------------
- ``ambient``: Subtle, flowing movement for atmospheric content
- ``bass_pulse``: Strong zoom pulses on bass hits
- ``beat_rotation``: Rotation and movement synchronized to beats
- ``spectrum``: Maps bass/mid/high to different motion axes
- ``immersive_3d``: Full 3D rotation and depth movement
- ``cinematic``: Slow, smooth movements for film-like content
- ``intense``: Aggressive movement for high-energy content

Usage
-----
::

    from audio_reactive import get_preset, list_presets

    # List available presets
    print(list_presets())

    # Get a preset
    config = get_preset("bass_pulse")
"""

from typing import Dict, List

from .types import CurveType
from .mapping import MappingConfig, FeatureMapping


def _create_presets() -> Dict[str, MappingConfig]:
    """Create all preset configurations."""

    presets = {}

    # =========================================================================
    # AMBIENT - Subtle, flowing movement
    # =========================================================================
    presets["ambient"] = MappingConfig(
        name="Ambient",
        description="Subtle, flowing movement for atmospheric content",
        mappings=[
            FeatureMapping(
                feature="energy",
                parameter="zoom",
                min_value=1.0,
                max_value=1.02,
                curve=CurveType.EASE_IN_OUT,
                smoothing=0.5,
            ),
            FeatureMapping(
                feature="spectral_centroid",
                parameter="translation_x",
                min_value=-5,
                max_value=5,
                curve=CurveType.SINE,
                smoothing=0.4,
            ),
            FeatureMapping(
                feature="spectral_bandwidth",
                parameter="translation_y",
                min_value=-3,
                max_value=3,
                curve=CurveType.SINE,
                smoothing=0.4,
            ),
        ],
        global_smoothing=0.3,
    )

    # =========================================================================
    # BASS PULSE - Strong zoom pulses on bass
    # =========================================================================
    presets["bass_pulse"] = MappingConfig(
        name="Bass Pulse",
        description="Strong zoom pulses on bass hits",
        mappings=[
            FeatureMapping(
                feature="bass",
                parameter="zoom",
                min_value=1.0,
                max_value=1.15,
                curve=CurveType.EASE_OUT,
                threshold=0.2,
            ),
            FeatureMapping(
                feature="beat_strength",
                parameter="strength",
                min_value=0.55,
                max_value=0.75,
                curve=CurveType.EASE_OUT,
            ),
            FeatureMapping(
                feature="mid",
                parameter="angle",
                min_value=-3,
                max_value=3,
                curve=CurveType.LINEAR,
                smoothing=0.2,
            ),
        ],
        global_smoothing=0.1,
    )

    # =========================================================================
    # BEAT ROTATION - Movement synchronized to beats
    # =========================================================================
    presets["beat_rotation"] = MappingConfig(
        name="Beat Rotation",
        description="Rotation and movement synchronized to beats",
        mappings=[
            FeatureMapping(
                feature="beat_strength",
                parameter="angle",
                min_value=-5,
                max_value=5,
                curve=CurveType.EASE_OUT,
            ),
            FeatureMapping(
                feature="beat_strength",
                parameter="zoom",
                min_value=1.0,
                max_value=1.08,
                curve=CurveType.EASE_OUT,
            ),
            FeatureMapping(
                feature="onset_strength",
                parameter="translation_z",
                min_value=0,
                max_value=15,
                curve=CurveType.EASE_OUT,
                threshold=0.3,
            ),
        ],
        global_smoothing=0.05,
    )

    # =========================================================================
    # SPECTRUM - Full frequency spectrum mapping
    # =========================================================================
    presets["spectrum"] = MappingConfig(
        name="Full Spectrum",
        description="Maps bass/mid/high to different motion axes",
        mappings=[
            FeatureMapping(
                feature="bass",
                parameter="zoom",
                min_value=1.0,
                max_value=1.12,
                curve=CurveType.EASE_OUT,
            ),
            FeatureMapping(
                feature="mid",
                parameter="translation_x",
                min_value=-10,
                max_value=10,
                curve=CurveType.LINEAR,
                smoothing=0.2,
            ),
            FeatureMapping(
                feature="high",
                parameter="translation_y",
                min_value=-8,
                max_value=8,
                curve=CurveType.EASE_IN_OUT,
                smoothing=0.3,
            ),
            FeatureMapping(
                feature="spectral_centroid",
                parameter="angle",
                min_value=-4,
                max_value=4,
                curve=CurveType.LINEAR,
                smoothing=0.2,
            ),
        ],
        global_smoothing=0.15,
    )

    # =========================================================================
    # IMMERSIVE 3D - Full 3D rotation and depth
    # =========================================================================
    presets["immersive_3d"] = MappingConfig(
        name="Immersive 3D",
        description="Full 3D rotation and depth movement",
        mappings=[
            FeatureMapping(
                feature="bass",
                parameter="translation_z",
                min_value=0,
                max_value=20,
                curve=CurveType.EASE_OUT,
            ),
            FeatureMapping(
                feature="mid",
                parameter="rotation_3d_x",
                min_value=-5,
                max_value=5,
                curve=CurveType.SINE,
                smoothing=0.3,
            ),
            FeatureMapping(
                feature="high",
                parameter="rotation_3d_y",
                min_value=-5,
                max_value=5,
                curve=CurveType.SINE,
                smoothing=0.3,
            ),
            FeatureMapping(
                feature="beat_strength",
                parameter="rotation_3d_z",
                min_value=-3,
                max_value=3,
                curve=CurveType.EASE_OUT,
            ),
            FeatureMapping(
                feature="energy",
                parameter="zoom",
                min_value=1.0,
                max_value=1.05,
                curve=CurveType.EASE_IN_OUT,
                smoothing=0.4,
            ),
        ],
        global_smoothing=0.2,
    )

    # =========================================================================
    # CINEMATIC - Slow, smooth movements
    # =========================================================================
    presets["cinematic"] = MappingConfig(
        name="Cinematic",
        description="Slow, smooth movements for cinematic content",
        mappings=[
            FeatureMapping(
                feature="energy",
                parameter="zoom",
                min_value=1.0,
                max_value=1.03,
                curve=CurveType.EASE_IN_OUT,
                smoothing=0.7,
            ),
            FeatureMapping(
                feature="spectral_centroid",
                parameter="translation_x",
                min_value=-3,
                max_value=3,
                curve=CurveType.SINE,
                smoothing=0.6,
            ),
            FeatureMapping(
                feature="spectral_flatness",
                parameter="strength",
                min_value=0.6,
                max_value=0.7,
                curve=CurveType.LINEAR,
                smoothing=0.5,
            ),
        ],
        global_smoothing=0.5,
    )

    # =========================================================================
    # INTENSE - Aggressive high-energy movement
    # =========================================================================
    presets["intense"] = MappingConfig(
        name="Intense",
        description="Aggressive movement for high-energy content",
        mappings=[
            FeatureMapping(
                feature="bass",
                parameter="zoom",
                min_value=1.0,
                max_value=1.25,
                curve=CurveType.EXPONENTIAL,
            ),
            FeatureMapping(
                feature="beat_strength",
                parameter="angle",
                min_value=-10,
                max_value=10,
                curve=CurveType.EASE_OUT,
            ),
            FeatureMapping(
                feature="onset_strength",
                parameter="translation_z",
                min_value=0,
                max_value=30,
                curve=CurveType.EXPONENTIAL,
                threshold=0.4,
            ),
            FeatureMapping(
                feature="high",
                parameter="noise",
                min_value=0.01,
                max_value=0.05,
                curve=CurveType.LINEAR,
            ),
            FeatureMapping(
                feature="energy",
                parameter="strength",
                min_value=0.5,
                max_value=0.8,
                curve=CurveType.EASE_OUT,
            ),
        ],
        global_smoothing=0.05,
    )

    # =========================================================================
    # PARSEQ STYLE - Strength drop on beats (classic music video look)
    # =========================================================================
    presets["parseq_drop"] = MappingConfig(
        name="Parseq Drop",
        description="Strength drops on beats for scene changes (Parseq-style)",
        mappings=[
            # DROP strength on beats (invert=True means loud=low strength)
            FeatureMapping(
                feature="beat_strength",
                parameter="strength",
                min_value=0.3,  # Low strength on beat = more change
                max_value=0.75,  # High strength between beats = stable
                curve=CurveType.EASE_OUT,
                invert=True,  # KEY: Inverts so beats DROP strength
                threshold=0.3,
            ),
            # Increase seed increment on beats for variation
            FeatureMapping(
                feature="onset_strength",
                parameter="seed_increment",
                min_value=0,  # No seed change normally
                max_value=5,  # Jump seed on strong onsets
                curve=CurveType.EASE_OUT,
                threshold=0.5,
            ),
            # Subtle zoom pulse
            FeatureMapping(
                feature="bass",
                parameter="zoom",
                min_value=1.0,
                max_value=1.08,
                curve=CurveType.EASE_OUT,
            ),
        ],
        global_smoothing=0.05,
    )

    # =========================================================================
    # BEAT MORPH - Scene changes on every beat
    # =========================================================================
    presets["beat_morph"] = MappingConfig(
        name="Beat Morph",
        description="Morphs/changes scene on every beat with seed jumps",
        mappings=[
            # Sharp strength drop on beats
            FeatureMapping(
                feature="beat_strength",
                parameter="strength",
                min_value=0.2,  # Very low = big change on beat
                max_value=0.7,
                curve=CurveType.EXPONENTIAL,
                invert=True,
            ),
            # Seed jumps on beats
            FeatureMapping(
                feature="beat_strength",
                parameter="seed_increment",
                min_value=0,
                max_value=10,  # Big seed jump = different image
                curve=CurveType.EASE_OUT,
                threshold=0.4,
            ),
            # CFG scale increases on beats for sharper images
            FeatureMapping(
                feature="beat_strength",
                parameter="cfg_scale",
                min_value=3.5,
                max_value=7.0,
                curve=CurveType.EASE_OUT,
            ),
        ],
        global_smoothing=0.02,  # Very snappy
    )

    # =========================================================================
    # SMOOTH RIDE - Consistent scene with subtle audio response
    # =========================================================================
    presets["smooth_ride"] = MappingConfig(
        name="Smooth Ride",
        description="High scene consistency with gentle audio modulation",
        mappings=[
            # Keep strength HIGH (stable scene)
            FeatureMapping(
                feature="energy",
                parameter="strength",
                min_value=0.7,  # Never too low
                max_value=0.85,
                curve=CurveType.EASE_IN_OUT,
                smoothing=0.5,
            ),
            # No seed changes - stay consistent
            FeatureMapping(
                feature="bass",
                parameter="seed_increment",
                min_value=0,
                max_value=0,  # Fixed at 0
                curve=CurveType.LINEAR,
            ),
            # Very subtle zoom
            FeatureMapping(
                feature="bass",
                parameter="zoom",
                min_value=1.0,
                max_value=1.03,
                curve=CurveType.EASE_IN_OUT,
                smoothing=0.4,
            ),
            # Gentle angle sway
            FeatureMapping(
                feature="mid",
                parameter="angle",
                min_value=-2,
                max_value=2,
                curve=CurveType.SINE,
                smoothing=0.5,
            ),
        ],
        global_smoothing=0.4,
    )

    # =========================================================================
    # DRUM REACTIVE - Responds to kick/snare patterns
    # =========================================================================
    presets["drum_reactive"] = MappingConfig(
        name="Drum Reactive",
        description="Strength and zoom react to drum hits",
        mappings=[
            # Kick drum = zoom in + strength drop
            FeatureMapping(
                feature="bass",
                parameter="zoom",
                min_value=1.0,
                max_value=1.15,
                curve=CurveType.EASE_OUT,
                threshold=0.3,
            ),
            FeatureMapping(
                feature="bass",
                parameter="strength",
                min_value=0.4,
                max_value=0.7,
                curve=CurveType.EASE_OUT,
                invert=True,  # Drop on kick
                threshold=0.3,
            ),
            # Snare/high = angle snap
            FeatureMapping(
                feature="high",
                parameter="angle",
                min_value=-8,
                max_value=8,
                curve=CurveType.EASE_OUT,
                threshold=0.4,
            ),
            # Mid = translation
            FeatureMapping(
                feature="mid",
                parameter="translation_x",
                min_value=-15,
                max_value=15,
                curve=CurveType.LINEAR,
                smoothing=0.2,
            ),
        ],
        global_smoothing=0.08,
    )

    return presets


# Global preset registry
PRESETS: Dict[str, MappingConfig] = _create_presets()


def list_presets() -> List[str]:
    """
    List available preset names.

    Returns
    -------
    List[str]
        Sorted list of preset names.
    """
    return sorted(PRESETS.keys())


def get_preset(name: str) -> MappingConfig:
    """
    Get a preset mapping configuration by name.

    Parameters
    ----------
    name : str
        Preset name (case-sensitive).

    Returns
    -------
    MappingConfig
        Copy of the preset configuration.

    Raises
    ------
    ValueError
        If preset name is not recognized.

    Examples
    --------
    >>> config = get_preset("bass_pulse")
    >>> print(config.name)
    'Bass Pulse'
    """
    if name not in PRESETS:
        available = ", ".join(list_presets())
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {available}"
        )

    # Return a copy to prevent modification of originals
    original = PRESETS[name]
    return MappingConfig.from_dict(original.to_dict())


def describe_preset(name: str) -> str:
    """
    Get a detailed description of a preset.

    Parameters
    ----------
    name : str
        Preset name.

    Returns
    -------
    str
        Multi-line description of the preset.
    """
    config = get_preset(name)
    return str(config)
