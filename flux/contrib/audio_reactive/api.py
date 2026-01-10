"""
High-level convenience API for audio reactive animations.

This module provides simple functions for common workflows, hiding the
complexity of the underlying classes.

Examples
--------
One-liner to generate a schedule::

    from audio_reactive import generate_schedule

    schedule = generate_schedule("music.mp3", mapping="bass_pulse")
    schedule.save("schedule.json")

Step-by-step with features reuse::

    from audio_reactive import extract_features, generate_schedule

    features = extract_features("music.mp3", fps=24)

    # Try different mappings
    schedule1 = generate_schedule(features, mapping="bass_pulse")
    schedule2 = generate_schedule(features, mapping="spectrum")

Custom mapping::

    from audio_reactive import extract_features, generate_schedule, create_mapping

    features = extract_features("music.mp3", fps=24)

    my_mapping = create_mapping(
        "My Style",
        bass_to_zoom=(1.0, 1.3),
        beat_strength_to_angle=(-10, 10),
    )

    schedule = generate_schedule(features, mapping=my_mapping)
"""

from pathlib import Path
from typing import Dict, Optional, Union

from .features import AudioFeatures
from .mapping import MappingConfig, FeatureMapping
from .schedule import ParseqSchedule
from .extractor import AudioFeatureExtractor, LIBROSA_AVAILABLE
from .generator import ScheduleGenerator
from .presets import get_preset
from .types import CurveType


def extract_features(
    audio_path: Union[str, Path],
    fps: float = 24.0,
    duration: Optional[float] = None,
    start_time: float = 0.0,
    **kwargs,
) -> AudioFeatures:
    """
    Extract audio features from an audio file.

    Parameters
    ----------
    audio_path : str or Path
        Path to audio file (mp3, wav, flac, etc.).
    fps : float, default=24.0
        Video frame rate for alignment.
    duration : float, optional
        Duration in seconds. If None, processes entire file.
    start_time : float, default=0.0
        Start time in seconds.
    **kwargs
        Additional arguments passed to AudioFeatureExtractor.

    Returns
    -------
    AudioFeatures
        Extracted features aligned to video frames.

    Raises
    ------
    ImportError
        If librosa is not installed.
    FileNotFoundError
        If audio file does not exist.

    Examples
    --------
    >>> features = extract_features("music.mp3", fps=24, duration=60)
    >>> print(f"Tempo: {features.tempo} BPM")
    >>> print(f"Beats: {len(features.beats)}")
    """
    extractor = AudioFeatureExtractor(**kwargs)
    return extractor.extract(
        audio_path,
        fps=fps,
        duration=duration,
        start_time=start_time,
    )


def generate_schedule(
    source: Union[str, Path, AudioFeatures],
    mapping: Union[str, MappingConfig] = "bass_pulse",
    output_path: Optional[Union[str, Path]] = None,
    fps: float = 24.0,
    duration: Optional[float] = None,
    keyframe_interval: int = 1,
    prompt: str = "",
    **kwargs,
) -> ParseqSchedule:
    """
    Generate an animation schedule from audio.

    This is the main convenience function for creating audio-reactive
    animation schedules. It can accept either an audio file path or
    pre-extracted AudioFeatures.

    Parameters
    ----------
    source : str, Path, or AudioFeatures
        Audio file path or pre-extracted features.
    mapping : str or MappingConfig, default="bass_pulse"
        Mapping preset name or custom configuration.
    output_path : str or Path, optional
        If provided, saves schedule to this path.
    fps : float, default=24.0
        Video frame rate (only used if source is audio path).
    duration : float, optional
        Duration in seconds (only used if source is audio path).
    keyframe_interval : int, default=1
        Frames between keyframes.
    prompt : str
        Default prompt for all frames.
    **kwargs
        Additional arguments for schedule generation.

    Returns
    -------
    ParseqSchedule
        Generated animation schedule.

    Examples
    --------
    >>> # From audio file
    >>> schedule = generate_schedule("music.mp3", mapping="bass_pulse")
    >>> schedule.save("schedule.json")

    >>> # From pre-extracted features
    >>> features = extract_features("music.mp3", fps=30)
    >>> schedule = generate_schedule(features, mapping="spectrum")

    >>> # With custom mapping
    >>> config = create_mapping("My Style", bass_to_zoom=(1.0, 1.2))
    >>> schedule = generate_schedule("music.mp3", mapping=config)
    """
    # Extract features if needed
    if isinstance(source, AudioFeatures):
        features = source
        audio_path = None
    else:
        audio_path = str(source)
        features = extract_features(
            source,
            fps=fps,
            duration=duration,
        )

    # Generate schedule
    generator = ScheduleGenerator()
    schedule = generator.generate(
        features,
        mapping=mapping,
        keyframe_interval=keyframe_interval,
        prompt=prompt,
        audio_path=audio_path,
        **kwargs,
    )

    # Save if output path provided
    if output_path:
        schedule.save(output_path)

    return schedule


def create_mapping(
    name: str,
    description: str = "",
    global_smoothing: float = 0.1,
    **feature_to_param,
) -> MappingConfig:
    """
    Create a custom mapping configuration using a simple syntax.

    Parameters
    ----------
    name : str
        Configuration name.
    description : str
        Configuration description.
    global_smoothing : float
        Global smoothing amount.
    **feature_to_param
        Mapping specifications in format:
        ``feature_to_param=(min_value, max_value)`` or
        ``feature_to_param=(min_value, max_value, curve)``

    Returns
    -------
    MappingConfig
        Custom mapping configuration.

    Examples
    --------
    >>> config = create_mapping(
    ...     "Bass Zoom",
    ...     "Zoom in on bass hits",
    ...     bass_to_zoom=(1.0, 1.3),
    ...     beat_strength_to_angle=(-5, 5),
    ...     mid_to_translation_x=(-10, 10, "ease_out"),
    ... )
    """
    mappings = []

    for key, value in feature_to_param.items():
        # Parse "feature_to_param" format
        if "_to_" not in key:
            continue

        parts = key.split("_to_", 1)
        if len(parts) != 2:
            continue

        feature = parts[0]
        parameter = parts[1]

        # Parse value tuple
        if isinstance(value, tuple):
            if len(value) >= 2:
                min_val = value[0]
                max_val = value[1]
                curve = CurveType.LINEAR

                if len(value) >= 3:
                    curve_val = value[2]
                    if isinstance(curve_val, str):
                        curve = CurveType(curve_val)
                    elif isinstance(curve_val, CurveType):
                        curve = curve_val

                mappings.append(FeatureMapping(
                    feature=feature,
                    parameter=parameter,
                    min_value=min_val,
                    max_value=max_val,
                    curve=curve,
                ))

    return MappingConfig(
        name=name,
        description=description,
        mappings=mappings,
        global_smoothing=global_smoothing,
    )


def get_deforum_strings(
    source: Union[str, Path, AudioFeatures],
    mapping: Union[str, MappingConfig] = "bass_pulse",
    fps: float = 24.0,
    duration: Optional[float] = None,
) -> Dict[str, str]:
    """
    Get Deforum keyframe strings from audio.

    This is a convenience function that extracts features, generates
    a schedule, and returns Deforum-format keyframe strings.

    Parameters
    ----------
    source : str, Path, or AudioFeatures
        Audio file path or pre-extracted features.
    mapping : str or MappingConfig
        Mapping configuration.
    fps : float
        Video frame rate.
    duration : float, optional
        Duration in seconds.

    Returns
    -------
    Dict[str, str]
        Parameter names mapped to keyframe strings.
        Example: {"zoom": "0:(1.0), 12:(1.05), ..."}

    Examples
    --------
    >>> strings = get_deforum_strings("music.mp3", mapping="bass_pulse")
    >>> print(strings["zoom"])
    '0:(1.0), 1:(1.02), 2:(1.05), ...'
    """
    schedule = generate_schedule(
        source,
        mapping=mapping,
        fps=fps,
        duration=duration,
        keyframe_interval=1,
    )
    return schedule.to_deforum_strings()
