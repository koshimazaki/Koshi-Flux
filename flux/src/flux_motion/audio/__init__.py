"""Audio feature extraction and schedule generation for Deforum animations."""

from .bridge import (
    AudioMotionResult,
    FeatureTracks,
    build_audio_motion_schedule,
    frame_bands,
    to_deforum_string,
    tracks_from_analysis_file,
    tracks_from_analysis_json,
    tracks_from_audio_features,
    tracks_from_audio_file,
    tracks_from_video,
    tracks_from_waveform,
)
from .extractor import AudioFeatureExtractor, AudioFeatures
from .mapping_config import (
    DEFAULT_MAPPINGS,
    FeatureMapping,
    MappingConfig,
    load_mapping_config,
    save_mapping_config,
)
from .schedule_generator import ParseqSchedule, ScheduleGenerator

__all__ = [
    "AudioFeatureExtractor",
    "AudioFeatures",
    "ScheduleGenerator",
    "ParseqSchedule",
    "MappingConfig",
    "FeatureMapping",
    "DEFAULT_MAPPINGS",
    "load_mapping_config",
    "save_mapping_config",
    "AudioMotionResult",
    "FeatureTracks",
    "build_audio_motion_schedule",
    "frame_bands",
    "to_deforum_string",
    "tracks_from_analysis_file",
    "tracks_from_analysis_json",
    "tracks_from_audio_features",
    "tracks_from_audio_file",
    "tracks_from_video",
    "tracks_from_waveform",
]
