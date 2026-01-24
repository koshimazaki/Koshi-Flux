"""Audio feature extraction and schedule generation for Deforum animations."""

from .extractor import AudioFeatureExtractor, AudioFeatures
from .schedule_generator import ScheduleGenerator, ParseqSchedule
from .mapping_config import (
    MappingConfig,
    FeatureMapping,
    DEFAULT_MAPPINGS,
    load_mapping_config,
    save_mapping_config,
)

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
]
