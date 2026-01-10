"""
Comprehensive tests for audio_reactive module.

Run with: python -m pytest tests/test_all.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import all components
from audio_reactive.types import (
    FeatureType,
    ParameterType,
    CurveType,
    FEATURE_NAMES,
    PARAMETER_NAMES,
    validate_feature,
    validate_parameter,
)
from audio_reactive.curves import (
    apply_curve,
    apply_smoothing,
    normalize,
    interpolate_frames,
)
from audio_reactive.features import AudioFeatures
from audio_reactive.mapping import FeatureMapping, MappingConfig
from audio_reactive.presets import PRESETS, get_preset, list_presets
from audio_reactive.schedule import ParseqKeyframe, ParseqSchedule
from audio_reactive.generator import ScheduleGenerator
from audio_reactive.api import create_mapping, generate_schedule


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_features():
    """Create sample audio features for testing."""
    num_frames = 100
    return AudioFeatures(
        duration=4.17,
        sample_rate=44100,
        num_frames=num_frames,
        fps=24.0,
        hop_length=1837,
        tempo=120.0,
        times=np.linspace(0, 4.17, num_frames),
        rms=np.random.random(num_frames),
        energy=np.random.random(num_frames),
        spectral_centroid=np.random.random(num_frames),
        spectral_bandwidth=np.random.random(num_frames),
        spectral_rolloff=np.random.random(num_frames),
        spectral_flatness=np.random.random(num_frames),
        bass=np.abs(np.sin(np.linspace(0, 4 * np.pi, num_frames))),
        mid=np.random.random(num_frames),
        high=np.random.random(num_frames),
        beats=np.array([0, 12, 24, 36, 48, 60, 72, 84, 96]),
        beat_strength=np.zeros(num_frames),
        onset_strength=np.random.random(num_frames),
        chroma=np.random.random((num_frames, 12)),
    )


# =============================================================================
# Type Tests
# =============================================================================

class TestTypes:
    """Test type definitions."""

    def test_feature_enum_values(self):
        """All feature types should have string values."""
        for ft in FeatureType:
            assert isinstance(ft.value, str)
            assert ft.value in FEATURE_NAMES

    def test_parameter_enum_values(self):
        """All parameter types should have string values."""
        for pt in ParameterType:
            assert isinstance(pt.value, str)
            assert pt.value in PARAMETER_NAMES

    def test_curve_enum_values(self):
        """All curve types should have string values."""
        for ct in CurveType:
            assert isinstance(ct.value, str)

    def test_validate_feature(self):
        """Feature validation should work correctly."""
        assert validate_feature("bass") is True
        assert validate_feature("invalid") is False

    def test_validate_parameter(self):
        """Parameter validation should work correctly."""
        assert validate_parameter("zoom") is True
        assert validate_parameter("invalid") is False


# =============================================================================
# Curve Tests
# =============================================================================

class TestCurves:
    """Test curve functions."""

    def test_linear_curve(self):
        """Linear curve returns unchanged values."""
        values = np.array([0.0, 0.5, 1.0])
        result = apply_curve(values, CurveType.LINEAR)
        np.testing.assert_array_almost_equal(result, values)

    def test_ease_in_curve(self):
        """Ease-in starts slow."""
        values = np.array([0.0, 0.5, 1.0])
        result = apply_curve(values, CurveType.EASE_IN)
        assert result[1] < 0.5  # Middle value should be below linear

    def test_ease_out_curve(self):
        """Ease-out ends slow."""
        values = np.array([0.0, 0.5, 1.0])
        result = apply_curve(values, CurveType.EASE_OUT)
        assert result[1] > 0.5  # Middle value should be above linear

    def test_curve_from_string(self):
        """Curves can be specified as strings."""
        values = np.array([0.0, 0.5, 1.0])
        result = apply_curve(values, "ease_out")
        assert result[1] > 0.5

    def test_smoothing_reduces_variance(self):
        """Smoothing should reduce signal variance."""
        values = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        smoothed = apply_smoothing(values, 0.5)
        assert np.var(smoothed) < np.var(values)

    def test_no_smoothing(self):
        """Zero smoothing returns unchanged values."""
        values = np.array([0.0, 1.0, 0.0])
        result = apply_smoothing(values, 0.0)
        np.testing.assert_array_almost_equal(result, values)

    def test_normalize(self):
        """Normalize should produce 0-1 range."""
        values = np.array([10.0, 20.0, 30.0])
        result = normalize(values)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_interpolate_frames(self):
        """Interpolation should change array length."""
        values = np.array([0.0, 1.0, 0.0])
        result = interpolate_frames(values, 5)
        assert len(result) == 5


# =============================================================================
# Features Tests
# =============================================================================

class TestAudioFeatures:
    """Test AudioFeatures class."""

    def test_get_feature(self, sample_features):
        """Can retrieve features by name."""
        bass = sample_features.get_feature("bass")
        assert len(bass) == sample_features.num_frames

    def test_get_invalid_feature(self, sample_features):
        """Invalid feature raises ValueError."""
        with pytest.raises(ValueError):
            sample_features.get_feature("invalid")

    def test_list_features(self, sample_features):
        """List features returns correct names."""
        features = sample_features.list_features()
        assert "bass" in features
        assert "beat_strength" in features

    def test_serialization(self, sample_features):
        """Features can be serialized to dict and back."""
        data = sample_features.to_dict()
        restored = AudioFeatures.from_dict(data)
        assert restored.num_frames == sample_features.num_frames
        np.testing.assert_array_almost_equal(
            restored.bass, sample_features.bass
        )

    def test_save_load(self, sample_features, tmp_path):
        """Features can be saved and loaded."""
        path = tmp_path / "features.json"
        sample_features.save(path)
        loaded = AudioFeatures.load(path)
        assert loaded.num_frames == sample_features.num_frames


# =============================================================================
# Mapping Tests
# =============================================================================

class TestMapping:
    """Test mapping configuration."""

    def test_feature_mapping_creation(self):
        """Can create a feature mapping."""
        m = FeatureMapping(
            feature="bass",
            parameter="zoom",
            min_value=1.0,
            max_value=1.2,
        )
        assert m.feature == "bass"
        assert m.parameter == "zoom"

    def test_feature_mapping_serialization(self):
        """Mapping can be serialized and restored."""
        m = FeatureMapping(
            feature="bass",
            parameter="zoom",
            min_value=1.0,
            max_value=1.2,
            curve=CurveType.EASE_OUT,
        )
        data = m.to_dict()
        restored = FeatureMapping.from_dict(data)
        assert restored.feature == m.feature
        assert restored.curve == m.curve

    def test_mapping_config_creation(self):
        """Can create a mapping config."""
        config = MappingConfig(
            name="Test",
            mappings=[
                FeatureMapping("bass", "zoom", 1.0, 1.2),
            ],
        )
        assert config.name == "Test"
        assert len(config.mappings) == 1

    def test_mapping_config_add(self):
        """Can add mappings with fluent API."""
        config = MappingConfig(name="Test")
        config.add_mapping("bass", "zoom", 1.0, 1.2)
        config.add_mapping("mid", "angle", -5, 5)
        assert len(config.mappings) == 2

    def test_mapping_config_serialization(self):
        """Config can be serialized and restored."""
        config = MappingConfig(
            name="Test",
            mappings=[FeatureMapping("bass", "zoom", 1.0, 1.2)],
        )
        data = config.to_dict()
        restored = MappingConfig.from_dict(data)
        assert restored.name == config.name


# =============================================================================
# Presets Tests
# =============================================================================

class TestPresets:
    """Test preset configurations."""

    def test_all_presets_exist(self):
        """Expected presets should exist."""
        expected = [
            "ambient", "bass_pulse", "beat_rotation",
            "spectrum", "immersive_3d", "cinematic", "intense"
        ]
        for name in expected:
            assert name in PRESETS

    def test_list_presets(self):
        """List presets returns names."""
        names = list_presets()
        assert "bass_pulse" in names

    def test_get_preset(self):
        """Can get preset by name."""
        config = get_preset("bass_pulse")
        assert config.name == "Bass Pulse"
        assert len(config.mappings) > 0

    def test_get_preset_returns_copy(self):
        """Get preset returns a copy, not the original."""
        config1 = get_preset("bass_pulse")
        config2 = get_preset("bass_pulse")
        config1.name = "Modified"
        assert config2.name == "Bass Pulse"

    def test_invalid_preset(self):
        """Invalid preset raises ValueError."""
        with pytest.raises(ValueError):
            get_preset("invalid_preset")


# =============================================================================
# Schedule Tests
# =============================================================================

class TestSchedule:
    """Test schedule generation."""

    def test_keyframe_creation(self):
        """Can create a keyframe."""
        kf = ParseqKeyframe(frame=0, zoom=1.0, angle=5.0)
        assert kf.frame == 0
        assert kf.zoom == 1.0

    def test_keyframe_to_dict(self):
        """Keyframe serialization omits None values."""
        kf = ParseqKeyframe(frame=0, zoom=1.0)
        data = kf.to_dict()
        assert "frame" in data
        assert "zoom" in data
        assert "angle" not in data  # None value

    def test_schedule_creation(self):
        """Can create a schedule."""
        schedule = ParseqSchedule(
            name="Test",
            fps=24,
            num_frames=100,
            keyframes=[
                ParseqKeyframe(frame=0, zoom=1.0),
                ParseqKeyframe(frame=99, zoom=1.2),
            ],
        )
        assert len(schedule.keyframes) == 2

    def test_schedule_to_deforum_strings(self):
        """Can convert to Deforum format."""
        schedule = ParseqSchedule(
            keyframes=[
                ParseqKeyframe(frame=0, zoom=1.0),
                ParseqKeyframe(frame=30, zoom=1.1),
                ParseqKeyframe(frame=60, zoom=1.0),
            ],
        )
        strings = schedule.to_deforum_strings()
        assert "zoom" in strings
        assert "0:(1.0)" in strings["zoom"]

    def test_schedule_serialization(self, tmp_path):
        """Schedule can be saved and loaded."""
        schedule = ParseqSchedule(
            name="Test",
            keyframes=[ParseqKeyframe(frame=0, zoom=1.0)],
        )
        path = tmp_path / "schedule.json"
        schedule.save(path)
        loaded = ParseqSchedule.load(path)
        assert loaded.name == schedule.name


# =============================================================================
# Generator Tests
# =============================================================================

class TestGenerator:
    """Test schedule generation from features."""

    def test_generate_with_preset(self, sample_features):
        """Can generate schedule with preset."""
        generator = ScheduleGenerator()
        schedule = generator.generate(sample_features, mapping="bass_pulse")
        assert len(schedule.keyframes) > 0
        assert schedule.num_frames == sample_features.num_frames

    def test_generate_with_custom_config(self, sample_features):
        """Can generate with custom config."""
        config = MappingConfig(
            name="Custom",
            mappings=[FeatureMapping("bass", "zoom", 1.0, 1.3)],
        )
        generator = ScheduleGenerator()
        schedule = generator.generate(sample_features, mapping=config)
        assert len(schedule.keyframes) > 0

    def test_keyframe_interval(self, sample_features):
        """Keyframe interval affects output count."""
        generator = ScheduleGenerator()

        schedule1 = generator.generate(
            sample_features, mapping="bass_pulse", keyframe_interval=1
        )
        schedule10 = generator.generate(
            sample_features, mapping="bass_pulse", keyframe_interval=10
        )

        assert len(schedule1.keyframes) > len(schedule10.keyframes)

    def test_all_presets_work(self, sample_features):
        """All presets generate valid schedules."""
        generator = ScheduleGenerator()

        for preset_name in PRESETS.keys():
            schedule = generator.generate(
                sample_features, mapping=preset_name
            )
            assert len(schedule.keyframes) > 0


# =============================================================================
# API Tests
# =============================================================================

class TestAPI:
    """Test convenience API functions."""

    def test_create_mapping(self):
        """Can create mapping with convenience syntax."""
        config = create_mapping(
            "Test",
            bass_to_zoom=(1.0, 1.2),
            mid_to_angle=(-5, 5),
        )
        assert config.name == "Test"
        assert len(config.mappings) == 2

    def test_create_mapping_with_curve(self):
        """Can specify curve in convenience syntax."""
        config = create_mapping(
            "Test",
            bass_to_zoom=(1.0, 1.2, "ease_out"),
        )
        assert config.mappings[0].curve == CurveType.EASE_OUT

    def test_generate_schedule_from_features(self, sample_features):
        """Can generate schedule from features."""
        schedule = generate_schedule(sample_features, mapping="bass_pulse")
        assert len(schedule.keyframes) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, sample_features, tmp_path):
        """Test complete workflow."""
        # Generate schedule
        schedule = generate_schedule(
            sample_features,
            mapping="spectrum",
            keyframe_interval=5,
        )

        # Verify structure
        assert schedule.num_frames == sample_features.num_frames
        assert len(schedule.keyframes) > 0

        # Save and reload
        path = tmp_path / "schedule.json"
        schedule.save(path)
        loaded = ParseqSchedule.load(path)
        assert len(loaded.keyframes) == len(schedule.keyframes)

        # Convert to Deforum strings
        strings = schedule.to_deforum_strings()
        assert "zoom" in strings

    def test_custom_mapping_workflow(self, sample_features):
        """Test with custom mapping."""
        config = create_mapping(
            "Custom",
            "My custom mapping",
            bass_to_zoom=(1.0, 1.5, "exponential"),
            beat_strength_to_angle=(-15, 15, "ease_out"),
            high_to_translation_y=(-10, 10),
        )

        schedule = generate_schedule(sample_features, mapping=config)
        assert len(schedule.keyframes) > 0

        # First keyframe should have zoom set
        assert schedule.keyframes[0].zoom is not None
