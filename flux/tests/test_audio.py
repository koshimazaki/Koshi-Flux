"""Tests for audio feature extraction and schedule generation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

# Try to import audio modules - skip tests if not available
try:
    from deforum_flux.audio.mapping_config import (
        FeatureMapping,
        MappingConfig,
        CurveType,
        apply_curve,
        apply_smoothing,
        DEFAULT_MAPPINGS,
        load_mapping_config,
        save_mapping_config,
        create_custom_mapping,
    )
    from deforum_flux.audio.schedule_generator import (
        ParseqKeyframe,
        ParseqSchedule,
        ScheduleGenerator,
    )
    from deforum_flux.audio.extractor import AudioFeatures
    HAS_AUDIO_MODULE = True
except ImportError:
    HAS_AUDIO_MODULE = False


pytestmark = pytest.mark.skipif(
    not HAS_AUDIO_MODULE,
    reason="deforum_flux.audio module not available"
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_features():
    """Create sample audio features for testing."""
    if not HAS_AUDIO_MODULE:
        pytest.skip("Audio module not available")

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


@pytest.fixture
def simple_mapping():
    """Create a simple mapping configuration for testing."""
    if not HAS_AUDIO_MODULE:
        pytest.skip("Audio module not available")

    return MappingConfig(
        name="Test Mapping",
        description="Test configuration",
        mappings=[
            FeatureMapping("bass", "zoom", 1.0, 1.1),
            FeatureMapping("beat_strength", "angle", -5, 5),
        ],
        global_smoothing=0.1,
    )


# =============================================================================
# Curve Tests
# =============================================================================

class TestCurves:
    """Test easing curve functions."""

    def test_linear_curve(self):
        """Linear curve should return unchanged values."""
        values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = apply_curve(values, CurveType.LINEAR)
        np.testing.assert_array_almost_equal(result, values)

    def test_ease_in_curve(self):
        """Ease-in should start slow."""
        values = np.array([0.0, 0.5, 1.0])
        result = apply_curve(values, CurveType.EASE_IN)
        assert result[0] == 0.0
        assert result[1] == 0.25  # 0.5^2
        assert result[2] == 1.0

    def test_ease_out_curve(self):
        """Ease-out should end slow."""
        values = np.array([0.0, 0.5, 1.0])
        result = apply_curve(values, CurveType.EASE_OUT)
        assert result[0] == 0.0
        assert result[1] == 0.75  # 1 - (1-0.5)^2
        assert result[2] == 1.0

    def test_ease_in_out_curve(self):
        """Ease-in-out should start and end slow."""
        values = np.array([0.0, 0.5, 1.0])
        result = apply_curve(values, CurveType.EASE_IN_OUT)
        assert result[0] == 0.0
        assert result[1] == 0.5
        assert result[2] == 1.0


class TestSmoothing:
    """Test temporal smoothing function."""

    def test_no_smoothing(self):
        """Zero smoothing should return unchanged values."""
        values = np.array([0.0, 1.0, 0.0, 1.0])
        result = apply_smoothing(values, 0.0)
        np.testing.assert_array_almost_equal(result, values)

    def test_smoothing_reduces_variance(self):
        """Smoothing should reduce variance."""
        values = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        result = apply_smoothing(values, 0.5)
        assert np.var(result) < np.var(values)

    def test_high_smoothing(self):
        """High smoothing should significantly reduce changes."""
        values = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        result = apply_smoothing(values, 0.9)
        # Should be much smoother
        diffs = np.abs(np.diff(result))
        assert np.max(diffs) < 0.5


# =============================================================================
# FeatureMapping Tests
# =============================================================================

class TestFeatureMapping:
    """Test FeatureMapping class."""

    def test_basic_mapping_creation(self):
        """Test creating a basic mapping."""
        mapping = FeatureMapping(
            feature="bass",
            parameter="zoom",
            min_value=1.0,
            max_value=1.2,
        )
        assert mapping.feature == "bass"
        assert mapping.parameter == "zoom"
        assert mapping.min_value == 1.0
        assert mapping.max_value == 1.2

    def test_mapping_to_dict(self):
        """Test converting mapping to dictionary."""
        mapping = FeatureMapping(
            feature="bass",
            parameter="zoom",
            min_value=1.0,
            max_value=1.2,
            curve=CurveType.EASE_OUT,
            smoothing=0.3,
        )
        d = mapping.to_dict()
        assert d["feature"] == "bass"
        assert d["parameter"] == "zoom"
        assert d["curve"] == "ease_out"
        assert d["smoothing"] == 0.3

    def test_mapping_from_dict(self):
        """Test creating mapping from dictionary."""
        data = {
            "feature": "energy",
            "parameter": "angle",
            "min_value": -10,
            "max_value": 10,
            "curve": "ease_in_out",
        }
        mapping = FeatureMapping.from_dict(data)
        assert mapping.feature == "energy"
        assert mapping.parameter == "angle"
        assert mapping.curve == CurveType.EASE_IN_OUT


# =============================================================================
# MappingConfig Tests
# =============================================================================

class TestMappingConfig:
    """Test MappingConfig class."""

    def test_config_creation(self, simple_mapping):
        """Test creating a mapping config."""
        assert simple_mapping.name == "Test Mapping"
        assert len(simple_mapping.mappings) == 2

    def test_add_mapping(self, simple_mapping):
        """Test adding a mapping."""
        simple_mapping.add_mapping(
            FeatureMapping("high", "translation_x", -5, 5)
        )
        assert len(simple_mapping.mappings) == 3

    def test_remove_mapping(self, simple_mapping):
        """Test removing a mapping."""
        removed = simple_mapping.remove_mapping("bass", "zoom")
        assert removed is True
        assert len(simple_mapping.mappings) == 1

    def test_get_mappings_for_parameter(self, simple_mapping):
        """Test getting mappings for a specific parameter."""
        zoom_mappings = simple_mapping.get_mappings_for_parameter("zoom")
        assert len(zoom_mappings) == 1
        assert zoom_mappings[0].feature == "bass"

    def test_config_serialization(self, simple_mapping):
        """Test config to/from dict."""
        d = simple_mapping.to_dict()
        restored = MappingConfig.from_dict(d)
        assert restored.name == simple_mapping.name
        assert len(restored.mappings) == len(simple_mapping.mappings)

    def test_config_file_save_load(self, simple_mapping):
        """Test saving and loading config from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            save_mapping_config(simple_mapping, path)
            loaded = load_mapping_config(path)
            assert loaded.name == simple_mapping.name


class TestDefaultMappings:
    """Test default mapping presets."""

    def test_all_presets_exist(self):
        """Test all expected presets exist."""
        expected = ["ambient", "bass_pulse", "beat_rotation", "spectrum",
                    "immersive_3d", "cinematic", "intense"]
        for name in expected:
            assert name in DEFAULT_MAPPINGS

    def test_presets_have_mappings(self):
        """Test all presets have at least one mapping."""
        for name, config in DEFAULT_MAPPINGS.items():
            assert len(config.mappings) > 0, f"Preset '{name}' has no mappings"

    def test_presets_have_valid_features(self):
        """Test all presets use valid feature names."""
        valid_features = {
            "rms", "energy", "spectral_centroid", "spectral_bandwidth",
            "spectral_rolloff", "spectral_flatness", "bass", "mid", "high",
            "beat_strength", "onset_strength"
        }
        for name, config in DEFAULT_MAPPINGS.items():
            for mapping in config.mappings:
                assert mapping.feature in valid_features, \
                    f"Invalid feature '{mapping.feature}' in preset '{name}'"


# =============================================================================
# AudioFeatures Tests
# =============================================================================

class TestAudioFeatures:
    """Test AudioFeatures class."""

    def test_feature_access(self, sample_features):
        """Test accessing features by name."""
        bass = sample_features.get_feature("bass")
        assert len(bass) == sample_features.num_frames

    def test_unknown_feature_raises(self, sample_features):
        """Test that unknown features raise ValueError."""
        with pytest.raises(ValueError):
            sample_features.get_feature("unknown_feature")

    def test_list_features(self, sample_features):
        """Test listing available features."""
        features = sample_features.list_features()
        assert "bass" in features
        assert "energy" in features
        assert "beat_strength" in features

    def test_to_dict(self, sample_features):
        """Test converting features to dictionary."""
        d = sample_features.to_dict()
        assert "metadata" in d
        assert "features" in d
        assert d["metadata"]["num_frames"] == sample_features.num_frames

    def test_save_load(self, sample_features):
        """Test saving and loading features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            sample_features.save(path)
            loaded = AudioFeatures.load(path)
            assert loaded.num_frames == sample_features.num_frames
            np.testing.assert_array_almost_equal(
                loaded.bass, sample_features.bass
            )


# =============================================================================
# ParseqKeyframe Tests
# =============================================================================

class TestParseqKeyframe:
    """Test ParseqKeyframe class."""

    def test_basic_keyframe(self):
        """Test creating a basic keyframe."""
        kf = ParseqKeyframe(frame=0, zoom=1.0, angle=5.0)
        assert kf.frame == 0
        assert kf.zoom == 1.0
        assert kf.angle == 5.0

    def test_keyframe_to_dict(self):
        """Test keyframe to dictionary."""
        kf = ParseqKeyframe(
            frame=10,
            zoom=1.05,
            translation_x=5.0,
            info="beat",
        )
        d = kf.to_dict()
        assert d["frame"] == 10
        assert d["zoom"] == 1.05
        assert d["translation_x"] == 5.0
        assert d["info"] == "beat"

    def test_keyframe_omits_none_values(self):
        """Test that None values are not included in dict."""
        kf = ParseqKeyframe(frame=0, zoom=1.0)
        d = kf.to_dict()
        assert "angle" not in d  # Should be omitted, not None


# =============================================================================
# ParseqSchedule Tests
# =============================================================================

class TestParseqSchedule:
    """Test ParseqSchedule class."""

    def test_basic_schedule(self):
        """Test creating a basic schedule."""
        schedule = ParseqSchedule(
            name="Test Schedule",
            fps=24,
            num_frames=100,
        )
        assert schedule.name == "Test Schedule"
        assert schedule.fps == 24

    def test_schedule_with_keyframes(self):
        """Test schedule with keyframes."""
        keyframes = [
            ParseqKeyframe(frame=0, zoom=1.0),
            ParseqKeyframe(frame=30, zoom=1.1),
            ParseqKeyframe(frame=60, zoom=1.0),
        ]
        schedule = ParseqSchedule(
            name="Test",
            keyframes=keyframes,
            num_frames=61,
        )
        assert len(schedule.keyframes) == 3

    def test_to_json(self):
        """Test JSON conversion."""
        schedule = ParseqSchedule(
            name="Test",
            fps=24,
            keyframes=[ParseqKeyframe(frame=0, zoom=1.0)],
        )
        json_str = schedule.to_json()
        data = json.loads(json_str)
        assert "meta" in data
        assert "keyframes" in data
        assert data["meta"]["name"] == "Test"

    def test_to_deforum_strings(self):
        """Test Deforum keyframe string generation."""
        keyframes = [
            ParseqKeyframe(frame=0, zoom=1.0, angle=0.0),
            ParseqKeyframe(frame=30, zoom=1.1, angle=5.0),
            ParseqKeyframe(frame=60, zoom=1.0, angle=0.0),
        ]
        schedule = ParseqSchedule(keyframes=keyframes)
        strings = schedule.to_deforum_strings()

        assert "zoom" in strings
        assert "0:(1.0)" in strings["zoom"]
        assert "30:(1.1)" in strings["zoom"]

    def test_save_load(self):
        """Test saving and loading schedule."""
        keyframes = [
            ParseqKeyframe(frame=0, zoom=1.0),
            ParseqKeyframe(frame=60, zoom=1.1),
        ]
        schedule = ParseqSchedule(
            name="Test",
            keyframes=keyframes,
            fps=24,
            num_frames=61,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schedule.json"
            schedule.save(path)
            loaded = ParseqSchedule.load(path)
            assert loaded.name == schedule.name
            assert len(loaded.keyframes) == 2


# =============================================================================
# ScheduleGenerator Tests
# =============================================================================

class TestScheduleGenerator:
    """Test ScheduleGenerator class."""

    def test_generate_with_preset(self, sample_features):
        """Test generating schedule with preset mapping."""
        generator = ScheduleGenerator()
        schedule = generator.generate(sample_features, mapping="bass_pulse")
        assert len(schedule.keyframes) > 0
        assert schedule.num_frames == sample_features.num_frames

    def test_generate_with_custom_mapping(self, sample_features, simple_mapping):
        """Test generating schedule with custom mapping."""
        generator = ScheduleGenerator()
        schedule = generator.generate(sample_features, mapping=simple_mapping)
        assert len(schedule.keyframes) > 0

    def test_keyframe_interval(self, sample_features):
        """Test keyframe interval setting."""
        generator = ScheduleGenerator()

        schedule_1 = generator.generate(
            sample_features, mapping="bass_pulse", keyframe_interval=1
        )
        schedule_10 = generator.generate(
            sample_features, mapping="bass_pulse", keyframe_interval=10
        )

        # More keyframes with smaller interval
        assert len(schedule_1.keyframes) > len(schedule_10.keyframes)

    def test_generate_deforum_strings(self, sample_features):
        """Test generating Deforum strings directly."""
        generator = ScheduleGenerator()
        strings = generator.generate_deforum_strings(
            sample_features, mapping="bass_pulse"
        )
        assert isinstance(strings, dict)
        assert "zoom" in strings

    def test_unknown_mapping_raises(self, sample_features):
        """Test that unknown mapping name raises error."""
        generator = ScheduleGenerator()
        with pytest.raises(ValueError):
            generator.generate(sample_features, mapping="nonexistent")


# =============================================================================
# Custom Mapping Creation Tests
# =============================================================================

class TestCreateCustomMapping:
    """Test custom mapping creation helper."""

    def test_create_simple_custom(self):
        """Test creating a simple custom mapping."""
        config = create_custom_mapping(
            "Custom",
            bass_to_zoom=(1.0, 1.2),
            energy_to_angle=(-10, 10),
        )
        assert config.name == "Custom"
        assert len(config.mappings) == 2

    def test_custom_with_curve(self):
        """Test custom mapping with curve specification."""
        config = create_custom_mapping(
            "Custom",
            bass_to_zoom=(1.0, 1.2, CurveType.EASE_OUT),
        )
        assert config.mappings[0].curve == CurveType.EASE_OUT


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self, sample_features):
        """Test complete workflow from features to schedule."""
        # Generate schedule
        generator = ScheduleGenerator()
        schedule = generator.generate(
            sample_features,
            mapping="spectrum",
            keyframe_interval=1,
            prompt="test prompt",
        )

        # Check schedule properties
        assert schedule.num_frames == sample_features.num_frames
        assert schedule.fps == int(sample_features.fps)
        assert schedule.bpm == sample_features.tempo

        # Check keyframes have values
        first_kf = schedule.keyframes[0]
        assert first_kf.frame == 0
        assert first_kf.zoom is not None

        # Convert to Deforum strings
        strings = schedule.to_deforum_strings()
        assert len(strings) > 0

        # Serialize and deserialize
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schedule.json"
            schedule.save(path)
            loaded = ParseqSchedule.load(path)
            assert len(loaded.keyframes) == len(schedule.keyframes)

    def test_all_presets_generate_valid_schedules(self, sample_features):
        """Test all preset mappings generate valid schedules."""
        generator = ScheduleGenerator()

        for preset_name in DEFAULT_MAPPINGS.keys():
            schedule = generator.generate(
                sample_features,
                mapping=preset_name,
                keyframe_interval=5,
            )
            assert len(schedule.keyframes) > 0
            assert schedule.num_frames == sample_features.num_frames

            # Verify JSON serialization works
            json_str = schedule.to_json()
            data = json.loads(json_str)
            assert data["meta"]["mapping"] == DEFAULT_MAPPINGS[preset_name].name
