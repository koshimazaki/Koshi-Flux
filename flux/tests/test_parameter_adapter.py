"""
Unit Tests for Parameter Adapter

Run with: pytest tests/test_parameter_adapter.py -v
"""

import pytest
from deforum_flux.adapters import FluxDeforumParameterAdapter, MotionFrame
from deforum_flux.core import ParameterError


class TestScheduleParsing:
    """Tests for schedule string parsing."""
    
    @pytest.fixture
    def adapter(self):
        return FluxDeforumParameterAdapter()
    
    def test_parse_simple_schedule(self, adapter):
        """Parse basic keyframe schedule."""
        schedule = "0:(1.0), 30:(1.5), 60:(1.0)"
        values = adapter.parse_schedule(schedule, num_frames=61, default=1.0)
        
        assert len(values) == 61
        assert values[0] == pytest.approx(1.0)
        assert values[30] == pytest.approx(1.5)
        assert values[60] == pytest.approx(1.0)
    
    def test_parse_interpolation(self, adapter):
        """Verify linear interpolation between keyframes."""
        schedule = "0:(0), 10:(10)"
        values = adapter.parse_schedule(schedule, num_frames=11, default=0)
        
        # Should interpolate linearly
        for i in range(11):
            assert values[i] == pytest.approx(float(i))
    
    def test_parse_negative_values(self, adapter):
        """Handle negative values in schedules."""
        schedule = "0:(0), 30:(-10), 60:(0)"
        values = adapter.parse_schedule(schedule, num_frames=61, default=0)
        
        assert values[30] == pytest.approx(-10)
        assert values[15] == pytest.approx(-5)  # Interpolated
    
    def test_parse_without_parentheses(self, adapter):
        """Handle schedules without parentheses."""
        schedule = "0:1.0, 30:1.5, 60:1.0"
        values = adapter.parse_schedule(schedule, num_frames=61, default=1.0)
        
        assert values[0] == pytest.approx(1.0)
        assert values[30] == pytest.approx(1.5)
    
    def test_parse_empty_schedule(self, adapter):
        """Empty schedule should return defaults."""
        values = adapter.parse_schedule("", num_frames=10, default=5.0)
        
        assert len(values) == 10
        assert all(v == 5.0 for v in values)
    
    def test_parse_single_keyframe(self, adapter):
        """Single keyframe should fill all frames."""
        schedule = "0:(2.5)"
        values = adapter.parse_schedule(schedule, num_frames=30, default=1.0)
        
        assert all(v == pytest.approx(2.5) for v in values)
    
    def test_parse_float_precision(self, adapter):
        """Handle high-precision floats."""
        schedule = "0:(1.12345), 10:(2.98765)"
        values = adapter.parse_schedule(schedule, num_frames=11, default=0)
        
        assert values[0] == pytest.approx(1.12345)
        assert values[10] == pytest.approx(2.98765)


class TestDeforumConversion:
    """Tests for full Deforum parameter conversion."""
    
    @pytest.fixture
    def adapter(self):
        return FluxDeforumParameterAdapter()
    
    def test_convert_string_params(self, adapter):
        """Convert string schedule parameters."""
        params = {
            "zoom": "0:(1.0), 10:(1.1)",
            "angle": "0:(0), 10:(5)",
        }
        
        frames = adapter.convert_deforum_params(params, num_frames=11)
        
        assert len(frames) == 11
        assert frames[0].zoom == pytest.approx(1.0)
        assert frames[10].zoom == pytest.approx(1.1)
        assert frames[10].angle == pytest.approx(5.0)
    
    def test_convert_constant_params(self, adapter):
        """Convert constant (non-schedule) parameters."""
        params = {
            "zoom": 1.5,  # Constant
            "angle": 0.0,
            "translation_x": 10,
        }
        
        frames = adapter.convert_deforum_params(params, num_frames=5)
        
        assert all(f.zoom == 1.5 for f in frames)
        assert all(f.translation_x == 10 for f in frames)
    
    def test_convert_with_prompts(self, adapter):
        """Convert parameters including keyframed prompts."""
        params = {
            "zoom": "0:(1.0), 30:(1.1)",
            "prompts": {
                0: "first prompt",
                15: "second prompt",
            }
        }
        
        frames = adapter.convert_deforum_params(params, num_frames=31)
        
        assert frames[0].prompt == "first prompt"
        assert frames[14].prompt == "first prompt"  # Before second keyframe
        assert frames[15].prompt == "second prompt"
        assert frames[30].prompt == "second prompt"
    
    def test_convert_strength_schedule(self, adapter):
        """Convert strength schedule."""
        params = {
            "strength_schedule": "0:(0.65), 30:(0.5), 60:(0.65)"
        }
        
        frames = adapter.convert_deforum_params(params, num_frames=61)
        
        assert frames[0].strength == pytest.approx(0.65)
        assert frames[30].strength == pytest.approx(0.5)
        assert frames[60].strength == pytest.approx(0.65)


class TestMotionFrame:
    """Tests for MotionFrame dataclass."""
    
    def test_default_values(self):
        """Check default values."""
        frame = MotionFrame(frame_index=0)
        
        assert frame.zoom == 1.0
        assert frame.angle == 0.0
        assert frame.translation_x == 0.0
        assert frame.translation_y == 0.0
        assert frame.translation_z == 0.0
        assert frame.strength == 0.65
        assert frame.prompt is None
    
    def test_to_dict(self):
        """Convert frame to motion dict."""
        frame = MotionFrame(
            frame_index=5,
            zoom=1.2,
            angle=15,
            translation_x=10,
            translation_y=-5,
            translation_z=20,
        )
        
        d = frame.to_dict()
        
        assert d["zoom"] == 1.2
        assert d["angle"] == 15
        assert d["translation_x"] == 10
        assert d["translation_y"] == -5
        assert d["translation_z"] == 20


class TestSimpleAnimation:
    """Tests for simple animation helper."""
    
    @pytest.fixture
    def adapter(self):
        return FluxDeforumParameterAdapter()
    
    def test_zoom_animation(self, adapter):
        """Create simple zoom animation."""
        frames = adapter.create_simple_animation(
            num_frames=11,
            zoom_start=1.0,
            zoom_end=2.0,
        )
        
        assert len(frames) == 11
        assert frames[0].zoom == pytest.approx(1.0)
        assert frames[5].zoom == pytest.approx(1.5)
        assert frames[10].zoom == pytest.approx(2.0)
    
    def test_rotation_animation(self, adapter):
        """Create animation with cumulative rotation."""
        frames = adapter.create_simple_animation(
            num_frames=10,
            rotation=5.0,  # 5 degrees per frame
        )
        
        assert frames[0].angle == pytest.approx(0)
        assert frames[5].angle == pytest.approx(25)  # 5 * 5
        assert frames[9].angle == pytest.approx(45)  # 5 * 9


class TestValidation:
    """Tests for parameter validation."""
    
    def test_strict_validation_out_of_range(self):
        """Strict mode should reject out-of-range values."""
        adapter = FluxDeforumParameterAdapter(strict_validation=True)
        
        params = {
            "zoom": 5.0,  # Way out of typical range
        }
        
        with pytest.raises(ParameterError):
            adapter.convert_deforum_params(params, num_frames=1)
    
    def test_lenient_validation(self):
        """Non-strict mode should allow out-of-range values."""
        adapter = FluxDeforumParameterAdapter(strict_validation=False)
        
        params = {
            "zoom": 5.0,
        }
        
        frames = adapter.convert_deforum_params(params, num_frames=1)
        assert frames[0].zoom == 5.0  # Should work


class TestMotionScheduleGeneration:
    """Tests for motion schedule dict generation."""
    
    @pytest.fixture
    def adapter(self):
        return FluxDeforumParameterAdapter()
    
    def test_generate_schedule_dict(self, adapter):
        """Generate schedule dictionary from frames."""
        frames = [
            MotionFrame(frame_index=0, zoom=1.0, angle=0),
            MotionFrame(frame_index=1, zoom=1.1, angle=5),
            MotionFrame(frame_index=2, zoom=1.2, angle=10),
        ]
        
        schedule = adapter.generate_motion_schedule(frames)
        
        assert len(schedule) == 3
        assert schedule[0]["zoom"] == 1.0
        assert schedule[1]["zoom"] == 1.1
        assert schedule[2]["angle"] == 10
