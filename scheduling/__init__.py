"""
Universal Scheduling Module for Deforum.

Model-agnostic keyframe scheduling with audio reactivity,
inspired by Parseq. Works with FLUX, LTX, and other video models.

Usage:
    from scheduling import Schedule, ScheduleRenderer, FluxAdapter

    # Create schedule
    schedule = Schedule(fps=30, total_frames=300, bpm=120)
    schedule.add_keyframe(0, zoom=1.0, strength=0.8)
    schedule.add_keyframe(60, zoom=1.1, strength=0.6, easing={"zoom": "easeOut"})

    # Render with audio
    renderer = ScheduleRenderer()
    rendered = renderer.render(
        schedule,
        audio_features=audio,
        audio_mappings={"strength": {"feature": "bass", "invert": True}}
    )

    # Adapt for FLUX
    adapter = FluxAdapter()
    adapted = adapter.adapt(rendered)
"""

from .core import (
    # Keyframe
    InterpolationType,
    Keyframe,
    Schedule,
    DEFAULT_PARAMS,
    # Interpolation
    linear_interpolation,
    step_interpolation,
    cubic_spline_interpolation,
    interpolate_array,
    lerp,
    inverse_lerp,
    remap,
    smoothstep,
    smootherstep,
    # Easing
    bezier_easing,
    apply_easing,
    apply_easing_to_range,
    list_easings,
    get_easing_points,
    EASING_PRESETS,
    # Oscillators
    WaveType,
    oscillator,
    oscillator_array,
    beat_oscillator,
    lfo,
    envelope,
    noise,
)

from .audio import (
    TimeSeries,
    AudioAnalyzer,
    AudioFeatures,
)

from .renderer import (
    RenderContext,
    ScheduleRenderer,
    render_schedule,
)

from .adapters import (
    # Protocol
    AdapterConfig,
    AdaptedSchedule,
    ScheduleAdapter,
    BaseAdapter,
    # FLUX
    FluxAdapter,
    FluxConfig,
    # LTX
    LTXAdapter,
    LTXConfig,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core - Keyframe
    "InterpolationType",
    "Keyframe",
    "Schedule",
    "DEFAULT_PARAMS",
    # Core - Interpolation
    "linear_interpolation",
    "step_interpolation",
    "cubic_spline_interpolation",
    "interpolate_array",
    "lerp",
    "inverse_lerp",
    "remap",
    "smoothstep",
    "smootherstep",
    # Core - Easing
    "bezier_easing",
    "apply_easing",
    "apply_easing_to_range",
    "list_easings",
    "get_easing_points",
    "EASING_PRESETS",
    # Core - Oscillators
    "WaveType",
    "oscillator",
    "oscillator_array",
    "beat_oscillator",
    "lfo",
    "envelope",
    "noise",
    # Audio
    "TimeSeries",
    "AudioAnalyzer",
    "AudioFeatures",
    # Renderer
    "RenderContext",
    "ScheduleRenderer",
    "render_schedule",
    # Adapters - Protocol
    "AdapterConfig",
    "AdaptedSchedule",
    "ScheduleAdapter",
    "BaseAdapter",
    # Adapters - FLUX
    "FluxAdapter",
    "FluxConfig",
    # Adapters - LTX
    "LTXAdapter",
    "LTXConfig",
]
