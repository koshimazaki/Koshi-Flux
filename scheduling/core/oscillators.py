"""
Oscillator functions for rhythmic animation.

Ported from Parseq - waveform generators for audio-reactive motion.
"""

import numpy as np
from typing import Literal, Optional
from enum import Enum


class WaveType(Enum):
    """Available oscillator waveforms."""
    SINE = "sin"
    TRIANGLE = "tri"
    SAWTOOTH = "saw"
    SQUARE = "sq"
    PULSE = "pulse"


def oscillator(
    wave_type: str,
    frame: int,
    period: float,
    phase: float = 0.0,
    amplitude: float = 1.0,
    center: float = 0.0,
    limit: int = 0,
    pulse_width: float = 0.5,
    active_keyframe: int = 0,
) -> float:
    """
    Generate oscillator value at a given frame.

    Args:
        wave_type: "sin", "tri", "saw", "sq", "pulse"
        frame: Current frame number
        period: Oscillation period in frames
        phase: Phase offset in frames
        amplitude: Peak amplitude (half of full range)
        center: Center/offset value
        limit: Max repetitions (0 = unlimited)
        pulse_width: For pulse wave, fraction of period that is "on" (0-1)
        active_keyframe: Frame where oscillation started (for limit)

    Returns:
        Oscillator value at frame
    """
    # Guard against zero period
    if period <= 0:
        return center

    # Check repetition limit
    if limit > 0:
        elapsed = frame - active_keyframe
        if elapsed > limit * period:
            return center

    pos = frame + phase

    if wave_type == "sin":
        return center + np.sin(pos * 2 * np.pi / period) * amplitude

    elif wave_type == "tri":
        # Triangle wave using arcsin
        return center + np.arcsin(np.sin(pos * 2 * np.pi / period)) * (2 * amplitude) / np.pi

    elif wave_type == "saw":
        # Sawtooth wave
        return center + ((pos % period) / period - 0.5) * 2 * amplitude

    elif wave_type == "sq":
        # Square wave
        return center + (1 if np.sin(pos * 2 * np.pi / period) >= 0 else -1) * amplitude

    elif wave_type == "pulse":
        # Pulse wave with configurable width
        phase_in_period = (pos % period) / period
        return center + amplitude if phase_in_period < pulse_width else center - amplitude

    return center


def oscillator_array(
    wave_type: str,
    total_frames: int,
    period: float,
    phase: float = 0.0,
    amplitude: float = 1.0,
    center: float = 0.0,
    limit: int = 0,
    pulse_width: float = 0.5,
    active_keyframe: int = 0,
) -> np.ndarray:
    """
    Generate oscillator values for all frames.

    Returns:
        Array of oscillator values
    """
    # Guard against zero period
    if period <= 0:
        return np.full(total_frames, center)

    frames = np.arange(total_frames)
    pos = frames + phase

    if limit > 0:
        elapsed = frames - active_keyframe
        mask = elapsed <= limit * period
    else:
        mask = np.ones(total_frames, dtype=bool)

    result = np.full(total_frames, center)

    if wave_type == "sin":
        result[mask] = center + np.sin(pos[mask] * 2 * np.pi / period) * amplitude

    elif wave_type == "tri":
        result[mask] = center + np.arcsin(np.sin(pos[mask] * 2 * np.pi / period)) * (2 * amplitude) / np.pi

    elif wave_type == "saw":
        result[mask] = center + ((pos[mask] % period) / period - 0.5) * 2 * amplitude

    elif wave_type == "sq":
        result[mask] = center + np.where(np.sin(pos[mask] * 2 * np.pi / period) >= 0, 1, -1) * amplitude

    elif wave_type == "pulse":
        phase_in_period = (pos[mask] % period) / period
        result[mask] = np.where(phase_in_period < pulse_width, center + amplitude, center - amplitude)

    return result


def beat_oscillator(
    frame: int,
    bpm: float,
    fps: float,
    wave_type: str = "sin",
    beat_multiplier: float = 1.0,
    phase_beats: float = 0.0,
    amplitude: float = 1.0,
    center: float = 0.0,
) -> float:
    """
    Oscillator synchronized to BPM.

    Args:
        frame: Current frame
        bpm: Beats per minute
        fps: Frames per second
        wave_type: Waveform type
        beat_multiplier: Oscillate every N beats (0.5 = twice per beat)
        phase_beats: Phase offset in beats
        amplitude: Peak amplitude
        center: Center value

    Returns:
        Oscillator value synced to beat
    """
    # Calculate period in frames
    frames_per_beat = (60.0 / bpm) * fps
    period = frames_per_beat * beat_multiplier

    # Convert phase from beats to frames
    phase_frames = phase_beats * frames_per_beat

    return oscillator(wave_type, frame, period, phase_frames, amplitude, center)


def lfo(
    frame: int,
    rate_hz: float,
    fps: float,
    wave_type: str = "sin",
    amplitude: float = 1.0,
    center: float = 0.0,
    phase: float = 0.0,
) -> float:
    """
    Low Frequency Oscillator (LFO) at specified Hz rate.

    Args:
        frame: Current frame
        rate_hz: Oscillation rate in Hz
        fps: Frames per second
        wave_type: Waveform type
        amplitude: Peak amplitude
        center: Center value
        phase: Phase offset (0-1, fraction of cycle)

    Returns:
        LFO value
    """
    if rate_hz <= 0:
        return center
    period = fps / rate_hz
    phase_frames = phase * period
    return oscillator(wave_type, frame, period, phase_frames, amplitude, center)


def envelope(
    frame: int,
    attack_frames: int,
    decay_frames: int,
    sustain_level: float,
    release_frames: int,
    trigger_frame: int,
    release_frame: Optional[int] = None,
) -> float:
    """
    ADSR envelope generator.

    Args:
        frame: Current frame
        attack_frames: Attack duration
        decay_frames: Decay duration
        sustain_level: Sustain level (0-1)
        release_frames: Release duration
        trigger_frame: Frame when envelope triggered
        release_frame: Frame when release started (None = not released)

    Returns:
        Envelope value 0-1
    """
    elapsed = frame - trigger_frame

    if elapsed < 0:
        return 0.0

    # Attack phase
    if attack_frames > 0 and elapsed < attack_frames:
        return elapsed / attack_frames
    elif attack_frames <= 0:
        pass  # Skip attack, go straight to decay
    else:
        elapsed -= attack_frames

    # Decay phase
    if attack_frames > 0:
        elapsed -= attack_frames
    if decay_frames > 0 and elapsed < decay_frames:
        return 1.0 - (1.0 - sustain_level) * (elapsed / decay_frames)

    # Sustain phase (or release)
    if release_frame is None:
        return sustain_level

    # Release phase
    release_elapsed = frame - release_frame
    if release_elapsed < 0:
        return sustain_level
    if release_frames <= 0 or release_elapsed >= release_frames:
        return 0.0

    return sustain_level * (1.0 - release_elapsed / release_frames)


def noise(
    frame: int,
    smoothing: float = 1.0,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: int = 42,
    octaves: int = 1,
) -> float:
    """
    Smooth noise generator using simplex-like interpolation.

    Args:
        frame: Current frame
        smoothing: Higher = smoother (frames between changes)
        min_val: Minimum output value
        max_val: Maximum output value
        seed: Random seed
        octaves: Number of noise octaves (1 = simple, more = detailed)

    Returns:
        Smooth random value
    """
    total = 0.0
    amplitude = 1.0
    max_amplitude = 0.0

    for i in range(octaves):
        # Scale position by octave
        pos = frame / (smoothing * (2 ** i))

        # Get integer positions
        p0 = int(np.floor(pos))
        p1 = p0 + 1

        # Generate deterministic random values at integer positions
        rng_p0 = np.random.default_rng(seed + p0 * 1000 + i)
        rng_p1 = np.random.default_rng(seed + p1 * 1000 + i)
        v0 = rng_p0.random()
        v1 = rng_p1.random()

        # Smooth interpolation
        t = pos - p0
        t = t * t * (3 - 2 * t)  # Smoothstep

        total += (v0 + t * (v1 - v0)) * amplitude
        max_amplitude += amplitude
        amplitude *= 0.5

    # Normalize and scale to range
    normalized = total / max_amplitude
    return min_val + normalized * (max_val - min_val)


__all__ = [
    "WaveType",
    "oscillator",
    "oscillator_array",
    "beat_oscillator",
    "lfo",
    "envelope",
    "noise",
]
