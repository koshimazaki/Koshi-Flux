"""
TimeSeries class for audio feature data.

Ported from Parseq - handles temporal audio data with interpolation.
"""

import numpy as np
from typing import Literal, Optional, List, Tuple
from scipy.interpolate import interp1d
from dataclasses import dataclass


@dataclass
class TimeSeries:
    """
    Time-indexed audio feature data with interpolation support.

    Attributes:
        times: Array of timestamps (frames or milliseconds)
        values: Array of feature values
        timestamp_type: "frame" or "ms"
        name: Optional feature name
    """
    times: np.ndarray
    values: np.ndarray
    timestamp_type: Literal["frame", "ms"] = "ms"
    name: str = ""

    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.times = np.asarray(self.times, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.times)

    @property
    def min_value(self) -> float:
        return float(np.min(self.values))

    @property
    def max_value(self) -> float:
        return float(np.max(self.values))

    @property
    def mean_value(self) -> float:
        return float(np.mean(self.values))

    @property
    def duration(self) -> float:
        """Duration in same units as timestamps."""
        return float(self.times[-1] - self.times[0]) if len(self.times) > 0 else 0.0

    def normalize(self, target_min: float = 0.0, target_max: float = 1.0) -> 'TimeSeries':
        """
        Normalize values to target range.

        Args:
            target_min: Minimum output value
            target_max: Maximum output value

        Returns:
            New TimeSeries with normalized values
        """
        min_val = self.min_value
        max_val = self.max_value
        range_val = max_val - min_val

        if range_val < 1e-10:
            # Constant value - map to middle of target range
            normalized = np.full_like(self.values, (target_min + target_max) / 2)
        else:
            normalized = ((self.values - min_val) / range_val) * (target_max - target_min) + target_min

        return TimeSeries(
            times=self.times.copy(),
            values=normalized,
            timestamp_type=self.timestamp_type,
            name=self.name,
        )

    def get_value_at(
        self,
        frame: int,
        fps: float,
        interpolation: Literal["step", "linear"] = "linear"
    ) -> float:
        """
        Get interpolated value at a specific frame.

        Args:
            frame: Frame number
            fps: Frames per second (for ms conversion)
            interpolation: "step" or "linear"

        Returns:
            Interpolated value
        """
        # Convert frame to timestamp type
        if self.timestamp_type == "ms":
            t = (frame / fps) * 1000
        else:
            t = float(frame)

        if len(self.times) == 0:
            return 0.0
        if len(self.times) == 1:
            return float(self.values[0])

        # Clamp to range
        if t <= self.times[0]:
            return float(self.values[0])
        if t >= self.times[-1]:
            return float(self.values[-1])

        if interpolation == "step":
            # Find last point at or before t
            idx = np.searchsorted(self.times, t, side='right') - 1
            return float(self.values[max(0, idx)])
        else:
            # Linear interpolation
            f = interp1d(self.times, self.values, kind='linear',
                        bounds_error=False,
                        fill_value=(self.values[0], self.values[-1]))
            return float(f(t))

    def to_frame_array(self, total_frames: int, fps: float,
                       interpolation: Literal["step", "linear"] = "linear") -> np.ndarray:
        """
        Convert to per-frame array.

        Args:
            total_frames: Number of output frames
            fps: Frames per second
            interpolation: Interpolation method

        Returns:
            Array with one value per frame
        """
        result = np.zeros(total_frames)
        for f in range(total_frames):
            result[f] = self.get_value_at(f, fps, interpolation)
        return result

    def moving_average(self, window_size: int) -> 'TimeSeries':
        """
        Apply moving average smoothing.

        Args:
            window_size: Number of samples to average

        Returns:
            Smoothed TimeSeries
        """
        if window_size <= 1:
            return TimeSeries(self.times.copy(), self.values.copy(),
                            self.timestamp_type, self.name)

        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(self.values, kernel, mode='same')

        return TimeSeries(
            times=self.times.copy(),
            values=smoothed,
            timestamp_type=self.timestamp_type,
            name=self.name,
        )

    def filter_range(self, min_value: float, max_value: float) -> 'TimeSeries':
        """
        Filter values outside range (set to 0).

        Args:
            min_value: Minimum threshold
            max_value: Maximum threshold

        Returns:
            Filtered TimeSeries
        """
        filtered = np.where(
            (self.values >= min_value) & (self.values <= max_value),
            self.values,
            0.0
        )
        return TimeSeries(self.times.copy(), filtered, self.timestamp_type, self.name)

    def threshold(self, thresh: float, above: bool = True) -> 'TimeSeries':
        """
        Apply threshold - values below/above become 0.

        Args:
            thresh: Threshold value
            above: If True, keep values above threshold

        Returns:
            Thresholded TimeSeries
        """
        if above:
            filtered = np.where(self.values >= thresh, self.values, 0.0)
        else:
            filtered = np.where(self.values <= thresh, self.values, 0.0)

        return TimeSeries(self.times.copy(), filtered, self.timestamp_type, self.name)

    def abs(self) -> 'TimeSeries':
        """Return absolute values."""
        return TimeSeries(self.times.copy(), np.abs(self.values),
                         self.timestamp_type, self.name)

    def clip(self, min_value: float, max_value: float) -> 'TimeSeries':
        """Clip values to range."""
        clipped = np.clip(self.values, min_value, max_value)
        return TimeSeries(self.times.copy(), clipped, self.timestamp_type, self.name)

    def invert(self) -> 'TimeSeries':
        """Invert normalized values (1 - x)."""
        return TimeSeries(self.times.copy(), 1.0 - self.values,
                         self.timestamp_type, self.name)

    def scale(self, factor: float) -> 'TimeSeries':
        """Scale values by factor."""
        return TimeSeries(self.times.copy(), self.values * factor,
                         self.timestamp_type, self.name)

    def offset(self, amount: float) -> 'TimeSeries':
        """Add offset to values."""
        return TimeSeries(self.times.copy(), self.values + amount,
                         self.timestamp_type, self.name)

    def derivative(self) -> 'TimeSeries':
        """Calculate rate of change."""
        if len(self.values) < 2:
            return TimeSeries(self.times.copy(), np.zeros_like(self.values),
                            self.timestamp_type, self.name)

        dt = np.diff(self.times)
        dv = np.diff(self.values)
        derivative = np.zeros_like(self.values)
        derivative[1:] = dv / np.maximum(dt, 1e-10)

        return TimeSeries(self.times.copy(), derivative, self.timestamp_type, self.name)

    def peaks(self, min_distance: int = 5, threshold: float = 0.5) -> List[Tuple[float, float]]:
        """
        Find peaks in the signal.

        Args:
            min_distance: Minimum samples between peaks
            threshold: Minimum peak value (relative to max)

        Returns:
            List of (time, value) tuples for peaks
        """
        from scipy.signal import find_peaks

        height = self.max_value * threshold
        peak_indices, _ = find_peaks(self.values, distance=min_distance, height=height)

        return [(float(self.times[i]), float(self.values[i])) for i in peak_indices]

    def resample(self, new_length: int) -> 'TimeSeries':
        """Resample to new length."""
        if len(self.times) < 2:
            return TimeSeries(
                times=np.linspace(0, 1, new_length),
                values=np.full(new_length, self.values[0] if len(self.values) > 0 else 0),
                timestamp_type=self.timestamp_type,
                name=self.name,
            )

        new_times = np.linspace(self.times[0], self.times[-1], new_length)
        f = interp1d(self.times, self.values, kind='linear',
                    bounds_error=False, fill_value=(self.values[0], self.values[-1]))
        new_values = f(new_times)

        return TimeSeries(new_times, new_values, self.timestamp_type, self.name)


__all__ = ["TimeSeries"]
