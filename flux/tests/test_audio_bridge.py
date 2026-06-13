"""Tests for dashboard/audio/video feature bridge schedules."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from flux_motion.audio.bridge import (
    build_audio_motion_schedule,
    tracks_from_analysis_json,
    tracks_from_video,
    tracks_from_waveform,
)
from flux_motion.shared import MotionFrame


def make_bfl_analysis(seconds=4.0):
    """Synthetic BFL dashboard AudioAnalysisResult."""
    waveform = []
    for index in range(int(seconds * 90)):
        t = index / 90.0
        beat_phase = (t % 0.5) / 0.5
        low = max(0.0, 1.0 - beat_phase * 4.0)
        mid = 0.3 + 0.1 * math.sin(t * 8.0)
        high = max(0.0, 0.2 + 0.2 * math.sin(t * 40.0))
        amplitude = max(low, mid, high)
        waveform.append(
            {
                "time": t,
                "peak": amplitude,
                "rms": amplitude * 0.8,
                "amplitude": amplitude,
                "low": low,
                "mid": mid,
                "high": high,
            }
        )

    markers = []
    hit = 0
    while hit * 0.5 < seconds:
        t = hit * 0.5
        markers.append(
            {
                "id": f"hit-{hit}",
                "time": t,
                "relativeTime": t,
                "kind": "kick",
                "band": "low",
                "amplitude": 0.9,
                "low": 0.95,
                "mid": 0.3,
                "high": 0.1,
                "confidence": 0.9,
            }
        )
        hit += 1

    return {
        "fileName": "test.wav",
        "duration": seconds,
        "sampleRate": 44100,
        "start": 0.0,
        "analyzedDuration": seconds,
        "waveform": waveform,
        "markers": markers,
    }


def test_bfl_dashboard_json_to_flux_motion_schedule():
    tracks = tracks_from_analysis_json(json.dumps(make_bfl_analysis()))
    result = build_audio_motion_schedule(
        tracks,
        num_frames=48,
        fps=24.0,
        feature="auto",
        base_zoom=1.0,
        zoom_gain=0.3,
        base_strength=0.2,
        strength_gain=0.2,
        smoothing=0.0,
    )

    assert result.driver == "waveform"
    assert result.tracks.meta["format"] == "bfl"
    assert len(result.motion_frames) == 48
    assert isinstance(result.motion_frames[0], MotionFrame)
    assert max(result.values["zoom"]) - min(result.values["zoom"]) > 0.02
    assert "zoom" in result.deforum_strings
    assert result.motion_schedule[0]["strength"] == pytest.approx(
        result.motion_frames[0].strength
    )
    assert result.engine_schedule["motion_frames"] == result.motion_frames


def test_forced_driver_falls_back_to_available_data():
    waveform_only = {
        "analyzedDuration": 4.0,
        "markers": [],
        "waveform": [
            {
                "time": index / 90.0,
                "amplitude": abs(math.sin(index / 8.0)),
                "low": abs(math.sin(index / 8.0)),
                "mid": 0.3,
                "high": 0.2,
            }
            for index in range(360)
        ],
    }
    waveform_tracks = tracks_from_analysis_json(waveform_only)
    result = build_audio_motion_schedule(
        waveform_tracks,
        48,
        24.0,
        feature="markers",
        zoom_gain=0.3,
        smoothing=0.0,
    )
    assert result.driver == "waveform"
    assert max(result.values["zoom"]) - min(result.values["zoom"]) > 0.02

    marker_tracks = tracks_from_analysis_json(
        {"kick_times": [0.0, 0.5, 1.0, 1.5], "duration": 2.0}
    )
    result = build_audio_motion_schedule(
        marker_tracks,
        48,
        24.0,
        feature="waveform",
        zoom_gain=0.3,
        smoothing=0.0,
    )
    assert result.driver == "markers"


def test_marker_only_schedule_uses_absolute_clip_time():
    marker_tracks = tracks_from_analysis_json({"kick_times": [1.0], "duration": 4.0})
    result = build_audio_motion_schedule(
        marker_tracks,
        40,
        10.0,
        feature="markers",
        zoom_gain=0.3,
        smoothing=0.0,
    )

    assert result.driver == "markers"
    assert result.bands["low"][0] == 0.0
    assert int(np.argmax(result.bands["low"])) == 10
    assert result.values["zoom"][10] > result.values["zoom"][0]


def test_waveform_audio_path_uses_numpy_stft():
    sample_rate = 22050
    seconds = 2.0
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    envelope = np.clip(np.sin(2 * math.pi * 2.0 * t), 0.0, 1.0)
    waveform = (np.sin(2 * math.pi * 110.0 * t) * envelope).reshape(1, 1, -1)

    tracks = tracks_from_waveform(waveform, sample_rate=sample_rate)
    result = build_audio_motion_schedule(tracks, 48, 24.0, zoom_gain=0.3, smoothing=0.0)

    assert result.driver == "waveform"
    assert tracks.meta["format"] == "waveform"
    assert max(result.values["zoom"]) - min(result.values["zoom"]) > 0.01


def test_waveform_rejects_invalid_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        tracks_from_waveform(np.zeros((1, 1, 100)), sample_rate=0)


def test_video_feature_source_extracts_mp4(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")

    path = tmp_path / "motion.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (64, 64),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter could not open mp4v writer")

    for index in range(24):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        x = min(48, index * 2)
        frame[16:48, x : x + 12] = 255
        writer.write(frame)
    writer.release()

    tracks = tracks_from_video(path)
    result = build_audio_motion_schedule(tracks, 24, 12.0, zoom_gain=0.2, smoothing=0.0)

    assert tracks.meta["format"] == "video"
    assert tracks.meta["frames"] == 24
    assert result.driver == "waveform"
    assert len(result.motion_frames) == 24
