# Audio Feature Extraction for Koshi

Generate animation schedules from audio files for use with Koshi and Parseq.

## Quick Start

```bash
# Install audio dependencies
pip install librosa audioread soundfile

# Generate schedule from audio
python -m koshi_flux.audio music.mp3 -o schedule.json -m bass_pulse
```

## Features Extracted

| Feature | Description | Range |
|---------|-------------|-------|
| `rms` | Root mean square energy | 0-1 |
| `energy` | Smoothed RMS envelope | 0-1 |
| `bass` | Low frequency energy (20-250Hz) | 0-1 |
| `mid` | Mid frequency energy (250-4kHz) | 0-1 |
| `high` | High frequency energy (4k-20kHz) | 0-1 |
| `beat_strength` | Beat envelope with decay | 0-1 |
| `onset_strength` | Note onset detection | 0-1 |
| `spectral_centroid` | Brightness/sharpness | 0-1 |
| `spectral_bandwidth` | Spectral spread | 0-1 |
| `spectral_rolloff` | High-frequency cutoff | 0-1 |
| `spectral_flatness` | Tonal vs noisy | 0-1 |

## Mapping Presets

| Preset | Description | Best For |
|--------|-------------|----------|
| `ambient` | Subtle, flowing movement | Atmospheric, ambient music |
| `bass_pulse` | Zoom pulses on bass hits | EDM, hip-hop, bass-heavy |
| `beat_rotation` | Rotation synced to beats | Rhythmic, percussive |
| `spectrum` | Bass/mid/high → different axes | Full-range music |
| `immersive_3d` | Full 3D rotation + depth | Immersive experiences |
| `cinematic` | Slow, smooth movements | Film scores, orchestral |
| `intense` | Aggressive movement | Metal, hardcore, high-energy |

## Python API

```python
from koshi_flux.audio import (
    AudioFeatureExtractor,
    ScheduleGenerator,
    DEFAULT_MAPPINGS,
)

# 1. Extract features
extractor = AudioFeatureExtractor()
features = extractor.extract("music.mp3", fps=24, duration=60)

print(f"Tempo: {features.tempo} BPM")
print(f"Frames: {features.num_frames}")
print(f"Beats: {len(features.beats)}")

# 2. Generate Parseq schedule
generator = ScheduleGenerator()
schedule = generator.generate(
    features,
    mapping="bass_pulse",  # or custom MappingConfig
    keyframe_interval=1,   # every frame
    prompt="cosmic nebula",
)

# 3. Save for Parseq
schedule.save("schedule.json")

# 4. Or get Koshi keyframe strings
deforum_params = schedule.to_deforum_strings()
# {"zoom": "0:(1.0), 12:(1.05), ...", "angle": "0:(0), ..."}
```

## Custom Mappings

```python
from koshi_flux.audio import MappingConfig, FeatureMapping, CurveType

config = MappingConfig(
    name="My Custom",
    description="Bass to zoom, beats to rotation",
    mappings=[
        FeatureMapping(
            feature="bass",
            parameter="zoom",
            min_value=1.0,
            max_value=1.2,
            curve=CurveType.EASE_OUT,
            threshold=0.2,  # Only activate above 20%
        ),
        FeatureMapping(
            feature="beat_strength",
            parameter="angle",
            min_value=-10,
            max_value=10,
            smoothing=0.1,
        ),
    ],
    global_smoothing=0.15,
)

schedule = generator.generate(features, mapping=config)
```

## CLI Usage

```bash
# Basic usage
python -m koshi_flux.audio music.mp3 -o schedule.json

# Different preset
python -m koshi_flux.audio music.mp3 -m spectrum -o schedule.json

# Custom settings
python -m koshi_flux.audio music.mp3 \
    --fps 30 \
    --duration 120 \
    --keyframe-interval 2 \
    -o schedule.json

# List all presets
python -m koshi_flux.audio --list-presets

# Show Koshi strings
python -m koshi_flux.audio music.mp3 --format deforum

# Save extracted features for reuse
python -m koshi_flux.audio music.mp3 --save-features features.json
```

## Output Formats

### Parseq JSON
```json
{
  "meta": {
    "name": "Audio Schedule - Bass Pulse",
    "fps": 24,
    "bpm": 120.0,
    "num_frames": 1440
  },
  "keyframes": [
    {"frame": 0, "zoom": 1.0, "angle": 0.0},
    {"frame": 1, "zoom": 1.02, "angle": 0.5},
    ...
  ]
}
```

### Koshi Strings
```
zoom: 0:(1.0), 1:(1.02), 2:(1.05), ...
angle: 0:(0), 1:(0.5), 2:(1.2), ...
```

## Curve Types

| Curve | Effect |
|-------|--------|
| `LINEAR` | Constant rate of change |
| `EASE_IN` | Slow start, fast end |
| `EASE_OUT` | Fast start, slow end |
| `EASE_IN_OUT` | Slow start and end |
| `EXPONENTIAL` | Accelerating curve |
| `LOGARITHMIC` | Decelerating curve |
| `SINE` | Smooth wave-like |

## Validation

Run tests to validate installation:

```bash
# Full test suite
./scripts/test_audio_runpod.sh

# Quick import test
python -c "from koshi_flux.audio import AudioFeatureExtractor; print('OK')"
```

## Troubleshooting

**librosa not found:**
```bash
pip install librosa audioread soundfile
```

**No audio backend:**
```bash
# Linux
apt-get install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile
```

**Empty beat detection:**
- Try different audio with clearer rhythm
- Adjust audio levels (normalize)
- Check sample rate compatibility
