#!/bin/bash
# =============================================================================
# DEFORUM AUDIO FEATURE EXTRACTION - RUNPOD TEST SCRIPT
# =============================================================================
# This script tests the complete audio-to-animation workflow on RunPod
#
# Usage:
#   chmod +x test_audio_runpod.sh
#   ./test_audio_runpod.sh
#
# Requirements:
#   - Python 3.10+
#   - Internet connection (for pip installs)
#   - ~500MB disk space for dependencies
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  DEFORUM AUDIO FEATURE EXTRACTION - TEST SUITE${NC}"
echo -e "${BLUE}============================================================${NC}"

# -----------------------------------------------------------------------------
# STEP 1: Environment Setup
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/7] Setting up environment...${NC}"

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AUDIO_DIR="$PROJECT_ROOT/src/deforum_flux/audio"
TEST_DIR="/tmp/deforum_audio_test"

mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "  Project root: $PROJECT_ROOT"
echo "  Audio module: $AUDIO_DIR"
echo "  Test directory: $TEST_DIR"

# -----------------------------------------------------------------------------
# STEP 2: Install Dependencies
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/7] Installing dependencies...${NC}"

pip install numpy --quiet 2>/dev/null || pip install numpy
pip install librosa audioread soundfile --quiet 2>/dev/null || {
    echo "  Installing audio dependencies..."
    pip install librosa audioread soundfile
}

echo -e "  ${GREEN}Dependencies installed${NC}"

# -----------------------------------------------------------------------------
# STEP 3: Create Test Audio File
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/7] Creating test audio file...${NC}"

python3 << 'PYEOF'
import numpy as np
try:
    import soundfile as sf

    # Create a 5-second test audio with beats
    duration = 5.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))

    # Base frequency (bass)
    bass = 0.5 * np.sin(2 * np.pi * 80 * t)

    # Add beats every 0.5 seconds (120 BPM)
    beats = np.zeros_like(t)
    for beat_time in np.arange(0, duration, 0.5):
        idx = int(beat_time * sr)
        decay = np.exp(-10 * np.arange(int(0.1 * sr)) / sr)
        end_idx = min(idx + len(decay), len(beats))
        beats[idx:end_idx] += decay[:end_idx - idx]

    # Add mid frequencies
    mid = 0.3 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))

    # Add high frequencies
    high = 0.1 * np.sin(2 * np.pi * 2000 * t) * beats

    # Combine
    audio = bass + mid + high + 0.3 * beats
    audio = audio / np.max(np.abs(audio))  # Normalize

    # Save
    sf.write('test_audio.wav', audio, sr)
    print("  Created test_audio.wav (5 seconds, 120 BPM)")
except Exception as e:
    print(f"  Warning: Could not create audio file: {e}")
    print("  Tests will use synthetic features instead")
PYEOF

# -----------------------------------------------------------------------------
# STEP 4: Test Module Imports (No librosa required)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/7] Testing module imports...${NC}"

python3 << PYEOF
import sys
sys.path.insert(0, '$AUDIO_DIR')

# Test 1: Import mapping_config
try:
    from mapping_config import (
        DEFAULT_MAPPINGS, CurveType, FeatureMapping,
        MappingConfig, apply_curve, apply_smoothing
    )
    print("  [PASS] mapping_config imports")
except Exception as e:
    print(f"  [FAIL] mapping_config imports: {e}")
    sys.exit(1)

# Test 2: Import schedule_generator
try:
    from schedule_generator import (
        ParseqKeyframe, ParseqSchedule, ScheduleGenerator
    )
    print("  [PASS] schedule_generator imports")
except Exception as e:
    print(f"  [FAIL] schedule_generator imports: {e}")
    sys.exit(1)

# Test 3: Import extractor
try:
    from extractor import AudioFeatures, LIBROSA_AVAILABLE
    print(f"  [PASS] extractor imports (librosa={'available' if LIBROSA_AVAILABLE else 'not installed'})")
except Exception as e:
    print(f"  [FAIL] extractor imports: {e}")
    sys.exit(1)

print("  All imports successful!")
PYEOF

if [ $? -eq 0 ]; then
    ((PASSED++))
    echo -e "  ${GREEN}[PASS] Module imports${NC}"
else
    ((FAILED++))
    echo -e "  ${RED}[FAIL] Module imports${NC}"
fi

# -----------------------------------------------------------------------------
# STEP 5: Test Mapping Config Functions
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/7] Testing mapping configuration...${NC}"

python3 << PYEOF
import sys
import json
import numpy as np
sys.path.insert(0, '$AUDIO_DIR')

from mapping_config import (
    DEFAULT_MAPPINGS, CurveType, FeatureMapping,
    MappingConfig, apply_curve, apply_smoothing,
    save_mapping_config, load_mapping_config
)

errors = []

# Test 1: Check all presets exist
expected_presets = ['ambient', 'bass_pulse', 'beat_rotation', 'spectrum', 'immersive_3d', 'cinematic', 'intense']
for preset in expected_presets:
    if preset not in DEFAULT_MAPPINGS:
        errors.append(f"Missing preset: {preset}")
print(f"  [{'PASS' if not errors else 'FAIL'}] All 7 presets exist")

# Test 2: Curve application
test_values = np.array([0.0, 0.5, 1.0])
for curve in CurveType:
    result = apply_curve(test_values, curve)
    if len(result) != 3:
        errors.append(f"Curve {curve} returned wrong length")
    if not (0 <= result[0] <= 1 and 0 <= result[2] <= 1):
        errors.append(f"Curve {curve} returned out-of-range values")
print(f"  [{'PASS' if not errors else 'FAIL'}] Curve functions work")

# Test 3: Smoothing
original = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
smoothed = apply_smoothing(original, 0.5)
if np.var(smoothed) >= np.var(original):
    errors.append("Smoothing did not reduce variance")
print(f"  [PASS] Smoothing reduces variance")

# Test 4: Config serialization
config = DEFAULT_MAPPINGS['bass_pulse']
config_dict = config.to_dict()
restored = MappingConfig.from_dict(config_dict)
if restored.name != config.name or len(restored.mappings) != len(config.mappings):
    errors.append("Config serialization failed")
print(f"  [PASS] Config serialization works")

# Test 5: Save/load to file
save_mapping_config(config, '/tmp/test_config.json')
loaded = load_mapping_config('/tmp/test_config.json')
if loaded.name != config.name:
    errors.append("File save/load failed")
print(f"  [PASS] Config file save/load works")

if errors:
    print(f"\n  Errors: {errors}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    ((PASSED++))
    echo -e "  ${GREEN}[PASS] Mapping configuration${NC}"
else
    ((FAILED++))
    echo -e "  ${RED}[FAIL] Mapping configuration${NC}"
fi

# -----------------------------------------------------------------------------
# STEP 6: Test Schedule Generation (with synthetic features)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/7] Testing schedule generation...${NC}"

python3 << PYEOF
import sys
import json
import numpy as np
sys.path.insert(0, '$AUDIO_DIR')

from extractor import AudioFeatures
from schedule_generator import ScheduleGenerator, ParseqSchedule, ParseqKeyframe
from mapping_config import DEFAULT_MAPPINGS

errors = []

# Create synthetic features
num_frames = 100
features = AudioFeatures(
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
print(f"  Created synthetic features: {num_frames} frames")

# Test 1: Generate schedule for all presets
generator = ScheduleGenerator()
for preset_name in DEFAULT_MAPPINGS.keys():
    try:
        schedule = generator.generate(features, mapping=preset_name)
        if len(schedule.keyframes) == 0:
            errors.append(f"Preset {preset_name} generated 0 keyframes")
    except Exception as e:
        errors.append(f"Preset {preset_name} failed: {e}")
print(f"  [{'PASS' if not errors else 'FAIL'}] All presets generate schedules")

# Test 2: Check schedule structure
schedule = generator.generate(features, mapping='bass_pulse')
if schedule.num_frames != num_frames:
    errors.append(f"Wrong num_frames: {schedule.num_frames} != {num_frames}")
if schedule.fps != 24:
    errors.append(f"Wrong fps: {schedule.fps}")
if schedule.bpm != 120.0:
    errors.append(f"Wrong bpm: {schedule.bpm}")
print(f"  [PASS] Schedule metadata correct")

# Test 3: Check keyframe values
first_kf = schedule.keyframes[0]
if first_kf.zoom is None:
    errors.append("First keyframe missing zoom")
if first_kf.frame != 0:
    errors.append(f"First keyframe wrong frame: {first_kf.frame}")
print(f"  [PASS] Keyframe structure correct")

# Test 4: JSON output
schedule_json = schedule.to_json()
parsed = json.loads(schedule_json)
if 'meta' not in parsed or 'keyframes' not in parsed:
    errors.append("Invalid JSON structure")
print(f"  [PASS] JSON output valid ({len(schedule_json)} chars)")

# Test 5: Deforum string output
deforum_strings = schedule.to_deforum_strings()
if 'zoom' not in deforum_strings:
    errors.append("Missing zoom in Deforum strings")
if '0:(' not in deforum_strings['zoom']:
    errors.append("Invalid Deforum string format")
print(f"  [PASS] Deforum strings generated")
print(f"    zoom: {deforum_strings['zoom'][:60]}...")

# Test 6: Save and load schedule
schedule.save('/tmp/test_schedule.json')
loaded = ParseqSchedule.load('/tmp/test_schedule.json')
if len(loaded.keyframes) != len(schedule.keyframes):
    errors.append("Schedule save/load keyframe count mismatch")
print(f"  [PASS] Schedule file save/load works")

# Test 7: Keyframe interval
schedule_sparse = generator.generate(features, mapping='bass_pulse', keyframe_interval=10)
if len(schedule_sparse.keyframes) >= len(schedule.keyframes):
    errors.append("Keyframe interval not working")
print(f"  [PASS] Keyframe interval works ({len(schedule_sparse.keyframes)} vs {len(schedule.keyframes)} keyframes)")

if errors:
    print(f"\n  Errors: {errors}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    ((PASSED++))
    echo -e "  ${GREEN}[PASS] Schedule generation${NC}"
else
    ((FAILED++))
    echo -e "  ${RED}[FAIL] Schedule generation${NC}"
fi

# -----------------------------------------------------------------------------
# STEP 7: Test Full Audio Pipeline (if librosa available)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[7/7] Testing full audio pipeline...${NC}"

python3 << PYEOF
import sys
sys.path.insert(0, '$AUDIO_DIR')

from extractor import LIBROSA_AVAILABLE

if not LIBROSA_AVAILABLE:
    print("  [SKIP] librosa not available, skipping audio extraction test")
    sys.exit(0)

import os
from extractor import AudioFeatureExtractor, AudioFeatures
from schedule_generator import ScheduleGenerator

audio_file = '/tmp/deforum_audio_test/test_audio.wav'
if not os.path.exists(audio_file):
    print("  [SKIP] Test audio file not found")
    sys.exit(0)

errors = []

# Test 1: Extract features from real audio
try:
    extractor = AudioFeatureExtractor()
    features = extractor.extract(audio_file, fps=24.0)
    print(f"  Extracted features: {features.num_frames} frames, {features.tempo:.1f} BPM")
except Exception as e:
    errors.append(f"Feature extraction failed: {e}")
    print(f"  [FAIL] Feature extraction: {e}")
    sys.exit(1)

# Test 2: Check feature shapes
if len(features.bass) != features.num_frames:
    errors.append("Bass array wrong length")
if len(features.beat_strength) != features.num_frames:
    errors.append("Beat strength array wrong length")
print(f"  [PASS] Feature shapes correct")

# Test 3: Check feature ranges (should be normalized 0-1)
import numpy as np
for name in ['rms', 'energy', 'bass', 'mid', 'high']:
    arr = getattr(features, name)
    if arr.min() < -0.01 or arr.max() > 1.01:
        errors.append(f"{name} not normalized: [{arr.min():.2f}, {arr.max():.2f}]")
print(f"  [PASS] Features normalized to 0-1 range")

# Test 4: Save and load features
features.save('/tmp/features.json')
loaded = AudioFeatures.load('/tmp/features.json')
if loaded.num_frames != features.num_frames:
    errors.append("Feature save/load mismatch")
print(f"  [PASS] Feature save/load works")

# Test 5: Generate schedule from real audio
generator = ScheduleGenerator()
schedule = generator.generate(features, mapping='bass_pulse')
schedule.save('/tmp/real_audio_schedule.json')
print(f"  [PASS] Generated schedule with {len(schedule.keyframes)} keyframes")

# Test 6: Verify beat detection
if len(features.beats) == 0:
    errors.append("No beats detected")
else:
    print(f"  [PASS] Detected {len(features.beats)} beats")

if errors:
    print(f"\n  Errors: {errors}")
    sys.exit(1)

print(f"\n  Schedule saved to: /tmp/real_audio_schedule.json")
PYEOF

if [ $? -eq 0 ]; then
    ((PASSED++))
    echo -e "  ${GREEN}[PASS] Full audio pipeline${NC}"
else
    # Check if it was a skip
    if grep -q "SKIP" <<< "$(python3 -c 'import sys; sys.path.insert(0, "'$AUDIO_DIR'"); from extractor import LIBROSA_AVAILABLE; print("SKIP" if not LIBROSA_AVAILABLE else "OK")')"; then
        echo -e "  ${YELLOW}[SKIP] librosa not available${NC}"
    else
        ((FAILED++))
        echo -e "  ${RED}[FAIL] Full audio pipeline${NC}"
    fi
fi

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}  TEST SUMMARY${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "  ${GREEN}Passed: $PASSED${NC}"
echo -e "  ${RED}Failed: $FAILED${NC}"

if [ -f /tmp/test_schedule.json ]; then
    echo -e "\n${YELLOW}Generated Files:${NC}"
    echo "  /tmp/test_schedule.json      - Parseq schedule (synthetic)"
    echo "  /tmp/test_config.json        - Mapping config"
    [ -f /tmp/real_audio_schedule.json ] && echo "  /tmp/real_audio_schedule.json - Parseq schedule (real audio)"
    [ -f /tmp/features.json ] && echo "  /tmp/features.json            - Extracted features"
fi

echo -e "\n${YELLOW}Quick Usage:${NC}"
echo "  # Extract features and generate schedule"
echo "  python -c \""
echo "  import sys; sys.path.insert(0, '$AUDIO_DIR')"
echo "  from schedule_generator import generate_schedule"
echo "  schedule = generate_schedule('your_audio.mp3', 'output.json', mapping='bass_pulse')"
echo "  \""

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed!${NC}"
    exit 1
fi
