"""
Music Parameter Mapping Nodes for ComfyUI

These nodes enable music-reactive video generation:
- Map audio features to generation parameters
- Beat detection and onset tracking
- Generate Deforum-style parameter schedules
- Preset configurations for common effects
"""

import torch
from typing import Tuple, Dict, List, Optional, Any

try:
    from ltx_audio_injection.models.audio_parameter_mapper import (
        AudioFeatureExtractor,
        ParameterScheduler,
        AudioReactiveConfig,
        ParameterMapping,
        AudioFeature,
        MappingCurve,
        AudioReactivePresets,
        MusicReactiveGenerator,
    )
    LTX_MUSIC_AVAILABLE = True
except ImportError:
    LTX_MUSIC_AVAILABLE = False


class AudioParameterMapper:
    """
    Map audio features to generation parameters over time.

    Creates per-frame parameter values based on audio analysis:
    - Beats → zoom pulses
    - Energy → motion intensity
    - Frequency bands → color shifts
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "forceInput": True}),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 1,
                    "max": 1000,
                }),
                "audio_feature": ([
                    "energy", "beat", "onset", "bass", "mid", "high",
                    "spectral_centroid", "spectral_bandwidth", "spectral_flux"
                ], {"default": "beat"}),
                "target_param": ("STRING", {
                    "default": "zoom",
                }),
                "min_value": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                }),
                "max_value": ("FLOAT", {
                    "default": 1.3,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "curve": ([
                    "linear", "exponential", "logarithmic", "sigmoid",
                    "sine", "bounce", "pulse", "smooth_step"
                ], {"default": "smooth_step"}),
                "attack": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "decay": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("PARAM_SCHEDULE", "TENSOR", "STRING")
    RETURN_NAMES = ("schedule", "values", "deforum_string")
    FUNCTION = "map_parameters"
    CATEGORY = "LTX-Audio/Music"

    def map_parameters(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        num_frames: int,
        audio_feature: str,
        target_param: str,
        min_value: float,
        max_value: float,
        fps: float = 24.0,
        curve: str = "smooth_step",
        attack: float = 0.05,
        decay: float = 0.2,
        threshold: float = 0.1,
        smoothing: float = 0.0,
        invert: bool = False,
    ) -> Tuple[Dict, torch.Tensor, str]:
        if not LTX_MUSIC_AVAILABLE:
            raise ImportError("ltx_audio_injection music module is required.")

        # Create mapping
        mapping = ParameterMapping(
            audio_feature=AudioFeature(audio_feature),
            target_param=target_param,
            min_value=min_value,
            max_value=max_value,
            curve=MappingCurve(curve),
            attack=attack,
            decay=decay,
            threshold=threshold,
            smoothing=smoothing,
            invert=invert,
        )

        # Create config with single mapping
        config = AudioReactiveConfig(
            mappings=[mapping],
            fps=fps,
        )

        # Generate schedule
        scheduler = ParameterScheduler(config)
        schedules = scheduler.generate_schedule(audio, num_frames, sample_rate)

        # Get values for the target parameter
        values = schedules.get(target_param, torch.zeros(num_frames))

        # Convert to Deforum format
        deforum_schedules = scheduler.to_deforum_schedule(schedules)
        deforum_string = deforum_schedules.get(target_param, "")

        schedule_dict = {
            "param_name": target_param,
            "values": values.tolist(),
            "num_frames": num_frames,
            "fps": fps,
        }

        return (schedule_dict, values, deforum_string)


class AudioReactivePresetNode:
    """
    Load a preset configuration for audio-reactive effects.

    Available presets:
    - beat_zoom: Zoom in on beats
    - energy_motion: More motion during loud sections
    - spectrum_colors: Map frequency bands to colors
    - music_video: Optimized for music videos
    - full_reactive: Comprehensive audio reactivity
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ([
                    "beat_zoom",
                    "energy_motion",
                    "spectrum_colors",
                    "music_video",
                    "full_reactive",
                ], {"default": "full_reactive"}),
            },
            "optional": {
                "global_reactivity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
                "beat_sensitivity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "load_preset"
    CATEGORY = "LTX-Audio/Music"

    def load_preset(
        self,
        preset: str,
        global_reactivity: float = 1.0,
        beat_sensitivity: float = 1.0,
    ) -> Tuple[Dict]:
        if not LTX_MUSIC_AVAILABLE:
            raise ImportError("ltx_audio_injection music module is required.")

        # Load preset
        if preset == "beat_zoom":
            config = AudioReactivePresets.beat_zoom()
        elif preset == "energy_motion":
            config = AudioReactivePresets.energy_motion()
        elif preset == "spectrum_colors":
            config = AudioReactivePresets.spectrum_colors()
        elif preset == "music_video":
            config = AudioReactivePresets.music_video()
        else:
            config = AudioReactivePresets.full_reactive()

        # Override settings
        config.global_reactivity = global_reactivity
        config.beat_sensitivity = beat_sensitivity

        # Convert to serializable format
        config_dict = {
            "preset": preset,
            "global_reactivity": global_reactivity,
            "beat_sensitivity": beat_sensitivity,
            "mappings": [
                {
                    "audio_feature": m.audio_feature.value,
                    "target_param": m.target_param,
                    "min_value": m.min_value,
                    "max_value": m.max_value,
                    "curve": m.curve.value,
                    "attack": m.attack,
                    "decay": m.decay,
                }
                for m in config.mappings
            ],
        }

        return (config_dict,)


class BeatDetectorNode:
    """
    Detect beats and onsets in audio.

    Outputs:
    - Beat times in seconds
    - Onset times in seconds
    - Per-frame beat strength
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "forceInput": True}),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 1,
                    "max": 1000,
                }),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0}),
                "sensitivity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("beat_strength", "onset_strength", "beat_frames")
    FUNCTION = "detect"
    CATEGORY = "LTX-Audio/Music"

    def detect(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        num_frames: int,
        fps: float = 24.0,
        sensitivity: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if not LTX_MUSIC_AVAILABLE:
            raise ImportError("ltx_audio_injection music module is required.")

        extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        features = extractor.extract_all(audio, num_frames, fps)

        beat_strength = features.get(AudioFeature.BEAT, torch.zeros(num_frames))
        onset_strength = features.get(AudioFeature.ONSET, torch.zeros(num_frames))

        # Apply sensitivity
        beat_strength = beat_strength * sensitivity
        onset_strength = onset_strength * sensitivity

        # Find frames with significant beats
        beat_frames = []
        for i, strength in enumerate(beat_strength.tolist()):
            if strength > 0.5:
                beat_frames.append(str(i))

        beat_frames_string = ", ".join(beat_frames)

        return (beat_strength, onset_strength, beat_frames_string)


class AudioToDeforumSchedule:
    """
    Generate Deforum-compatible parameter schedules from audio.

    Outputs schedule strings that can be directly used in
    Deforum or similar animation tools.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "forceInput": True}),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 1,
                    "max": 1000,
                }),
                "config": ("AUDIO_CONFIG",),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0}),
            },
        }

    RETURN_TYPES = ("DEFORUM_SCHEDULES", "STRING")
    RETURN_NAMES = ("schedules", "schedules_text")
    FUNCTION = "generate"
    CATEGORY = "LTX-Audio/Music"

    def generate(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        num_frames: int,
        config: Dict,
        fps: float = 24.0,
    ) -> Tuple[Dict[str, str], str]:
        if not LTX_MUSIC_AVAILABLE:
            raise ImportError("ltx_audio_injection music module is required.")

        # Recreate config from dict
        mappings = [
            ParameterMapping(
                audio_feature=AudioFeature(m["audio_feature"]),
                target_param=m["target_param"],
                min_value=m["min_value"],
                max_value=m["max_value"],
                curve=MappingCurve(m["curve"]),
                attack=m.get("attack", 0.1),
                decay=m.get("decay", 0.3),
            )
            for m in config.get("mappings", [])
        ]

        audio_config = AudioReactiveConfig(
            mappings=mappings,
            global_reactivity=config.get("global_reactivity", 1.0),
            beat_sensitivity=config.get("beat_sensitivity", 1.0),
            fps=fps,
        )

        # Generate schedules
        scheduler = ParameterScheduler(audio_config)
        param_schedules = scheduler.generate_schedule(audio, num_frames, sample_rate)
        deforum_schedules = scheduler.to_deforum_schedule(param_schedules)

        # Create readable text output
        text_lines = ["# Audio-Reactive Parameter Schedules", ""]
        for param_name, schedule_string in deforum_schedules.items():
            text_lines.append(f"{param_name}:")
            text_lines.append(f"  {schedule_string[:200]}...")
            text_lines.append("")

        schedules_text = "\n".join(text_lines)

        return (deforum_schedules, schedules_text)


class CombineParameterSchedules:
    """
    Combine multiple parameter schedules into one dictionary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_1": ("PARAM_SCHEDULE",),
            },
            "optional": {
                "schedule_2": ("PARAM_SCHEDULE",),
                "schedule_3": ("PARAM_SCHEDULE",),
                "schedule_4": ("PARAM_SCHEDULE",),
                "schedule_5": ("PARAM_SCHEDULE",),
            },
        }

    RETURN_TYPES = ("COMBINED_SCHEDULES", "STRING")
    RETURN_NAMES = ("schedules", "summary")
    FUNCTION = "combine"
    CATEGORY = "LTX-Audio/Music"

    def combine(
        self,
        schedule_1: Dict,
        schedule_2: Dict = None,
        schedule_3: Dict = None,
        schedule_4: Dict = None,
        schedule_5: Dict = None,
    ) -> Tuple[Dict, str]:
        combined = {}
        all_schedules = [schedule_1, schedule_2, schedule_3, schedule_4, schedule_5]

        for schedule in all_schedules:
            if schedule is not None:
                param_name = schedule.get("param_name", "unknown")
                values = schedule.get("values", [])
                combined[param_name] = values

        # Create summary
        summary_lines = [f"Combined {len(combined)} parameter schedules:"]
        for param_name, values in combined.items():
            if values:
                min_val = min(values)
                max_val = max(values)
                summary_lines.append(f"  {param_name}: range [{min_val:.3f}, {max_val:.3f}]")

        summary = "\n".join(summary_lines)

        return (combined, summary)


class CreateCustomMapping:
    """
    Create a custom audio-to-parameter mapping.

    Allows fine-grained control over how audio features
    affect generation parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_feature": ([
                    "energy", "beat", "onset", "bass", "mid", "high",
                    "spectral_centroid", "spectral_bandwidth", "spectral_flux",
                    "loudness", "pitch"
                ], {"default": "beat"}),
                "target_param": ("STRING", {"default": "zoom"}),
                "min_value": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.5, "step": 0.01}),
            },
            "optional": {
                "curve": ([
                    "linear", "exponential", "logarithmic", "sigmoid",
                    "sine", "bounce", "pulse", "smooth_step"
                ], {"default": "smooth_step"}),
                "attack": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 2.0, "step": 0.01}),
                "decay": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("PARAM_MAPPING",)
    RETURN_NAMES = ("mapping",)
    FUNCTION = "create"
    CATEGORY = "LTX-Audio/Music"

    def create(
        self,
        audio_feature: str,
        target_param: str,
        min_value: float,
        max_value: float,
        curve: str = "smooth_step",
        attack: float = 0.05,
        decay: float = 0.2,
        threshold: float = 0.1,
        multiplier: float = 1.0,
        delay: float = 0.0,
        smoothing: float = 0.0,
        invert: bool = False,
    ) -> Tuple[Dict]:
        mapping_dict = {
            "audio_feature": audio_feature,
            "target_param": target_param,
            "min_value": min_value,
            "max_value": max_value,
            "curve": curve,
            "attack": attack,
            "decay": decay,
            "threshold": threshold,
            "multiplier": multiplier,
            "delay": delay,
            "smoothing": smoothing,
            "invert": invert,
        }
        return (mapping_dict,)


class CombineMappings:
    """
    Combine multiple parameter mappings into a config.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mapping_1": ("PARAM_MAPPING",),
            },
            "optional": {
                "mapping_2": ("PARAM_MAPPING",),
                "mapping_3": ("PARAM_MAPPING",),
                "mapping_4": ("PARAM_MAPPING",),
                "mapping_5": ("PARAM_MAPPING",),
                "mapping_6": ("PARAM_MAPPING",),
                "global_reactivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            },
        }

    RETURN_TYPES = ("AUDIO_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "combine"
    CATEGORY = "LTX-Audio/Music"

    def combine(
        self,
        mapping_1: Dict,
        mapping_2: Dict = None,
        mapping_3: Dict = None,
        mapping_4: Dict = None,
        mapping_5: Dict = None,
        mapping_6: Dict = None,
        global_reactivity: float = 1.0,
    ) -> Tuple[Dict]:
        mappings = [mapping_1]
        for m in [mapping_2, mapping_3, mapping_4, mapping_5, mapping_6]:
            if m is not None:
                mappings.append(m)

        config_dict = {
            "preset": "custom",
            "global_reactivity": global_reactivity,
            "mappings": mappings,
        }

        return (config_dict,)
