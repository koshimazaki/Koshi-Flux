"""
Centralized Configuration Management for Koshi Flux

This module provides a unified configuration system that merges the previous
Config and KoshiConfig classes, eliminating duplication and providing a
single source of truth for all configuration settings.

"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Unified Deforum configuration - merges Config and KoshiConfig


@dataclass
class Config:
    """
    Unified configuration class for Koshi Flux backend.
    
    This class merges the previous Config and KoshiConfig classes to eliminate
    duplication and provide a single, comprehensive configuration system.
    
    Configuration is organized into logical groups:
    - Core Settings: Basic system configuration
    - Generation Settings: Image/video generation parameters
    - Animation Settings: Motion and keyframe parameters
    - Performance Settings: Optimization and memory management
    - API Settings: Server and network configuration
    - Security Settings: Authentication and validation
    - Testing Settings: Test-specific options
    """
    
    # ===== CORE SETTINGS =====
    device: str = "auto"
    models_path: str = "models"
    output_path: str = "outputs" 
    cache_path: str = "cache"
    model_name: str = "flux-schnell"
    
    # ===== GENERATION SETTINGS =====
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    prompt: Optional[str] = None
    max_prompt_length: int = 256
    
    # ===== ANIMATION SETTINGS =====
    # Basic animation parameters
    animation_mode: str = "2D"  # "2D", "3D", "Video Input"
    max_frames: int = 10
    fps: int = 24
    
    # Motion schedules (keyframe strings) - from KoshiConfig
    zoom: str = "0:(1.0)"
    angle: str = "0:(0)"
    translation_x: str = "0:(0)"
    translation_y: str = "0:(0)"
    translation_z: str = "0:(0)"
    rotation_3d_x: str = "0:(0)"
    rotation_3d_y: str = "0:(0)"
    rotation_3d_z: str = "0:(0)"
    
    # Strength schedules - from KoshiConfig
    strength_schedule: str = "0:(0.65)"
    noise_schedule: str = "0:(0.02)"
    contrast_schedule: str = "0:(1.0)"
    
    # Classic Deforum motion settings (simplified parameters) - from Config
    enable_learned_motion: bool = False  # Classic mode only
    motion_strength: float = 0.5
    motion_coherence: float = 0.7
    motion_schedule: str = "0:(0.5)"
    depth_strength: float = 0.3
    perspective_flip_theta: str = "0:(0)"
    perspective_flip_phi: str = "0:(0)"
    perspective_flip_gamma: str = "0:(0)"
    perspective_flip_fv: str = "0:(53)"
    motion_mode: str = "geometric"  # "geometric", "learned", "hybrid"
    
    # 3D settings - from KoshiConfig
    midas_weight: float = 0.3
    near_plane: int = 200
    far_plane: int = 10000
    fov: int = 40
    
    # Prompts (frame -> prompt mapping) - from KoshiConfig
    positive_prompts: Dict[str, str] = field(default_factory=lambda: {"0": "a beautiful landscape"})
    negative_prompts: Dict[str, str] = field(default_factory=dict)
    
    # ===== PERFORMANCE SETTINGS =====
    batch_size: int = 1
    memory_efficient: bool = True
    enable_attention_slicing: bool = True
    enable_vae_tiling: bool = False
    enable_vae_slicing: bool = False
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    offload: bool = False  # Model offloading to CPU (alias for enable_cpu_offload)
    precision: str = "fp16"  # fp16, fp32, bf16
    enable_xformers: bool = True
    enable_flash_attention: bool = False
    
    # ===== SAMPLING SETTINGS =====
    scheduler: str = "euler"
    eta: float = 0.0
    clip_skip: int = 1
    
    # ===== QUANTIZATION SETTINGS =====
    enable_quantization: bool = False
    quantization_type: str = "none"  # "none", "fp8", "fp4", "bnb4"
    
    # ===== LOGGING SETTINGS =====
    log_level: str = "INFO"
    enable_tensorboard: bool = False
    
    # ===== API SETTINGS =====
    api_host: str = "127.0.0.1"
    api_port: int = 7860
    enable_cors: bool = True
    
    # ===== SECURITY SETTINGS =====
    api_key_required: bool = False
    api_key: Optional[str] = None

    # ===== TESTING SETTINGS =====
    skip_model_loading: bool = False
    allow_mocks: bool = False
    

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Sync offload settings
        if self.offload:
            self.enable_cpu_offload = True
        
        # Create directories
        for path_attr in ["models_path", "output_path", "cache_path"]:
            path = Path(getattr(self, path_attr))
            path.mkdir(parents=True, exist_ok=True)
        
        # Environment variable overrides
        self.api_host = os.getenv("API_HOST", self.api_host)
        self.api_port = int(os.getenv("API_PORT", str(self.api_port)))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        
        # GPU Cloud optimizations
        if os.getenv("GPU_CLOUD_MODE"):
            self.api_host = "0.0.0.0"
            self.enable_cpu_offload = False  # Keep models in GPU memory
            self.memory_efficient = True
            
        logger.debug(f"Config initialized - device: {self.device}, animation_mode: {self.animation_mode}")
    
    # Sensitive fields that should not be serialized
    _SENSITIVE_FIELDS = frozenset({"api_key"})

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert config to dictionary.

        Args:
            include_sensitive: If True, include sensitive fields like api_key
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if include_sensitive or field.name not in self._SENSITIVE_FIELDS
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        # Filter out any keys that aren't valid KoshiConfig fields
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Create config from JSON file."""
        with open(config_path, "r") as f:
            config_data = json.load(f)
        return cls.from_dict(config_data)
    
    def update(self, **kwargs) -> "Config":
        """Create a new KoshiConfig with updated values."""
        current_values = {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
        current_values.update(kwargs)
        return Config(**current_values)
    
    def get_motion_parameters(self) -> Dict[str, Any]:
        """Get motion-related parameters for animation (simplified parameters)."""
        return {
            "motion_strength": self.motion_strength,
            "motion_coherence": self.motion_coherence,
            "motion_schedule": self.motion_schedule,
            "depth_strength": self.depth_strength,
            "perspective_flip_theta": self.perspective_flip_theta,
            "perspective_flip_phi": self.perspective_flip_phi,
            "perspective_flip_gamma": self.perspective_flip_gamma,
            "perspective_flip_fv": self.perspective_flip_fv,
        }
    
    def to_motion_schedule(self) -> Dict[str, Any]:
        """Convert keyframe strings to motion schedule dictionary (comprehensive parameters)."""
        motion_schedule = {}
        
        # Parse each motion parameter
        motion_params = {
            "zoom": self.zoom,
            "angle": self.angle,
            "translation_x": self.translation_x,
            "translation_y": self.translation_y,
            "translation_z": self.translation_z,
            "rotation_3d_x": self.rotation_3d_x,
            "rotation_3d_y": self.rotation_3d_y,
            "rotation_3d_z": self.rotation_3d_z
        }
        
        for param_name, keyframe_string in motion_params.items():
            parsed = self._parse_keyframes(keyframe_string)
            for frame, value in parsed.items():
                if frame not in motion_schedule:
                    motion_schedule[frame] = {}
                motion_schedule[frame][param_name] = value
        
        return motion_schedule
    
    def _parse_keyframes(self, keyframe_string: str) -> Dict[int, float]:
        """Parse keyframe string like '0:(1.0), 10:(1.5)' into frame->value dict."""
        result = {}
        if not keyframe_string:
            return result
        
        try:
            parts = keyframe_string.split(",")
            for part in parts:
                part = part.strip()
                if ":" in part and "(" in part:
                    frame_part, value_part = part.split(":", 1)
                    frame = int(frame_part.strip())
                    value_str = value_part.strip()
                    if value_str.startswith("(") and value_str.endswith(")"):
                        value = float(value_str[1:-1])
                        result[frame] = value
        except (ValueError, IndexError):
            # If parsing fails, return default
            result[0] = 1.0 if "zoom" in keyframe_string else 0.0
        
        return result


# Default configuration instance
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the current configuration."""
    return DEFAULT_CONFIG


def update_config(updates: Dict[str, Any]) -> None:
    """Update the global configuration."""
    global DEFAULT_CONFIG
    for key, value in updates.items():
        if hasattr(DEFAULT_CONFIG, key):
            setattr(DEFAULT_CONFIG, key, value)
        else:
            logger.warning(f"Unknown configuration key: {key}") 


# ===== CONFIGURATION PRESETS =====
PRESETS = {
    "fast": Config(
        model_name="flux-schnell",
        steps=18,
        guidance_scale=3.5,
        width=1024,
        height=1024,
        enable_cpu_offload=True,
        enable_attention_slicing=True,
    ),
    "balanced": Config(
        model_name="flux-dev",
        steps=20,
        guidance_scale=7.5,
        width=1024,
        height=1024,
        enable_cpu_offload=False,
        enable_attention_slicing=True,
    ),
    "quality": Config(
        model_name="flux-dev",
        steps=28,
        guidance_scale=7.5,
        width=1024,
        height=1024,
        enable_cpu_offload=False,
        enable_attention_slicing=False,
    ),
    "production": Config(
        model_name="flux-dev",
        steps=50,
        guidance_scale=7.5,
        width=2048,
        height=2048,
        enable_cpu_offload=False,
        enable_attention_slicing=False,
        memory_efficient=False,
    ),
    # TEST-SPECIFIC PRESETS
    "test_minimal": Config(
        model_name="flux-schnell",
        steps=2,
        guidance_scale=3.5,
        width=768,
        height=768,
        skip_model_loading=True,
        allow_mocks=True,  # CI/unit testing only
        max_frames=2,
        enable_cpu_offload=True,
    ),
    "test_GPU_Cloud": Config(
        model_name="flux-dev",
        steps=4,
        guidance_scale=7.5,
        width=1024,
        height=1024,
        skip_model_loading=False,
        api_host="0.0.0.0",
        api_port=7860,
        memory_efficient=True,
    )
}


def get_preset(preset_name: str) -> Config:
    """Get a configuration preset by name."""
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    # Return a copy to avoid modifying the original preset
    preset = PRESETS[preset_name]
    return Config(
        **{field.name: getattr(preset, field.name) for field in preset.__dataclass_fields__.values()}
    )


