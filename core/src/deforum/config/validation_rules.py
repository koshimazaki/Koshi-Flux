"""
Centralized Validation Rules for Deforum Flux

This module provides a single source of truth for all validation constants,
eliminating duplication across the codebase and ensuring consistency.
"""

from typing import Tuple


class ValidationRules:
    """Centralized validation constants and rules."""
    
    # Image dimensions
    DIMENSIONS = {
        "min": 64,
        "max": 4096,
        "divisible_by": 8
    }
    
    # Generation parameters
    STEPS = {
        "min": 1,
        "max": 200
    }
    
    GUIDANCE_SCALE = {
        "min": 0.0,
        "max": 30.0
    }
    
    BATCH_SIZE = {
        "min": 1,
        "max": 32,
        "performance_max": 16
    }
    
    # Animation parameters
    MAX_FRAMES = {
        "min": 1,
        "max": 10000
    }
    
    FPS = {
        "min": 1,
        "max": 120
    }
    
    # Motion parameters with their valid ranges
    MOTION_RANGES = {
        "zoom": (0.1, 10.0),
        "angle": (-360.0, 360.0),
        "translation_x": (-2000.0, 2000.0),
        "translation_y": (-2000.0, 2000.0),
        "translation_z": (-2000.0, 2000.0),
        "rotation_3d_x": (-360.0, 360.0),
        "rotation_3d_y": (-360.0, 360.0),
        "rotation_3d_z": (-360.0, 360.0)
    }
    
    # Strength and scheduling parameters
    STRENGTH_RANGES = {
        "midas_weight": (0.0, 1.0),
        "strength_schedule": (0.0, 1.0),
        "noise_schedule": (0.0, 1.0),
        "contrast_schedule": (0.0, 10.0),
        "motion_strength": (0.0, 1.0),
        "motion_coherence": (0.0, 1.0),
        "depth_strength": (0.0, 1.0)
    }
    
    # 3D rendering parameters
    RENDERING_3D = {
        "near_plane": {"min": 1, "max": 1000},
        "far_plane": {"min": 100, "max": 50000},
        "fov": {"min": 1, "max": 180}
    }
    
    # Prompt limits
    PROMPT = {
        "max_length": 2048,
        "min_length": 1
    }
    
    # Device types
    VALID_DEVICES = ["cpu", "cuda", "mps"]
    
    # Model names
    VALID_MODELS = ["flux-schnell", "flux-dev"]
    
    # Animation modes
    VALID_ANIMATION_MODES = ["2D", "3D", "Video Input", "Interpolation"]
    
    # Motion modes
    VALID_MOTION_MODES = ["grouped", "independent", "mixed"]
    
    # Log levels
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    # Seed limits
    SEED = {
        "min": 0,
        "max": 2**32 - 1
    }
    
    # File extensions
    ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ALLOWED_CONFIG_EXTENSIONS = [".json", ".yaml", ".yml"]
    
    @classmethod
    def get_dimension_range(cls) -> Tuple[int, int]:
        """Get dimension min/max as tuple."""
        return cls.DIMENSIONS["min"], cls.DIMENSIONS["max"]
    
    @classmethod
    def get_steps_range(cls) -> Tuple[int, int]:
        """Get steps min/max as tuple."""
        return cls.STEPS["min"], cls.STEPS["max"]
    
    @classmethod
    def get_guidance_range(cls) -> Tuple[float, float]:
        """Get guidance scale min/max as tuple."""
        return cls.GUIDANCE_SCALE["min"], cls.GUIDANCE_SCALE["max"]
    
    @classmethod
    def get_motion_range(cls, motion_param: str) -> Tuple[float, float]:
        """Get motion parameter range."""
        if motion_param not in cls.MOTION_RANGES:
            raise ValueError(f"Unknown motion parameter: {motion_param}")
        return cls.MOTION_RANGES[motion_param]
    
    @classmethod
    def get_strength_range(cls, strength_param: str) -> Tuple[float, float]:
        """Get strength parameter range."""
        if strength_param not in cls.STRENGTH_RANGES:
            raise ValueError(f"Unknown strength parameter: {strength_param}")
        return cls.STRENGTH_RANGES[strength_param]
    
    @classmethod
    def is_valid_device(cls, device: str) -> bool:
        """Check if device string is valid."""
        device_type = device.split(":")[0]  # Handle cuda:0, cuda:1, etc.
        return device_type in cls.VALID_DEVICES
    
    @classmethod
    def is_valid_model(cls, model_name: str) -> bool:
        """Check if model name is valid."""
        return model_name in cls.VALID_MODELS
    
    @classmethod
    def is_valid_animation_mode(cls, mode: str) -> bool:
        """Check if animation mode is valid."""
        return mode in cls.VALID_ANIMATION_MODES
    
    @classmethod
    def is_valid_motion_mode(cls, mode: str) -> bool:
        """Check if motion mode is valid."""
        return mode in cls.VALID_MOTION_MODES
    
    @classmethod
    def is_valid_log_level(cls, level: str) -> bool:
        """Check if log level is valid."""
        return level in cls.VALID_LOG_LEVELS
