"""
Input validation utilities for Koshi Flux

This module provides comprehensive input validation using the hybrid validation
approach with centralized rules and reusable utilities from the config module.
"""

import re
from typing import Any, List, Dict, Optional

from koshi.core.exceptions import ValidationError
from koshi.core.logging_config import get_logger
from koshi.config.validation_rules import ValidationRules
from koshi.config.validation_utils import ValidationUtils, DomainValidators


class InputValidator:
    """Utility class for validating various types of inputs using hybrid validation approach."""
    
    def __init__(self, max_prompt_length: Optional[int] = None):
        """
        Initialize input validator.
        
        Args:
            max_prompt_length: Maximum allowed prompt length (uses ValidationRules default if None)
        """
        self.max_prompt_length = max_prompt_length or ValidationRules.PROMPT["max_length"]
        self.logger = get_logger(__name__)
    
    def validate_prompt(self, prompt: str) -> None:
        """
        Validate text prompt using centralized rules.
        
        Args:
            prompt: Text prompt to validate
            
        Raises:
            ValidationError: If prompt is invalid
        """
        errors = ValidationUtils.validate_string_not_empty(prompt, "prompt", self.max_prompt_length)
        
        # Check for potentially problematic characters
        if re.search(r'[<>{}]', prompt):
            self.logger.warning("Prompt contains potentially problematic characters: < > { }")
        
        if errors:
            raise ValidationError("Prompt validation failed", validation_errors=errors)
    
    def validate_dimensions(self, width: int, height: int) -> None:
        """
        Validate image dimensions using domain validators.
        
        Args:
            width: Image width
            height: Image height
            
        Raises:
            ValidationError: If dimensions are invalid
        """
        errors = DomainValidators.validate_dimensions(width, height)
        
        if errors:
            raise ValidationError("Dimension validation failed", validation_errors=errors)
    
    def validate_generation_params(
        self,
        steps: int,
        guidance_scale: float,
        seed: Optional[int] = None
    ) -> None:
        """
        Validate generation parameters using domain validators.
        
        Args:
            steps: Number of generation steps
            guidance_scale: Guidance scale value
            seed: Random seed (optional)
            
        Raises:
            ValidationError: If parameters are invalid
        """
        errors = DomainValidators.validate_generation_params(steps, guidance_scale, seed)
        
        if errors:
            raise ValidationError("Generation parameter validation failed", validation_errors=errors)
    
    def validate_motion_params(self, motion_params: Dict[str, float]) -> None:
        """
        Validate motion parameters using domain validators.
        
        Args:
            motion_params: Dictionary of motion parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        errors = DomainValidators.validate_motion_params(motion_params)
        
        if errors:
            raise ValidationError("Motion parameter validation failed", validation_errors=errors)
    
    def validate_file_path(
        self, 
        file_path: str, 
        must_exist: bool = True,
        allowed_extensions: Optional[List[str]] = None
    ) -> None:
        """
        Validate file path using centralized utilities.
        
        Args:
            file_path: Path to validate
            must_exist: Whether file must exist
            allowed_extensions: List of allowed file extensions
            
        Raises:
            ValidationError: If path is invalid
        """
        errors = ValidationUtils.validate_file_path(file_path, must_exist, allowed_extensions)
        
        if errors:
            raise ValidationError("File path validation failed", validation_errors=errors)
    
    def validate_animation_config(self, config: Dict[str, Any]) -> None:
        """
        Validate complete animation configuration using centralized approach.
        
        Args:
            config: Animation configuration dictionary
            
        Raises:
            ValidationError: If configuration is invalid
        """
        errors = []
        
        # Required fields validation
        required_fields = ["prompt", "max_frames"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate prompt
        if "prompt" in config:
            prompt_errors = ValidationUtils.validate_string_not_empty(
                config["prompt"], "prompt", self.max_prompt_length
            )
            errors.extend(prompt_errors)
        
        # Validate dimensions if present
        if "width" in config and "height" in config:
            dimension_errors = DomainValidators.validate_dimensions(
                config["width"], config["height"]
            )
            errors.extend(dimension_errors)
        
        # Validate generation parameters if present
        if all(key in config for key in ["steps", "guidance_scale"]):
            gen_errors = DomainValidators.validate_generation_params(
                config["steps"], 
                config["guidance_scale"], 
                config.get("seed")
            )
            errors.extend(gen_errors)
        
        # Validate animation settings
        if "max_frames" in config and "fps" in config:
            anim_errors = DomainValidators.validate_animation_settings(
                config["max_frames"], config["fps"]
            )
            errors.extend(anim_errors)
        elif "max_frames" in config:
            # Validate just max_frames
            min_frames, max_frames_limit = ValidationRules.MAX_FRAMES["min"], ValidationRules.MAX_FRAMES["max"]
            frame_errors = ValidationUtils.validate_range(
                config["max_frames"], min_frames, max_frames_limit, "max_frames", int
            )
            errors.extend(frame_errors)
        
        # Validate motion schedule
        if "motion_schedule" in config:
            motion_schedule = config["motion_schedule"]
            if not isinstance(motion_schedule, dict):
                errors.append("motion_schedule must be a dictionary")
            else:
                for frame, motion_params in motion_schedule.items():
                    frame_errors = ValidationUtils.validate_frame_number(frame, "frame")
                    errors.extend(frame_errors)
                    
                    if isinstance(motion_params, dict):
                        motion_errors = DomainValidators.validate_motion_params(motion_params)
                        errors.extend(motion_errors)
                    else:
                        errors.append(f"Motion parameters for frame {frame} must be a dictionary")
        
        if errors:
            raise ValidationError("Animation configuration validation failed", validation_errors=errors)
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe file system usage using centralized utilities.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        return ValidationUtils.sanitize_filename(filename)
    
    def validate_device_string(self, device: str) -> None:
        """
        Validate device string using centralized rules.
        
        Args:
            device: Device string to validate
            
        Raises:
            ValidationError: If device string is invalid
        """
        if not ValidationRules.is_valid_device(device):
            raise ValidationError(f"Invalid device: {device}. Valid devices: {ValidationRules.VALID_DEVICES}")
    
    def validate_batch_size(self, batch_size: int, max_batch_size: Optional[int] = None) -> None:
        """
        Validate batch size using centralized rules.
        
        Args:
            batch_size: Batch size to validate
            max_batch_size: Maximum allowed batch size (uses ValidationRules default if None)
            
        Raises:
            ValidationError: If batch size is invalid
        """
        max_batch = max_batch_size or ValidationRules.BATCH_SIZE["max"]
        min_batch = ValidationRules.BATCH_SIZE["min"]
        
        errors = ValidationUtils.validate_range(batch_size, min_batch, max_batch, "batch_size", int)
        
        if errors:
            raise ValidationError("Batch size validation failed", validation_errors=errors)
    
    def validate_model_name(self, model_name: str) -> None:
        """
        Validate model name using centralized rules.
        
        Args:
            model_name: Model name to validate
            
        Raises:
            ValidationError: If model name is invalid
        """
        if not ValidationRules.is_valid_model(model_name):
            raise ValidationError(f"Invalid model: {model_name}. Valid models: {ValidationRules.VALID_MODELS}")
    
    def validate_animation_mode(self, animation_mode: str) -> None:
        """
        Validate animation mode using centralized rules.
        
        Args:
            animation_mode: Animation mode to validate
            
        Raises:
            ValidationError: If animation mode is invalid
        """
        if not ValidationRules.is_valid_animation_mode(animation_mode):
            raise ValidationError(f"Invalid animation mode: {animation_mode}. Valid modes: {ValidationRules.VALID_ANIMATION_MODES}")
    
    def validate_log_level(self, log_level: str) -> None:
        """
        Validate log level using centralized rules.
        
        Args:
            log_level: Log level to validate
            
        Raises:
            ValidationError: If log level is invalid
        """
        if not ValidationRules.is_valid_log_level(log_level):
            raise ValidationError(f"Invalid log level: {log_level}. Valid levels: {ValidationRules.VALID_LOG_LEVELS}")
    
    def validate_image_file(self, file_path: str, must_exist: bool = True) -> None:
        """
        Validate image file path using centralized rules.
        
        Args:
            file_path: Path to image file
            must_exist: Whether file must exist
            
        Raises:
            ValidationError: If file is invalid
        """
        self.validate_file_path(file_path, must_exist, ValidationRules.ALLOWED_IMAGE_EXTENSIONS)


