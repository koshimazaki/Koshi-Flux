"""
Reusable Validation Utilities for Koshi Flux

This module provides common validation helper functions that can be used
across different modules, reducing code duplication and ensuring consistency.
"""

import re
import os
from typing import Any, List, Dict, Optional, Union
from pathlib import Path

from .validation_rules import ValidationRules


class ValidationUtils:
    """Reusable validation utility functions."""
    
    @staticmethod
    def is_in_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> bool:
        """
        Check if value is within specified range (inclusive).
        
        Args:
            value: Value to check
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            True if value is in range, False otherwise
        """
        return min_val <= value <= max_val
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, param_name: str) -> List[str]:
        """
        Validate value type and return errors if invalid.
        
        Args:
            value: Value to validate
            expected_type: Expected type
            param_name: Parameter name for error messages
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        if not isinstance(value, expected_type):
            errors.append(f"{param_name} must be {expected_type.__name__}, got {type(value).__name__}")
        return errors
    
    @staticmethod
    def validate_range(
        value: Union[int, float], 
        min_val: Union[int, float], 
        max_val: Union[int, float], 
        param_name: str,
        value_type: type = None
    ) -> List[str]:
        """
        Validate value is within range and optionally validate type.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            param_name: Parameter name for error messages
            value_type: Optional type to validate
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Type validation if specified
        if value_type and not isinstance(value, value_type):
            errors.append(f"{param_name} must be {value_type.__name__}, got {type(value).__name__}")
            return errors  # Return early if type is wrong
        
        # Range validation
        if not ValidationUtils.is_in_range(value, min_val, max_val):
            errors.append(f"{param_name} must be between {min_val} and {max_val}, got {value}")
        
        return errors
    
    @staticmethod
    def validate_positive_integer(value: Any, param_name: str, max_val: Optional[int] = None) -> List[str]:
        """
        Validate value is a positive integer.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            max_val: Optional maximum value
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(value, int):
            errors.append(f"{param_name} must be an integer, got {type(value).__name__}")
            return errors
        
        if value <= 0:
            errors.append(f"{param_name} must be positive, got {value}")
        
        if max_val is not None and value > max_val:
            errors.append(f"{param_name} must be <= {max_val}, got {value}")
        
        return errors
    
    @staticmethod
    def validate_divisible_by(value: int, divisor: int, param_name: str) -> List[str]:
        """
        Validate value is divisible by divisor.
        
        Args:
            value: Value to validate
            divisor: Required divisor
            param_name: Parameter name for error messages
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        if value % divisor != 0:
            errors.append(f"{param_name} must be divisible by {divisor}, got {value}")
        return errors
    
    @staticmethod
    def validate_string_not_empty(value: Any, param_name: str, max_length: Optional[int] = None) -> List[str]:
        """
        Validate string is not empty and optionally check length.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            max_length: Optional maximum length
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(value, str):
            errors.append(f"{param_name} must be a string, got {type(value).__name__}")
            return errors
        
        if not value.strip():
            errors.append(f"{param_name} cannot be empty")
        
        if max_length is not None and len(value) > max_length:
            errors.append(f"{param_name} too long: {len(value)} > {max_length}")
        
        return errors
    
    @staticmethod
    def validate_choice(value: Any, choices: List[Any], param_name: str) -> List[str]:
        """
        Validate value is one of allowed choices.
        
        Args:
            value: Value to validate
            choices: List of allowed values
            param_name: Parameter name for error messages
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        if value not in choices:
            errors.append(f"Invalid {param_name}: {value}. Must be one of {choices}")
        return errors
    
    @staticmethod
    def validate_keyframe_syntax(keyframe_string: str) -> bool:
        """
        Validate keyframe syntax (e.g., "0:(1.0), 30:(1.5)").
        
        Args:
            keyframe_string: Keyframe string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(keyframe_string, str):
            return False
        
        try:
            # Basic validation - should contain frame:value pairs
            if ":" not in keyframe_string or "(" not in keyframe_string:
                return False
            
            # Split by comma and validate each part
            parts = keyframe_string.split(",")
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                if ":" not in part or "(" not in part or ")" not in part:
                    return False
                
                frame_part, value_part = part.split(":", 1)
                frame_num = int(frame_part.strip())
                value = value_part.strip()
                
                if not value.startswith("(") or not value.endswith(")"):
                    return False
                
                # Try to parse the value
                float(value[1:-1])
                
                # Frame number should be non-negative
                if frame_num < 0:
                    return False
            
            return True
            
        except (ValueError, IndexError):
            return False
    
    @staticmethod
    def validate_file_path(
        file_path: str,
        must_exist: bool = True,
        allowed_extensions: Optional[List[str]] = None,
        param_name: str = "file_path"
    ) -> List[str]:
        """
        Validate file path with security checks.
        
        Args:
            file_path: Path to validate
            must_exist: Whether file must exist
            allowed_extensions: List of allowed file extensions
            param_name: Parameter name for error messages
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(file_path, str):
            errors.append(f"{param_name} must be a string, got {type(file_path).__name__}")
            return errors
        
        try:
            path = Path(file_path).resolve()
        except (OSError, ValueError) as e:
            errors.append(f"Invalid {param_name}: {e}")
            return errors
        
        if must_exist and not path.exists():
            errors.append(f"File does not exist: {file_path}")
        
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                errors.append(f"File extension not allowed. Got {path.suffix}, allowed: {allowed_extensions}")
        
        return errors
    
    @staticmethod
    def validate_frame_number(frame: Any, param_name: str = "frame") -> List[str]:
        """
        Validate frame number is a non-negative integer.
        
        Args:
            frame: Frame value to validate
            param_name: Parameter name for error messages
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        try:
            frame_num = int(frame)
            if frame_num < 0:
                errors.append(f"{param_name} number must be non-negative, got {frame_num}")
        except (ValueError, TypeError):
            errors.append(f"Invalid {param_name} number: {frame}")
        
        return errors
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for safe file system usage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        sanitized = ''.join(c for c in sanitized if ord(c) >= 32)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        # Ensure it's not empty
        if not sanitized.strip():
            sanitized = "untitled"
        
        return sanitized
    
    @staticmethod
    def collect_errors(*error_lists: List[str]) -> List[str]:
        """
        Collect and flatten multiple error lists.
        
        Args:
            *error_lists: Variable number of error lists
            
        Returns:
            Flattened list of all errors
        """
        all_errors = []
        for error_list in error_lists:
            all_errors.extend(error_list)
        return all_errors


# Convenience functions using ValidationRules
class DomainValidators:
    """Domain-specific validators using ValidationRules and ValidationUtils."""
    
    @staticmethod
    def validate_dimensions(width: int, height: int) -> List[str]:
        """Validate image dimensions using centralized rules."""
        min_dim, max_dim = ValidationRules.get_dimension_range()
        divisor = ValidationRules.DIMENSIONS["divisible_by"]
        
        errors = []
        errors.extend(ValidationUtils.validate_range(width, min_dim, max_dim, "width", int))
        errors.extend(ValidationUtils.validate_range(height, min_dim, max_dim, "height", int))
        errors.extend(ValidationUtils.validate_divisible_by(width, divisor, "width"))
        errors.extend(ValidationUtils.validate_divisible_by(height, divisor, "height"))
        
        return errors
    
    @staticmethod
    def validate_generation_params(steps: int, guidance_scale: float, seed: Optional[int] = None) -> List[str]:
        """Validate generation parameters using centralized rules."""
        min_steps, max_steps = ValidationRules.get_steps_range()
        min_guidance, max_guidance = ValidationRules.get_guidance_range()
        
        errors = []
        errors.extend(ValidationUtils.validate_range(steps, min_steps, max_steps, "steps", int))
        errors.extend(ValidationUtils.validate_range(guidance_scale, min_guidance, max_guidance, "guidance_scale", (int, float)))
        
        if seed is not None:
            min_seed, max_seed = ValidationRules.SEED["min"], ValidationRules.SEED["max"]
            errors.extend(ValidationUtils.validate_range(seed, min_seed, max_seed, "seed", int))
        
        return errors
    
    @staticmethod
    def validate_motion_params(motion_params: Dict[str, float]) -> List[str]:
        """Validate motion parameters using centralized rules."""
        errors = []
        
        for param_name, param_value in motion_params.items():
            if param_name not in ValidationRules.MOTION_RANGES:
                errors.append(f"Unknown motion parameter: {param_name}")
                continue
            
            min_val, max_val = ValidationRules.get_motion_range(param_name)
            errors.extend(ValidationUtils.validate_range(param_value, min_val, max_val, param_name, (int, float)))
        
        return errors
    
    @staticmethod
    def validate_animation_settings(max_frames: int, fps: int) -> List[str]:
        """Validate animation settings using centralized rules."""
        errors = []
        
        min_frames, max_frames_limit = ValidationRules.MAX_FRAMES["min"], ValidationRules.MAX_FRAMES["max"]
        min_fps, max_fps = ValidationRules.FPS["min"], ValidationRules.FPS["max"]
        
        errors.extend(ValidationUtils.validate_range(max_frames, min_frames, max_frames_limit, "max_frames", int))
        errors.extend(ValidationUtils.validate_range(fps, min_fps, max_fps, "fps", int))
        
        return errors
