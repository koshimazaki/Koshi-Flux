"""
File utilities for Koshi Flux

This module provides utilities for file operations, including saving animations,
managing output directories, and handling configuration files.

SECURITY ENHANCEMENTS:
- Input validation and sanitization for all user inputs
- Path traversal attack prevention
- Command injection protection
- Secure subprocess execution
"""

import os
import json
import shutil
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np

from koshi.core.exceptions import KoshiException, SecurityError
from koshi.core.logging_config import get_logger


class SecurityValidator:
    """Security validation utilities for input sanitization."""
    
    # Allowed characters for file patterns (alphanumeric, underscore, dash, dot, percent, digit specifiers)
    SAFE_PATTERN_REGEX = re.compile(r'^[a-zA-Z0-9_\-\.%d]+$')
    
    # Maximum allowed path depth to prevent excessive directory traversal
    MAX_PATH_DEPTH = 20
    
    @staticmethod
    def validate_file_pattern(pattern: str) -> str:
        """
        Validate and sanitize file pattern for ffmpeg.
        
        Args:
            pattern: File pattern string
            
        Returns:
            Sanitized pattern
            
        Raises:
            SecurityError: If pattern contains unsafe characters
        """
        if not isinstance(pattern, str):
            raise SecurityError(f"Pattern must be string, got {type(pattern)}")
        
        if not pattern:
            raise SecurityError("Pattern cannot be empty")
        
        if len(pattern) > 255:  # Reasonable max filename length
            raise SecurityError("Pattern too long (max 255 characters)")
        
        # Check for suspicious patterns
        dangerous_patterns = ['..', '/', '\\', '|', ';', '&', '$', '`', '(', ')', '{', '}', '[', ']', '<', '>']
        for dangerous in dangerous_patterns:
            if dangerous in pattern:
                raise SecurityError(f"Pattern contains unsafe sequence: {dangerous}")
        
        # Allow only safe characters
        if not SecurityValidator.SAFE_PATTERN_REGEX.match(pattern):
            raise SecurityError(f"Pattern contains unsafe characters: {pattern}")
        
        return pattern
    
    @staticmethod
    def validate_safe_path(path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Validate path against traversal attacks.
        
        Args:
            path: Path to validate
            base_path: Base path to restrict operations to (optional)
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path is unsafe
        """
        if not isinstance(path, (str, Path)):
            raise SecurityError(f"Path must be string or Path, got {type(path)}")
        
        path_obj = Path(path).resolve()
        
        # Check path depth
        parts = path_obj.parts
        if len(parts) > SecurityValidator.MAX_PATH_DEPTH:
            raise SecurityError(f"Path too deep (max {SecurityValidator.MAX_PATH_DEPTH} levels)")
        
        # Check for suspicious path components
        for part in parts:
            if part in ['..', '.', '']:
                continue  # These are handled by resolve()
            if part.startswith('.') and len(part) > 1:
                # Allow .gitignore, .env, etc. but be cautious
                pass
            # Check for suspicious characters in path components
            if any(char in part for char in ['|', ';', '&', '$', '`']):
                raise SecurityError(f"Path component contains unsafe characters: {part}")
        
        # If base_path provided, ensure path is within it
        if base_path is not None:
            base_path_obj = Path(base_path).resolve()
            try:
                path_obj.relative_to(base_path_obj)
            except ValueError:
                raise SecurityError(f"Path {path_obj} is outside allowed base path {base_path_obj}")
        
        return path_obj


class FileUtils:
    """Utility class for file operations with security enhancements."""
    
    def __init__(self):
        """Initialize file utilities."""
        self.logger = get_logger(__name__)
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Ensure directory exists, create if it doesn't.
        
        SECURITY: Validates path against traversal attacks.
        
        Args:
            directory: Directory path
            base_path: Base path to restrict operations to (optional)
            
        Returns:
            Path object of the directory
            
        Raises:
            SecurityError: If path is unsafe
            KoshiException: If directory creation fails
        """
        try:
            # Validate path security
            dir_path = SecurityValidator.validate_safe_path(directory, base_path)
            
            # Create directory securely
            dir_path.mkdir(parents=True, exist_ok=True, mode=0o755)  # Secure permissions
            
            logger = get_logger(__name__)
            logger.debug(f"Ensured directory exists: {dir_path}")
            
            return dir_path
            
        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            raise KoshiException(f"Failed to create directory {directory}: {e}")
    
    @staticmethod
    def save_animation_frames(
        frames: List[np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "frame",
        format: str = "png"
    ) -> List[Path]:
        """
        Save animation frames to files.
        
        SECURITY: Validates output directory and filename components.
        
        Args:
            frames: List of frame arrays
            output_dir: Output directory
            prefix: Filename prefix
            format: Image format
            
        Returns:
            List of saved file paths
            
        Raises:
            SecurityError: If inputs are unsafe
            KoshiException: If saving fails
        """
        # Validate inputs
        if not isinstance(frames, list) or not frames:
            raise KoshiException("Frames must be a non-empty list")
        
        # Validate prefix for safety
        if not re.match(r'^[a-zA-Z0-9_\-]+$', prefix):
            raise SecurityError(f"Unsafe filename prefix: {prefix}")
        
        # Validate format
        allowed_formats = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        if format.lower() not in allowed_formats:
            raise SecurityError(f"Unsupported format: {format}")
        
        output_path = FileUtils.ensure_directory(output_dir)
        saved_files = []
        
        try:
            from PIL import Image
            
            for i, frame in enumerate(frames):
                # Secure filename generation
                filename = f"{prefix}_{i:04d}.{format}"
                file_path = output_path / filename
                
                # Validate final path
                SecurityValidator.validate_safe_path(file_path, output_path)
                
                # Convert numpy array to PIL Image
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                
                image = Image.fromarray(frame)
                image.save(file_path)
                saved_files.append(file_path)
            
            return saved_files
            
        except ImportError:
            raise KoshiException("PIL (Pillow) is required for saving images. Install with: pip install Pillow")
        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            raise KoshiException(f"Failed to save animation frames: {e}")
    
    @staticmethod
    def create_video_from_frames(
        frame_dir: Union[str, Path],
        output_path: Union[str, Path],
        fps: int = 24,
        pattern: str = "frame_%04d.png"
    ) -> Path:
        """
        Create video from frame images using ffmpeg.
        
        SECURITY: Validates all inputs and uses secure subprocess execution.
        
        Args:
            frame_dir: Directory containing frames
            output_path: Output video path
            fps: Frames per second
            pattern: Frame filename pattern
            
        Returns:
            Path to created video
            
        Raises:
            SecurityError: If inputs are unsafe
            KoshiException: If video creation fails
        """
        # Validate FPS
        if not isinstance(fps, int) or fps <= 0 or fps > 120:
            raise SecurityError(f"Invalid FPS value: {fps} (must be 1-120)")
        
        # Validate and sanitize pattern (CRITICAL SECURITY FIX)
        pattern = SecurityValidator.validate_file_pattern(pattern)
        
        # Validate paths
        frame_dir = SecurityValidator.validate_safe_path(frame_dir)
        output_path = SecurityValidator.validate_safe_path(output_path)
        
        # Ensure frame directory exists
        if not frame_dir.exists() or not frame_dir.is_dir():
            raise KoshiException(f"Frame directory does not exist: {frame_dir}")
        
        # Ensure output directory exists
        FileUtils.ensure_directory(output_path.parent)
        
        try:
            import subprocess
            
            # Build ffmpeg command with secure parameters
            input_pattern = str(frame_dir / pattern)
            
            # Use explicit parameter list to prevent injection
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-framerate", str(fps),  # Convert to string securely
                "-i", input_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p", 
                "-crf", "18",  # High quality
                str(output_path)
            ]
            
            logger = get_logger(__name__)
            logger.info(f"Executing ffmpeg command: {' '.join(cmd)}")
            
            # Execute with security measures
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout
                cwd=str(frame_dir.parent),  # Set working directory
                env={"PATH": os.environ.get("PATH", "")},  # Minimal environment
            )
            
            if result.returncode != 0:
                logger.error(f"ffmpeg stderr: {result.stderr}")
                raise KoshiException(f"ffmpeg failed with return code {result.returncode}: {result.stderr}")
            
            logger.info(f"Successfully created video: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise KoshiException("ffmpeg command timed out (5 minutes)")
        except FileNotFoundError:
            raise KoshiException("ffmpeg not found. Please install ffmpeg to create videos.")
        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            raise KoshiException(f"Failed to create video: {e}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.
        
        SECURITY: Validates file path and config content.
        
        Args:
            config: Configuration dictionary
            file_path: Output file path
            
        Raises:
            SecurityError: If inputs are unsafe
            KoshiException: If saving fails
        """
        # Validate file path
        file_path = SecurityValidator.validate_safe_path(file_path)
        
        # Validate config content
        if not isinstance(config, dict):
            raise SecurityError("Config must be a dictionary")
        
        FileUtils.ensure_directory(file_path.parent)
        
        # Convert any non-serializable objects
        serializable_config = FileUtils._make_serializable(config)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_config, f, indent=2, ensure_ascii=False)
                
            logger = get_logger(__name__)
            logger.debug(f"Saved config to: {file_path}")
            
        except Exception as e:
            raise KoshiException(f"Failed to save config: {e}")
    
    @staticmethod
    def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        SECURITY: Validates file path and size limits.
        
        Args:
            file_path: Input file path
            
        Returns:
            Configuration dictionary
            
        Raises:
            SecurityError: If file is unsafe
            KoshiException: If loading fails
        """
        # Validate file path
        file_path = SecurityValidator.validate_safe_path(file_path)
        
        if not file_path.exists():
            raise KoshiException(f"Configuration file not found: {file_path}")
        
        # Check file size (prevent huge files)
        file_size = file_path.stat().st_size
        max_size = 10 * 1024 * 1024  # 10MB limit
        if file_size > max_size:
            raise SecurityError(f"Config file too large: {file_size} bytes (max {max_size})")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if not isinstance(config, dict):
                raise KoshiException("Config file must contain a JSON object")
                
            logger = get_logger(__name__)
            logger.debug(f"Loaded config from: {file_path}")
            
            return config
            
        except json.JSONDecodeError as e:
            raise KoshiException(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise KoshiException(f"Failed to load configuration: {e}")
    
    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.
        
        SECURITY: Prevents code injection through object serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, dict):
            return {k: FileUtils._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FileUtils._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif hasattr(obj, '__dict__'):
            # Only serialize safe attributes (no private/dunder attributes)
            safe_dict = {}
            for k, v in obj.__dict__.items():
                if not k.startswith('_'):  # Skip private attributes
                    safe_dict[k] = FileUtils._make_serializable(v)
            return safe_dict
        else:
            # Convert to string but sanitize
            str_repr = str(obj)
            if len(str_repr) > 1000:  # Prevent huge strings
                str_repr = str_repr[:1000] + "..."
            return str_repr

    # Additional secure file operations with remaining methods...
    @staticmethod
    def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
        """Create backup of a file with security validation."""
        file_path = SecurityValidator.validate_safe_path(file_path)
        
        if not file_path.exists():
            raise KoshiException(f"File to backup does not exist: {file_path}")
        
        if backup_dir is None:
            backup_dir = file_path.parent / "backups"
        else:
            backup_dir = SecurityValidator.validate_safe_path(backup_dir)
        
        FileUtils.ensure_directory(backup_dir)
        
        # Create unique backup filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        return backup_path

    @staticmethod
    def find_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
        """Find files with security validation."""
        directory = SecurityValidator.validate_safe_path(directory)
        
        if not directory.exists():
            return []
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))


# Export classes
__all__ = ["FileUtils", "SecurityValidator"]
