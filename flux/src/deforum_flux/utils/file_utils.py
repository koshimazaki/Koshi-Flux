"""
File Utilities for FLUX Deforum Pipeline

Secure file operations with path traversal protection,
video encoding, and resource management.
"""

import os
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Union
from contextlib import contextmanager

from deforum_flux.core import SecurityError, get_logger


logger = get_logger(__name__)


# ============================================================================
# Security Functions
# ============================================================================

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename to prevent path traversal and injection.
    
    Args:
        filename: Raw filename
        max_length: Maximum allowed length
        
    Returns:
        Sanitized filename
        
    Raises:
        SecurityError: If filename is inherently unsafe
    """
    # Remove null bytes
    filename = filename.replace("\x00", "")
    
    # Get basename to prevent directory traversal
    filename = os.path.basename(filename)
    
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Truncate to max length
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        name = name[:max_length - len(ext) - 1]
        filename = name + ext
    
    # Check for empty result
    if not filename:
        raise SecurityError("Filename sanitization resulted in empty string")
    
    return filename


def validate_path(
    path: Union[str, Path],
    allowed_base: Union[str, Path],
    must_exist: bool = False
) -> Path:
    """
    Validate path is within allowed directory.
    
    Args:
        path: Path to validate
        allowed_base: Allowed base directory
        must_exist: Whether path must exist
        
    Returns:
        Validated absolute path
        
    Raises:
        SecurityError: If path escapes allowed base
        FileNotFoundError: If must_exist and path doesn't exist
    """
    path = Path(path).resolve()
    allowed_base = Path(allowed_base).resolve()
    
    # Check path is under allowed base
    try:
        path.relative_to(allowed_base)
    except ValueError:
        raise SecurityError(
            f"Path traversal detected: {path} is not under {allowed_base}"
        )
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    return path


def safe_makedirs(path: Union[str, Path], mode: int = 0o755) -> Path:
    """
    Safely create directory and parents.
    
    Args:
        path: Directory path to create
        mode: Permission mode
        
    Returns:
        Created path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    return path


# ============================================================================
# Video Encoding
# ============================================================================

def encode_video_ffmpeg(
    frames_dir: Union[str, Path],
    output_path: Union[str, Path],
    fps: int = 24,
    codec: str = "libx264",
    crf: int = 18,
    pix_fmt: str = "yuv420p",
    frame_pattern: str = "frame_%05d.png"
) -> Path:
    """
    Encode frames to video using FFmpeg.
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
        crf: Constant rate factor (quality, lower=better)
        pix_fmt: Pixel format
        frame_pattern: Frame filename pattern
        
    Returns:
        Output video path
        
    Raises:
        SecurityError: If command injection detected
        RuntimeError: If FFmpeg fails
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    
    # Validate inputs (prevent command injection)
    if not re.match(r'^[a-zA-Z0-9_\-\.%]+$', frame_pattern):
        raise SecurityError(f"Invalid frame pattern: {frame_pattern}")
    
    if not re.match(r'^[a-z0-9]+$', codec):
        raise SecurityError(f"Invalid codec: {codec}")
    
    if not isinstance(fps, int) or fps < 1 or fps > 120:
        raise SecurityError(f"Invalid fps: {fps}")
    
    if not isinstance(crf, int) or crf < 0 or crf > 51:
        raise SecurityError(f"Invalid crf: {crf}")
    
    # Build command as list (safe from injection)
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-framerate", str(fps),
        "-i", str(frames_dir / frame_pattern),
        "-c:v", codec,
        "-crf", str(crf),
        "-pix_fmt", pix_fmt,
        "-movflags", "+faststart",
        str(output_path)
    ]
    
    logger.info(f"Encoding video: {len(list(frames_dir.glob('*.png')))} frames → {output_path}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"FFmpeg stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr}")
        raise RuntimeError(f"FFmpeg encoding failed: {e.stderr}")
    
    return output_path


def probe_video(video_path: Union[str, Path]) -> dict:
    """
    Get video metadata using FFprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    import json
    
    video_path = Path(video_path)
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFprobe failed: {e.stderr}")
        return {}


# ============================================================================
# Temporary File Management
# ============================================================================

@contextmanager
def temp_directory(prefix: str = "deforum_", cleanup: bool = True):
    """
    Context manager for temporary directory.
    
    Args:
        prefix: Directory name prefix
        cleanup: Whether to cleanup on exit
        
    Yields:
        Path to temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    
    try:
        yield temp_dir
    finally:
        if cleanup:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")


@contextmanager
def temp_file(suffix: str = "", prefix: str = "deforum_", cleanup: bool = True):
    """
    Context manager for temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        cleanup: Whether to cleanup on exit
        
    Yields:
        Path to temporary file
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    path = Path(path)
    
    try:
        yield path
    finally:
        if cleanup and path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


# ============================================================================
# File Operations
# ============================================================================

def save_frames(
    frames: List,
    output_dir: Union[str, Path],
    prefix: str = "frame",
    format: str = "png"
) -> List[Path]:
    """
    Save list of frames to directory.
    
    Args:
        frames: List of PIL Images or numpy arrays
        output_dir: Output directory
        prefix: Filename prefix
        format: Image format
        
    Returns:
        List of saved file paths
    """
    from PIL import Image
    import numpy as np
    
    output_dir = safe_makedirs(output_dir)
    paths = []
    
    for i, frame in enumerate(frames):
        # Convert numpy to PIL if needed
        if isinstance(frame, np.ndarray):
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame = Image.fromarray(frame)
        
        filename = f"{prefix}_{i:05d}.{format}"
        filepath = output_dir / filename
        frame.save(filepath)
        paths.append(filepath)
    
    logger.info(f"Saved {len(paths)} frames to {output_dir}")
    return paths


def load_frames(
    input_dir: Union[str, Path],
    pattern: str = "*.png",
    max_frames: Optional[int] = None
) -> List:
    """
    Load frames from directory.
    
    Args:
        input_dir: Input directory
        pattern: Glob pattern for files
        max_frames: Maximum frames to load
        
    Returns:
        List of PIL Images
    """
    from PIL import Image
    
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob(pattern))
    
    if max_frames:
        paths = paths[:max_frames]
    
    frames = []
    for path in paths:
        frames.append(Image.open(path))
    
    logger.info(f"Loaded {len(frames)} frames from {input_dir}")
    return frames


__all__ = [
    "sanitize_filename",
    "validate_path",
    "safe_makedirs",
    "encode_video_ffmpeg",
    "probe_video",
    "temp_directory",
    "temp_file",
    "save_frames",
    "load_frames",
]
