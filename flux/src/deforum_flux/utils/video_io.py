"""Video I/O utilities for Deforum animations.

Provides video loading, saving, and frame extraction aligned with
the Deforum pipeline requirements.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import subprocess

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Allowed FFmpeg parameters (security whitelist)
ALLOWED_CODECS = {"libx264", "libx265", "libvpx", "libvpx-vp9", "h264", "hevc"}
ALLOWED_PIXEL_FORMATS = {"yuv420p", "yuv444p", "rgb24", "yuv422p"}
CRF_RANGE = (0, 51)

# Optional cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not installed. Install with: pip install opencv-python")


class VideoReader:
    """Read video frames as PIL Images.

    Example:
        reader = VideoReader("input.mp4")
        for frame in reader:
            process(frame)

        # Or with context manager
        with VideoReader("input.mp4") as reader:
            frames = reader.read_all(max_frames=100)
    """

    def __init__(
        self,
        path: Union[str, Path],
        target_fps: Optional[float] = None,
        resize: Optional[Tuple[int, int]] = None,
        start_frame: int = 0,
    ):
        """Initialize video reader.

        Args:
            path: Path to video file
            target_fps: Resample to this FPS (None = original)
            resize: (width, height) to resize frames
            start_frame: Frame index to start reading from
        """
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python required for video reading")

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self.target_fps = target_fps
        self.resize = resize
        self.start_frame = start_frame

        self._cap = None
        self._init_capture()

    def _init_capture(self):
        """Initialize video capture."""
        self._cap = cv2.VideoCapture(str(self.path))
        try:
            if not self._cap.isOpened():
                raise ValueError(f"Cannot open video: {self.path}")

            self.original_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.total_frames / self.original_fps if self.original_fps > 0 else 0

            # Calculate frame skip for target FPS
            self.frame_skip = 1
            if self.target_fps:
                if self.target_fps <= 0:
                    raise ValueError(f"target_fps must be positive, got {self.target_fps}")
                if self.target_fps < self.original_fps:
                    self.frame_skip = int(self.original_fps / self.target_fps)

            self.effective_fps = self.original_fps / self.frame_skip if self.frame_skip > 0 else self.original_fps

            # Seek to start frame
            if self.start_frame > 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            logger.info(f"Opened video: {self.path.name}")
            logger.info(f"  Original: {self.width}x{self.height}, {self.original_fps:.1f}fps, {self.total_frames} frames")
            if self.target_fps:
                logger.info(f"  Resampled: {self.effective_fps:.1f}fps (skip every {self.frame_skip} frames)")
        except Exception:
            self._cap.release()
            self._cap = None
            raise

    def read_frame(self) -> Optional[Image.Image]:
        """Read next frame as PIL Image."""
        if self._cap is None:
            return None

        # Skip frames if resampling
        for _ in range(self.frame_skip):
            ret, frame = self._cap.read()
            if not ret:
                return None

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Resize if specified
        if self.resize:
            img = img.resize(self.resize, Image.LANCZOS)

        return img

    def read_all(self, max_frames: Optional[int] = None) -> List[Image.Image]:
        """Read all frames (or up to max_frames)."""
        frames = []
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break

        logger.info(f"Read {len(frames)} frames")
        return frames

    def __iter__(self):
        """Iterate over frames."""
        return self

    def __next__(self) -> Image.Image:
        """Get next frame."""
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

    def close(self):
        """Release video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        """Destructor."""
        self.close()


class VideoWriter:
    """Write frames to video file.

    Example:
        with VideoWriter("output.mp4", fps=24) as writer:
            for frame in frames:
                writer.write(frame)
    """

    def __init__(
        self,
        path: Union[str, Path],
        fps: float = 24,
        codec: str = "libx264",
        crf: int = 18,
        pixel_format: str = "yuv420p",
    ):
        """Initialize video writer.

        Args:
            path: Output video path
            fps: Frame rate
            codec: Video codec (libx264, libx265, etc.)
            crf: Constant rate factor (quality, 0-51, lower = better)
            pixel_format: Pixel format

        Raises:
            ValueError: If codec, crf, or pixel_format are invalid
        """
        # Validate parameters (security: prevent command injection)
        if codec not in ALLOWED_CODECS:
            raise ValueError(f"Invalid codec '{codec}'. Allowed: {ALLOWED_CODECS}")
        if pixel_format not in ALLOWED_PIXEL_FORMATS:
            raise ValueError(f"Invalid pixel_format '{pixel_format}'. Allowed: {ALLOWED_PIXEL_FORMATS}")
        if not CRF_RANGE[0] <= crf <= CRF_RANGE[1]:
            raise ValueError(f"CRF must be {CRF_RANGE[0]}-{CRF_RANGE[1]}, got {crf}")
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")

        self.path = Path(path)
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.pixel_format = pixel_format

        self._temp_dir = None
        self._frame_count = 0

    def _ensure_temp_dir(self):
        """Create temp directory for frames."""
        if self._temp_dir is None:
            self._temp_dir = self.path.parent / f".temp_{self.path.stem}"
            self._temp_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame: Image.Image):
        """Write a frame."""
        self._ensure_temp_dir()
        frame_path = self._temp_dir / f"frame_{self._frame_count:06d}.png"
        frame.save(frame_path)
        self._frame_count += 1

    def write_all(self, frames: List[Image.Image]):
        """Write all frames."""
        for frame in frames:
            self.write(frame)

    def _cleanup_temp(self):
        """Clean up temporary directory and files."""
        if self._temp_dir is not None and self._temp_dir.exists():
            for f in self._temp_dir.glob("*.png"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                self._temp_dir.rmdir()
            except OSError:
                pass
            self._temp_dir = None

    def finalize(self):
        """Encode frames to video using ffmpeg."""
        if self._frame_count == 0:
            logger.warning("No frames written")
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", str(self._temp_dir / "frame_%06d.png"),
            "-c:v", self.codec,
            "-pix_fmt", self.pixel_format,
            "-crf", str(self.crf),
            str(self.path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            logger.info(f"Saved video: {self.path} ({self._frame_count} frames at {self.fps}fps)")
        finally:
            # Always cleanup temp files, even on error
            self._cleanup_temp()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit - finalize video."""
        self.finalize()


def load_video(
    path: Union[str, Path],
    max_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Image.Image], float]:
    """Convenience function to load video frames.

    Args:
        path: Path to video file
        max_frames: Maximum frames to load
        target_fps: Resample to this FPS
        resize: (width, height) to resize

    Returns:
        (list of PIL Images, effective fps)
    """
    with VideoReader(path, target_fps=target_fps, resize=resize) as reader:
        frames = reader.read_all(max_frames=max_frames)
        return frames, reader.effective_fps


def save_video(
    frames: List[Image.Image],
    path: Union[str, Path],
    fps: float = 24,
    **kwargs,
):
    """Convenience function to save frames as video.

    Args:
        frames: List of PIL Images
        path: Output video path
        fps: Frame rate
        **kwargs: Additional VideoWriter options
    """
    with VideoWriter(path, fps=fps, **kwargs) as writer:
        writer.write_all(frames)


def extract_frame(
    video_path: Union[str, Path],
    frame_index: int,
    resize: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """Extract a single frame from video.

    Args:
        video_path: Path to video
        frame_index: Frame index to extract
        resize: Optional resize

    Returns:
        PIL Image
    """
    with VideoReader(video_path, start_frame=frame_index, resize=resize) as reader:
        frame = reader.read_frame()
        if frame is None:
            raise ValueError(f"Could not read frame {frame_index}")
        return frame


__all__ = [
    "VideoReader",
    "VideoWriter",
    "load_video",
    "save_video",
    "extract_frame",
]
