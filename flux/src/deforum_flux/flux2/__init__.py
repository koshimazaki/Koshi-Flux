"""FLUX.2 Deforum Pipeline - 128-channel latent space."""

from .motion_engine import Flux2MotionEngine, Flux2DevMotionEngine
from .config import FLUX2_CONFIG

# Pipeline will be added when flux2 API is finalized
# from .pipeline import Flux2DeforumPipeline

__all__ = [
    "Flux2MotionEngine",
    "Flux2DevMotionEngine",
    "FLUX2_CONFIG",
]
