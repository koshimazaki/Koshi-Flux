"""LoRA loading and management for FLUX.2 Klein.

Supports loading LoRAs for both native BFL SDK and diffusers backends.
Klein-compatible LoRAs can be loaded from HuggingFace or local paths.

Usage:
    # With diffusers backend
    lora_manager = KleinLoRAManager(pipe, backend="diffusers")
    lora_manager.load("path/to/lora.safetensors", strength=0.8)

    # With native SDK (experimental)
    lora_manager = KleinLoRAManager(model, backend="native")
    lora_manager.load("path/to/lora", strength=1.0)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# Validation constants
STRENGTH_RANGE = (0.0, 2.0)


@dataclass
class LoRAInfo:
    """Information about a loaded LoRA."""
    name: str
    path: str
    strength: float
    backend: str
    adapter_name: str = ""  # Track actual adapter name for diffusers
    fused: bool = False


class KleinLoRAManager:
    """Manage LoRA loading for Klein models.

    Supports:
    - Diffusers-format LoRAs (recommended for Klein)
    - Native BFL SDK LoRAs (experimental)
    - Multiple LoRAs with different strengths
    - Fusing/unfusing for performance

    Example:
        # Load single LoRA
        manager = KleinLoRAManager(pipe)
        manager.load("stabilityai/sd-vae-ft-mse", strength=0.7)

        # Load multiple LoRAs
        manager.load("style_lora.safetensors", strength=0.5)
        manager.load("detail_lora.safetensors", strength=0.3)

        # Fuse for faster inference
        manager.fuse_all()

        # Later, unfuse to change
        manager.unfuse_all()
    """

    def __init__(
        self,
        pipe_or_model,
        backend: str = "diffusers",
    ):
        """Initialize LoRA manager.

        Args:
            pipe_or_model: Diffusers pipeline or native model
            backend: "diffusers" or "native"
        """
        self.pipe = pipe_or_model
        self.backend = backend
        self.loaded_loras: Dict[str, LoRAInfo] = {}

        logger.info(f"KleinLoRAManager initialized (backend={backend})")

    def load(
        self,
        lora_path: Union[str, Path],
        strength: float = 1.0,
        name: Optional[str] = None,
        adapter_name: Optional[str] = None,
    ) -> LoRAInfo:
        """Load a LoRA.

        Args:
            lora_path: Path to LoRA (local file or HuggingFace repo)
            strength: LoRA strength/scale (0.0-2.0, 1.0 = full effect)
            name: Optional name for this LoRA
            adapter_name: Adapter name for diffusers (auto-generated if None)

        Returns:
            LoRAInfo with loaded LoRA details

        Raises:
            ValueError: If strength is out of range
        """
        # Validate strength
        if not STRENGTH_RANGE[0] <= strength <= STRENGTH_RANGE[1]:
            raise ValueError(
                f"Strength must be {STRENGTH_RANGE[0]}-{STRENGTH_RANGE[1]}, got {strength}"
            )

        lora_path = str(lora_path)
        name = name or Path(lora_path).stem
        adapter_name = adapter_name or f"lora_{len(self.loaded_loras)}"

        if self.backend == "diffusers":
            info = self._load_diffusers(lora_path, strength, name, adapter_name)
        else:
            info = self._load_native(lora_path, strength, name)

        self.loaded_loras[name] = info
        logger.info(f"Loaded LoRA: {name} (strength={strength})")
        return info

    def _load_diffusers(
        self,
        lora_path: str,
        strength: float,
        name: str,
        adapter_name: str,
    ) -> LoRAInfo:
        """Load LoRA using diffusers PEFT integration."""
        try:
            # Check if it's a HuggingFace repo or local file
            if "/" in lora_path and not Path(lora_path).exists():
                # HuggingFace repo
                self.pipe.load_lora_weights(
                    lora_path,
                    adapter_name=adapter_name,
                )
            else:
                # Local file
                lora_path = Path(lora_path)
                if lora_path.is_file():
                    self.pipe.load_lora_weights(
                        str(lora_path.parent),
                        weight_name=lora_path.name,
                        adapter_name=adapter_name,
                    )
                else:
                    self.pipe.load_lora_weights(
                        str(lora_path),
                        adapter_name=adapter_name,
                    )

            # Set scale
            self.pipe.set_adapters([adapter_name], adapter_weights=[strength])

            return LoRAInfo(
                name=name,
                path=str(lora_path),
                strength=strength,
                backend="diffusers",
                adapter_name=adapter_name,
                fused=False,
            )

        except Exception as e:
            logger.error(f"Failed to load LoRA {lora_path}: {e}")
            raise

    def _load_native(
        self,
        lora_path: str,
        strength: float,
        name: str,
    ) -> LoRAInfo:
        """Load LoRA for native BFL SDK.

        Raises:
            NotImplementedError: Native LoRA merging is not yet implemented.
                Use backend="diffusers" instead.
        """
        raise NotImplementedError(
            "Native BFL SDK LoRA loading is not implemented. "
            "The LoRA merge operation (W' = W + alpha * B @ A) requires "
            "model-specific weight mapping that varies by architecture. "
            "Please use backend='diffusers' for LoRA support. "
            f"Attempted to load: {lora_path}"
        )

    def set_strength(self, name: str, strength: float):
        """Adjust strength of a loaded LoRA.

        Args:
            name: LoRA name
            strength: New strength value

        Raises:
            ValueError: If LoRA not loaded or strength out of range
        """
        if name not in self.loaded_loras:
            raise ValueError(f"LoRA not loaded: {name}")

        if not STRENGTH_RANGE[0] <= strength <= STRENGTH_RANGE[1]:
            raise ValueError(
                f"Strength must be {STRENGTH_RANGE[0]}-{STRENGTH_RANGE[1]}, got {strength}"
            )

        info = self.loaded_loras[name]

        if self.backend == "diffusers" and not info.fused:
            # Use actual adapter names stored in LoRAInfo
            adapter_names = []
            adapter_weights = []
            for lora_name, lora_info in self.loaded_loras.items():
                if lora_info.adapter_name:
                    adapter_names.append(lora_info.adapter_name)
                    adapter_weights.append(
                        strength if lora_name == name else lora_info.strength
                    )

            if adapter_names:
                self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

        info.strength = strength
        logger.info(f"Set LoRA {name} strength to {strength}")

    def fuse(self, name: str):
        """Fuse a LoRA into base model for faster inference.

        Args:
            name: LoRA name to fuse
        """
        if name not in self.loaded_loras:
            raise ValueError(f"LoRA not loaded: {name}")

        info = self.loaded_loras[name]
        if info.fused:
            logger.warning(f"LoRA {name} already fused")
            return

        if self.backend == "diffusers":
            self.pipe.fuse_lora(lora_scale=info.strength)
            info.fused = True
            logger.info(f"Fused LoRA: {name}")

    def unfuse(self, name: str):
        """Unfuse a LoRA from base model.

        Args:
            name: LoRA name to unfuse
        """
        if name not in self.loaded_loras:
            raise ValueError(f"LoRA not loaded: {name}")

        info = self.loaded_loras[name]
        if not info.fused:
            logger.warning(f"LoRA {name} not fused")
            return

        if self.backend == "diffusers":
            self.pipe.unfuse_lora()
            info.fused = False
            logger.info(f"Unfused LoRA: {name}")

    def fuse_all(self):
        """Fuse all loaded LoRAs."""
        for name in self.loaded_loras:
            if not self.loaded_loras[name].fused:
                self.fuse(name)

    def unfuse_all(self):
        """Unfuse all LoRAs."""
        for name in self.loaded_loras:
            if self.loaded_loras[name].fused:
                self.unfuse(name)

    def unload(self, name: str):
        """Unload a LoRA.

        Args:
            name: LoRA name to unload
        """
        if name not in self.loaded_loras:
            raise ValueError(f"LoRA not loaded: {name}")

        info = self.loaded_loras[name]

        if info.fused:
            self.unfuse(name)

        if self.backend == "diffusers":
            try:
                self.pipe.unload_lora_weights()
            except Exception as e:
                logger.warning(f"Could not unload LoRA weights: {e}")

        del self.loaded_loras[name]
        logger.info(f"Unloaded LoRA: {name}")

    def unload_all(self):
        """Unload all LoRAs."""
        names = list(self.loaded_loras.keys())
        for name in names:
            self.unload(name)

    def list_loaded(self) -> List[LoRAInfo]:
        """List all loaded LoRAs."""
        return list(self.loaded_loras.values())

    def get_info(self, name: str) -> Optional[LoRAInfo]:
        """Get info for a loaded LoRA."""
        return self.loaded_loras.get(name)


# Convenience functions

def load_klein_lora(
    pipe,
    lora_path: Union[str, Path],
    strength: float = 1.0,
) -> KleinLoRAManager:
    """Quick helper to load a LoRA onto a Klein pipeline.

    Args:
        pipe: Diffusers Klein pipeline
        lora_path: Path to LoRA
        strength: LoRA strength

    Returns:
        KleinLoRAManager instance
    """
    manager = KleinLoRAManager(pipe, backend="diffusers")
    manager.load(lora_path, strength=strength)
    return manager


# Known good Klein LoRAs (community recommendations - Jan 2026)
# Note: Verify paths on HuggingFace before use, community LoRAs may move
RECOMMENDED_LORAS = {
    "animatediff_style": {
        "path": "Nebsh/LTX2_Animatediff_style",
        "strength": 0.7,
        "description": "Animatediff aesthetic for video generation",
    },
    "deforum_morph": {
        "path": "s4f3tymarc/Ltxv-Deforum-Morphing-Style_v1-2025",
        "strength": 0.6,
        "description": "Classic Deforum morphing style",
    },
    # Community anatomy fix LoRA - path TBD when published
    # "anatomy_fix": {"path": "...", "strength": 0.8, "description": "Fixes hand/limb issues"},
}


__all__ = [
    "LoRAInfo",
    "KleinLoRAManager",
    "load_klein_lora",
    "RECOMMENDED_LORAS",
]
