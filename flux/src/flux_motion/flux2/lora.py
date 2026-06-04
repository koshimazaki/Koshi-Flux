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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

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
    matched: int = 0  # Native: number of LoRA modules merged into the model
    total: int = 0    # Native: total LoRA modules found in the file


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
        # Native merge bookkeeping: name -> {"backups", "applied", "strength"}.
        # backups = clones of original weights (for exact restore), applied =
        # the delta tensors actually added (cast to param dtype/device).
        self._native_state: Dict[str, dict] = {}

        logger.info(f"KleinLoRAManager initialized (backend={backend})")

    def load(
        self,
        lora_path: Union[str, Path],
        strength: float = 1.0,
        name: Optional[str] = None,
        adapter_name: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> LoRAInfo:
        """Load a LoRA.

        Args:
            lora_path: Path to LoRA (local file or HuggingFace repo)
            strength: LoRA strength/scale (0.0-2.0, 1.0 = full effect)
            name: Optional name for this LoRA
            adapter_name: Adapter name for diffusers (auto-generated if None)
            alpha: Native PEFT alpha override (per-adapter ``lora_alpha``). If
                None, native load auto-reads it from an adjacent
                ``adapter_config.json``. Ignored by the diffusers backend.

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
            info = self._load_native(lora_path, strength, name, alpha=alpha)

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

    def _resolve_native_module(self):
        """Return the underlying ``nn.Module`` to merge LoRA weights into.

        Accepts either a raw model or a ``Flux2Pipeline``. For a pipeline we
        ensure weights are loaded (``load_models``) and return ``_model``.

        Raises:
            RuntimeError: If no ``nn.Module`` with parameters can be located.
        """
        target = self.pipe
        # A Flux2Pipeline lazily loads its DiT; force it before merging.
        if hasattr(target, "load_models") and not getattr(target, "_loaded", False):
            target.load_models()
        # Prefer the explicit native DiT attribute, then the public property.
        module = getattr(target, "_model", None)
        if module is None:
            module = getattr(target, "model", None)
        if module is None and hasattr(target, "named_parameters"):
            module = target  # target itself is the model
        if module is None or not hasattr(module, "named_parameters"):
            raise RuntimeError(
                "Could not locate a native nn.Module to merge the LoRA into. "
                "Expected the pipeline to expose `_model`/`model` or to be an "
                "nn.Module itself."
            )
        return module

    def _read_lora_file(self, lora_path: str) -> dict:
        """Load a local ``.safetensors`` LoRA into a flat tensor dict.

        Args:
            lora_path: Path to a local ``.safetensors`` file.

        Raises:
            RuntimeError: If safetensors is unavailable.
            FileNotFoundError: If the path is not a local ``.safetensors`` file.
                Native merge needs raw tensors, so HF-repo / directory inputs are
                not auto-resolved here (use the diffusers backend for those).
        """
        if not SAFETENSORS_AVAILABLE:
            raise RuntimeError(
                "safetensors is required for native LoRA merge. "
                "Install with: pip install safetensors"
            )
        path = Path(lora_path)
        if not (path.is_file() and path.suffix == ".safetensors"):
            raise FileNotFoundError(
                f"Native LoRA merge expects a local .safetensors file, got: "
                f"{lora_path}. For HuggingFace repos or directories use "
                f"backend='diffusers'."
            )
        return load_file(str(path))

    def _read_peft_alpha(self, lora_path: str) -> Optional[float]:
        """Read ``lora_alpha`` from a PEFT ``adapter_config.json`` next to the file.

        PEFT/diffusers LoRAs store ``lora_alpha`` in their adapter config rather
        than the weights; without it a non-default alpha would silently mis-scale
        the merge. Returns the alpha as a float, or None if no config/key exists.
        """
        try:
            import json
            config = Path(lora_path).parent / "adapter_config.json"
            if config.is_file():
                value = json.loads(config.read_text()).get("lora_alpha")
                if value is not None:
                    return float(value)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not read adapter_config.json alpha: %s", exc)
        return None

    def _load_native(
        self,
        lora_path: str,
        strength: float,
        name: str,
        alpha: Optional[float] = None,
    ) -> LoRAInfo:
        """Merge a LoRA directly into the native flux2 DiT weights.

        Reads a ``.safetensors`` LoRA (PEFT or kohya format), maps each module to
        the model's Linear weights, and applies ``W += scale * (B @ A)`` in place.
        Original weights are backed up so :meth:`unfuse` restores exactly.

        Args:
            lora_path: Local ``.safetensors`` path.
            strength: Strength multiplier (already range-validated by ``load``).
            name: Name to register this LoRA under.

        Returns:
            LoRAInfo describing the merged LoRA (``fused=True``).

        Raises:
            RuntimeError: If the LoRA matched zero model modules (nothing applied).
        """
        from .lora_merge import (
            apply_deltas_to_module,
            merge_lora_into_state_dict,
        )

        module = self._resolve_native_module()
        lora_state = self._read_lora_file(lora_path)

        # PEFT/diffusers LoRAs keep lora_alpha in adapter_config.json (not the
        # weights); use it (or an explicit override) so a non-default alpha is not
        # silently mis-scaled. kohya alpha (in-weight) still takes precedence.
        alpha_override = alpha if alpha is not None else self._read_peft_alpha(lora_path)
        deltas, report = merge_lora_into_state_dict(
            module.state_dict(), lora_state, strength, alpha_override=alpha_override
        )
        if alpha_override is not None:
            logger.info("Native LoRA alpha override: %s", alpha_override)

        if report.low_match:
            logger.warning(
                "Native LoRA merge: %s. Most modules did NOT map onto the model "
                "- check the LoRA key format / target architecture.",
                report.summary(),
            )
        else:
            logger.info("Native LoRA merge: %s", report.summary())
        if report.unmatched:
            logger.warning(
                "Native LoRA: %d unmatched module(s), e.g. %s",
                len(report.unmatched),
                report.unmatched[:5],
            )

        if not report.matched:
            raise RuntimeError(
                f"Native LoRA merge matched 0 of {report.total} modules for "
                f"'{lora_path}'. No weights changed. The key names likely do not "
                f"correspond to this model's modules ({report.summary()})."
            )

        backups, applied = apply_deltas_to_module(module, deltas)
        self._native_state[name] = {
            "backups": backups,
            "applied": applied,
            "strength": strength,
        }

        return LoRAInfo(
            name=name,
            path=str(lora_path),
            strength=strength,
            backend="native",
            fused=True,  # merged straight into the weights
            matched=len(report.matched),
            total=report.total,
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
        elif self.backend == "native":
            # Deltas are linear in strength: rescale the applied delta in place.
            self._rescale_native(name, strength)

        info.strength = strength
        logger.info(f"Set LoRA {name} strength to {strength}")

    def _rescale_native(self, name: str, strength: float):
        """Rescale a native LoRA's stored delta to a new strength.

        The merge is linear in strength, so the new delta is the stored delta
        scaled by ``new / old``. Model weights are only touched when the LoRA is
        currently fused; when it is unfused the weights stay at their originals and
        we just update the stored delta + strength, so a later ``fuse()`` applies
        the right amount (prevents an unfuse -> set_strength -> fuse double-apply).
        """
        state = self._native_state.get(name)
        if not state:
            logger.warning(f"No native merge state for {name}; cannot set strength")
            return
        old = state.get("strength", 0.0)
        if not old:
            logger.warning(
                "Native LoRA %s was applied at strength 0; cannot rescale to %s "
                "(reload at the desired strength).", name, strength
            )
        factor = (strength / old) if old else 0.0
        new_applied: Dict[str, "torch.Tensor"] = {
            pname: tensor * factor for pname, tensor in state["applied"].items()
        }
        if self.loaded_loras[name].fused:
            module = self._resolve_native_module()
            params = dict(module.named_parameters())
            with torch.no_grad():
                for pname, original in state["backups"].items():
                    param = params.get(pname)
                    if param is not None and pname in new_applied:
                        param.copy_(original + new_applied[pname])
        state["applied"] = new_applied
        state["strength"] = strength

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
        elif self.backend == "native":
            # Native LoRAs are merged into the weights at load time. If a prior
            # unfuse restored the weights, re-apply the stored delta.
            self._refuse_native(name)
            info.fused = True
            logger.info(f"Re-fused native LoRA: {name}")

    def _refuse_native(self, name: str):
        """Re-apply a previously-unfused native LoRA from its stored delta."""
        state = self._native_state.get(name)
        if not state:
            logger.warning(f"No native merge state for {name}; cannot fuse")
            return
        module = self._resolve_native_module()
        params = dict(module.named_parameters())
        with torch.no_grad():
            for pname, delta in state["applied"].items():
                param = params.get(pname)
                if param is not None:
                    param.add_(delta)

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
        elif self.backend == "native":
            from .lora_merge import restore_module

            state = self._native_state.get(name)
            if state:
                restore_module(self._resolve_native_module(), state["backups"])
            info.fused = False
            logger.info(f"Unfused native LoRA: {name} (weights restored)")

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
        elif self.backend == "native":
            # Weights already restored by unfuse above; drop the backup tensors.
            self._native_state.pop(name, None)

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


def load_klein_lora_native(
    pipe_or_model,
    lora_path: Union[str, Path],
    strength: float = 1.0,
) -> KleinLoRAManager:
    """Merge a LoRA into the native flux2 DiT weights (no diffusers needed).

    Args:
        pipe_or_model: ``Flux2Pipeline`` or a raw flux2 ``nn.Module``.
        lora_path: Local ``.safetensors`` LoRA path.
        strength: LoRA strength (0.0-2.0).

    Returns:
        KleinLoRAManager with the LoRA merged into the model weights.
    """
    manager = KleinLoRAManager(pipe_or_model, backend="native")
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
    "load_klein_lora_native",
    "merge_lora_into_state_dict",
    "RECOMMENDED_LORAS",
]

# Re-export the pure merge entrypoint for convenience / testing.
from .lora_merge import merge_lora_into_state_dict  # noqa: E402,F401
