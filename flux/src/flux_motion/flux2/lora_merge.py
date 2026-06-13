"""Pure LoRA merge math and key-mapping for the native FLUX.2 DiT.

This module is intentionally free of any BFL SDK / pipeline / GPU dependency so
the merge logic can be unit-tested on CPU with plain dicts of tensors. The only
third-party requirement is ``torch``.

The native flux2 DiT (``flux2.model.Flux2``) is a plain ``nn.Module`` whose
Linear weights are named like::

    img_in.weight
    double_blocks.0.img_attn.qkv.weight
    double_blocks.0.img_mlp.0.weight
    single_blocks.0.linear1.weight
    final_layer.linear.weight

LoRA checkpoints reference those modules under one of two conventions:

* PEFT / diffusers: ``<module>.lora_A.weight`` (shape ``(rank, in)``) paired with
  ``<module>.lora_B.weight`` (shape ``(out, rank)``).
* kohya: ``<module>.lora_down.weight`` / ``<module>.lora_up.weight`` plus an
  optional scalar ``<module>.alpha``. kohya also flattens dots to underscores and
  prefixes names with ``lora_unet_`` / ``lora_te_``.

The merge applied to each matched Linear weight ``W`` is::

    W' = W + scale * (B @ A)      with scale = strength * (alpha / rank)

``alpha`` defaults to ``rank`` when absent, so ``scale == strength`` for plain
PEFT LoRAs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

# Suffixes that identify the two halves of a LoRA pair, mapped to a role.
_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight")
_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight")
_ALPHA_SUFFIX = ".alpha"

# Prefixes commonly prepended by trainers/exporters; stripped before matching.
_KNOWN_PREFIXES = (
    "lora_unet_",
    "lora_transformer_",
    "lora_te_",
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "lora_",
)

# Prefixes added by wrappers such as torch.compile or DistributedDataParallel.
_MODEL_WRAPPER_PREFIXES = ("_orig_mod.", "module.")

# Warn when fewer than this fraction of LoRA modules map onto the model.
LOW_MATCH_WARN_FRACTION = 0.5


def _squash(name: str) -> str:
    """Normalize a module name to a separator-insensitive key.

    kohya replaces ``.`` with ``_`` while the original module names already
    contain underscores, so reconstructing the dotted path is ambiguous. Instead
    we collapse every non-alphanumeric character and lowercase, which is robust to
    ``.``/``_`` differences and stable across both conventions.

    Args:
        name: Raw module name (already stripped of its LoRA/weight suffix).

    Returns:
        Lowercase alphanumeric-only key.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _strip_lora_prefix(name: str) -> str:
    """Remove known trainer prefixes from a LoRA module name."""
    changed = True
    while changed:
        changed = False
        for prefix in _KNOWN_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
                break
    return name


def _strip_model_wrapper_prefix(name: str) -> str:
    """Remove wrapper prefixes while keeping the real state-dict key intact."""
    changed = True
    while changed:
        changed = False
        for prefix in _MODEL_WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
                break
    return name


def _base_and_role(key: str) -> Tuple[str, str] | Tuple[None, None]:
    """Split a LoRA tensor key into ``(base_module, role)``.

    Returns ``(None, None)`` for keys that are not LoRA up/down/alpha tensors.
    """
    for suffix in _DOWN_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "down"
    for suffix in _UP_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "up"
    if key.endswith(_ALPHA_SUFFIX):
        return key[: -len(_ALPHA_SUFFIX)], "alpha"
    return None, None


@dataclass
class LoRAPair:
    """A paired LoRA module (down/up matrices plus optional alpha)."""

    base: str
    down: torch.Tensor | None = None
    up: torch.Tensor | None = None
    alpha: float | None = None


def pair_lora_modules(lora_state: Dict[str, torch.Tensor]) -> Dict[str, LoRAPair]:
    """Group a flat LoRA state dict into per-module up/down/alpha pairs.

    Args:
        lora_state: Mapping of tensor name -> tensor from the ``.safetensors`` file.

    Returns:
        Mapping of base module name -> :class:`LoRAPair`. Only modules that have
        at least one of up/down are returned.
    """
    pairs: Dict[str, LoRAPair] = {}
    for key, tensor in lora_state.items():
        base, role = _base_and_role(key)
        if base is None:
            continue
        pair = pairs.setdefault(base, LoRAPair(base=base))
        if role == "down":
            pair.down = tensor
        elif role == "up":
            pair.up = tensor
        elif role == "alpha":
            # alpha may be a 0-dim tensor or python scalar.
            pair.alpha = float(tensor.item() if hasattr(tensor, "item") else tensor)
    return pairs


def build_model_weight_index(
    model_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, str], set]:
    """Index 2D ``*.weight`` model params by their squashed module name.

    Args:
        model_state: ``module.state_dict()`` (only keys/shapes are inspected).

    Returns:
        ``(index, ambiguous)`` where ``index`` maps squashed-name -> real key for
        unambiguous Linear weights, and ``ambiguous`` is the set of squashed names
        that collide across multiple real keys (excluded from ``index``).
    """
    index: Dict[str, str] = {}
    ambiguous: set = set()
    for key, tensor in model_state.items():
        if not key.endswith(".weight"):
            continue
        if not hasattr(tensor, "ndim") or tensor.ndim != 2:
            continue  # Linear only; conv/other handled by caller as unsupported.
        module_name = _strip_model_wrapper_prefix(key[: -len(".weight")])
        squashed = _squash(module_name)
        if squashed in index and index[squashed] != key:
            ambiguous.add(squashed)
        else:
            index[squashed] = key
    for squashed in ambiguous:
        index.pop(squashed, None)
    return index, ambiguous


@dataclass
class MergeReport:
    """Summary of a LoRA -> model merge attempt."""

    total: int = 0
    matched: List[str] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)
    shape_mismatch: List[str] = field(default_factory=list)
    incomplete: List[str] = field(default_factory=list)
    ambiguous: List[str] = field(default_factory=list)
    scale_by_key: Dict[str, float] = field(default_factory=dict)

    @property
    def match_fraction(self) -> float:
        """Fraction of LoRA modules that mapped onto a model weight."""
        return len(self.matched) / self.total if self.total else 0.0

    @property
    def low_match(self) -> bool:
        """True when the match rate is below the warning threshold."""
        return self.total > 0 and self.match_fraction < LOW_MATCH_WARN_FRACTION

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"matched {len(self.matched)}/{self.total} LoRA modules "
            f"({self.match_fraction:.0%}); unmatched={len(self.unmatched)} "
            f"shape_mismatch={len(self.shape_mismatch)} "
            f"incomplete={len(self.incomplete)} ambiguous={len(self.ambiguous)}"
        )


def merge_lora_into_state_dict(
    model_state: Dict[str, torch.Tensor],
    lora_state: Dict[str, torch.Tensor],
    strength: float = 1.0,
    alpha_override: float | None = None,
) -> Tuple[Dict[str, torch.Tensor], MergeReport]:
    """Compute per-weight LoRA deltas for a native model state dict.

    Pure function: no SDK/pipeline import required. Deltas are returned keyed by
    the model's real ``*.weight`` name so the caller can apply ``W += delta`` and
    later restore exactly.

    Args:
        model_state: ``module.state_dict()`` of the target model.
        lora_state: Flat LoRA tensor dict (PEFT or kohya format).
        strength: User strength multiplier (typically 0.0-2.0).
        alpha_override: Alpha for pairs that carry no in-weight ``.alpha`` (i.e.
            PEFT/diffusers LoRAs, whose ``lora_alpha`` lives in
            ``adapter_config.json``). In-weight kohya alpha always wins; when both
            are absent alpha defaults to ``rank`` (scale == strength).

    Returns:
        ``(deltas, report)``. ``deltas`` maps model weight name -> float32 delta
        tensor (``scale * (B @ A)``). ``report`` is a :class:`MergeReport`.
    """
    index, ambiguous_squashed = build_model_weight_index(model_state)
    pairs = pair_lora_modules(lora_state)

    deltas: Dict[str, torch.Tensor] = {}
    report = MergeReport(total=len(pairs))

    for base, pair in pairs.items():
        if pair.down is None or pair.up is None:
            report.incomplete.append(base)
            continue

        squashed = _squash(_strip_lora_prefix(base))
        if squashed in ambiguous_squashed:
            report.ambiguous.append(base)
            continue
        real_key = index.get(squashed)
        if real_key is None:
            report.unmatched.append(base)
            continue

        # Compute the delta in float32 for numerical stability.
        down = pair.down.to(torch.float32)
        up = pair.up.to(torch.float32)
        if down.ndim != 2 or up.ndim != 2:
            # Conv or other layout - not supported here; surface it explicitly.
            report.shape_mismatch.append(base)
            continue

        rank = down.shape[0]
        if pair.alpha is not None:
            alpha = pair.alpha               # in-weight alpha (kohya) wins
        elif alpha_override is not None:
            alpha = float(alpha_override)     # PEFT alpha from config / caller
        else:
            alpha = float(rank)              # default -> scale == strength
        scale = strength * (alpha / rank if rank else 0.0)
        delta = scale * (up @ down)

        target_shape = model_state[real_key].shape
        if tuple(delta.shape) != tuple(target_shape):
            report.shape_mismatch.append(base)
            continue

        # Accumulate in case multiple LoRA modules target the same weight.
        if real_key in deltas:
            deltas[real_key] = deltas[real_key] + delta
        else:
            deltas[real_key] = delta
        report.matched.append(base)
        report.scale_by_key[real_key] = scale

    return deltas, report


def apply_deltas_to_module(
    module: "torch.nn.Module",
    deltas: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Apply ``W += delta`` in place to a module's parameters.

    Args:
        module: Target ``nn.Module`` whose ``named_parameters`` are mutated.
        deltas: Mapping of parameter name -> delta tensor (any dtype/device).

    Returns:
        ``(backups, applied)``: ``backups`` holds a clone of each original weight
        (for exact restore) and ``applied`` holds the delta cast to the parameter
        dtype/device that was actually added.
    """
    backups: Dict[str, torch.Tensor] = {}
    applied: Dict[str, torch.Tensor] = {}
    params = dict(module.named_parameters())
    for name, delta in deltas.items():
        param = params.get(name)
        if param is None:
            continue
        backups[name] = param.detach().clone()
        cast = delta.to(device=param.device, dtype=param.dtype)
        with torch.no_grad():
            param.add_(cast)
        applied[name] = cast
    return backups, applied


def restore_module(
    module: "torch.nn.Module",
    backups: Dict[str, torch.Tensor],
) -> None:
    """Restore original weights from ``backups`` (exact, in place)."""
    params = dict(module.named_parameters())
    with torch.no_grad():
        for name, original in backups.items():
            param = params.get(name)
            if param is not None:
                param.copy_(original)


__all__ = [
    "LoRAPair",
    "MergeReport",
    "pair_lora_modules",
    "build_model_weight_index",
    "merge_lora_into_state_dict",
    "apply_deltas_to_module",
    "restore_module",
    "LOW_MATCH_WARN_FRACTION",
]
