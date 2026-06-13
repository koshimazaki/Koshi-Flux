"""CPU-only tests for native FLUX.2 Klein LoRA merging.

These tests avoid importing the full ``flux_motion`` package (its ``__init__``
chain pulls heavy/SDK deps that are unavailable in CI). Instead the two LoRA
modules under test are loaded in isolation as a small synthetic package, so the
relative imports inside ``lora.py`` still resolve.

Covered:
- key mapping for PEFT/diffusers and kohya conventions (incl. underscore flatten)
- merged delta equals ``strength * (alpha / rank) * (B @ A)``
- unmatched / incomplete / ambiguous modules are reported (not silently dropped)
- accumulation when two LoRA modules target the same weight
- strength linearity (used by native ``set_strength``)
- ``apply_deltas_to_module`` + ``restore_module`` restore weights exactly
- full manager flow: load -> merge -> set_strength -> unfuse (file IO included)
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

# --- Load lora_merge + lora as a tiny synthetic package (no flux_motion init) ---
_FLUX2_DIR = Path(__file__).resolve().parents[1] / "src" / "flux_motion" / "flux2"
_PKG = "klein_lora_under_test"


def _load_module(modname: str, filename: str):
    """Load ``filename`` as ``{_PKG}.{modname}`` so relative imports resolve."""
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_FLUX2_DIR)]
        sys.modules[_PKG] = pkg
    full = f"{_PKG}.{modname}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _FLUX2_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


lora_merge = _load_module("lora_merge", "lora_merge.py")
lora = _load_module("lora", "lora.py")

merge_lora_into_state_dict = lora_merge.merge_lora_into_state_dict
apply_deltas_to_module = lora_merge.apply_deltas_to_module
restore_module = lora_merge.restore_module
KleinLoRAManager = lora.KleinLoRAManager


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _model_state():
    """Realistic flux2-style Linear weights (out, in) plus a 1D weight to ignore."""
    return {
        "img_in.weight": torch.zeros(8, 4),
        "double_blocks.0.img_attn.qkv.weight": torch.zeros(24, 8),
        "double_blocks.0.img_mlp.0.weight": torch.zeros(16, 8),
        "single_blocks.0.linear1.weight": torch.zeros(40, 8),
        "final_layer.linear.weight": torch.zeros(4, 8),
        "double_blocks.0.img_attn.norm.weight": torch.zeros(8),  # 1D -> ignored
    }


def _peft_pair(out_dim, in_dim, rank, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(rank, in_dim, generator=g)
    b = torch.randn(out_dim, rank, generator=g)
    return a, b


# --------------------------------------------------------------------------- #
# Pure merge math
# --------------------------------------------------------------------------- #
def test_peft_keys_map_and_delta_matches():
    state = _model_state()
    rank = 4
    a, b = _peft_pair(24, 8, rank, seed=1)
    lora_state = {
        # diffusers-style dotted name with a transformer. prefix
        "transformer.double_blocks.0.img_attn.qkv.lora_A.weight": a,
        "transformer.double_blocks.0.img_attn.qkv.lora_B.weight": b,
    }
    strength = 0.5
    deltas, report = merge_lora_into_state_dict(state, lora_state, strength)

    key = "double_blocks.0.img_attn.qkv.weight"
    assert report.matched == ["transformer.double_blocks.0.img_attn.qkv"]
    assert report.unmatched == []
    # alpha absent -> scale == strength
    expected = strength * (b @ a)
    assert torch.allclose(deltas[key], expected, atol=1e-5)
    assert deltas[key].shape == state[key].shape


def test_peft_keys_map_through_nested_and_compiled_prefixes():
    state = {"_orig_mod.double_blocks.0.img_attn.qkv.weight": torch.zeros(24, 8)}
    rank = 4
    a, b = _peft_pair(24, 8, rank, seed=16)
    lora_state = {
        "base_model.model.transformer.double_blocks.0.img_attn.qkv.lora_A.weight": a,
        "base_model.model.transformer.double_blocks.0.img_attn.qkv.lora_B.weight": b,
    }

    deltas, report = merge_lora_into_state_dict(state, lora_state, strength=1.0)

    key = "_orig_mod.double_blocks.0.img_attn.qkv.weight"
    assert report.matched == [
        "base_model.model.transformer.double_blocks.0.img_attn.qkv"
    ]
    assert report.unmatched == []
    assert key in deltas
    assert torch.allclose(deltas[key], b @ a, atol=1e-5)


def test_kohya_keys_map_with_alpha():
    state = _model_state()
    rank = 4
    g = torch.Generator().manual_seed(2)
    down = torch.randn(rank, 8, generator=g)
    up = torch.randn(40, rank, generator=g)
    alpha = 2.0
    lora_state = {
        # kohya: lora_unet_ prefix + dots flattened to underscores + alpha scalar
        "lora_unet_single_blocks_0_linear1.lora_down.weight": down,
        "lora_unet_single_blocks_0_linear1.lora_up.weight": up,
        "lora_unet_single_blocks_0_linear1.alpha": torch.tensor(alpha),
    }
    deltas, report = merge_lora_into_state_dict(state, lora_state, strength=1.0)

    key = "single_blocks.0.linear1.weight"
    assert len(report.matched) == 1
    expected = (alpha / rank) * (up @ down)
    assert torch.allclose(deltas[key], expected, atol=1e-5)


def test_unmatched_and_incomplete_reported():
    state = _model_state()
    rank = 2
    a, b = _peft_pair(4, 8, rank, seed=3)
    lora_state = {
        # matches final_layer.linear
        "final_layer.linear.lora_A.weight": a,
        "final_layer.linear.lora_B.weight": b,
        # no model counterpart -> unmatched
        "transformer.ghost_block.lora_A.weight": torch.randn(rank, 8),
        "transformer.ghost_block.lora_B.weight": torch.randn(4, rank),
        # missing the up matrix -> incomplete
        "img_in.lora_down.weight": torch.randn(rank, 4),
    }
    deltas, report = merge_lora_into_state_dict(state, lora_state, 1.0)

    assert "final_layer.linear.weight" in deltas
    assert report.unmatched == ["transformer.ghost_block"]
    assert report.incomplete == ["img_in"]
    assert report.total == 3
    assert 0.0 < report.match_fraction < 1.0


def test_shape_mismatch_reported_not_applied():
    state = _model_state()
    rank = 2
    # up @ down -> (10, 8) but target qkv is (24, 8): shape mismatch.
    lora_state = {
        "double_blocks.0.img_attn.qkv.lora_down.weight": torch.randn(rank, 8),
        "double_blocks.0.img_attn.qkv.lora_up.weight": torch.randn(10, rank),
    }
    deltas, report = merge_lora_into_state_dict(state, lora_state, 1.0)
    assert deltas == {}
    assert report.shape_mismatch == ["double_blocks.0.img_attn.qkv"]
    assert report.matched == []


def test_accumulates_when_two_modules_hit_same_weight():
    state = {"x.weight": torch.zeros(6, 3)}
    a1, b1 = _peft_pair(6, 3, 2, seed=4)
    a2, b2 = _peft_pair(6, 3, 2, seed=5)
    lora_state = {
        "x.lora_A.weight": a1,
        "x.lora_B.weight": b1,
        # second module squashes to the same target via separator difference
        "transformer.x.lora_down.weight": a2,
        "transformer.x.lora_up.weight": b2,
    }
    deltas, report = merge_lora_into_state_dict(state, lora_state, 1.0)
    assert len(report.matched) == 2
    expected = (b1 @ a1) + (b2 @ a2)
    assert torch.allclose(deltas["x.weight"], expected, atol=1e-5)


def test_ambiguous_model_keys_excluded():
    # Two distinct real keys squash to the same alphanumeric form.
    state = {
        "a.b.weight": torch.zeros(4, 3),
        "a_b.weight": torch.zeros(4, 3),
    }
    a, b = _peft_pair(4, 3, 2, seed=6)
    lora_state = {"a.b.lora_A.weight": a, "a.b.lora_B.weight": b}
    deltas, report = merge_lora_into_state_dict(state, lora_state, 1.0)
    assert deltas == {}
    assert report.ambiguous == ["a.b"]
    assert report.matched == []


def test_strength_is_linear():
    state = {"x.weight": torch.zeros(6, 3)}
    a, b = _peft_pair(6, 3, 2, seed=7)
    lora_state = {"x.lora_A.weight": a, "x.lora_B.weight": b}
    d1, _ = merge_lora_into_state_dict(state, lora_state, 0.5)
    d2, _ = merge_lora_into_state_dict(state, lora_state, 1.0)
    assert torch.allclose(d2["x.weight"], 2.0 * d1["x.weight"], atol=1e-6)


# --------------------------------------------------------------------------- #
# apply / restore exactness
# --------------------------------------------------------------------------- #
def test_apply_then_restore_is_exact():
    module = nn.Module()
    module.lin = nn.Linear(3, 6, bias=False)
    original = module.lin.weight.detach().clone()

    delta = torch.randn(6, 3)
    backups, applied = apply_deltas_to_module(module, {"lin.weight": delta})
    assert not torch.allclose(module.lin.weight, original)
    assert torch.allclose(applied["lin.weight"], delta, atol=1e-6)

    restore_module(module, backups)
    assert torch.equal(module.lin.weight, original)  # bit-exact


# --------------------------------------------------------------------------- #
# Full manager flow (file IO + merge + set_strength + unfuse)
# --------------------------------------------------------------------------- #
class _FakePipe:
    """Minimal stand-in for Flux2Pipeline exposing a native nn.Module."""

    def __init__(self, module):
        self._model = module
        self._loaded = True


def _build_klein_like_module():
    module = nn.Module()
    module.img_in = nn.Linear(4, 8, bias=False)
    # nested block to exercise dotted names
    block = nn.Module()
    attn = nn.Module()
    attn.qkv = nn.Linear(8, 24, bias=False)
    block.img_attn = attn
    module.double_blocks = nn.ModuleList([block])
    return module


def test_manager_native_load_set_strength_unfuse(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file

    module = _build_klein_like_module()
    qkv_orig = module.double_blocks[0].img_attn.qkv.weight.detach().clone()

    rank = 4
    g = torch.Generator().manual_seed(8)
    down = torch.randn(rank, 8, generator=g)
    up = torch.randn(24, rank, generator=g)
    lora_state = {
        # kohya names targeting double_blocks.0.img_attn.qkv
        "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": down,
        "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": up,
        "lora_unet_double_blocks_0_img_attn_qkv.alpha": torch.tensor(float(rank)),
    }
    lora_file = tmp_path / "cyber_flowers.safetensors"
    save_file(lora_state, str(lora_file))

    pipe = _FakePipe(module)
    manager = KleinLoRAManager(pipe, backend="native")
    info = manager.load(str(lora_file), strength=1.0)

    # Merge applied: matched 1, weights changed by expected delta.
    assert info.backend == "native"
    assert info.matched == 1 and info.total == 1
    assert info.fused is True
    expected = up.float() @ down.float()  # alpha == rank -> scale == strength == 1
    applied_delta = (module.double_blocks[0].img_attn.qkv.weight - qkv_orig).float()
    assert torch.allclose(applied_delta, expected, atol=1e-3)

    # set_strength halves the contribution.
    manager.set_strength(info.name, 0.5)
    half_delta = (module.double_blocks[0].img_attn.qkv.weight - qkv_orig).float()
    assert torch.allclose(half_delta, 0.5 * expected, atol=1e-3)

    # unfuse restores exactly.
    manager.unfuse(info.name)
    assert torch.equal(module.double_blocks[0].img_attn.qkv.weight, qkv_orig)


def test_manager_native_zero_match_raises(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    module = _build_klein_like_module()
    lora_state = {
        "lora_unet_totally_unknown_layer.lora_down.weight": torch.randn(2, 8),
        "lora_unet_totally_unknown_layer.lora_up.weight": torch.randn(24, 2),
    }
    lora_file = tmp_path / "bad.safetensors"
    save_file(lora_state, str(lora_file))

    manager = KleinLoRAManager(_FakePipe(module), backend="native")
    with pytest.raises(RuntimeError, match="matched 0"):
        manager.load(str(lora_file), strength=1.0)


# --------------------------------------------------------------------------- #
# alpha handling (PEFT alpha lives in adapter_config.json, not the weights)
# --------------------------------------------------------------------------- #
def test_alpha_override_scales_peft_without_in_weight_alpha():
    state = {"x.weight": torch.zeros(6, 3)}
    a, b = _peft_pair(6, 3, 2, seed=11)
    lora_peft = {"x.lora_A.weight": a, "x.lora_B.weight": b}
    rank = 2

    d_default, _ = merge_lora_into_state_dict(state, lora_peft, 1.0)
    # alpha_override = 2*rank -> scale doubles.
    d_override, _ = merge_lora_into_state_dict(
        state, lora_peft, 1.0, alpha_override=2 * rank
    )
    assert torch.allclose(d_override["x.weight"], 2.0 * d_default["x.weight"], atol=1e-6)

    # In-weight kohya alpha still wins over the override.
    lora_kohya = {
        "x.lora_down.weight": a,
        "x.lora_up.weight": b,
        "x.alpha": torch.tensor(float(rank)),
    }
    d_kohya, _ = merge_lora_into_state_dict(
        state, lora_kohya, 1.0, alpha_override=2 * rank
    )
    assert torch.allclose(d_kohya["x.weight"], d_default["x.weight"], atol=1e-6)


def test_manager_native_reads_peft_alpha_from_config(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    module = _build_klein_like_module()
    qkv_orig = module.double_blocks[0].img_attn.qkv.weight.detach().clone()

    rank = 4
    g = torch.Generator().manual_seed(12)
    a = torch.randn(rank, 8, generator=g)    # lora_A (rank, in)
    b = torch.randn(24, rank, generator=g)   # lora_B (out, rank)
    lora_state = {
        "transformer.double_blocks.0.img_attn.qkv.lora_A.weight": a,
        "transformer.double_blocks.0.img_attn.qkv.lora_B.weight": b,
    }
    lora_file = tmp_path / "peft_lora.safetensors"
    save_file(lora_state, str(lora_file))
    # PEFT keeps lora_alpha in adapter_config.json (here 2*rank -> scale 2).
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": 2 * rank, "r": rank})
    )

    manager = KleinLoRAManager(_FakePipe(module), backend="native")
    manager.load(str(lora_file), strength=1.0)
    applied = (module.double_blocks[0].img_attn.qkv.weight - qkv_orig).float()
    expected = (2 * rank / rank) * (b.float() @ a.float())
    assert torch.allclose(applied, expected, atol=1e-4)


def test_native_unfuse_set_strength_fuse_no_double_apply(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    module = _build_klein_like_module()
    qkv_orig = module.double_blocks[0].img_attn.qkv.weight.detach().clone()

    rank = 4
    g = torch.Generator().manual_seed(13)
    down = torch.randn(rank, 8, generator=g)
    up = torch.randn(24, rank, generator=g)
    lora_state = {
        "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": down,
        "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": up,
        "lora_unet_double_blocks_0_img_attn_qkv.alpha": torch.tensor(float(rank)),
    }
    lora_file = tmp_path / "k.safetensors"
    save_file(lora_state, str(lora_file))

    manager = KleinLoRAManager(_FakePipe(module), backend="native")
    info = manager.load(str(lora_file), strength=1.0)
    full = up.float() @ down.float()  # alpha == rank -> scale == strength

    # Unfuse -> weights back to original.
    manager.unfuse(info.name)
    assert torch.equal(module.double_blocks[0].img_attn.qkv.weight, qkv_orig)

    # set_strength while unfused must NOT touch weights (no premature re-apply).
    manager.set_strength(info.name, 0.5)
    assert torch.equal(module.double_blocks[0].img_attn.qkv.weight, qkv_orig)

    # Re-fuse -> applies exactly 0.5x (not 1.5x): no double-apply.
    manager.fuse(info.name)
    applied = (module.double_blocks[0].img_attn.qkv.weight - qkv_orig).float()
    assert torch.allclose(applied, 0.5 * full, atol=1e-3)


def test_manager_native_stacked_loras_preserve_each_other_on_unfuse(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    module = _build_klein_like_module()
    qkv_orig = module.double_blocks[0].img_attn.qkv.weight.detach().clone()

    def make_lora(path: Path, seed: int):
        rank = 4
        g = torch.Generator().manual_seed(seed)
        down = torch.randn(rank, 8, generator=g)
        up = torch.randn(24, rank, generator=g)
        save_file(
            {
                "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": down,
                "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": up,
                "lora_unet_double_blocks_0_img_attn_qkv.alpha": torch.tensor(float(rank)),
            },
            str(path),
        )
        return up.float() @ down.float()

    delta_a = make_lora(tmp_path / "style_a.safetensors", seed=14)
    delta_b = make_lora(tmp_path / "style_b.safetensors", seed=15)

    manager = KleinLoRAManager(_FakePipe(module), backend="native")
    info_a = manager.load(str(tmp_path / "style_a.safetensors"), strength=1.0)
    info_b = manager.load(str(tmp_path / "style_b.safetensors"), strength=0.25)

    applied = (module.double_blocks[0].img_attn.qkv.weight - qkv_orig).float()
    assert torch.allclose(applied, delta_a + 0.25 * delta_b, atol=1e-3)

    manager.set_strength(info_a.name, 0.5)
    applied = (module.double_blocks[0].img_attn.qkv.weight - qkv_orig).float()
    assert torch.allclose(applied, 0.5 * delta_a + 0.25 * delta_b, atol=1e-3)

    manager.unfuse(info_a.name)
    applied = (module.double_blocks[0].img_attn.qkv.weight - qkv_orig).float()
    assert torch.allclose(applied, 0.25 * delta_b, atol=1e-3)

    manager.fuse(info_a.name)
    applied = (module.double_blocks[0].img_attn.qkv.weight - qkv_orig).float()
    assert torch.allclose(applied, 0.5 * delta_a + 0.25 * delta_b, atol=1e-3)

    manager.unload(info_b.name)
    applied = (module.double_blocks[0].img_attn.qkv.weight - qkv_orig).float()
    assert torch.allclose(applied, 0.5 * delta_a, atol=1e-3)

    manager.unfuse(info_a.name)
    assert torch.equal(module.double_blocks[0].img_attn.qkv.weight, qkv_orig)
