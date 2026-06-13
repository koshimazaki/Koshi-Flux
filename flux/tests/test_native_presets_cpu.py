"""CPU-tier contract tests for the native preset lane.

No GPU and no model downloads required - runs on RunPod (or any env with
torch + the repo deps) BEFORE burning GPU time:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=flux/src:core/src:presets \
        pytest flux/tests/test_native_presets_cpu.py -q

Guards the exact contracts the presets rely on, including SDK-drift canaries
(if BFL changes the flux2 interface upstream, these fail before a preset does).
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (REPO_ROOT / "flux/src", REPO_ROOT / "core/src", REPO_ROOT / "presets",
          REPO_ROOT / "presets/native", REPO_ROOT.parent / "flux2-main/src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------- klein_utils

def test_path_priority_first_listed_wins(tmp_path, monkeypatch):
    """_prepend_existing_paths must keep the FIRST listed path first on sys.path
    (regression: stale /workspace copies shadowing the checkout)."""
    klein_utils = pytest.importorskip("klein_utils")
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    monkeypatch.setattr(sys, "path", list(sys.path))
    klein_utils._prepend_existing_paths(a, b, c)
    assert sys.path[:3] == [str(a), str(b), str(c)]


def test_generation_context_json_contract(tmp_path):
    klein_utils = pytest.importorskip("klein_utils")
    import json

    out = tmp_path / "ok.mp4"
    with klein_utils.GenerationContext(str(out)) as gen:
        gen.update(preset="t", seed=1)
    data = json.loads(out.with_suffix(".json").read_text())
    assert data["status"] == "completed" and data["preset"] == "t"

    out2 = tmp_path / "boom.mp4"
    with pytest.raises(RuntimeError):
        with klein_utils.GenerationContext(str(out2)) as gen:
            gen.update(preset="t2")
            raise RuntimeError("boom")
    data2 = json.loads(out2.with_suffix(".json").read_text())
    assert data2["status"] == "failed" and "boom" in data2["error"]


def test_generation_helper_records_failure(tmp_path):
    """generation() contextmanager must not record completed on crash."""
    klein_utils = pytest.importorskip("klein_utils")
    import json

    out = tmp_path / "g.mp4"
    with pytest.raises(ValueError):
        with klein_utils.generation(str(out), preset="g"):
            raise ValueError("nope")
    assert json.loads(out.with_suffix(".json").read_text())["status"] == "failed"


# ------------------------------------------------------------- motion engine

@pytest.fixture(scope="module")
def engine():
    flux2_me = pytest.importorskip("flux_motion.flux2.motion_engine")
    return flux2_me.Flux2MotionEngine(device="cpu")


def test_motion_engine_identity_on_neutral_params(engine):
    z = torch.randn(1, 128, 16, 16)
    out = engine.apply_motion(z, {"zoom": 1.0, "angle": 0.0,
                                  "translation_x": 0.0, "translation_y": 0.0,
                                  "translation_z": 0.0})
    assert out.shape == z.shape
    assert torch.allclose(out, z, atol=1e-5)


def test_motion_engine_zoom_changes_latent_preserves_shape_dtype(engine):
    for dtype in (torch.float32, torch.bfloat16):
        z = torch.randn(1, 128, 16, 16, dtype=dtype)
        out = engine.apply_motion(z, {"zoom": 1.05})
        assert out.shape == z.shape
        assert out.dtype == dtype, "engine must be dtype-agnostic (bf16 feedback latents)"
        assert not torch.allclose(out.float(), z.float(), atol=1e-5)


def test_motion_frame_to_dict_contract():
    pa = pytest.importorskip("flux_motion.shared.parameter_adapter")
    mf = pa.MotionFrame(frame_index=0, zoom=1.1, strength=0.3)
    d = mf.to_dict()
    assert set(d) == {"zoom", "angle", "translation_x", "translation_y", "translation_z"}
    assert "strength" not in d, "strength is img2img denoise, must stay out of motion params"


# ----------------------------------------------------------- audio schedules

def test_audio_schedule_frame_alignment():
    np = pytest.importorskip("numpy")
    audio = pytest.importorskip("flux_motion.audio")
    t = np.linspace(0, 2.0, 2 * 22050, dtype=np.float32)
    wave = (np.sin(2 * np.pi * 60 * t) * np.exp(-8 * (t % 0.5))).astype(np.float32)
    tracks = audio.tracks_from_waveform(wave, sample_rate=22050)
    motion = audio.build_audio_motion_schedule(tracks, num_frames=24, fps=12)
    assert len(motion.motion_frames) == 24
    assert len(motion.bands["low"]) == 24
    assert all(0.0 <= f.strength <= 1.0 for f in motion.motion_frames)


# -------------------------------------------------------- SDK drift canaries

def test_flux2_text_encoder_has_no_encode_method():
    """klein_native.py shipped with text_enc.encode() for months and never ran.
    Canary: BFL embedders are callables (forward(list[str])), not .encode()."""
    te = pytest.importorskip("flux2.text_encoder")
    for cls_name in ("Qwen3Embedder", "Mistral3SmallEmbedder"):
        cls = getattr(te, cls_name, None)
        if cls is None:
            continue
        assert not hasattr(cls, "encode"), f"{cls_name} grew .encode - update native_utils"
        assert hasattr(cls, "forward")


def test_flux2_sampling_contracts():
    sampling = pytest.importorskip("flux2.sampling")
    import inspect

    assert list(inspect.signature(sampling.get_schedule).parameters) == \
        ["num_steps", "image_seq_len"]
    den = list(inspect.signature(sampling.denoise).parameters)
    assert den[:7] == ["model", "img", "img_ids", "txt", "txt_ids", "timesteps", "guidance"]

    # scatter_ids: (1, C, T, H, W) per element, dtype preserved -> .squeeze(2) valid
    tokens = torch.randn(1, 6, 4, dtype=torch.bfloat16)  # (b, seq, ch)
    ids = torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                         [0, 1, 1, 0], [0, 2, 0, 0], [0, 2, 1, 0]]])
    out = sampling.scatter_ids(tokens, ids)
    assert isinstance(out, list) and out[0].dim() == 5
    assert out[0].squeeze(2).shape == (1, 4, 3, 2)
    assert out[0].dtype == torch.bfloat16


def test_native_pipeline_interface():
    """NativePipeline public surface the presets depend on (no model load)."""
    import inspect

    nu = pytest.importorskip("native_utils")
    sig = inspect.signature(nu.NativePipeline.generate_from_latent)
    assert "motion_params" in sig.parameters
    assert {"strength", "num_steps", "guidance", "seed"} <= set(sig.parameters)
    assert hasattr(nu.NativePipeline, "encode")
    assert hasattr(nu.NativePipeline, "decode")
    assert hasattr(nu.NativePipeline, "generate")
