"""Smoke tests for activation extraction hooks and SAE architectures.

Run with: ./env/bin/python -m pytest tests/test_hooks.py -v
"""

import sys
import pytest
import torch

sys.path.insert(0, "src")

DEVICE = "cpu"  # Use CPU for tests; GPU may be under load


def test_esm2_loads():
    """Test that ESM-2 loads via HuggingFace and produces output."""
    from models.esm2_hooks import load_esm2

    model, tokenizer = load_esm2("esm2_t6_8M_UR50D", device=DEVICE)
    assert model is not None
    assert tokenizer is not None

    inputs = tokenizer("MKTAYIAKQRQISFVKSHFSRQLE", return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)

    assert output.hidden_states is not None
    # 6 layers + 1 embedding = 7 hidden states
    assert len(output.hidden_states) == 7


def test_esm2_hook_extraction():
    """Test that hooks correctly capture activations."""
    from models.esm2_hooks import load_esm2, ESM2HookManager

    model, tokenizer = load_esm2("esm2_t6_8M_UR50D", device=DEVICE)
    hook_mgr = ESM2HookManager(model, layers=[0, 3, 5], extract_attention=True)

    inputs = tokenizer("MKTAYIAKQRQISFVKSHFSRQLE", return_tensors="pt").to(DEVICE)

    with hook_mgr:
        with torch.no_grad():
            model(**inputs, output_attentions=True)

        cache = hook_mgr.cache

        # Check residual stream was captured at requested layers
        assert 0 in cache.residual_stream
        assert 3 in cache.residual_stream
        assert 5 in cache.residual_stream
        # Unrequested layers should not be present
        assert 1 not in cache.residual_stream

        # Check shapes: (batch=1, seq_len, hidden_dim)
        seq_len = inputs["input_ids"].size(1)
        for layer_idx in [0, 3, 5]:
            rs = cache.residual_stream[layer_idx]
            assert rs.ndim == 3
            assert rs.size(0) == 1
            assert rs.size(1) == seq_len

        # Check attention weights were captured
        assert len(cache.attention_weights) > 0
        for layer_idx, attn in cache.attention_weights.items():
            assert attn.ndim == 4  # (batch, heads, seq_len, seq_len)


def test_esm2_batch_extraction():
    """Test activation extraction with multiple sequences."""
    from models.esm2_hooks import load_esm2, extract_activations

    model, tokenizer = load_esm2("esm2_t6_8M_UR50D", device=DEVICE)

    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLE",
        "ACDEFGHIKLMNPQRSTVWY",
        "MHQAIIHFGHKLL",
    ]

    caches = extract_activations(
        model, tokenizer, sequences,
        layers=[0, 5],
        extract_attention=False,
        batch_size=2,
        device=DEVICE,
    )

    assert len(caches) == 3
    for cache in caches:
        assert 0 in cache.residual_stream
        assert 5 in cache.residual_stream


def test_sae_architectures():
    """Test that all SAE architectures forward pass correctly."""
    from sae.model import SAEConfig, build_sae

    for arch in ["vanilla", "topk", "gated"]:
        config = SAEConfig(
            input_dim=64,
            expansion_factor=4,
            k=8,
            architecture=arch,
        )
        sae = build_sae(config)
        x = torch.randn(16, 64)
        output = sae(x)

        assert "loss" in output
        assert "recon_loss" in output
        assert "z" in output
        assert "x_hat" in output
        assert "l0" in output

        assert output["x_hat"].shape == x.shape
        assert output["z"].shape == (16, 64 * 4)
        assert output["loss"].ndim == 0  # scalar


def test_sae_topk_sparsity():
    """Test that TopK SAE enforces exactly K active features."""
    from sae.model import SAEConfig, build_sae

    config = SAEConfig(input_dim=64, expansion_factor=4, k=8, architecture="topk")
    sae = build_sae(config)
    x = torch.randn(32, 64)
    output = sae(x)

    z = output["z"]
    # Each input should have exactly K nonzero features
    l0_per_input = (z > 0).float().sum(dim=-1)
    assert torch.all(l0_per_input == 8), f"Expected L0=8, got {l0_per_input}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ESM-3 requires GPU (bfloat16)")
def test_esm3_loads():
    """Test that ESM-3 loads and produces output."""
    from models.esm3_hooks import load_esm3, tokenize_sequence

    model, tokenizers = load_esm3(device="cuda:0")
    assert model is not None
    inputs = tokenize_sequence("MKTAYIAKQRQISFVKSHFSRQLE", tokenizers, device="cuda:0")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**inputs)
    assert output.embeddings.shape[-1] == 1536


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ESM-3 requires GPU (bfloat16)")
def test_esm3_hook_extraction():
    """Test that hooks correctly capture ESM-3 activations."""
    from models.esm3_hooks import load_esm3, ESM3HookManager, extract_esm3_activations

    model, tokenizers = load_esm3(device="cuda:0")
    seqs = ["MKTAYIAKQRQISFVKSHFSRQLE", "ACDEFGHIKLMNPQRSTVWY"]
    layers = [0, 24, 47]
    caches = extract_esm3_activations(model, tokenizers, seqs, layers=layers, device="cuda:0")

    assert len(caches) == 2
    for i, cache in enumerate(caches):
        for layer_idx in layers:
            rs = cache.residual_stream[layer_idx]
            assert rs.ndim == 2  # (seq_len, 1536), batch dim removed
            assert rs.shape[1] == 1536
            assert rs.dtype == torch.float32
            assert rs.shape[0] == len(seqs[i]) + 2  # BOS + seq + EOS


def test_activation_save_load():
    """Test HDF5 save/load roundtrip."""
    import tempfile
    from pathlib import Path
    from models.utils import save_activations_h5, load_activations_h5

    residual = {0: torch.randn(10, 64), 5: torch.randn(10, 64)}
    attention = {0: torch.randn(4, 10, 10), 5: torch.randn(4, 10, 10)}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        save_activations_h5(path, residual, attention, metadata={"model": "test"})

        loaded_rs, loaded_attn = load_activations_h5(path, load_attention=True)

        assert set(loaded_rs.keys()) == {0, 5}
        assert set(loaded_attn.keys()) == {0, 5}
        for k in [0, 5]:
            assert loaded_rs[k].shape == residual[k].shape
            assert loaded_attn[k].shape == attention[k].shape
