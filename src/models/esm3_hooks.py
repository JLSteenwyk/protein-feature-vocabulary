"""Activation extraction hooks for ESM-3 (EvolutionaryScale).

ESM-3 architecture (esm3-sm-open-v1, 1.4B):
    ESM3
    ├── encoder (EncodeInputs) — additive multimodal embedding
    ├── transformer (TransformerStack)
    │   ├── blocks (ModuleList of UnifiedTransformerBlock)
    │   │   Each block:
    │   │   ├── attn (MultiHeadAttention / FlashMultiHeadAttention)
    │   │   ├── geom_attn (GeometricReasoningOriginalImpl, layer 0 only)
    │   │   └── ffn (SwiGLU FFN)
    │   └── norm (LayerNorm)
    └── output_heads (OutputHeads)

d_model=1536, n_heads=24, d_head=64, n_layers=48
Rotary positional embeddings (RoPE).
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ESM3ActivationCache:
    """Stores extracted activations from an ESM-3 forward pass."""
    residual_stream: dict[int, torch.Tensor] = field(default_factory=dict)
    attention_weights: dict[int, torch.Tensor] = field(default_factory=dict)
    model_name: str = "esm3"

    def clear(self):
        self.residual_stream.clear()
        self.attention_weights.clear()


class ESM3HookManager:
    """Manages forward hooks for ESM-3 activation extraction.

    Usage:
        model = load_esm3()
        hook_mgr = ESM3HookManager(model, layers=[0, 12, 24, 36, 47])

        with hook_mgr:
            output = model(sequence_tokens=tokens)
            cache = hook_mgr.cache
            layer_24 = cache.residual_stream[24]  # (batch, seq_len, 1536)
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[list[int]] = None,
        extract_attention: bool = False,
        extract_residual: bool = True,
    ):
        self.model = model
        self.extract_attention = extract_attention
        self.extract_residual = extract_residual
        self.cache = ESM3ActivationCache()
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        self._transformer_blocks = self._find_blocks()
        self.num_layers = len(self._transformer_blocks)

        if layers is None:
            self.layers = list(range(self.num_layers))
        else:
            self.layers = [l if l >= 0 else self.num_layers + l for l in layers]

    def _find_blocks(self) -> nn.ModuleList:
        """Locate transformer blocks in ESM-3."""
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "blocks"):
            return self.model.transformer.blocks
        raise ValueError(
            "Could not find transformer blocks. "
            f"Top-level attributes: {[n for n, _ in self.model.named_children()]}"
        )

    def _make_residual_hook(self, layer_idx: int):
        """Hook capturing block output (residual stream after attn+ffn)."""
        def hook_fn(module, input, output):
            # UnifiedTransformerBlock output is just the hidden state tensor
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.cache.residual_stream[layer_idx] = hidden.detach().cpu()
        return hook_fn

    def _make_attention_hook(self, layer_idx: int):
        """Hook on attention module to capture Q, K for weight computation.

        ESM-3 uses scaled_dot_product_attention which doesn't expose weights
        directly. We hook after QKV projection and compute weights manually.
        """
        def hook_fn(module, input, output):
            # MultiHeadAttention output is the projected output tensor
            # To get attention weights, we'd need to hook inside the forward
            # For now, we store the output of the attention sublayer
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.cache.attention_weights[layer_idx] = attn_weights.detach().cpu()
        return hook_fn

    def register(self):
        """Register forward hooks on specified layers."""
        self.remove()
        self.cache.clear()

        for layer_idx in self.layers:
            if layer_idx >= self.num_layers:
                raise IndexError(
                    f"Layer {layer_idx} requested but model only has {self.num_layers} layers"
                )
            block = self._transformer_blocks[layer_idx]

            if self.extract_residual:
                hook = block.register_forward_hook(
                    self._make_residual_hook(layer_idx)
                )
                self._hooks.append(hook)

            if self.extract_attention:
                attn_module = block.attn
                hook = attn_module.register_forward_hook(
                    self._make_attention_hook(layer_idx)
                )
                self._hooks.append(hook)

    def remove(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


# --- Model Loading ---

def load_esm3(
    model_name: str = "esm3_sm_open_v1",
    device: str = "cuda",
) -> tuple[nn.Module, object]:
    """Load ESM-3 model via EvolutionaryScale's esm package.

    Args:
        model_name: model identifier (default: esm3_sm_open_v1)
        device: device to load model on

    Returns:
        (model, tokenizers) tuple
    """
    from esm.models.esm3 import ESM3

    model = ESM3.from_pretrained(model_name).to(device)
    model.eval()
    tokenizers = model.tokenizers
    return model, tokenizers


def tokenize_sequence(
    sequence: str,
    tokenizers: object,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Tokenize a protein sequence for ESM-3 (sequence-only input).

    Args:
        sequence: amino acid sequence string
        tokenizers: ESM-3 tokenizer collection
        device: target device

    Returns:
        Dict with 'sequence_tokens' tensor (B=1, L+2) including BOS/EOS
    """
    seq_tokens = tokenizers.sequence.encode(sequence)
    seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
    return {"sequence_tokens": seq_tensor}


def extract_esm3_activations(
    model: nn.Module,
    tokenizers: object,
    sequences: list[str],
    layers: Optional[list[int]] = None,
    batch_size: int = 1,
    device: str = "cuda",
    max_length: int = 1024,
) -> list[ESM3ActivationCache]:
    """Extract activations from ESM-3 for a list of sequences.

    Uses sequence-only input mode (no structure/function tokens).
    For multimodal extraction, use extract_esm3_multimodal.

    Args:
        model: ESM-3 model
        tokenizers: ESM-3 tokenizer collection
        sequences: list of amino acid sequences
        layers: which layers to extract (None = all 48)
        batch_size: batch size (ESM-3 is large, use 1-2)
        device: device for inference
        max_length: max sequence length

    Returns:
        List of ESM3ActivationCache, one per sequence
    """
    hook_mgr = ESM3HookManager(model, layers=layers, extract_attention=False)

    caches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = [s for s in sequences[i:i + batch_size] if len(s) <= max_length]
        if not batch_seqs:
            continue

        for seq in batch_seqs:
            inputs = tokenize_sequence(seq, tokenizers, device=device)

            hook_mgr.cache.clear()
            hook_mgr.register()

            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                model(**inputs)

            cache = ESM3ActivationCache()
            for layer_idx, tensor in hook_mgr.cache.residual_stream.items():
                # Remove batch dim, cast to float32 for downstream use
                cache.residual_stream[layer_idx] = tensor[0].float()
            caches.append(cache)

            hook_mgr.remove()

    return caches


def extract_esm3_multimodal(
    model: nn.Module,
    tokenizers: object,
    sequence: str,
    structure_tokens: Optional[torch.Tensor] = None,
    function_tokens: Optional[torch.Tensor] = None,
    layers: Optional[list[int]] = None,
    device: str = "cuda",
) -> ESM3ActivationCache:
    """Extract activations from ESM-3 under specific modality conditions.

    This is the core function for modality dropout experiments.
    Pass None for a modality to exclude it.

    Args:
        model: ESM-3 model
        tokenizers: ESM-3 tokenizer collection
        sequence: amino acid sequence (always required as base)
        structure_tokens: optional structure tokens (B, L) int64
        function_tokens: optional function tokens (B, L, 8) int64
        layers: which layers to extract
        device: device for inference

    Returns:
        ESM3ActivationCache with extracted activations
    """
    hook_mgr = ESM3HookManager(model, layers=layers, extract_attention=False)

    inputs = tokenize_sequence(sequence, tokenizers, device=device)
    if structure_tokens is not None:
        inputs["structure_tokens"] = structure_tokens.to(device)
    if function_tokens is not None:
        inputs["function_tokens"] = function_tokens.to(device)

    hook_mgr.cache.clear()
    hook_mgr.register()

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        model(**inputs)

    cache = ESM3ActivationCache()
    for layer_idx, tensor in hook_mgr.cache.residual_stream.items():
        cache.residual_stream[layer_idx] = tensor[0].float()

    hook_mgr.remove()
    return cache
