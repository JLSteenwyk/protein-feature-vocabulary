"""Activation extraction hooks for ESM-2 via HuggingFace Transformers.

Phase 0 pilot: extract residual stream activations and attention weights
from ESM-2 at specified layers.

ESM-2 architecture (HuggingFace):
    EsmForMaskedLM
    └── esm (EsmModel)
        ├── embeddings (EsmEmbeddings)
        └── encoder (EsmEncoder)
            └── layer (ModuleList of EsmLayer)
                Each EsmLayer:
                ├── attention (EsmAttention)
                │   ├── self (EsmSelfAttention)
                │   └── output (EsmSelfOutput)
                └── intermediate + output (FFN)
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ActivationCache:
    """Stores extracted activations from a forward pass."""
    residual_stream: dict[int, torch.Tensor] = field(default_factory=dict)
    attention_weights: dict[int, torch.Tensor] = field(default_factory=dict)
    model_name: str = "esm2"

    def clear(self):
        self.residual_stream.clear()
        self.attention_weights.clear()


class ESM2HookManager:
    """Manages forward hooks for ESM-2 activation extraction (HuggingFace).

    Usage:
        model, tokenizer = load_esm2()
        hook_mgr = ESM2HookManager(model, layers=[1, 8, 16, 24, 32])
        hook_mgr.register()

        inputs = tokenizer("MKTAYIAK", return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(**inputs, output_attentions=True)

        cache = hook_mgr.cache
        layer_24_activations = cache.residual_stream[24]  # (batch, seq_len, hidden_dim)

        hook_mgr.remove()
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[list[int]] = None,
        extract_attention: bool = True,
        extract_residual: bool = True,
    ):
        self.model = model
        self.extract_attention = extract_attention
        self.extract_residual = extract_residual
        self.cache = ActivationCache(model_name="esm2")
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        # HuggingFace ESM-2: model.esm.encoder.layer
        self._transformer_layers = self._find_transformer_layers()
        self.num_layers = len(self._transformer_layers)

        if layers is None:
            self.layers = list(range(self.num_layers))
        else:
            self.layers = [l if l >= 0 else self.num_layers + l for l in layers]

    def _find_transformer_layers(self) -> nn.ModuleList:
        """Locate transformer layers in HuggingFace ESM-2."""
        # EsmForMaskedLM -> .esm.encoder.layer
        if hasattr(self.model, "esm"):
            return self.model.esm.encoder.layer
        # EsmModel -> .encoder.layer
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            return self.model.encoder.layer
        raise ValueError(
            "Could not find transformer layers. "
            f"Top-level attributes: {[n for n, _ in self.model.named_children()]}"
        )

    def _make_residual_hook(self, layer_idx: int):
        """Hook capturing transformer layer output (residual stream)."""
        def hook_fn(module, input, output):
            # EsmLayer output: (hidden_states,) or (hidden_states, attention_weights)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.cache.residual_stream[layer_idx] = hidden.detach().cpu()
        return hook_fn

    def _make_attention_hook(self, layer_idx: int):
        """Hook capturing attention weights from self-attention."""
        def hook_fn(module, input, output):
            # EsmSelfAttention output: (context, attention_weights)
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
            layer_module = self._transformer_layers[layer_idx]

            if self.extract_residual:
                hook = layer_module.register_forward_hook(
                    self._make_residual_hook(layer_idx)
                )
                self._hooks.append(hook)

            if self.extract_attention:
                # HuggingFace ESM-2: layer.attention.self
                attn_module = layer_module.attention.self
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

# Map short names to HuggingFace model IDs
ESM2_MODELS = {
    "esm2_t6_8M_UR50D": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D": "facebook/esm2_t36_3B_UR50D",
}


def load_esm2(
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cuda",
) -> tuple[nn.Module, object]:
    """Load ESM-2 model and tokenizer via HuggingFace.

    Args:
        model_name: short name (e.g., "esm2_t33_650M_UR50D") or HF model ID
        device: device to load model on

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_name = ESM2_MODELS.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForMaskedLM.from_pretrained(hf_name, attn_implementation="eager")
    model = model.to(device)
    model.eval()
    return model, tokenizer


def extract_activations(
    model: nn.Module,
    tokenizer: object,
    sequences: list[str],
    layers: Optional[list[int]] = None,
    extract_attention: bool = True,
    batch_size: int = 4,
    device: str = "cuda",
    max_length: int = 1024,
) -> list[ActivationCache]:
    """Extract activations from ESM-2 for a list of sequences.

    Args:
        model: ESM-2 model (HuggingFace)
        tokenizer: ESM-2 tokenizer
        sequences: list of amino acid sequences
        layers: which layers to extract (None = all)
        extract_attention: whether to extract attention weights
        batch_size: batch size for inference
        device: device for inference
        max_length: max sequence length

    Returns:
        List of ActivationCache objects, one per sequence
    """
    hook_mgr = ESM2HookManager(
        model, layers=layers, extract_attention=extract_attention
    )

    caches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = [s for s in sequences[i : i + batch_size] if len(s) <= max_length]
        if not batch_seqs:
            continue

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        hook_mgr.cache.clear()
        hook_mgr.register()

        with torch.no_grad():
            model(**inputs, output_attentions=extract_attention)

        # Split batch into per-sequence caches
        for seq_idx in range(len(batch_seqs)):
            cache = ActivationCache(model_name="esm2")
            for layer_idx, tensor in hook_mgr.cache.residual_stream.items():
                cache.residual_stream[layer_idx] = tensor[seq_idx]
            for layer_idx, tensor in hook_mgr.cache.attention_weights.items():
                cache.attention_weights[layer_idx] = tensor[seq_idx]
            caches.append(cache)

        hook_mgr.remove()

    return caches
