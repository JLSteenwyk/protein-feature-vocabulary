"""Shared intervention engine for causal analysis of protein language models.

Supports layer ablation, head ablation, activation patching, and SAE feature
ablation for ESM-2 (HuggingFace) and ESM-3 (EvolutionaryScale).

Design: Uses register_forward_pre_hook() on layer L+1 to modify input (which
is the output of layer L). This avoids interfering with the forward pass of
the target layer itself and works consistently across architectures.
"""

import copy
import functools
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


# ============================================================
# Model-agnostic layer access
# ============================================================

def get_layers(model, model_type: str) -> nn.ModuleList:
    """Get the list of transformer layers for a model."""
    if model_type == "esm2":
        return model.esm.encoder.layer
    elif model_type == "esm3":
        return model.transformer.blocks
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_lm_head(model, model_type: str):
    """Get the language model head for logit computation."""
    if model_type == "esm2":
        return model.lm_head
    elif model_type == "esm3":
        return model.output_heads.sequence_head
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_num_heads(model, model_type: str) -> int:
    """Get the number of attention heads."""
    if model_type == "esm2":
        return model.esm.encoder.layer[0].attention.self.num_attention_heads
    elif model_type == "esm3":
        return model.transformer.blocks[0].attn.n_heads
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_head_dim(model, model_type: str) -> int:
    """Get per-head dimension."""
    if model_type == "esm2":
        layer0 = model.esm.encoder.layer[0].attention.self
        return layer0.attention_head_size
    elif model_type == "esm3":
        return model.transformer.blocks[0].attn.d_head
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# Layer Ablation
# ============================================================

@contextmanager
def layer_ablation(model, model_type: str, layer_idx: int,
                   method: str = "mean", ref_mean: torch.Tensor = None):
    """Context manager that ablates a layer's output during forward pass.

    Registers a hook on layer L+1 (or final norm) that replaces its input
    with a constant (zero or mean activation).

    Args:
        model: the model
        model_type: "esm2" or "esm3"
        layer_idx: which layer to ablate
        method: "mean" (replace with ref_mean) or "zero"
        ref_mean: (d_model,) mean activation tensor (required if method="mean")

    Yields:
        None — model can be called normally within the context
    """
    layers = get_layers(model, model_type)
    n_layers = len(layers)

    def make_hook(replacement):
        def hook_fn(module, args):
            # args is a tuple; first element is the hidden state
            if isinstance(args, tuple):
                hidden = args[0]
            else:
                hidden = args
            # Replace with constant
            if method == "zero":
                new_hidden = torch.zeros_like(hidden)
            else:
                new_hidden = replacement.to(hidden.device, hidden.dtype).expand_as(hidden)
            if isinstance(args, tuple):
                return (new_hidden,) + args[1:]
            return new_hidden
        return hook_fn

    replacement = ref_mean if ref_mean is not None else torch.zeros(1)

    # Hook target: the next layer, or final norm if ablating the last layer
    if layer_idx < n_layers - 1:
        target = layers[layer_idx + 1]
    else:
        # For the last layer, hook the final layer norm
        if model_type == "esm3":
            target = model.transformer.norm
        else:
            target = model.esm.encoder.emb_layer_norm_after if hasattr(model.esm.encoder, 'emb_layer_norm_after') else layers[-1]

    handle = target.register_forward_pre_hook(make_hook(replacement))
    try:
        yield
    finally:
        handle.remove()


# ============================================================
# Head Ablation
# ============================================================

@contextmanager
def head_ablation_esm2(model, layer_idx: int, head_idx: int):
    """Zero out a specific attention head in ESM-2.

    Hooks the attention output projection to zero the contribution
    of one head before projection.
    """
    layer = model.esm.encoder.layer[layer_idx]
    attn_self = layer.attention.self
    n_heads = attn_self.num_attention_heads
    head_dim = attn_self.attention_head_size

    original_forward = attn_self.forward

    def patched_forward(*args, **kwargs):
        # Run normal forward
        output = original_forward(*args, **kwargs)
        # output is (context_layer, attention_probs) or just (context_layer,)
        context = output[0]  # (B, L, H*D)
        B, L, _ = context.shape
        # Reshape, zero one head, reshape back
        context = context.view(B, L, n_heads, head_dim)
        context[:, :, head_idx, :] = 0.0
        context = context.view(B, L, n_heads * head_dim)
        if len(output) > 1:
            return (context,) + output[1:]
        return (context,)

    attn_self.forward = patched_forward
    try:
        yield
    finally:
        attn_self.forward = original_forward


@contextmanager
def head_ablation_esm3(model, layer_idx: int, head_idx: int):
    """Zero out a specific attention head in ESM-3.

    Monkey-patches the attention forward to zero one head's QKV contribution.
    """
    block = model.transformer.blocks[layer_idx]
    attn = block.attn
    original_forward = attn.forward

    def patched_forward(x, seq_id):
        qkv_BLD3 = attn.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            attn.q_ln(query_BLD).to(query_BLD.dtype),
            attn.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = attn._apply_rotary(query_BLD, key_BLD)

        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=attn.n_heads
        )
        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        # Zero the target head
        query_BHLD[:, head_idx, :, :] = 0.0
        key_BHLD[:, head_idx, :, :] = 0.0
        value_BHLD[:, head_idx, :, :] = 0.0

        # Compute attention with SDPA
        if seq_id is not None:
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)
        else:
            mask_BHLL = None

        context_BHLD = F.scaled_dot_product_attention(
            query_BHLD, key_BHLD, value_BHLD, attn_mask=mask_BHLL
        )
        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return attn.out_proj(context_BLD)

    attn.forward = patched_forward
    try:
        yield
    finally:
        attn.forward = original_forward


@contextmanager
def head_ablation(model, model_type: str, layer_idx: int, head_idx: int):
    """Unified head ablation context manager."""
    if model_type == "esm2":
        with head_ablation_esm2(model, layer_idx, head_idx):
            yield
    elif model_type == "esm3":
        with head_ablation_esm3(model, layer_idx, head_idx):
            yield
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# Activation Patching
# ============================================================

@contextmanager
def activation_patch(model, model_type: str, layer_idx: int,
                     source_activation: torch.Tensor):
    """Patch the residual stream at a given layer with source activations.

    Replaces the output of layer_idx with source_activation during forward.

    Args:
        model: the model
        model_type: "esm2" or "esm3"
        layer_idx: which layer's output to replace
        source_activation: (1, L, D) tensor to substitute
    """
    layers = get_layers(model, model_type)
    n_layers = len(layers)

    def make_hook(source):
        def hook_fn(module, args):
            if isinstance(args, tuple):
                hidden = args[0]
            else:
                hidden = args
            # Use source activation, matching sequence length
            src = source.to(hidden.device, hidden.dtype)
            L_src, L_tgt = src.size(1), hidden.size(1)
            if L_src != L_tgt:
                # Truncate or pad to match
                min_L = min(L_src, L_tgt)
                new_hidden = hidden.clone()
                new_hidden[:, :min_L, :] = src[:, :min_L, :]
            else:
                new_hidden = src
            if isinstance(args, tuple):
                return (new_hidden,) + args[1:]
            return new_hidden
        return hook_fn

    # Hook target: layer L+1 or final norm
    if layer_idx < n_layers - 1:
        target = layers[layer_idx + 1]
    else:
        if model_type == "esm3":
            target = model.transformer.norm
        else:
            target = layers[-1]

    handle = target.register_forward_pre_hook(make_hook(source_activation))
    try:
        yield
    finally:
        handle.remove()


# ============================================================
# Interchange Intervention (Causal Probing)
# ============================================================

@contextmanager
def interchange_intervention(model, model_type: str, layer_idx: int,
                              direction: torch.Tensor,
                              source_activation: torch.Tensor):
    """Interchange intervention along a specific direction.

    Replaces the component of the residual stream along `direction` with the
    corresponding component from `source_activation`. This tests whether
    the model causally uses the information encoded along that direction.

    Args:
        model: the model
        model_type: "esm2" or "esm3"
        layer_idx: which layer's output to modify
        direction: (D,) unit vector (probe direction)
        source_activation: (1, L, D) source activations to take the projection from
    """
    layers = get_layers(model, model_type)
    n_layers = len(layers)

    def make_hook(src, d):
        def hook_fn(module, args):
            if isinstance(args, tuple):
                hidden = args[0]
            else:
                hidden = args
            d_dev = d.to(hidden.device, hidden.dtype)
            s = src.to(hidden.device, hidden.dtype)

            # Match sequence lengths
            min_L = min(hidden.size(1), s.size(1))

            # Project hidden and source onto direction
            h_proj = (hidden[:, :min_L] * d_dev).sum(dim=-1, keepdim=True)  # (B, L, 1)
            s_proj = (s[:, :min_L] * d_dev).sum(dim=-1, keepdim=True)

            # Replace hidden's component along d with source's
            new_hidden = hidden.clone()
            new_hidden[:, :min_L] = hidden[:, :min_L] + (s_proj - h_proj) * d_dev

            if isinstance(args, tuple):
                return (new_hidden,) + args[1:]
            return new_hidden
        return hook_fn

    if layer_idx < n_layers - 1:
        target = layers[layer_idx + 1]
    else:
        if model_type == "esm3":
            target = model.transformer.norm
        else:
            target = layers[-1]

    handle = target.register_forward_pre_hook(make_hook(source_activation, direction))
    try:
        yield
    finally:
        handle.remove()


# ============================================================
# SAE Feature Ablation
# ============================================================

@contextmanager
def sae_feature_ablation(model, model_type: str, sae, layer_idx: int,
                         feature_indices: list[int]):
    """Ablate specific SAE features at a given layer.

    Hooks the layer to: encode -> zero features -> decode -> substitute.

    Args:
        model: the model
        model_type: "esm2" or "esm3"
        sae: trained SAE with encode() and decode() methods
        layer_idx: which layer to intervene on
        feature_indices: list of feature indices to zero out
    """
    layers = get_layers(model, model_type)
    target_layer = layers[layer_idx]
    feature_mask = torch.tensor(feature_indices, dtype=torch.long)

    def hook_fn(module, input, output):
        # Get the hidden state from output
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # SAE encode -> ablate -> decode
        B, L, D = hidden.shape
        flat = hidden.reshape(-1, D).float()
        z = sae.encode(flat)
        z[:, feature_mask] = 0.0
        reconstructed = sae.decode(z)
        new_hidden = reconstructed.reshape(B, L, D).to(hidden.dtype)

        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


# ============================================================
# Residual Stream Cache
# ============================================================

@contextmanager
def cache_residual_stream(model, model_type: str, layers: list[int]):
    """Cache residual stream activations at specified layers.

    Yields a dict that gets populated during forward pass.

    Args:
        model: the model
        model_type: "esm2" or "esm3"
        layers: which layers to cache

    Yields:
        dict mapping layer_idx -> tensor (detached, on CPU)
    """
    all_layers = get_layers(model, model_type)
    cache = {}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            cache[layer_idx] = hidden.detach().cpu()
        return hook_fn

    for layer_idx in layers:
        handle = all_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


# ============================================================
# Reference Mean Computation
# ============================================================

# ============================================================
# Logit Lens — project intermediate layers through LM head
# ============================================================

def get_final_norm(model, model_type: str):
    """Get the final layer norm applied before the LM head."""
    if model_type == "esm2":
        # ESM-2 (HuggingFace): post-norm architecture, no separate final norm
        # The LM head includes its own layer norm internally
        return None
    elif model_type == "esm3":
        return model.transformer.norm
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def logit_lens_project(hidden: torch.Tensor, model, model_type: str) -> torch.Tensor:
    """Project a hidden state through the final norm + LM head to get logits.

    Args:
        hidden: (B, L, D) hidden state from an intermediate layer
        model: the model
        model_type: "esm2" or "esm3"

    Returns:
        (B, L, V) logits
    """
    lm_head = get_lm_head(model, model_type)
    final_norm = get_final_norm(model, model_type)

    with torch.no_grad():
        x = hidden
        if final_norm is not None:
            # Match model dtype (ESM-3 uses bfloat16 LayerNorm)
            norm_dtype = next(final_norm.parameters()).dtype
            x = final_norm(x.to(norm_dtype))
        logits = lm_head(x)
    return logits


@contextmanager
def cache_residual_with_logit_lens(model, model_type: str, layers: list[int]):
    """Cache residual stream and compute logit lens projections at each layer.

    Yields a dict: {layer_idx: {'hidden': tensor, 'logits': tensor}}
    """
    all_layers = get_layers(model, model_type)
    cache = {}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            h = hidden.detach()
            logits = logit_lens_project(h, model, model_type)
            cache[layer_idx] = {
                'hidden': h.cpu(),
                'logits': logits.cpu(),
            }
        return hook_fn

    for layer_idx in layers:
        handle = all_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


# ============================================================
# Residual Stream Decomposition (Attn vs MLP)
# ============================================================

@contextmanager
def cache_component_outputs(model, model_type: str, layers: list[int]):
    """Cache attention and MLP sub-layer outputs separately.

    For each hooked layer, captures:
    - attn_out: the output of the attention sublayer (before residual add)
    - mlp_out: the output of the MLP/FFN sublayer (before residual add)

    Yields: {layer_idx: {'attn': tensor, 'mlp': tensor}}
    """
    all_layers = get_layers(model, model_type)
    cache = {}
    handles = []

    if model_type == "esm2":
        # ESM-2: layer.attention.output is the attention sublayer output module
        # layer.output is the FFN output module (adds residual)
        # Hook attention.output to get attn contribution
        # Hook layer.output to get FFN contribution
        for layer_idx in layers:
            layer = all_layers[layer_idx]
            cache[layer_idx] = {}

            # Hook to capture input to the attention sublayer (the residual before attn)
            def make_attn_pre_hook(lidx):
                def hook_fn(module, args):
                    if isinstance(args, tuple):
                        cache[lidx]['_pre_attn'] = args[0].detach().cpu()
                    else:
                        cache[lidx]['_pre_attn'] = args.detach().cpu()
                return hook_fn

            h = layer.register_forward_pre_hook(make_attn_pre_hook(layer_idx))
            handles.append(h)

            # Hook attention output (includes residual in HF ESM-2)
            def make_attn_hook(lidx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        attn_out = output[0]
                    else:
                        attn_out = output
                    pre = cache[lidx].get('_pre_attn')
                    if pre is not None:
                        cache[lidx]['attn'] = (attn_out.detach().cpu() - pre)
                return hook_fn

            h = layer.attention.register_forward_hook(make_attn_hook(layer_idx))
            handles.append(h)

            # Hook FFN: capture layer output and subtract attention output
            def make_layer_hook(lidx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        full_out = output[0]
                    else:
                        full_out = output
                    attn_contrib = cache[lidx].get('attn')
                    pre = cache[lidx].get('_pre_attn')
                    if attn_contrib is not None and pre is not None:
                        cache[lidx]['mlp'] = full_out.detach().cpu() - pre - attn_contrib
                return hook_fn

            h = layer.register_forward_hook(make_layer_hook(layer_idx))
            handles.append(h)

    elif model_type == "esm3":
        # ESM-3: block.attn(x) → attn_out, block.ffn(x) → ffn_out
        # Block forward: x = x + attn(x); x = x + ffn(x)
        for layer_idx in layers:
            block = all_layers[layer_idx]
            cache[layer_idx] = {}

            def make_attn_hook(lidx):
                def hook_fn(module, input, output):
                    cache[lidx]['attn'] = output.detach().cpu()
                return hook_fn

            h = block.attn.register_forward_hook(make_attn_hook(layer_idx))
            handles.append(h)

            def make_ffn_hook(lidx):
                def hook_fn(module, input, output):
                    cache[lidx]['mlp'] = output.detach().cpu()
                return hook_fn

            h = block.ffn.register_forward_hook(make_ffn_hook(layer_idx))
            handles.append(h)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


# ============================================================
# Steering Vectors
# ============================================================

def compute_steering_vector(activations_pos: list[torch.Tensor],
                           activations_neg: list[torch.Tensor]) -> torch.Tensor:
    """Compute a steering vector as the mean difference between two groups.

    Args:
        activations_pos: list of (L_i, D) tensors for positive examples
        activations_neg: list of (L_i, D) tensors for negative examples

    Returns:
        (D,) steering vector
    """
    # Pool each activation to a single vector (mean over residues)
    pos_mean = torch.stack([a.float().mean(dim=0) for a in activations_pos]).mean(dim=0)
    neg_mean = torch.stack([a.float().mean(dim=0) for a in activations_neg]).mean(dim=0)
    return pos_mean - neg_mean


def compute_residue_steering_vector(activations: list[torch.Tensor],
                                    labels: list[torch.Tensor],
                                    pos_label: int, neg_label: int) -> torch.Tensor:
    """Compute a residue-level steering vector from labeled residues.

    Args:
        activations: list of (L_i, D) tensors per protein
        labels: list of (L_i,) int tensors per protein (per-residue labels)
        pos_label: label for positive class
        neg_label: label for negative class

    Returns:
        (D,) steering vector
    """
    pos_vecs, neg_vecs = [], []
    for act, lab in zip(activations, labels):
        pos_mask = (lab == pos_label)
        neg_mask = (lab == neg_label)
        if pos_mask.any():
            pos_vecs.append(act[pos_mask].float().mean(dim=0))
        if neg_mask.any():
            neg_vecs.append(act[neg_mask].float().mean(dim=0))
    if not pos_vecs or not neg_vecs:
        raise ValueError("Need at least one positive and one negative example")
    return torch.stack(pos_vecs).mean(0) - torch.stack(neg_vecs).mean(0)


@contextmanager
def apply_steering_vector(model, model_type: str, layer_idx: int,
                          vector: torch.Tensor, scale: float = 1.0):
    """Add a steering vector to the residual stream at a given layer.

    Hooks layer L+1 pre-hook to add the vector to the hidden state.

    Args:
        model: the model
        model_type: "esm2" or "esm3"
        layer_idx: which layer's output to steer
        vector: (D,) steering vector
        scale: multiplier for the steering vector
    """
    layers = get_layers(model, model_type)
    n_layers = len(layers)

    def make_hook(vec, s):
        def hook_fn(module, args):
            if isinstance(args, tuple):
                hidden = args[0]
            else:
                hidden = args
            delta = vec.to(hidden.device, hidden.dtype) * s
            new_hidden = hidden + delta.unsqueeze(0).unsqueeze(0)
            if isinstance(args, tuple):
                return (new_hidden,) + args[1:]
            return new_hidden
        return hook_fn

    if layer_idx < n_layers - 1:
        target = layers[layer_idx + 1]
    else:
        if model_type == "esm3":
            target = model.transformer.norm
        else:
            target = layers[-1]

    handle = target.register_forward_pre_hook(make_hook(vector, scale))
    try:
        yield
    finally:
        handle.remove()


# ============================================================
# Direct Logit Attribution
# ============================================================

def get_unembedding_matrix(model, model_type: str) -> torch.Tensor:
    """Get the unembedding (output projection) weight matrix.

    Returns:
        (V, D) weight matrix that maps hidden states to logits.
    """
    lm_head = get_lm_head(model, model_type)
    # Most LM heads have a dense layer or are just a linear projection
    if model_type == "esm2":
        # ESM-2 lm_head: LayerNorm → Dense → GELU → LayerNorm → Decoder
        # The final projection is model.lm_head.decoder.weight (V, D)
        return lm_head.decoder.weight.detach()
    elif model_type == "esm3":
        # ESM-3 sequence_head: the last linear layer
        # Iterate to find the final Linear layer
        last_linear = None
        for module in lm_head.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            return last_linear.weight.detach()
        raise RuntimeError("Could not find linear layer in ESM-3 sequence head")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


@contextmanager
def cache_residual_for_logit_attribution(model, model_type: str, layers: list[int]):
    """Cache per-layer residual stream contributions for direct logit attribution.

    For each layer, caches the *change* in residual stream (layer output minus input).
    This gives each layer's additive contribution to the final hidden state.

    Yields: {layer_idx: (B, L, D) contribution tensor}
    """
    all_layers = get_layers(model, model_type)
    cache = {}
    handles = []

    for layer_idx in layers:
        layer = all_layers[layer_idx]
        cache[layer_idx] = {}

        def make_pre_hook(lidx):
            def hook_fn(module, args):
                if isinstance(args, tuple):
                    cache[lidx]['_input'] = args[0].detach()
                else:
                    cache[lidx]['_input'] = args.detach()
            return hook_fn

        def make_post_hook(lidx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                inp = cache[lidx].get('_input')
                if inp is not None:
                    cache[lidx]['contribution'] = (out.detach() - inp).cpu()
            return hook_fn

        h1 = layer.register_forward_pre_hook(make_pre_hook(layer_idx))
        h2 = layer.register_forward_hook(make_post_hook(layer_idx))
        handles.extend([h1, h2])

    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


# ============================================================
# OV / QK Circuit Decomposition
# ============================================================

def extract_ov_qk_matrices_esm2(model, layer_idx: int):
    """Extract OV and QK weight matrices for an ESM-2 attention head.

    Returns:
        W_OV: (n_heads, d_model, d_model) — what information each head moves
        W_QK: (n_heads, d_model, d_model) — what each head attends to
    """
    layer = model.esm.encoder.layer[layer_idx]
    attn = layer.attention.self
    n_heads = attn.num_attention_heads
    d_head = attn.attention_head_size
    d_model = n_heads * d_head

    W_Q = attn.query.weight.detach()  # (d_model, d_model)
    W_K = attn.key.weight.detach()
    W_V = attn.value.weight.detach()
    W_O = layer.attention.output.dense.weight.detach()  # (d_model, d_model)

    # Reshape to per-head: (n_heads, d_head, d_model)
    W_Q_h = W_Q.view(n_heads, d_head, d_model)
    W_K_h = W_K.view(n_heads, d_head, d_model)
    W_V_h = W_V.view(n_heads, d_head, d_model)
    W_O_h = W_O.view(d_model, n_heads, d_head).permute(1, 0, 2)  # (n_heads, d_model, d_head)

    # W_OV[h] = W_O[h] @ W_V[h] : (d_model, d_model)
    W_OV = torch.bmm(W_O_h, W_V_h)  # (n_heads, d_model, d_model)
    # W_QK[h] = W_Q[h]^T @ W_K[h] : (d_model, d_model)
    W_QK = torch.bmm(W_Q_h.transpose(1, 2), W_K_h)  # (n_heads, d_model, d_model)

    return W_OV, W_QK


def extract_ov_qk_matrices_esm3(model, layer_idx: int):
    """Extract OV and QK weight matrices for an ESM-3 attention head.

    ESM-3 uses fused QKV projection (layernorm_qkv) and separate out_proj.
    """
    block = model.transformer.blocks[layer_idx]
    attn = block.attn
    n_heads = attn.n_heads
    d_head = attn.d_head
    d_model = n_heads * d_head

    # layernorm_qkv projects to 3*d_model (Q, K, V concatenated)
    # It's a Sequential with LayerNorm → Linear
    qkv_linear = None
    for module in attn.layernorm_qkv.modules():
        if isinstance(module, nn.Linear):
            qkv_linear = module
    if qkv_linear is None:
        raise RuntimeError("Could not find QKV linear in ESM-3 attention")

    W_QKV = qkv_linear.weight.detach()  # (3*d_model, d_model)
    W_Q_all = W_QKV[:d_model]     # (d_model, d_model)
    W_K_all = W_QKV[d_model:2*d_model]
    W_V_all = W_QKV[2*d_model:]

    W_O = attn.out_proj.weight.detach()  # (d_model, d_model)

    W_Q_h = W_Q_all.view(n_heads, d_head, d_model)
    W_K_h = W_K_all.view(n_heads, d_head, d_model)
    W_V_h = W_V_all.view(n_heads, d_head, d_model)
    W_O_h = W_O.view(d_model, n_heads, d_head).permute(1, 0, 2)

    W_OV = torch.bmm(W_O_h, W_V_h)
    W_QK = torch.bmm(W_Q_h.transpose(1, 2), W_K_h)

    return W_OV, W_QK


def extract_ov_qk_matrices(model, model_type: str, layer_idx: int):
    """Unified OV/QK extraction."""
    if model_type == "esm2":
        return extract_ov_qk_matrices_esm2(model, layer_idx)
    elif model_type == "esm3":
        return extract_ov_qk_matrices_esm3(model, layer_idx)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# Attribution Patching (Gradient-based)
# ============================================================

@contextmanager
def enable_gradient_cache(model, model_type: str, layers: list[int]):
    """Enable gradient tracking on specified layer outputs for attribution patching.

    Hooks each layer to store its output and enable grad tracking.
    After forward + backward, each layer's .grad contains the gradient
    of the loss w.r.t. that layer's output.

    Yields: dict {layer_idx: tensor} where tensors have requires_grad=True
    """
    all_layers = get_layers(model, model_type)
    cache = {}
    handles = []

    for layer_idx in layers:
        layer = all_layers[layer_idx]

        def make_hook(lidx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Create a leaf tensor that requires grad
                h_grad = h.detach().requires_grad_(True)
                cache[lidx] = h_grad
                if isinstance(output, tuple):
                    return (h_grad,) + output[1:]
                return h_grad
            return hook_fn

        handle = layer.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


# ============================================================
# Reference Mean Computation
# ============================================================

def compute_reference_means(model, model_type: str, dataloader, layers: list[int],
                            device: str = "cuda", max_batches: int = 50) -> dict:
    """Compute per-layer mean activations from a dataset for mean-ablation.

    Args:
        model: the model
        model_type: "esm2" or "esm3"
        dataloader: yields input dicts
        layers: which layers to compute means for
        device: computation device
        max_batches: maximum batches to process

    Returns:
        {layer_idx: (1, 1, D) mean activation tensor}
    """
    sums = {}
    counts = {}

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        with cache_residual_stream(model, model_type, layers) as cache:
            with torch.no_grad():
                if model_type == "esm3":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model(**{k: v.to(device) for k, v in batch.items()})
                else:
                    model(**{k: v.to(device) for k, v in batch.items()})

        for layer_idx, tensor in cache.items():
            # tensor: (B, L, D) — average over B and L
            mean = tensor.float().mean(dim=(0, 1))  # (D,)
            if layer_idx not in sums:
                sums[layer_idx] = mean
                counts[layer_idx] = 1
            else:
                sums[layer_idx] += mean
                counts[layer_idx] += 1

    means = {}
    for layer_idx in sums:
        means[layer_idx] = (sums[layer_idx] / counts[layer_idx]).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

    return means
