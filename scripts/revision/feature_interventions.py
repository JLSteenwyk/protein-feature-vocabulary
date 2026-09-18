"""Explicit SAE interventions; historical substitution code remains unchanged."""
from contextlib import contextmanager
import torch


def intervene_hidden(hidden, sae, feature_indices, mode='residual_preserving', token_mask=None):
    """Remove encoded features, retaining reconstruction error unless requested.

    residual_preserving: h + decode(z_removed) - decode(z)
    reconstruction: decode(z_removed), including for an empty feature list.
    The latter's matched comparator is reconstruction with an empty list, not h.
    A boolean token mask selects intervention positions, not readout positions.
    """
    if mode not in {'residual_preserving', 'reconstruction'}:
        raise ValueError('Unknown intervention mode')
    if hidden.ndim != 3 or not hidden.is_floating_point():
        raise ValueError('Expected floating-point batch/token/hidden tensor')
    indices = list(feature_indices)
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
        raise ValueError('Feature indices must be nonnegative integers')
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate feature indices')
    if token_mask is not None:
        if token_mask.dtype != torch.bool or token_mask.shape != hidden.shape[:-1]:
            raise ValueError('Token mask must be boolean and match batch/token dimensions')
        token_mask = token_mask.to(hidden.device)
    if not torch.isfinite(hidden).all():
        raise ValueError('Nonfinite hidden state')
    flat = hidden.reshape(-1, hidden.shape[-1]).float()
    # Avoid low-precision decode subtraction under model-level autocast.
    with torch.autocast(hidden.device.type, enabled=False):
        z = sae.encode(flat)
        if z.ndim != 2 or z.shape[0] != flat.shape[0] or not torch.isfinite(z).all():
            raise ValueError('Invalid encoded activations')
        if indices and max(indices) >= z.shape[-1]:
            raise ValueError('Feature index outside dictionary')
        if not indices and mode == 'residual_preserving':
            return hidden
        removed = z.clone()
        removed[:, indices] = 0
        decoded = sae.decode(removed)
        replacement = decoded if mode == 'reconstruction' else flat + (decoded-sae.decode(z))
    if replacement.shape != flat.shape or not torch.isfinite(replacement).all():
        raise ValueError('Invalid decoded replacement')
    replacement = replacement.reshape_as(hidden).to(hidden.dtype)
    if not torch.isfinite(replacement).all():
        raise ValueError('Replacement overflow after dtype conversion')
    return replacement if token_mask is None else torch.where(token_mask[..., None], replacement, hidden)


@contextmanager
def feature_intervention(block, sae, feature_indices, mode='residual_preserving', token_mask=None):
    """Hook a block output, preserving tuple metadata and removing hook on failure."""
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        value = intervene_hidden(hidden, sae, feature_indices, mode, token_mask)
        return (value,) + output[1:] if isinstance(output, tuple) else value

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
