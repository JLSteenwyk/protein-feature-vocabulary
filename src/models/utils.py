"""Shared utilities for model activation extraction."""

import os
import h5py
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass


def save_activations_h5(
    filepath: str | Path,
    residual_stream: dict[int, torch.Tensor],
    attention_weights: dict[int, torch.Tensor] | None = None,
    metadata: dict | None = None,
):
    """Save activations to HDF5 file.

    Args:
        filepath: output path
        residual_stream: {layer_idx: tensor} mapping
        attention_weights: optional {layer_idx: tensor} mapping
        metadata: optional metadata dict (stored as HDF5 attributes)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        rs_group = f.create_group("residual_stream")
        for layer_idx, tensor in residual_stream.items():
            rs_group.create_dataset(
                str(layer_idx),
                data=tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor,
                compression="gzip",
                compression_opts=4,
            )

        if attention_weights:
            attn_group = f.create_group("attention_weights")
            for layer_idx, tensor in attention_weights.items():
                attn_group.create_dataset(
                    str(layer_idx),
                    data=tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor,
                    compression="gzip",
                    compression_opts=4,
                )

        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value


def load_activations_h5(
    filepath: str | Path,
    layers: list[int] | None = None,
    load_attention: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray] | None]:
    """Load activations from HDF5 file.

    Args:
        filepath: input path
        layers: specific layers to load (None = all)
        load_attention: whether to load attention weights

    Returns:
        (residual_stream, attention_weights) tuple
    """
    residual_stream = {}
    attention_weights = None

    with h5py.File(filepath, "r") as f:
        rs_group = f["residual_stream"]
        layer_keys = list(rs_group.keys()) if layers is None else [str(l) for l in layers]
        for key in layer_keys:
            if key in rs_group:
                residual_stream[int(key)] = rs_group[key][:]

        if load_attention and "attention_weights" in f:
            attention_weights = {}
            attn_group = f["attention_weights"]
            attn_keys = list(attn_group.keys()) if layers is None else [str(l) for l in layers]
            for key in attn_keys:
                if key in attn_group:
                    attention_weights[int(key)] = attn_group[key][:]

    return residual_stream, attention_weights


def get_device(gpu_id: int = 0) -> torch.device:
    """Get torch device, preferring CUDA if available."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


