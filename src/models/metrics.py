"""Consolidated measurement functions for model comparison.

Provides CKA, KL divergence, effective rank, and other metrics
used across the unified analysis pipeline.
"""

import numpy as np
import torch
import torch.nn.functional as F


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear Centered Kernel Alignment between two representations.

    Args:
        X: (n, d1) representation matrix
        Y: (n, d2) representation matrix

    Returns:
        CKA similarity in [0, 1]
    """
    X = X.astype(np.float64) - X.astype(np.float64).mean(axis=0, keepdims=True)
    Y = Y.astype(np.float64) - Y.astype(np.float64).mean(axis=0, keepdims=True)

    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def effective_rank(X: np.ndarray) -> float:
    """Compute effective rank (exponential of entropy of normalized singular values).

    Args:
        X: (n, d) matrix

    Returns:
        Effective rank (float >= 1)
    """
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def kl_divergence(logits_orig: torch.Tensor, logits_ablated: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between original and ablated output distributions.

    Args:
        logits_orig: (B, L, V) original logits
        logits_ablated: (B, L, V) ablated logits

    Returns:
        (B, L) KL divergence per position
    """
    log_p = F.log_softmax(logits_orig, dim=-1)
    log_q = F.log_softmax(logits_ablated, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def mean_kl_divergence(logits_orig: torch.Tensor, logits_ablated: torch.Tensor) -> float:
    """Mean KL divergence across all positions."""
    kl = kl_divergence(logits_orig, logits_ablated)
    return float(kl.mean())


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy from logits.

    Args:
        logits: (B, L, V) or (B*L, V)
        targets: (B, L) or (B*L,) integer labels

    Returns:
        Accuracy in [0, 1]
    """
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
    if targets.dim() == 2:
        targets = targets.reshape(-1)
    preds = logits.argmax(dim=-1)
    mask = targets >= 0  # ignore padding
    if mask.sum() == 0:
        return 0.0
    return float((preds[mask] == targets[mask]).float().mean())


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence between two distributions.

    Args:
        p, q: probability distributions (1D, same shape, sum to 1)

    Returns:
        JSD in [0, log(2)]
    """
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def attention_entropy(attn_weights: np.ndarray) -> float:
    """Compute entropy of attention distribution.

    Args:
        attn_weights: (L,) attention distribution for one query position

    Returns:
        Entropy (nats)
    """
    p = np.asarray(attn_weights, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def contact_precision_from_attention(attn_weights: np.ndarray,
                                     contact_map: np.ndarray,
                                     min_sep: int = 6,
                                     top_L_fraction: float = 1.0) -> dict:
    """Compute contact prediction precision from attention weights.

    Following Rao et al. 2021: symmetrize attention, apply APC correction,
    then evaluate precision at L, L/2, L/5 contacts.

    Args:
        attn_weights: (n_heads, L, L) attention weights for one layer
        contact_map: (L, L) binary contact map (1 = contact, 0 = no contact)
        min_sep: minimum sequence separation for a valid contact
        top_L_fraction: fraction of L for top-L evaluation

    Returns:
        dict with precision at L, L/2, L/5
    """
    n_heads, L, _ = attn_weights.shape

    # Symmetrize: average A[i,j] and A[j,i] across heads
    attn_sym = (attn_weights + attn_weights.transpose(0, 2, 1)) / 2
    # Average across heads
    attn_avg = attn_sym.mean(axis=0)  # (L, L)

    # APC correction
    row_mean = attn_avg.mean(axis=1, keepdims=True)
    col_mean = attn_avg.mean(axis=0, keepdims=True)
    total_mean = attn_avg.mean()
    apc = row_mean * col_mean / (total_mean + 1e-10)
    attn_corrected = attn_avg - apc

    # Mask short-range contacts and diagonal
    mask = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= min_sep
    scores = attn_corrected * mask

    # Flatten and sort
    triu_idx = np.triu_indices(L, k=min_sep)
    flat_scores = scores[triu_idx]
    flat_contacts = contact_map[triu_idx]
    sorted_idx = np.argsort(-flat_scores)

    results = {}
    for label, k in [("L", L), ("L/2", L // 2), ("L/5", L // 5)]:
        k = max(1, int(k * top_L_fraction))
        top_k = sorted_idx[:k]
        if len(top_k) == 0:
            results[label] = 0.0
        else:
            results[label] = float(flat_contacts[top_k].mean())

    return results


def local_attention_fraction(attn_weights: np.ndarray, window: int = 5) -> float:
    """Fraction of attention within +-window of diagonal.

    Args:
        attn_weights: (L, L) attention matrix for one head
        window: local window half-size

    Returns:
        Fraction in [0, 1]
    """
    L = attn_weights.shape[0]
    mask = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) <= window
    return float(attn_weights[mask].sum() / (attn_weights.sum() + 1e-10))
