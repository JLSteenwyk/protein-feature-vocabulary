#!/usr/bin/env python3
"""Attention atlas: map structure-responsive attention heads across all ESM-3 layers.

For each protein, compares attention patterns under S-only vs S+St conditions.
Identifies heads that change attention with structure (JSD metric).

Uses monkey-patched attention forward to capture attention weights
(F.scaled_dot_product_attention doesn't expose them).

Usage:
    CUDA_VISIBLE_DEVICES=1 ./env/bin/python scripts/scaled_1.5M/20_attention_atlas.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("attn_atlas")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

N_PROTEINS = 500
N_LAYERS = 48
N_HEADS = 24
D_HEAD = 64
SEED = 42


class AttentionCapture:
    """Context manager to capture attention weights from ESM-3.

    Monkey-patches the attention forward to compute Q@K^T/sqrt(d) manually
    instead of using F.scaled_dot_product_attention which doesn't expose weights.

    ESM-3 attention uses layernorm_qkv, q_ln, k_ln, and rotary embeddings.
    See scripts/run_phase1_attention_atlas.py for reference implementation.
    """

    def __init__(self, model):
        self.model = model
        self.attention_weights = {}  # layer -> (n_heads, L, L)
        self._original_forwards = {}

    def __enter__(self):
        import functools
        import einops

        def make_forward(attn_mod, lidx, store):
            def fwd(x, seq_id=None):
                qkv = attn_mod.layernorm_qkv(x)
                q, k, v = torch.chunk(qkv, 3, dim=-1)
                q = attn_mod.q_ln(q).to(q.dtype)
                k = attn_mod.k_ln(k).to(k.dtype)
                q, k = attn_mod._apply_rotary(q, k)
                reshaper = functools.partial(
                    einops.rearrange, pattern="b s (h d) -> b h s d", h=attn_mod.n_heads
                )
                q, k, v = map(reshaper, (q, k, v))
                d = q.shape[-1]
                logits = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
                if seq_id is not None:
                    mask = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
                    logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
                weights = F.softmax(logits, dim=-1)
                store[lidx] = weights.detach().cpu().float().squeeze(0)
                out = torch.matmul(weights, v)
                out = einops.rearrange(out, "b h s d -> b s (h d)")
                out = attn_mod.out_proj(out)
                return out
            return fwd

        for layer_idx, block in enumerate(self.model.transformer.blocks):
            attn = block.attn
            self._original_forwards[layer_idx] = attn.forward
            attn.forward = make_forward(attn, layer_idx, self.attention_weights)
        return self

    def __exit__(self, *args):
        for layer_idx, block in enumerate(self.model.transformer.blocks):
            if layer_idx in self._original_forwards:
                block.attn.forward = self._original_forwards[layer_idx]
        self._original_forwards.clear()
        # NOTE: don't clear attention_weights — caller reads them after __exit__


def jsd(p, q, eps=1e-10):
    """Jensen-Shannon divergence between two distributions."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-proteins", type=int, default=N_PROTEINS)
    args = parser.parse_args()

    from models.esm3_hooks import load_esm3
    from esm.sdk.api import ESMProtein

    # Load structure manifest
    with open(STRUCT_DIR / "manifest.json") as f:
        manifest = json.load(f)
    available = manifest["available"]

    # Sample proteins (shorter ones for memory, max 400 residues)
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    candidates = [acc for acc in sorted(available.keys())
                  if acc in sequences and len(sequences[acc]) <= 400]
    rng = np.random.RandomState(SEED)
    if len(candidates) > args.n_proteins:
        selected = rng.choice(candidates, args.n_proteins, replace=False).tolist()
    else:
        selected = candidates
    log.info(f"Selected {len(selected)} proteins (max 400 residues) for attention atlas")

    # Load ESM-3
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=args.device)
    log.info("ESM-3 loaded")

    # Per-head JSD accumulator: (n_layers, n_heads) sum + count
    jsd_sum = np.zeros((N_LAYERS, N_HEADS))
    jsd_count = np.zeros((N_LAYERS, N_HEADS))

    # Per-head statistics
    head_entropy_S = np.zeros((N_LAYERS, N_HEADS))
    head_entropy_SSt = np.zeros((N_LAYERS, N_HEADS))

    n_done = 0
    start_time = time.time()

    for pi, acc in enumerate(selected):
        pdb_path = available[acc]

        try:
            protein_with_struct = ESMProtein.from_pdb(pdb_path)
            if protein_with_struct.sequence is None:
                continue
            pdb_seq = protein_with_struct.sequence
            if len(pdb_seq) < 50 or len(pdb_seq) > 400:
                continue

            # S-only
            protein_s = ESMProtein(sequence=pdb_seq)
            tokens_s = model.encode(protein_s)

            capture = AttentionCapture(model)
            with capture, torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                model(sequence_tokens=tokens_s.sequence.to(args.device).unsqueeze(0))
            attn_S = {L: w.numpy() for L, w in capture.attention_weights.items()}

            # S+St
            tokens_sst = model.encode(protein_with_struct)

            capture2 = AttentionCapture(model)
            with capture2, torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                model(
                    sequence_tokens=tokens_sst.sequence.to(args.device).unsqueeze(0),
                    structure_tokens=tokens_sst.structure.to(args.device).unsqueeze(0),
                )
            attn_SSt = {L: w.numpy() for L, w in capture2.attention_weights.items()}

            # Compute per-head JSD between S and S+St attention
            for L in range(N_LAYERS):
                if L not in attn_S or L not in attn_SSt:
                    continue
                for h in range(N_HEADS):
                    # Mean attention distribution across query positions
                    attn_s_mean = attn_S[L][h].mean(axis=0)  # (L_seq,)
                    attn_sst_mean = attn_SSt[L][h].mean(axis=0)

                    # Truncate to same length
                    min_len = min(len(attn_s_mean), len(attn_sst_mean))
                    j = jsd(attn_s_mean[:min_len], attn_sst_mean[:min_len])

                    jsd_sum[L, h] += j
                    jsd_count[L, h] += 1

                    # Entropy
                    p = np.clip(attn_s_mean, 1e-10, None)
                    p = p / p.sum()
                    head_entropy_S[L, h] += -np.sum(p * np.log(p))

                    p = np.clip(attn_sst_mean, 1e-10, None)
                    p = p / p.sum()
                    head_entropy_SSt[L, h] += -np.sum(p * np.log(p))

            n_done += 1
            torch.cuda.empty_cache()

        except Exception as e:
            if n_done < 5:
                log.warning(f"  Error on {acc}: {e}")
            continue

        if n_done % 50 == 0:
            elapsed = time.time() - start_time
            rate = n_done / elapsed
            eta = (len(selected) - n_done) / rate / 60
            log.info(f"  {n_done}/{len(selected)} proteins | Rate: {rate:.2f}/s | ETA: {eta:.0f} min")

    log.info(f"\nProcessed {n_done} proteins in {(time.time()-start_time)/60:.1f} min")

    # Compute mean JSD per head
    mean_jsd = np.divide(jsd_sum, jsd_count, where=jsd_count > 0, out=np.zeros_like(jsd_sum))
    mean_entropy_S = np.divide(head_entropy_S, jsd_count, where=jsd_count > 0, out=np.zeros_like(head_entropy_S))
    mean_entropy_SSt = np.divide(head_entropy_SSt, jsd_count, where=jsd_count > 0, out=np.zeros_like(head_entropy_SSt))

    # Classify structure-responsive heads (JSD > 0.1)
    responsive_mask = mean_jsd > 0.1
    n_responsive = responsive_mask.sum()

    log.info(f"\n{'='*60}")
    log.info("ATTENTION ATLAS RESULTS")
    log.info(f"{'='*60}")
    log.info(f"Structure-responsive heads (JSD > 0.1): {n_responsive}/{N_LAYERS * N_HEADS}")

    # Per-layer summary
    layer_summary = {}
    for L in range(N_LAYERS):
        n_resp = responsive_mask[L].sum()
        mean_j = mean_jsd[L].mean()
        max_j = mean_jsd[L].max()
        layer_summary[L] = {
            "n_responsive_heads": int(n_resp),
            "mean_jsd": float(mean_j),
            "max_jsd": float(max_j),
        }
        if n_resp > 0:
            log.info(f"  Layer {L:3d}: {n_resp} responsive heads, "
                     f"mean JSD={mean_j:.4f}, max JSD={max_j:.4f}")

    # Find peak structure responsiveness
    layer_mean_jsd = [mean_jsd[L].mean() for L in range(N_LAYERS)]
    peak_layer = int(np.argmax(layer_mean_jsd))
    log.info(f"\nPeak structure responsiveness: layer {peak_layer} "
             f"(mean JSD={layer_mean_jsd[peak_layer]:.4f})")

    output = {
        "n_proteins": n_done,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "n_responsive_heads": int(n_responsive),
        "jsd_threshold": 0.1,
        "peak_layer": peak_layer,
        "per_layer": layer_summary,
        "mean_jsd_matrix": mean_jsd.tolist(),
    }

    out_path = OUTPUT_DIR / "attention_atlas.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
