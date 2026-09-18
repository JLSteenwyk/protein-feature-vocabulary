"""Phase 1.4: Cross-Modal Attention Atlas for ESM-3.

Run as: ./env/bin/python scripts/run_phase1_attention_atlas.py
Logs to: results/phase1/attention_atlas/attention_atlas_log.txt

Extracts per-head attention patterns under S-only and S+St conditions,
computes how each head responds to structure information, and classifies
heads into functional categories.

Since ESM-3 uses additive multimodal embeddings (all modalities at same
positions), "cross-modal attention" means comparing how attention patterns
change when structure tokens are provided.

Metrics per head per layer:
  - Structure responsiveness: JSD(attn_S, attn_S+St)
  - Attention entropy under each condition
  - Local attention fraction (within ±5 positions)
  - Mean maximum attention weight

Output: Head classification atlas, layer-head heatmaps, CKA correlation.
"""

import sys
import os
import json
import time
import logging
import functools
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

log_dir = ROOT / "results" / "phase1" / "attention_atlas"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "attention_atlas_log.txt"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("attention_atlas")

import torch
import torch.nn.functional as F
import numpy as np
import einops


# ============================================================
# Attention Weight Capture
# ============================================================
class AttentionCapture:
    """Monkey-patches MultiHeadAttention to capture attention weights.

    ESM-3 uses F.scaled_dot_product_attention (fused op, no weight access).
    We replace forward() to also compute softmax(QK^T/sqrt(d)) explicitly.
    """

    def __init__(self, model, layers=None):
        self.model = model
        blocks = model.transformer.blocks
        self.layers = layers if layers is not None else list(range(len(blocks)))
        self.weights = {}  # layer_idx -> (1, H, L, L) tensor on CPU
        self._orig_forwards = {}

    def install(self):
        """Replace forward methods to capture attention weights."""
        for layer_idx in self.layers:
            attn = self.model.transformer.blocks[layer_idx].attn
            self._orig_forwards[layer_idx] = attn.forward
            attn.forward = self._make_capturing_forward(attn, layer_idx)

    def _make_capturing_forward(self, attn_mod, layer_idx):
        storage = self.weights

        def capturing_forward(x, seq_id):
            # Replicate MultiHeadAttention.forward, adding weight capture
            qkv_BLD3 = attn_mod.layernorm_qkv(x)
            query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
            query_BLD, key_BLD = (
                attn_mod.q_ln(query_BLD).to(query_BLD.dtype),
                attn_mod.k_ln(key_BLD).to(query_BLD.dtype),
            )
            query_BLD, key_BLD = attn_mod._apply_rotary(query_BLD, key_BLD)

            reshaper = functools.partial(
                einops.rearrange, pattern="b s (h d) -> b h s d", h=attn_mod.n_heads
            )
            query_BHLD, key_BHLD, value_BHLD = map(
                reshaper, (query_BLD, key_BLD, value_BLD)
            )

            # Compute attention weights explicitly
            d_head = attn_mod.d_head
            attn_logits = torch.matmul(
                query_BHLD, key_BHLD.transpose(-2, -1)
            ) / (d_head**0.5)

            if seq_id is not None:
                mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
                mask_BHLL = mask_BLL.unsqueeze(1)
                attn_logits = attn_logits.masked_fill(~mask_BHLL, float("-inf"))
            else:
                mask_BHLL = None

            attn_w = torch.softmax(attn_logits, dim=-1)
            storage[layer_idx] = attn_w.detach().float().cpu()

            # Use SDPA for actual context (efficient + numerically consistent)
            if mask_BHLL is not None:
                context_BHLD = F.scaled_dot_product_attention(
                    query_BHLD, key_BHLD, value_BHLD, mask_BHLL
                )
            else:
                context_BHLD = F.scaled_dot_product_attention(
                    query_BHLD, key_BHLD, value_BHLD
                )

            context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
            return attn_mod.out_proj(context_BLD)

        return capturing_forward

    def remove(self):
        """Restore original forward methods."""
        for layer_idx, orig_fn in self._orig_forwards.items():
            self.model.transformer.blocks[layer_idx].attn.forward = orig_fn
        self._orig_forwards.clear()

    def clear(self):
        self.weights.clear()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *args):
        self.remove()


# ============================================================
# Per-head statistics
# ============================================================
def compute_head_stats(attn_weights):
    """Compute per-head summary statistics from attention weights.

    Args:
        attn_weights: (1, H, L, L) attention weight tensor (float32, CPU)

    Returns:
        dict with (H,) arrays: entropy, local_fraction, mean_max_attn
    """
    attn = attn_weights[0]  # (H, L, L)
    H, L, _ = attn.shape

    # Skip BOS (pos 0) and EOS (last pos) as query positions
    if L > 2:
        attn_q = attn[:, 1:-1, :]  # (H, L-2, L)
        q_offset = 1
    else:
        attn_q = attn
        q_offset = 0

    eps = 1e-10
    Lq, Lk = attn_q.shape[1], attn_q.shape[2]

    # Entropy per query position, averaged
    entropy = -(attn_q * torch.log(attn_q + eps)).sum(-1).mean(-1)  # (H,)

    # Local attention fraction (within ±5 positions)
    q_pos = torch.arange(Lq).unsqueeze(-1) + q_offset  # (Lq, 1)
    k_pos = torch.arange(Lk).unsqueeze(0)  # (1, Lk)
    local_mask = (q_pos - k_pos).abs() <= 5  # (Lq, Lk)
    local_frac = (attn_q * local_mask.unsqueeze(0).float()).sum(-1).mean(-1)  # (H,)

    # Mean max attention per query
    mean_max = attn_q.max(-1).values.mean(-1)  # (H,)

    return {
        "entropy": entropy.numpy(),
        "local_fraction": local_frac.numpy(),
        "mean_max_attn": mean_max.numpy(),
    }


def compute_jsd(attn_s, attn_st):
    """Compute JSD between attention under S vs S+St conditions.

    Args:
        attn_s:  (1, H, L, L) attention under S condition
        attn_st: (1, H, L, L) attention under S+St condition

    Returns:
        (H,) JSD per head, averaged over query positions
    """
    p = attn_s[0]  # (H, L, L)
    q = attn_st[0]

    if p.shape[1] > 2:
        p = p[:, 1:-1, :]
        q = q[:, 1:-1, :]

    eps = 1e-10
    m = 0.5 * (p + q)
    kl_p = (p * torch.log((p + eps) / (m + eps))).sum(-1)  # (H, Lq)
    kl_q = (q * torch.log((q + eps) / (m + eps))).sum(-1)
    jsd = 0.5 * kl_p + 0.5 * kl_q  # (H, Lq)

    return jsd.mean(-1).numpy()  # (H,)


# ============================================================
# Step 1: Load model and data
# ============================================================
def step1_load_data():
    log.info("=" * 60)
    log.info("STEP 1: Loading model and data")
    log.info("=" * 60)

    with open(ROOT / "data" / "pilot" / "sequences" / "sequences.json") as f:
        seq_dict = json.load(f)
    with open(ROOT / "data" / "pilot" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    # Find proteins with AlphaFold structures (length ≤ 500)
    struct_dir = ROOT / "data" / "pilot" / "structures"
    available = {}
    for pdb_path in sorted(struct_dir.glob("*.pdb")):
        acc = pdb_path.stem
        if acc in seq_dict and len(seq_dict[acc]) <= 500:
            available[acc] = str(pdb_path)

    log.info(f"Found {len(available)} proteins with structures (length ≤ 500)")

    log.info("Loading ESM-3 model...")
    from models.esm3_hooks import load_esm3

    model, tokenizers = load_esm3(device="cuda:0")
    vram = torch.cuda.memory_allocated(0) / 1e9
    log.info(f"Model loaded. VRAM: {vram:.2f} GB")

    return model, tokenizers, available, seq_dict, metadata


# ============================================================
# Step 2: Extract attention patterns under S and S+St
# ============================================================
def step2_extract_attention(model, tokenizers, available, seq_dict):
    log.info("=" * 60)
    log.info("STEP 2: Extracting attention patterns")
    log.info("=" * 60)

    from esm.sdk.api import ESMProtein

    device = next(model.parameters()).device
    n_layers = len(model.transformer.blocks)
    n_heads = model.transformer.blocks[0].attn.n_heads
    all_layers = list(range(n_layers))

    log.info(f"Model: {n_layers} layers, {n_heads} heads per layer")
    log.info(f"Capturing attention weights at all {n_layers} layers")

    # Accumulators (n_layers × n_heads)
    jsd_sum = np.zeros((n_layers, n_heads))
    entropy_s_sum = np.zeros((n_layers, n_heads))
    entropy_st_sum = np.zeros((n_layers, n_heads))
    local_s_sum = np.zeros((n_layers, n_heads))
    local_st_sum = np.zeros((n_layers, n_heads))
    max_s_sum = np.zeros((n_layers, n_heads))
    max_st_sum = np.zeros((n_layers, n_heads))

    n_ok = 0
    start_time = time.time()

    capture = AttentionCapture(model, layers=all_layers)
    capture.install()

    try:
        for i, (acc, pdb_path) in enumerate(available.items()):
            try:
                # Load from PDB → get sequence + structure
                protein_st = ESMProtein.from_pdb(pdb_path)
                if protein_st.sequence is None:
                    continue
                pdb_seq = protein_st.sequence

                # Tokenize both conditions
                protein_s = ESMProtein(sequence=pdb_seq)
                tensor_s = model.encode(protein_s)
                tensor_st = model.encode(protein_st)

                # --- S condition ---
                capture.clear()
                kwargs_s = {
                    "sequence_tokens": tensor_s.sequence.unsqueeze(0).to(device)
                }
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(**kwargs_s)

                attn_s = {l: capture.weights[l] for l in all_layers}

                # --- S+St condition ---
                capture.clear()
                kwargs_st = {
                    "sequence_tokens": tensor_st.sequence.unsqueeze(0).to(device)
                }
                if tensor_st.structure is not None:
                    kwargs_st["structure_tokens"] = (
                        tensor_st.structure.unsqueeze(0).to(device)
                    )

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(**kwargs_st)

                attn_st_dict = {l: capture.weights[l] for l in all_layers}

                # Compute per-head statistics at each layer
                for l in all_layers:
                    ws = attn_s[l]
                    wst = attn_st_dict[l]

                    if ws.shape != wst.shape:
                        continue
                    if torch.isnan(ws).any() or torch.isnan(wst).any():
                        continue

                    stats_s = compute_head_stats(ws)
                    stats_st = compute_head_stats(wst)
                    jsd = compute_jsd(ws, wst)

                    jsd_sum[l] += jsd
                    entropy_s_sum[l] += stats_s["entropy"]
                    entropy_st_sum[l] += stats_st["entropy"]
                    local_s_sum[l] += stats_s["local_fraction"]
                    local_st_sum[l] += stats_st["local_fraction"]
                    max_s_sum[l] += stats_s["mean_max_attn"]
                    max_st_sum[l] += stats_st["mean_max_attn"]

                n_ok += 1

                if n_ok % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = n_ok / elapsed
                    vram = torch.cuda.memory_allocated(0) / 1e9
                    log.info(
                        f"  {n_ok}/{len(available)} proteins "
                        f"({rate:.2f} prot/s, VRAM: {vram:.1f} GB)"
                    )

            except Exception as e:
                if n_ok < 3:
                    log.warning(f"  {acc}: {e}")
                continue

    finally:
        capture.remove()

    if n_ok == 0:
        raise RuntimeError("No proteins processed successfully")

    # Average
    jsd_mean = jsd_sum / n_ok
    entropy_s_mean = entropy_s_sum / n_ok
    entropy_st_mean = entropy_st_sum / n_ok
    local_s_mean = local_s_sum / n_ok
    local_st_mean = local_st_sum / n_ok
    max_s_mean = max_s_sum / n_ok
    max_st_mean = max_st_sum / n_ok

    elapsed = time.time() - start_time
    log.info(f"Step 2 complete: {n_ok} proteins in {elapsed:.0f}s ({elapsed/n_ok:.1f}s/prot)")

    return {
        "jsd": jsd_mean,
        "entropy_s": entropy_s_mean,
        "entropy_st": entropy_st_mean,
        "local_s": local_s_mean,
        "local_st": local_st_mean,
        "max_s": max_s_mean,
        "max_st": max_st_mean,
        "n_proteins": n_ok,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }


# ============================================================
# Step 3: Classify heads and build atlas
# ============================================================
def step3_classify_heads(stats):
    log.info("=" * 60)
    log.info("STEP 3: Classifying attention heads")
    log.info("=" * 60)

    jsd = stats["jsd"]  # (n_layers, n_heads)
    n_layers, n_heads = jsd.shape

    all_jsd = jsd.flatten()
    p25 = float(np.percentile(all_jsd, 25))
    p75 = float(np.percentile(all_jsd, 75))
    median = float(np.median(all_jsd))

    log.info(
        f"JSD distribution: min={all_jsd.min():.4f}, p25={p25:.4f}, "
        f"median={median:.4f}, p75={p75:.4f}, max={all_jsd.max():.4f}"
    )

    # Classify
    classifications = np.full((n_layers, n_heads), "mixed", dtype=object)
    classifications[jsd < p25] = "sequence_only"
    classifications[jsd > p75] = "structure_responsive"

    # Per-layer summary
    log.info(f"\n  {'Layer':>6} {'SeqOnly':>8} {'Mixed':>6} {'StrResp':>8} {'MeanJSD':>9}")
    for l in range(n_layers):
        n_seq = int((classifications[l] == "sequence_only").sum())
        n_mix = int((classifications[l] == "mixed").sum())
        n_str = int((classifications[l] == "structure_responsive").sum())
        log.info(
            f"  {l:>6} {n_seq:>8} {n_mix:>6} {n_str:>8} {jsd[l].mean():>9.5f}"
        )

    # Identify structure-integration layers
    layer_mean_jsd = jsd.mean(axis=1)
    top_layers = np.argsort(layer_mean_jsd)[::-1][:12]
    log.info(f"\nTop 12 layers by mean JSD: {sorted(top_layers.tolist())}")

    # Entropy change summary
    ent_change = stats["entropy_st"] - stats["entropy_s"]
    log.info(
        f"\nEntropy change (S+St − S): mean={ent_change.mean():.4f}, "
        f"std={ent_change.std():.4f}"
    )

    # Local attention change
    loc_change = stats["local_st"] - stats["local_s"]
    log.info(
        f"Local fraction change: mean={loc_change.mean():.4f}, "
        f"std={loc_change.std():.4f}"
    )

    return classifications, {"p25": p25, "median": median, "p75": p75}


# ============================================================
# Step 4: Correlate with CKA integration layers
# ============================================================
def step4_correlate_cka(stats):
    log.info("=" * 60)
    log.info("STEP 4: Correlating with CKA integration analysis")
    log.info("=" * 60)

    cka_path = ROOT / "results" / "phase1" / "modality_dropout" / "cka_results.json"
    if not cka_path.exists():
        log.warning("CKA results not found, skipping correlation")
        return {}

    with open(cka_path) as f:
        cka_data = json.load(f)

    # Structure impact = 1 − CKA(S, S+St)
    cka_layers = cka_data["layers"]
    struct_impact = {}
    for l in cka_layers:
        mat = np.array(cka_data["cka_matrices"][str(l)])
        struct_impact[l] = 1.0 - mat[0, 1]

    jsd = stats["jsd"]

    log.info(f"\n  {'CKA Layer':>10} {'StructImpact':>13} {'MeanJSD':>10} {'MaxJSD':>10}")
    for l in cka_layers:
        if l < jsd.shape[0]:
            log.info(
                f"  {l:>10} {struct_impact[l]:>13.4f} "
                f"{jsd[l].mean():>10.5f} {jsd[l].max():>10.5f}"
            )

    # Pearson correlation
    valid = [l for l in cka_layers if l < jsd.shape[0]]
    cka_vals = np.array([struct_impact[l] for l in valid])
    jsd_vals = np.array([jsd[l].mean() for l in valid])

    corr = float(np.corrcoef(cka_vals, jsd_vals)[0, 1]) if len(valid) > 2 else None
    if corr is not None:
        log.info(f"\nPearson r(CKA struct impact, mean JSD): {corr:.3f}")

    return {
        "cka_layers": [int(l) for l in valid],
        "cka_struct_impact": cka_vals.tolist(),
        "mean_jsd_at_cka_layers": jsd_vals.tolist(),
        "pearson_r": corr,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PHASE 1.4: Cross-Modal Attention Atlas")
    log.info("=" * 60)

    overall_start = time.time()

    try:
        model, tokenizers, available, seq_dict, metadata = step1_load_data()

        stats = step2_extract_attention(model, tokenizers, available, seq_dict)

        del model
        torch.cuda.empty_cache()

        classifications, thresholds = step3_classify_heads(stats)

        cka_corr = step4_correlate_cka(stats)

        # Save results
        output = {
            "n_proteins": stats["n_proteins"],
            "n_layers": int(stats["n_layers"]),
            "n_heads": int(stats["n_heads"]),
            "jsd": stats["jsd"].tolist(),
            "entropy_s": stats["entropy_s"].tolist(),
            "entropy_st": stats["entropy_st"].tolist(),
            "local_fraction_s": stats["local_s"].tolist(),
            "local_fraction_st": stats["local_st"].tolist(),
            "mean_max_attn_s": stats["max_s"].tolist(),
            "mean_max_attn_st": stats["max_st"].tolist(),
            "classifications": classifications.tolist(),
            "thresholds": thresholds,
            "cka_correlation": cka_corr,
        }

        with open(log_dir / "attention_atlas.json", "w") as f:
            json.dump(output, f, indent=2)

        elapsed = time.time() - overall_start
        log.info(f"\nPHASE 1.4 COMPLETE in {elapsed/60:.1f} minutes")
        log.info(f"Results saved to {log_dir}/")

    except Exception as e:
        log.error(f"Phase 1.4 failed: {e}", exc_info=True)
        raise
