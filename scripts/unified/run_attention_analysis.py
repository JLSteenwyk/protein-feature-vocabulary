#!/usr/bin/env python3
"""Unified attention head analysis for ESM-2 and ESM-3.

Run as:
    ./env/bin/python scripts/unified/run_attention_analysis.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_attention_analysis.py --model esm3 --device cuda:1

ESM-2: Uses HuggingFace output_attentions=True (native support).
ESM-3: Uses monkey-patched attention forward (from Phase 1.4).

Metrics per head per layer:
  - Attention entropy (mean over positions)
  - Local attention fraction (within +-5 positions)
  - Mean max attention weight
  - Head norm (L2 of output projection column for that head)

Output: results/unified/{model}/attention_analysis.json
"""

import sys
import os
import json
import time
import logging
import argparse
import functools
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
import torch.nn.functional as F
import numpy as np

try:
    import einops
except ImportError:
    einops = None


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "attention_analysis_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("attn_analysis"), out_dir


def load_protein_sequences(max_proteins=200, max_length=500):
    """Load protein sequences from the scaled dataset."""
    seq_file = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    if seq_file.exists():
        with open(seq_file) as f:
            all_seqs = json.load(f)
    else:
        seq_file = ROOT / "data" / "protein_dataset" / "sequences"
        fasta_file = ROOT / "data" / "scaled" / "sequences" / "sequences.fasta"
        all_seqs = {}
        if fasta_file.exists():
            acc, seq = None, []
            with open(fasta_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if acc and seq:
                            all_seqs[acc] = "".join(seq)
                        acc = line[1:].split()[0]
                        seq = []
                    else:
                        seq.append(line)
            if acc and seq:
                all_seqs[acc] = "".join(seq)

    # Filter by length and take subset
    filtered = {k: v for k, v in all_seqs.items() if len(v) <= max_length}
    # Sort for reproducibility, take first max_proteins
    sorted_accs = sorted(filtered.keys())[:max_proteins]
    return {k: filtered[k] for k in sorted_accs}


# ============================================================
# ESM-2 Attention Extraction
# ============================================================

def extract_esm2_attention(model, tokenizer, sequences, device, log):
    """Extract attention weights from ESM-2 using native output_attentions."""
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    log.info(f"ESM-2: {n_layers} layers, {n_heads} heads")

    # Per-head statistics: [layer][head] -> list of values
    head_entropy = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    head_local_frac = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    head_max_attn = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    for idx, (acc, seq) in enumerate(sequences.items()):
        t0 = time.time()
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attn_tuple = outputs.attentions  # tuple of (B, H, L, L) per layer
        for layer_idx in range(n_layers):
            attn = attn_tuple[layer_idx][0].float().cpu().numpy()  # (H, L, L)
            L = attn.shape[-1]
            for h in range(n_heads):
                w = attn[h]  # (L, L)
                # Entropy: mean over query positions (skip BOS/EOS)
                for qi in range(1, L - 1):
                    row = w[qi]
                    p = row[row > 0]
                    if len(p) > 0:
                        head_entropy[layer_idx][h].append(float(-np.sum(p * np.log(p))))
                # Local fraction
                diag_mask = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) <= 5
                local_frac = float(w[diag_mask].sum() / (w.sum() + 1e-10))
                head_local_frac[layer_idx][h].append(local_frac)
                # Max attention
                head_max_attn[layer_idx][h].append(float(w[1:-1, :].max()))

        if (idx + 1) % 20 == 0 or idx == 0:
            log.info(f"  [{idx+1}/{len(sequences)}] {acc} ({len(seq)} aa) {time.time()-t0:.1f}s")

    # Aggregate
    results = {
        "model": "esm2",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_proteins": len(sequences),
        "per_head": {}
    }
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            key = f"L{layer_idx}_H{h}"
            results["per_head"][key] = {
                "entropy": float(np.mean(head_entropy[layer_idx][h])) if head_entropy[layer_idx][h] else 0.0,
                "local_fraction": float(np.mean(head_local_frac[layer_idx][h])) if head_local_frac[layer_idx][h] else 0.0,
                "max_attention": float(np.mean(head_max_attn[layer_idx][h])) if head_max_attn[layer_idx][h] else 0.0,
            }

    return results


# ============================================================
# ESM-3 Attention Extraction
# ============================================================

class ESM3AttentionCapture:
    """Monkey-patches ESM-3 attention to capture weights."""

    def __init__(self, model, layers=None):
        self.model = model
        blocks = model.transformer.blocks
        self.layers = layers if layers is not None else list(range(len(blocks)))
        self.weights = {}
        self._orig_forwards = {}

    def install(self):
        for layer_idx in self.layers:
            attn = self.model.transformer.blocks[layer_idx].attn
            self._orig_forwards[layer_idx] = attn.forward
            attn.forward = self._make_capturing_forward(attn, layer_idx)

    def _make_capturing_forward(self, attn_mod, layer_idx):
        storage = self.weights

        def capturing_forward(x, seq_id):
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

            d_head = attn_mod.d_head
            attn_logits = torch.matmul(
                query_BHLD, key_BHLD.transpose(-2, -1)
            ) / (d_head ** 0.5)

            if seq_id is not None:
                mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
                mask_BHLL = mask_BLL.unsqueeze(1)
                attn_logits = attn_logits.masked_fill(~mask_BHLL, float("-inf"))

            attn_w = torch.softmax(attn_logits, dim=-1)
            storage[layer_idx] = attn_w.detach().float().cpu()

            context = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD,
                attn_mask=mask_BHLL if seq_id is not None else None,
            )
            context = einops.rearrange(context, "b h s d -> b s (h d)")
            return attn_mod.out_proj(context)

        return capturing_forward

    def uninstall(self):
        for layer_idx, orig in self._orig_forwards.items():
            self.model.transformer.blocks[layer_idx].attn.forward = orig
        self._orig_forwards.clear()
        self.weights.clear()


def extract_esm3_attention(model, tokenizers, sequences, device, log):
    """Extract attention weights from ESM-3 via monkey-patching."""
    n_layers = len(model.transformer.blocks)
    n_heads = model.transformer.blocks[0].attn.n_heads
    log.info(f"ESM-3: {n_layers} layers, {n_heads} heads")

    capture = ESM3AttentionCapture(model, layers=list(range(n_layers)))
    capture.install()

    head_entropy = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    head_local_frac = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    head_max_attn = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    try:
        for idx, (acc, seq) in enumerate(sequences.items()):
            t0 = time.time()
            seq_tokens = tokenizers.sequence.encode(seq)
            seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)

            capture.weights.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(sequence_tokens=seq_tensor)

            for layer_idx in range(n_layers):
                if layer_idx not in capture.weights:
                    continue
                attn = capture.weights[layer_idx][0].numpy()  # (H, L, L)
                L = attn.shape[-1]
                for h in range(n_heads):
                    w = attn[h]
                    for qi in range(1, L - 1):
                        row = w[qi]
                        p = row[row > 0]
                        if len(p) > 0:
                            head_entropy[layer_idx][h].append(float(-np.sum(p * np.log(p))))
                    diag_mask = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) <= 5
                    local_frac = float(w[diag_mask].sum() / (w.sum() + 1e-10))
                    head_local_frac[layer_idx][h].append(local_frac)
                    head_max_attn[layer_idx][h].append(float(w[1:-1, :].max()))

            if (idx + 1) % 20 == 0 or idx == 0:
                log.info(f"  [{idx+1}/{len(sequences)}] {acc} ({len(seq)} aa) {time.time()-t0:.1f}s")
    finally:
        capture.uninstall()

    results = {
        "model": "esm3",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_proteins": len(sequences),
        "per_head": {}
    }
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            key = f"L{layer_idx}_H{h}"
            results["per_head"][key] = {
                "entropy": float(np.mean(head_entropy[layer_idx][h])) if head_entropy[layer_idx][h] else 0.0,
                "local_fraction": float(np.mean(head_local_frac[layer_idx][h])) if head_local_frac[layer_idx][h] else 0.0,
                "max_attention": float(np.mean(head_max_attn[layer_idx][h])) if head_max_attn[layer_idx][h] else 0.0,
            }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Attention Analysis: {args.model} ===")

    sequences = load_protein_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        results = extract_esm2_attention(model, tokenizer, sequences, args.device, log)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        results = extract_esm3_attention(model, tokenizers, sequences, args.device, log)

    out_path = out_dir / "attention_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
