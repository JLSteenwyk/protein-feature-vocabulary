#!/usr/bin/env python3
"""Contact map prediction from attention weights for ESM-2 and ESM-3.

Extracts attention matrices and evaluates contact prediction precision:
- Symmetrize attention, apply APC correction (Rao et al. 2021)
- Compare against ground-truth contacts from AlphaFold structures
- Evaluate per-layer and per-head, plus best-layer ensemble
- For ESM-3: compare S-only vs S+St attention contact maps

Run as:
    ./env/bin/python scripts/unified/run_contact_map.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_contact_map.py --model esm3 --device cuda:1

Output: results/unified/{model}/contact_map.json
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
import numpy as np

from models.metrics import contact_precision_from_attention


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "contact_map_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("contact_map"), out_dir


def load_sequences_with_structures(max_proteins=500, max_length=500):
    """Load sequences that have AlphaFold structures available."""
    seq_file = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    struct_map_file = ROOT / "data" / "scaled" / "structure_map.json"

    with open(seq_file) as f:
        all_seqs = json.load(f)

    struct_accs = set()
    if struct_map_file.exists():
        with open(struct_map_file) as f:
            smap = json.load(f)
        if "available" in smap:
            struct_accs = set(smap["available"])
        else:
            struct_accs = set(smap.keys())

    filtered = {k: v for k, v in all_seqs.items()
                if len(v) <= max_length and k in struct_accs}
    sorted_accs = sorted(filtered.keys())[:max_proteins]
    return {k: filtered[k] for k in sorted_accs}


def load_contact_map(acc, seq_length, contact_threshold=8.0):
    """Load contact map from AlphaFold structure or PDB file.

    Uses Cb-Cb distance < threshold (Ca for Gly).
    """
    pdb_dir = ROOT / "data" / "scaled" / "structures"
    pdb_file = pdb_dir / f"{acc}.pdb"
    cif_file = pdb_dir / f"{acc}.cif"

    coords = None

    # Try PDB format
    for path in [pdb_file, cif_file]:
        if not path.exists():
            continue
        try:
            coords = _parse_cb_coords(path, seq_length)
            break
        except Exception:
            continue

    if coords is None:
        return None

    # Compute distance matrix
    dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    contact_map = (dist < contact_threshold).astype(np.float32)
    np.fill_diagonal(contact_map, 0)
    return contact_map


def _parse_cb_coords(pdb_path, seq_length):
    """Parse Cb coordinates from PDB/CIF file (Ca for Gly)."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                res_seq = int(line[22:26].strip())
                res_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                if atom_name == "CB" or (atom_name == "CA" and res_name == "GLY"):
                    coords[res_seq] = np.array([x, y, z])

    if not coords:
        return None

    # Build coordinate array aligned to sequence
    result = np.zeros((seq_length, 3))
    for i in range(seq_length):
        res_num = i + 1
        if res_num in coords:
            result[i] = coords[res_num]
        else:
            result[i] = np.nan

    return result


def extract_attention_esm2(model, tokenizer, seq, device):
    """Extract attention weights from ESM-2."""
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    # out.attentions is a tuple of (B, n_heads, L, L) tensors per layer
    n_layers = len(out.attentions)
    # Remove BOS/EOS tokens
    attn_per_layer = {}
    for layer_idx in range(n_layers):
        attn = out.attentions[layer_idx][0].cpu().numpy()  # (n_heads, L_full, L_full)
        attn = attn[:, 1:-1, 1:-1]  # Remove special tokens
        attn_per_layer[layer_idx] = attn
    return attn_per_layer


def extract_attention_esm3(model, tokenizers, seq, device):
    """Extract attention weights from ESM-3 via monkey-patching."""
    import functools
    import einops

    seq_tokens = tokenizers.sequence.encode(seq)
    seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)

    attn_per_layer = {}
    n_layers = len(model.transformer.blocks)

    for layer_idx in range(n_layers):
        block = model.transformer.blocks[layer_idx]
        attn = block.attn
        original_forward = attn.forward

        captured = {}

        def patched_forward(x, seq_id, _orig=original_forward, _attn=attn, _cap=captured):
            qkv_BLD3 = _attn.layernorm_qkv(x)
            query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
            query_BLD, key_BLD = (
                _attn.q_ln(query_BLD).to(query_BLD.dtype),
                _attn.k_ln(key_BLD).to(query_BLD.dtype),
            )
            query_BLD, key_BLD = _attn._apply_rotary(query_BLD, key_BLD)

            reshaper = functools.partial(
                einops.rearrange, pattern="b s (h d) -> b h s d", h=_attn.n_heads
            )
            query_BHLD, key_BHLD, value_BHLD = map(reshaper, (query_BLD, key_BLD, value_BLD))

            # Compute attention weights manually
            scale = query_BHLD.size(-1) ** -0.5
            scores = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * scale
            if seq_id is not None:
                mask = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
                scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            weights = torch.softmax(scores, dim=-1)
            _cap['weights'] = weights.detach().cpu()

            # Continue with normal forward
            context_BHLD = torch.matmul(weights, value_BHLD)
            context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
            return _attn.out_proj(context_BLD)

        attn.forward = patched_forward
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(sequence_tokens=seq_tensor)
        finally:
            attn.forward = original_forward

        if 'weights' in captured:
            w = captured['weights'][0].float().numpy()  # (n_heads, L_full, L_full)
            w = w[:, 1:-1, 1:-1]  # Remove BOS/EOS
            attn_per_layer[layer_idx] = w

    return attn_per_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--contact-threshold", type=float, default=8.0)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Contact Map from Attention: {args.model} ===")

    sequences = load_sequences_with_structures(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins with structures")

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        n_layers = len(model.esm.encoder.layer)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        n_layers = len(model.transformer.blocks)

    log.info(f"{args.model}: {n_layers} layers")

    # Evaluate subset of layers (every 4th + first/last)
    eval_layers = sorted(set([0, n_layers - 1] + list(range(0, n_layers, 4))))

    per_layer_precisions = {l: {"L": [], "L/2": [], "L/5": []} for l in eval_layers}
    n_evaluated = 0

    t0 = time.time()
    for seq_idx, (acc, seq) in enumerate(sequences.items()):
        contact_map = load_contact_map(acc, len(seq), args.contact_threshold)
        if contact_map is None:
            continue

        if len(seq) < 20:
            continue

        if args.model == "esm2":
            attn_per_layer = extract_attention_esm2(model, tokenizer, seq, args.device)
        else:
            attn_per_layer = extract_attention_esm3(model, tokenizers, seq, args.device)

        for layer_idx in eval_layers:
            if layer_idx not in attn_per_layer:
                continue
            attn = attn_per_layer[layer_idx]
            L_attn = attn.shape[1]
            L_contact = contact_map.shape[0]
            L = min(L_attn, L_contact)
            if L < 20:
                continue

            prec = contact_precision_from_attention(
                attn[:, :L, :L], contact_map[:L, :L]
            )
            for k in prec:
                per_layer_precisions[layer_idx][k].append(prec[k])

        n_evaluated += 1
        if (n_evaluated) % 20 == 0:
            log.info(f"  [{n_evaluated} proteins evaluated, {time.time()-t0:.0f}s]")

    log.info(f"Evaluated {n_evaluated} proteins in {time.time()-t0:.1f}s")

    # Aggregate results
    results = {}
    best_layer = -1
    best_prec = 0.0
    for l in eval_layers:
        if not per_layer_precisions[l]["L"]:
            continue
        results[l] = {
            k: float(np.mean(v)) for k, v in per_layer_precisions[l].items() if v
        }
        if results[l].get("L", 0) > best_prec:
            best_prec = results[l]["L"]
            best_layer = l
        if l % max(1, n_layers // 8) == 0 or l == n_layers - 1:
            log.info(f"  Layer {l}: P@L={results[l].get('L', 0):.3f}, "
                     f"P@L/2={results[l].get('L/2', 0):.3f}, "
                     f"P@L/5={results[l].get('L/5', 0):.3f}")

    log.info(f"Best layer: {best_layer} with P@L={best_prec:.3f}")

    output = {
        "model": args.model,
        "n_layers": n_layers,
        "n_proteins": n_evaluated,
        "contact_threshold": args.contact_threshold,
        "eval_layers": eval_layers,
        "best_layer": best_layer,
        "best_precision_L": best_prec,
        "per_layer": {str(k): v for k, v in results.items()},
    }

    out_path = out_dir / "contact_map.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
