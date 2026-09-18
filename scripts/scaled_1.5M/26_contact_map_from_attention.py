#!/usr/bin/env python3
"""Contact map prediction from attention: do structure-responsive heads predict 3D contacts?

Uses attention weights from the attention atlas. Computes APC-corrected contact
precision at L/5, L/2, L thresholds for structure-responsive vs non-responsive heads.

Usage:
    ./env/bin/python scripts/scaled_1.5M/26_contact_map_from_attention.py
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("contact_attn")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

N_PROTEINS = 500
CONTACT_THRESHOLD = 8.0  # Angstroms (Cb-Cb distance)
MIN_SEQ_SEP = 6  # minimum sequence separation for contacts
SEED = 42


def extract_cb_coords(pdb_path):
    """Extract Cb coordinates from PDB (Ca for glycine)."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                res_idx = int(line[22:26].strip()) - 1  # 0-indexed
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if atom_name == "CB":
                    coords[res_idx] = np.array([x, y, z])
                elif atom_name == "CA" and res_idx not in coords:
                    coords[res_idx] = np.array([x, y, z])
    return coords


def compute_contact_map(coords, L, threshold=CONTACT_THRESHOLD, min_sep=MIN_SEQ_SEP):
    """Compute binary contact map from coordinates."""
    contact_map = np.zeros((L, L), dtype=bool)
    for i in range(L):
        for j in range(i + min_sep, L):
            if i in coords and j in coords:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < threshold:
                    contact_map[i, j] = True
                    contact_map[j, i] = True
    return contact_map


def apc_correction(matrix):
    """Average Product Correction for contact prediction."""
    row_mean = matrix.mean(axis=1, keepdims=True)
    col_mean = matrix.mean(axis=0, keepdims=True)
    global_mean = matrix.mean()
    if global_mean == 0:
        return matrix
    corrected = matrix - (row_mean * col_mean) / global_mean
    return corrected


def contact_precision(pred_matrix, contact_map, L, min_sep=MIN_SEQ_SEP):
    """Compute contact precision at L/5, L/2, L thresholds."""
    # Mask for valid positions (sequence separation >= min_sep)
    mask = np.zeros_like(contact_map, dtype=bool)
    for i in range(L):
        for j in range(i + min_sep, L):
            mask[i, j] = True

    # Apply APC
    pred_apc = apc_correction(pred_matrix)

    # Get upper triangle scores
    scores = pred_apc[mask]
    labels = contact_map[mask]

    if labels.sum() == 0 or len(scores) == 0:
        return {}

    # Sort by predicted score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    results = {}
    for name, k in [("L/5", max(L // 5, 1)), ("L/2", max(L // 2, 1)), ("L", L)]:
        k = min(k, len(sorted_labels))
        precision = sorted_labels[:k].mean()
        results[name] = float(precision)

    return results


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from models.esm3_hooks import load_esm3
    from esm.sdk.api import ESMProtein
    import functools, einops
    import torch.nn.functional as F

    # Load attention atlas results to identify responsive heads
    atlas_path = OUTPUT_DIR / "attention_atlas.json"
    responsive_heads = set()
    if atlas_path.exists():
        with open(atlas_path) as f:
            atlas = json.load(f)
        jsd_matrix = np.array(atlas.get("mean_jsd_matrix", []))
        if jsd_matrix.size > 0:
            for L in range(jsd_matrix.shape[0]):
                for h in range(jsd_matrix.shape[1]):
                    if jsd_matrix[L, h] > 0.1:
                        responsive_heads.add((L, h))
    log.info(f"Structure-responsive heads: {len(responsive_heads)}")

    with open(STRUCT_DIR / "manifest.json") as f:
        available = json.load(f)["available"]
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    rng = np.random.RandomState(SEED)
    candidates = [acc for acc in sorted(available.keys())
                  if acc in sequences and 80 <= len(sequences[acc]) <= 300]
    selected = rng.choice(candidates, min(N_PROTEINS, len(candidates)), replace=False).tolist()
    log.info(f"Selected {len(selected)} proteins (length 80-300)")

    # Load ESM-3
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=args.device)

    # Set up attention capture (same pattern as attention atlas)
    def make_forward(attn_mod, lidx, store):
        def fwd(x, seq_id=None):
            qkv = attn_mod.layernorm_qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = attn_mod.q_ln(q).to(q.dtype)
            k = attn_mod.k_ln(k).to(k.dtype)
            q, k = attn_mod._apply_rotary(q, k)
            reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=attn_mod.n_heads)
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

    log.info("ESM-3 loaded")

    # Results
    all_head_precision = defaultdict(lambda: {"L/5": [], "L/2": [], "L": []})
    responsive_precision = {"L/5": [], "L/2": [], "L": []}
    nonresponsive_precision = {"L/5": [], "L/2": [], "L": []}

    # For supervised combination (Rao protocol): collect features and labels
    supervised_features = []  # list of (n_pairs, n_heads*n_layers) per protein
    supervised_labels = []    # list of (n_pairs,) binary contacts

    n_done = 0
    start_time = time.time()

    for acc in selected:
        pdb_path = available[acc]

        try:
            # Get true contacts
            coords = extract_cb_coords(pdb_path)
            seq = sequences[acc]
            L = len(seq)
            true_contacts = compute_contact_map(coords, L)
            n_contacts = true_contacts.sum() // 2

            if n_contacts < 10:
                continue

            # Get S-only attention (to measure intrinsic contact prediction)
            protein_s = ESMProtein(sequence=seq)
            tokens = model.encode(protein_s)

            # Install attention capture
            storage = {}
            originals = {}
            for i, block in enumerate(model.transformer.blocks):
                originals[i] = block.attn.forward
                block.attn.forward = make_forward(block.attn, i, storage)

            with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                model(sequence_tokens=tokens.sequence.to(args.device).unsqueeze(0))

            # Restore
            for i, block in enumerate(model.transformer.blocks):
                block.attn.forward = originals[i]

            # Compute contact precision per head
            for layer_idx, attn_weights in storage.items():
                # attn_weights: (n_heads, L+2, L+2) — remove BOS/EOS
                attn = attn_weights[:, 1:L+1, 1:L+1].numpy()

                for h in range(24):
                    # Symmetrize
                    attn_sym = (attn[h] + attn[h].T) / 2
                    prec = contact_precision(attn_sym, true_contacts, L)

                    for k, v in prec.items():
                        all_head_precision[(layer_idx, h)][k].append(v)

                        if (layer_idx, h) in responsive_heads:
                            responsive_precision[k].append(v)
                        else:
                            nonresponsive_precision[k].append(v)

            # Supervised combination: build feature matrix for this protein
            # Vectorized: precompute all APC-corrected attention maps, then index
            n_layers_stored = len(storage)
            if n_layers_stored > 0:
                # Stack all APC-corrected attention maps: (n_layers*n_heads, L, L)
                all_apc = []
                for layer_idx in sorted(storage.keys()):
                    attn = storage[layer_idx][:, 1:L+1, 1:L+1].numpy()  # (24, L, L)
                    for h in range(24):
                        attn_sym = (attn[h] + attn[h].T) / 2
                        all_apc.append(apc_correction(attn_sym))
                all_apc = np.stack(all_apc)  # (n_features, L, L)

                # Extract upper triangle pairs with min_sep
                pairs_i, pairs_j = [], []
                for i in range(L):
                    for j in range(i + MIN_SEQ_SEP, L):
                        pairs_i.append(i)
                        pairs_j.append(j)
                pairs_i = np.array(pairs_i)
                pairs_j = np.array(pairs_j)

                if len(pairs_i) > 0:
                    # Vectorized feature extraction: (n_pairs, n_features)
                    pair_feats = all_apc[:, pairs_i, pairs_j].T.astype(np.float32)
                    pair_labs = true_contacts[pairs_i, pairs_j].astype(np.int32)

                    supervised_features.append(pair_feats)
                    supervised_labels.append(pair_labs)

            n_done += 1
            torch.cuda.empty_cache()

        except Exception as e:
            if n_done < 3:
                log.warning(f"  Error on {acc}: {e}")
            continue

        if n_done % 50 == 0:
            elapsed = time.time() - start_time
            log.info(f"  {n_done}/{len(selected)} | Rate: {n_done/elapsed:.1f}/s")

    log.info(f"\nProcessed {n_done} proteins")

    # Supervised linear combination (Rao et al. protocol)
    supervised_result = {}
    if supervised_features:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score

        X_all = np.concatenate(supervised_features, axis=0)
        y_all = np.concatenate(supervised_labels, axis=0)
        log.info(f"\nSupervised combination: {X_all.shape[0]} pairs, "
                 f"{y_all.sum()} contacts ({100*y_all.mean():.1f}%)")

        # Train/test split (first 70% proteins train, rest test)
        n_train_proteins = int(0.7 * len(supervised_features))
        train_sizes = [len(f) for f in supervised_features[:n_train_proteins]]
        n_train_pairs = sum(train_sizes)

        X_train = X_all[:n_train_pairs]
        y_train = y_all[:n_train_pairs]
        X_test = X_all[n_train_pairs:]
        y_test = y_all[n_train_pairs:]

        # Subsample training pairs to avoid liblinear int overflow (max ~500K)
        MAX_TRAIN_PAIRS = 500_000
        if len(X_train) > MAX_TRAIN_PAIRS:
            rng_sub = np.random.RandomState(SEED)
            sub_idx = rng_sub.choice(len(X_train), MAX_TRAIN_PAIRS, replace=False)
            sub_idx.sort()
            X_train = X_train[sub_idx]
            y_train = y_train[sub_idx]
            log.info(f"  Subsampled training to {MAX_TRAIN_PAIRS} pairs")

        if y_train.sum() > 10 and y_test.sum() > 10:
            clf = LogisticRegression(penalty="l1", C=0.01, solver="liblinear",
                                     max_iter=300, class_weight="balanced", random_state=42)
            clf.fit(X_train, y_train)
            test_scores = clf.decision_function(X_test)

            # Precision at L/5 and L for test proteins
            for name, k_frac in [("L/5", 0.2), ("L/2", 0.5), ("L", 1.0)]:
                # Average across test proteins
                precisions = []
                offset = 0
                for pi in range(n_train_proteins, len(supervised_features)):
                    n_pairs = len(supervised_features[pi])
                    scores_pi = test_scores[offset:offset + n_pairs]
                    labels_pi = y_test[offset:offset + n_pairs]
                    offset += n_pairs

                    # Estimate L from n_pairs ≈ L*(L-1)/2
                    L_est = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
                    k = max(int(L_est * k_frac), 1)
                    k = min(k, len(scores_pi))

                    top_idx = np.argsort(-scores_pi)[:k]
                    prec = labels_pi[top_idx].mean()
                    precisions.append(prec)

                mean_prec = np.mean(precisions)
                supervised_result[name] = float(mean_prec)
                log.info(f"  Supervised @{name}: precision={mean_prec:.3f}")

            # Which heads have highest weights?
            coef = clf.coef_[0]  # (n_layers * n_heads,)
            n_total_heads = len(coef)
            head_weights = []
            idx = 0
            for layer_idx in sorted(storage.keys()):
                for h in range(24):
                    if idx < n_total_heads:
                        head_weights.append({
                            "layer": int(layer_idx), "head": h,
                            "weight": float(coef[idx]),
                            "is_responsive": (int(layer_idx), h) in responsive_heads,
                        })
                    idx += 1
            head_weights.sort(key=lambda x: -abs(x["weight"]))
            supervised_result["top_weighted_heads"] = head_weights[:20]
        else:
            log.warning("  Insufficient contacts for supervised training")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("CONTACT MAP FROM ATTENTION")
    log.info(f"{'='*60}")

    for k in ["L/5", "L/2", "L"]:
        resp = np.mean(responsive_precision[k]) if responsive_precision[k] else 0
        nonresp = np.mean(nonresponsive_precision[k]) if nonresponsive_precision[k] else 0
        log.info(f"  {k}: responsive={resp:.3f}, non-responsive={nonresp:.3f}")

    # Best heads
    best_heads = []
    for (L, h), precs in all_head_precision.items():
        mean_prec = np.mean(precs.get("L", [0]))
        is_resp = (L, h) in responsive_heads
        best_heads.append({"layer": L, "head": h, "precision_L": float(mean_prec),
                          "is_responsive": is_resp})
    best_heads.sort(key=lambda x: -x["precision_L"])

    log.info(f"\nTop 10 heads for contact prediction:")
    for bh in best_heads[:10]:
        resp_str = " *" if bh["is_responsive"] else ""
        log.info(f"  L{bh['layer']:2d} H{bh['head']:2d}: precision@L = {bh['precision_L']:.3f}{resp_str}")

    output = {
        "n_proteins": n_done,
        "n_responsive_heads": len(responsive_heads),
        "precision_summary": {
            k: {
                "responsive": float(np.mean(responsive_precision[k])) if responsive_precision[k] else None,
                "non_responsive": float(np.mean(nonresponsive_precision[k])) if nonresponsive_precision[k] else None,
            } for k in ["L/5", "L/2", "L"]
        },
        "top_heads": best_heads[:50],
        "supervised_combination": supervised_result,
    }

    out_path = OUTPUT_DIR / "contact_map_from_attention.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
