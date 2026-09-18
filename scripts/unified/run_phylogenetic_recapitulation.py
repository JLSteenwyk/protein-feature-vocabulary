#!/usr/bin/env python3
"""Phylogenetic recapitulation analysis for ESM-2 and ESM-3.

Tests whether model representations recapitulate evolutionary relationships
by comparing representation-based distance matrices with sequence-based
distance matrices across protein families.

For each protein family:
  1. Compute pairwise sequence distances (k-mer based)
  2. Extract model representations at multiple layers
  3. Compute pairwise representation distances (cosine)
  4. Correlate distance matrices (Mantel test — Spearman)
  5. Build dendrograms and compare topology (cophenetic correlation)

Run as:
    ./env/bin/python scripts/unified/run_phylogenetic_recapitulation.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_phylogenetic_recapitulation.py --model esm3 --device cuda:1

Output: results/unified/{model}/phylogenetic_recapitulation.json
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.cluster.hierarchy import linkage, cophenet


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "phylogenetic_recapitulation_log.txt", mode='w'),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("phylo"), out_dir


def kmer_vector(seq, k=3):
    """Compute normalized k-mer frequency vector for a sequence."""
    from itertools import product
    AAs = "ACDEFGHIKLMNPQRSTVWY"
    # Use 3-mer frequencies as feature vector
    kmer_to_idx = {}
    idx = 0
    for kmer in product(AAs, repeat=k):
        kmer_to_idx["".join(kmer)] = idx
        idx += 1

    vec = np.zeros(len(kmer_to_idx))
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_to_idx:
            vec[kmer_to_idx[kmer]] += 1

    # Normalize
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def sequence_distance_matrix(sequences, k=3):
    """Compute pairwise sequence distances using k-mer profiles."""
    vecs = [kmer_vector(seq, k=k) for seq in sequences]
    vecs = np.array(vecs)
    # Cosine distance between k-mer frequency vectors
    dists = pdist(vecs, metric='cosine')
    return dists


def representation_distance_matrix(representations):
    """Compute pairwise cosine distances between mean-pooled representations."""
    # representations: list of (L, d_model) arrays -> mean pool to (d_model,)
    pooled = []
    for rep in representations:
        pooled.append(rep.mean(axis=0))
    pooled = np.array(pooled)
    dists = pdist(pooled, metric='cosine')
    return dists


def mantel_test(dist_a, dist_b, n_permutations=999):
    """Mantel test: correlation between two distance matrices.

    Returns Spearman rho and permutation p-value.
    """
    rho, _ = scipy_stats.spearmanr(dist_a, dist_b)

    # Permutation test
    n = int(np.ceil(np.sqrt(2 * len(dist_a))))  # recover n from condensed form
    # Actually use exact formula: n*(n-1)/2 = len
    n = int(0.5 + np.sqrt(0.25 + 2 * len(dist_a)))

    count = 0
    sq_b = squareform(dist_b)
    for _ in range(n_permutations):
        perm = np.random.permutation(n)
        perm_b = sq_b[np.ix_(perm, perm)]
        perm_b_condensed = squareform(perm_b)
        perm_rho, _ = scipy_stats.spearmanr(dist_a, perm_b_condensed)
        if perm_rho >= rho:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return float(rho), float(p_value)


def select_families(metadata, sequences, min_members=8, max_members=25,
                    min_organisms=2, max_families=15):
    """Select protein families suitable for phylogenetic analysis."""
    pfam_counts = Counter()
    pfam_members = {}
    for acc, p in metadata.items():
        if acc not in sequences:
            continue
        for pf in p.get('pfam', []):
            pfam_counts[pf] += 1
            if pf not in pfam_members:
                pfam_members[pf] = []
            pfam_members[pf].append(acc)

    families = []
    for pf, count in pfam_counts.most_common():
        if count < min_members or count > max_members:
            continue
        members = pfam_members[pf]
        organisms = set(metadata[a].get('organism', '') for a in members)
        if len(organisms) < min_organisms:
            continue
        families.append({
            'pfam_id': pf,
            'members': members,
            'n_members': len(members),
            'n_organisms': len(organisms),
            'organisms': sorted(organisms),
        })
        if len(families) >= max_families:
            break

    return families


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-families", type=int, default=15)
    parser.add_argument("--max-length", type=int, default=500)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Phylogenetic Recapitulation: {args.model} ===")
    device = args.device

    # Load data
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        all_seqs = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    # Filter to manageable lengths
    seqs = {k: v for k, v in all_seqs.items() if len(v) <= args.max_length}
    log.info(f"Loaded {len(seqs)} sequences (≤{args.max_length} aa)")

    # Select families
    families = select_families(metadata, seqs, max_families=args.max_families)
    log.info(f"Selected {len(families)} protein families:")
    for fam in families:
        log.info(f"  {fam['pfam_id']}: {fam['n_members']} members, "
                 f"{fam['n_organisms']} organisms")

    # Load model
    if args.model == "esm2":
        from models.esm2_hooks import load_esm2, ESM2HookManager
        model, tokenizer = load_esm2(device=device)
        n_layers = 33
        target_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    else:
        from models.esm3_hooks import load_esm3, ESM3HookManager
        model, tokenizers = load_esm3(device=device)
        n_layers = 48
        target_layers = [0, 6, 12, 18, 24, 30, 36, 42, 47]

    log.info(f"Model loaded. Analyzing layers: {target_layers}")

    # Process each family
    per_family = {}
    all_mantel_by_layer = {str(l): [] for l in target_layers}
    all_cophenet_by_layer = {str(l): [] for l in target_layers}

    t0 = time.time()

    for fi, fam in enumerate(families):
        pfam_id = fam['pfam_id']
        members = [a for a in fam['members'] if a in seqs]
        if len(members) < 5:
            log.warning(f"  {pfam_id}: only {len(members)} members after filtering, skipping")
            continue

        log.info(f"\n=== [{fi+1}/{len(families)}] {pfam_id} ({len(members)} members) ===")

        family_seqs = [seqs[a] for a in members]

        # Compute sequence distance matrix (k-mer based)
        seq_dists = sequence_distance_matrix(family_seqs, k=3)
        log.info(f"  Sequence distances: mean={np.mean(seq_dists):.3f}, "
                 f"range=[{np.min(seq_dists):.3f}, {np.max(seq_dists):.3f}]")

        # Check for zero variance
        if np.std(seq_dists) < 1e-10:
            log.warning(f"  {pfam_id}: zero variance in sequence distances, skipping")
            continue

        # Extract representations at each target layer
        layer_results = {}
        try:
            for layer in target_layers:
                if args.model == "esm2":
                    hook_mgr = ESM2HookManager(model, layers=[layer],
                                                extract_attention=False)
                else:
                    hook_mgr = ESM3HookManager(model, layers=[layer],
                                                extract_attention=False)

                representations = []
                for acc in members:
                    seq = seqs[acc]
                    hook_mgr.cache.clear()
                    hook_mgr.register()

                    if args.model == "esm2":
                        inputs = tokenizer(seq, return_tensors="pt",
                                           truncation=True, max_length=1024).to(device)
                        with torch.no_grad():
                            model(**inputs)
                    else:
                        seq_tokens = tokenizers.sequence.encode(seq)
                        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                                  device=device).unsqueeze(0)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            model(sequence_tokens=seq_tensor)

                    hook_mgr.remove()

                    # Get residual stream, strip BOS/EOS
                    h = hook_mgr.cache.residual_stream[layer][0][1:-1].float().cpu().numpy()
                    representations.append(h)

                # Compute representation distances
                rep_dists = representation_distance_matrix(representations)

                if np.std(rep_dists) < 1e-10:
                    log.warning(f"  Layer {layer}: zero variance in rep distances")
                    layer_results[str(layer)] = {
                        "mantel_rho": 0.0, "mantel_p": 1.0,
                        "cophenetic_corr": 0.0,
                    }
                    continue

                # Mantel test
                rho, p_val = mantel_test(seq_dists, rep_dists, n_permutations=999)

                # Cophenetic correlation (how well dendrograms match)
                seq_linkage = linkage(seq_dists, method='average')
                rep_linkage = linkage(rep_dists, method='average')
                _, seq_coph = cophenet(seq_linkage, seq_dists)
                _, rep_coph = cophenet(rep_linkage, rep_dists)
                coph_corr, _ = scipy_stats.spearmanr(seq_coph, rep_coph)

                layer_results[str(layer)] = {
                    "mantel_rho": float(rho),
                    "mantel_p": float(p_val),
                    "cophenetic_corr": float(coph_corr) if not np.isnan(coph_corr) else 0.0,
                    "mean_rep_dist": float(np.mean(rep_dists)),
                }

                all_mantel_by_layer[str(layer)].append(rho)
                all_cophenet_by_layer[str(layer)].append(
                    float(coph_corr) if not np.isnan(coph_corr) else 0.0)

                log.info(f"  Layer {layer}: Mantel ρ={rho:.3f} (p={p_val:.3f}), "
                         f"cophenetic r={coph_corr:.3f}")

        except Exception as e:
            log.warning(f"  {pfam_id}: FAILED — {e}")
            torch.cuda.empty_cache()
            continue

        # Best layer
        best_layer = max(layer_results.keys(),
                         key=lambda l: layer_results[l]["mantel_rho"])
        best_rho = layer_results[best_layer]["mantel_rho"]

        per_family[pfam_id] = {
            "n_members": len(members),
            "n_organisms": fam['n_organisms'],
            "organisms": fam['organisms'],
            "mean_seq_dist": float(np.mean(seq_dists)),
            "best_layer": int(best_layer),
            "best_mantel_rho": float(best_rho),
            "per_layer": layer_results,
        }

        log.info(f"  Best: layer {best_layer}, Mantel ρ={best_rho:.3f}")
        log.info(f"  [{fi+1}/{len(families)} families, {time.time()-t0:.0f}s]")

    # ================================================================
    # Aggregate
    # ================================================================
    log.info(f"\n=== Aggregate Results ({len(per_family)} families) ===")

    aggregate_by_layer = {}
    for layer in target_layers:
        lk = str(layer)
        mantels = all_mantel_by_layer[lk]
        cophenets = all_cophenet_by_layer[lk]
        if mantels:
            aggregate_by_layer[lk] = {
                "mean_mantel_rho": float(np.mean(mantels)),
                "std_mantel_rho": float(np.std(mantels)),
                "median_mantel_rho": float(np.median(mantels)),
                "mean_cophenetic": float(np.mean(cophenets)),
                "n_families": len(mantels),
                "n_significant": int(sum(1 for r in mantels if r > 0.3)),
            }
            log.info(f"  Layer {layer}: mean Mantel ρ={np.mean(mantels):.3f} ± "
                     f"{np.std(mantels):.3f}, cophenetic={np.mean(cophenets):.3f}")

    # Best layer overall
    if aggregate_by_layer:
        best_layer = max(aggregate_by_layer.keys(),
                         key=lambda l: aggregate_by_layer[l]["mean_mantel_rho"])
        log.info(f"\n  Best layer overall: {best_layer} "
                 f"(mean ρ={aggregate_by_layer[best_layer]['mean_mantel_rho']:.3f})")

    output = {
        "model": args.model,
        "n_families": len(per_family),
        "target_layers": target_layers,
        "n_layers": n_layers,
        "aggregate_by_layer": aggregate_by_layer,
        "per_family": per_family,
    }

    out_path = out_dir / "phylogenetic_recapitulation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
