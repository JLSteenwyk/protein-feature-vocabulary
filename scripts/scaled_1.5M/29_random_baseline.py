#!/usr/bin/env python3
"""Random baseline / negative control for SAE features.

Compares SAE features against random projections for:
- GO enrichment fraction (using gokit ORA with BH-FDR, matching 10_go_enrichment.py)
- Functional site probing AUROC
- Feature-annotation correlation

Shows that SAE features are meaningfully structured, not spurious.

Usage:
    ./env/bin/python scripts/scaled_1.5M/29_random_baseline.py
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import h5py
from gokit.core.enrichment import OraRunner

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baseline")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

SEED = 42
N_RANDOM_SEEDS = 3  # multiple random seeds for error bars
MAX_RESIDUES_PROBE = 200_000  # for probing comparison
MAX_FEATURES_GO = 100  # top features for GO comparison (gokit ORA is thorough but slow)


def load_annotations():
    """Load functional site annotations for eval proteins."""
    for fname in ["annotations.json", "metadata.json"]:
        ann_path = EVAL_DIR / fname
        if ann_path.exists():
            with open(ann_path) as f:
                return json.load(f)
    return None


def build_functional_labels(protein_ids, offsets, annotations):
    """Build per-residue binary labels for functional sites."""
    n_total = offsets[-1][0] + offsets[-1][1] if len(offsets) > 0 else 0
    # Use the offset structure: offsets[i] = [start, length]
    labels = np.zeros(n_total, dtype=np.int8)

    for i, pid in enumerate(protein_ids):
        if pid not in annotations:
            continue
        ann = annotations[pid]
        features = ann.get("features", [])
        start, length = int(offsets[i][0]), int(offsets[i][1])

        for feat in features:
            feat_type = feat.get("type", "")
            if feat_type in ["Active site", "Binding site", "Metal binding",
                             "Site", "Disulfide bond", "Modified residue"]:
                # Handle both formats: flat (start/end) and nested (location.start.value)
                if "start" in feat:
                    pos_start = feat["start"] - 1  # 1-indexed to 0-indexed
                    pos_end = feat.get("end", pos_start + 1)
                elif "location" in feat:
                    loc = feat["location"]
                    pos_start = loc.get("start", {}).get("value", 1) - 1
                    pos_end = loc.get("end", {}).get("value", pos_start + 2)
                else:
                    continue
                for pos in range(max(0, pos_start), min(length, pos_end)):
                    labels[start + pos] = 1

    return labels


def random_projection_features(X_sparse, d_random, seed):
    """Create random sparse features matching SAE sparsity pattern.

    Instead of dense projection (OOM on 3M residues), randomly assign
    each residue's k=64 nonzero slots to random feature indices with
    random magnitudes drawn from the SAE activation distribution.
    """
    rng = np.random.RandomState(seed)
    n_residues = X_sparse.shape[0]

    # Match SAE's sparsity: k nonzeros per row, same magnitude distribution
    k_per_row = 64  # TopK SAE k value
    nnz_total = n_residues * k_per_row

    # Random feature indices (uniform over d_random features)
    col_indices = rng.randint(0, d_random, size=nnz_total)
    # Random magnitudes: sample from SAE activation distribution
    sae_vals = X_sparse.data
    mag_indices = rng.randint(0, len(sae_vals), size=nnz_total)
    values = sae_vals[mag_indices].astype(np.float32)

    # Row indices: k entries per row
    row_indices = np.repeat(np.arange(n_residues), k_per_row)

    result = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_residues, d_random)
    )
    return result


def shuffled_features(X_sparse, seed):
    """Create shuffled version: permute feature assignments across residues."""
    rng = np.random.RandomState(seed)
    X_csc = X_sparse.tocsc()
    n_residues = X_sparse.shape[0]

    # For each feature column, shuffle the row indices
    new_indices = X_csc.indices.copy()
    for j in range(X_csc.shape[1]):
        start = X_csc.indptr[j]
        end = X_csc.indptr[j + 1]
        if end > start:
            new_indices[start:end] = rng.permutation(new_indices[start:end])

    shuffled = sparse.csc_matrix(
        (X_csc.data.copy(), new_indices, X_csc.indptr.copy()),
        shape=X_csc.shape
    ).tocsr()
    return shuffled


def _safe_odds_ratio(study_count, study_n, pop_count, pop_n):
    """Compute odds ratio with denominator floor to avoid Inf (fixes M4).

    Uses enrichment ratio: (study_count/study_n) / (pop_count/pop_n)
    with a floor of 1e-6 on the denominator fraction.
    """
    study_frac = study_count / max(study_n, 1)
    pop_frac = pop_count / max(pop_n, 1)
    return study_frac / max(pop_frac, 1e-6)


# Namespace prefix mapping from metadata term strings ("F:", "P:", "C:")
_NS_PREFIX = {"F": "MF", "P": "BP", "C": "CC"}

# FDR threshold (matches 10_go_enrichment.py)
_FDR_THRESHOLD = 0.05


def _eval_single_feature(args):
    """Evaluate GO enrichment for a single feature. Top-level for pickling."""
    feat_idx, acts_bg, bg_pids_list, protein_to_go, go_to_namespace, background_pids = args

    if acts_bg.max() == 0:
        return None

    nonzero = acts_bg > 0
    if nonzero.sum() < 5:
        return (0, 1.0, 0.0)

    thresh = np.percentile(acts_bg[nonzero], 75)
    if thresh == 0:
        return (0, 1.0, 0.0)

    study_pids = set(
        pid for pid, act in zip(bg_pids_list, acts_bg) if act >= thresh
    )

    if len(study_pids) < 3:
        return (0, 1.0, 0.0)

    runner_local = OraRunner(
        population_genes=background_pids,
        gene_to_go=protein_to_go,
        go_to_namespace=go_to_namespace,
    )

    ora_results = runner_local.run_study(
        study_genes=study_pids,
        namespace_filter="all",
        method="fdr_bh",
        test_direction="over",
    )

    n_sig = 0
    best_odds = 1.0
    best_neglog_p = 0.0

    for r in ora_results:
        if r.p_adjusted < _FDR_THRESHOLD and r.study_count >= 2:
            n_sig += 1
            odds = _safe_odds_ratio(r.study_count, r.study_n, r.pop_count, r.pop_n)
            if odds > best_odds:
                best_odds = odds
            neglog_p = -np.log10(max(r.p_adjusted, 1e-300))
            if neglog_p > best_neglog_p:
                best_neglog_p = neglog_p

    return (n_sig, float(best_odds), float(best_neglog_p))


def evaluate_go_enrichment(X_features, protein_ids, offsets, annotations, n_top=MAX_FEATURES_GO):
    """Evaluate GO enrichment quality using gokit ORA with BH-FDR correction.

    Uses the same gokit OraRunner as 10_go_enrichment.py (fixes M1).
    Odds ratios capped via denominator floor (fixes M4).
    BH-FDR correction applied per feature (fixes C2 consistency).

    Measures effect size (odds ratio), specificity (n enriched terms per feature),
    and significance strength (-log10 adjusted p-value).
    """
    n_proteins = len(protein_ids)
    n_features = X_features.shape[1]

    # Build GO mappings: protein_id -> set of GO IDs, GO ID -> namespace
    protein_to_go = {}
    go_to_namespace = {}
    for pid in protein_ids:
        if pid in annotations:
            go_terms_raw = annotations[pid].get("go_terms", [])
            go_ids = set()
            for g in go_terms_raw:
                go_id = g.get("id", "")
                term_str = g.get("term", "")
                if go_id:
                    go_ids.add(go_id)
                    if go_id not in go_to_namespace and term_str:
                        prefix = term_str.split(":")[0]
                        go_to_namespace[go_id] = _NS_PREFIX.get(prefix, "BP")
            if go_ids:
                protein_to_go[pid] = go_ids

    if len(protein_to_go) < 100:
        return {"median_enriched_terms": 0, "n_tested": 0, "error": "too few GO annotations"}

    # Build OraRunner (population = proteins with GO annotations)
    background_pids = set(protein_to_go.keys())
    runner = OraRunner(
        population_genes=background_pids,
        gene_to_go=protein_to_go,
        go_to_namespace=go_to_namespace,
    )

    # Map protein index -> pid for background proteins
    bg_idx_map = {pid: i for i, pid in enumerate(protein_ids) if pid in background_pids}

    # Compute protein-level max activations
    prot_max = np.zeros((n_proteins, min(n_features, n_top)), dtype=np.float32)
    for i in range(n_proteins):
        start, length = int(offsets[i][0]), int(offsets[i][1])
        if length > 0:
            chunk = X_features[start:start + length, :n_top]
            if sparse.issparse(chunk):
                chunk = chunk.toarray()
            prot_max[i] = chunk.max(axis=0)

    # Per-feature metrics — computed in parallel for speed
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Indices and pids for background proteins
    bg_indices = sorted(bg_idx_map.values())
    bg_pids_list = [protein_ids[i] for i in bg_indices]

    # Build per-feature activation arrays
    feature_tasks = []
    for feat_idx in range(min(n_features, n_top)):
        acts_bg = prot_max[bg_indices, feat_idx]
        feature_tasks.append((feat_idx, acts_bg))

    # Run in parallel (use up to 16 workers)
    n_workers = min(16, len(feature_tasks))
    per_feature_n_enriched = []
    per_feature_best_odds = []
    per_feature_best_pval = []
    n_tested = 0

    log.info(f"  Running GO enrichment on {len(feature_tasks)} features ({n_workers} workers)...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for feat_idx, acts_bg in feature_tasks:
            fut = executor.submit(
                _eval_single_feature,
                (feat_idx, acts_bg, bg_pids_list, protein_to_go, go_to_namespace, background_pids)
            )
            futures[fut] = feat_idx

        done_count = 0
        for fut in as_completed(futures):
            result = fut.result()
            done_count += 1
            if done_count % 25 == 0:
                log.info(f"    {done_count}/{len(feature_tasks)} features done")
            if result is None:
                continue
            n_tested += 1
            per_feature_n_enriched.append(result[0])
            per_feature_best_odds.append(result[1])
            per_feature_best_pval.append(result[2])

    n_enriched_arr = np.array(per_feature_n_enriched) if per_feature_n_enriched else np.array([0])
    odds_arr = np.array(per_feature_best_odds) if per_feature_best_odds else np.array([1.0])
    pval_arr = np.array(per_feature_best_pval) if per_feature_best_pval else np.array([0.0])

    return {
        "n_tested": n_tested,
        "correction_method": "fdr_bh",
        "fdr_threshold": _FDR_THRESHOLD,
        "fraction_with_any_enrichment": float((n_enriched_arr > 0).mean()) if n_tested > 0 else 0,
        "median_enriched_terms_per_feature": float(np.median(n_enriched_arr)) if n_tested > 0 else 0,
        "mean_enriched_terms_per_feature": float(np.mean(n_enriched_arr)) if n_tested > 0 else 0,
        "median_best_odds_ratio": float(np.median(odds_arr)) if n_tested > 0 else 1.0,
        "mean_best_odds_ratio": float(np.mean(odds_arr)) if n_tested > 0 else 1.0,
        "median_best_neglog10_pval": float(np.median(pval_arr)) if n_tested > 0 else 0,
        "mean_best_neglog10_pval": float(np.mean(pval_arr)) if n_tested > 0 else 0,
        "frac_with_3plus_terms": float((n_enriched_arr >= 3).mean()) if n_tested > 0 else 0,
    }


def evaluate_probing(X_features, labels, protein_ids, offsets, max_residues=MAX_RESIDUES_PROBE):
    """Train functional site probe and measure AUROC.

    Uses protein-level train/test splits to avoid data leakage
    (residues from the same protein always stay in the same split).
    """
    n_proteins = len(protein_ids)
    n_train_prot = int(0.8 * n_proteins)

    # Shuffle protein indices
    rng = np.random.RandomState(SEED)
    prot_order = rng.permutation(n_proteins)
    train_prots = set(prot_order[:n_train_prot].tolist())
    test_prots = set(prot_order[n_train_prot:].tolist())

    # Gather residue indices for each split
    train_idx, test_idx = [], []
    for i in range(n_proteins):
        start, length = int(offsets[i][0]), int(offsets[i][1])
        indices = list(range(start, start + length))
        if i in train_prots:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)

    # Subsample train if needed
    if len(train_idx) > max_residues:
        train_idx = rng.choice(train_idx, max_residues, replace=False).tolist()
    if len(test_idx) > max_residues:
        test_idx = rng.choice(test_idx, max_residues, replace=False).tolist()

    X_train = X_features[train_idx]
    y_train = labels[train_idx]
    X_test = X_features[test_idx]
    y_test = labels[test_idx]

    if y_train.sum() < 20 or (y_train == 0).sum() < 20:
        return {"auroc": 0.5, "error": "insufficient train labels"}

    if y_test.sum() < 5:
        return {"auroc": 0.5, "error": "insufficient test positives"}

    clf = LogisticRegression(penalty="l1", C=0.01, solver="liblinear",
                             max_iter=200, class_weight="balanced", random_state=SEED)
    clf.fit(X_train, y_train)
    y_score = clf.decision_function(X_test)
    auroc = roc_auc_score(y_test, y_score)

    return {"auroc": float(auroc), "n_train": int(len(y_train)),
            "n_test": int(len(y_test)), "n_pos": int(y_test.sum()),
            "n_train_proteins": n_train_prot,
            "n_test_proteins": n_proteins - n_train_prot}


def main():
    annotations = load_annotations()
    if annotations is None:
        log.error("No annotations found")
        return

    results = {}
    out_path = OUTPUT_DIR / "random_baseline.json"

    def _save_incremental():
        """Save progress after each round so no work is lost."""
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  [checkpoint saved to {out_path.name}]")

    for model_name in ["esm3", "esm2"]:
        log.info(f"\n{'='*60}")
        log.info(f"  {model_name.upper()} RANDOM BASELINE")
        log.info(f"{'='*60}")

        sparse_path = FEATURE_ROOT / model_name / "residue" / "features_sparse.npz"
        summary_path = FEATURE_ROOT / model_name / "residue" / "protein_summaries.h5"

        if not sparse_path.exists():
            log.warning(f"  No features for {model_name}")
            continue

        log.info("Loading SAE features...")
        X_sae = sparse.load_npz(sparse_path)

        with h5py.File(summary_path, "r") as f:
            protein_ids = [s.decode() for s in f["protein_ids"][:]]
            offsets = f["offsets"][:]

        log.info(f"  {X_sae.shape[0]:,} residues, {X_sae.shape[1]:,} features")

        # Build labels
        log.info("Building functional site labels...")
        labels = build_functional_labels(protein_ids, offsets, annotations)
        n_pos = labels.sum()
        log.info(f"  {n_pos:,} positive residues ({100*n_pos/len(labels):.2f}%)")

        model_results = {"sae": {}, "random_projection": [], "shuffled": []}

        # 1. SAE features (real)
        log.info("\n--- SAE features (real) ---")
        sae_go = evaluate_go_enrichment(X_sae, protein_ids, offsets, annotations)
        sae_probe = evaluate_probing(X_sae, labels, protein_ids, offsets)
        model_results["sae"] = {
            "go_enrichment": sae_go,
            "probing": sae_probe,
        }
        log.info(f"  GO: median_terms={sae_go['median_enriched_terms_per_feature']:.1f}, "
                 f"median_OR={sae_go['median_best_odds_ratio']:.1f}, "
                 f"frac_3+={sae_go['frac_with_3plus_terms']:.3f}")
        log.info(f"  Probing AUROC: {sae_probe['auroc']:.4f}")
        results[model_name] = model_results
        _save_incremental()

        # 2. Random projection baselines
        for seed_i in range(N_RANDOM_SEEDS):
            log.info(f"\n--- Random projection (seed {seed_i}) ---")
            d_random = min(X_sae.shape[1], 5000)  # match reasonable feature count
            X_rand = random_projection_features(X_sae, d_random, seed=SEED + seed_i)
            rand_go = evaluate_go_enrichment(X_rand, protein_ids, offsets, annotations,
                                             n_top=min(d_random, MAX_FEATURES_GO))
            rand_probe = evaluate_probing(X_rand, labels, protein_ids, offsets)
            model_results["random_projection"].append({
                "seed": seed_i,
                "d_random": d_random,
                "go_enrichment": rand_go,
                "probing": rand_probe,
            })
            log.info(f"  GO: median_terms={rand_go['median_enriched_terms_per_feature']:.1f}, "
                     f"median_OR={rand_go['median_best_odds_ratio']:.1f}")
            log.info(f"  Probing AUROC: {rand_probe['auroc']:.4f}")
            results[model_name] = model_results
            _save_incremental()

        # 3. Shuffled features (permutation test)
        for seed_i in range(N_RANDOM_SEEDS):
            log.info(f"\n--- Shuffled features (seed {seed_i}) ---")
            X_shuf = shuffled_features(X_sae, seed=SEED + 100 + seed_i)
            shuf_go = evaluate_go_enrichment(X_shuf, protein_ids, offsets, annotations)
            shuf_probe = evaluate_probing(X_shuf, labels, protein_ids, offsets)
            model_results["shuffled"].append({
                "seed": seed_i,
                "go_enrichment": shuf_go,
                "probing": shuf_probe,
            })
            log.info(f"  GO: median_terms={shuf_go['median_enriched_terms_per_feature']:.1f}, "
                     f"median_OR={shuf_go['median_best_odds_ratio']:.1f}")
            log.info(f"  Probing AUROC: {shuf_probe['auroc']:.4f}")
            results[model_name] = model_results
            _save_incremental()

        # Summary using discriminating metrics
        def _mean(lst, key): return float(np.mean([r["go_enrichment"][key] for r in lst]))
        def _std(lst, key): return float(np.std([r["go_enrichment"][key] for r in lst]))

        rand_aurocs = [r["probing"]["auroc"] for r in model_results["random_projection"]]
        shuf_aurocs = [r["probing"]["auroc"] for r in model_results["shuffled"]]

        log.info(f"\n--- SUMMARY ({model_name}) ---")
        log.info(f"  SAE:      terms/feat={sae_go['mean_enriched_terms_per_feature']:.1f}, "
                 f"OR={sae_go['median_best_odds_ratio']:.1f}, "
                 f"AUROC={sae_probe['auroc']:.4f}")
        log.info(f"  Random:   terms/feat={_mean(model_results['random_projection'], 'mean_enriched_terms_per_feature'):.1f}, "
                 f"OR={_mean(model_results['random_projection'], 'median_best_odds_ratio'):.1f}, "
                 f"AUROC={np.mean(rand_aurocs):.4f}")
        log.info(f"  Shuffled: terms/feat={_mean(model_results['shuffled'], 'mean_enriched_terms_per_feature'):.1f}, "
                 f"OR={_mean(model_results['shuffled'], 'median_best_odds_ratio'):.1f}, "
                 f"AUROC={np.mean(shuf_aurocs):.4f}")

        model_results["summary"] = {
            "sae_go_mean_terms": float(sae_go["mean_enriched_terms_per_feature"]),
            "sae_go_median_odds": float(sae_go["median_best_odds_ratio"]),
            "sae_go_frac_3plus": float(sae_go["frac_with_3plus_terms"]),
            "sae_auroc": float(sae_probe["auroc"]),
            "random_go_mean_terms": _mean(model_results["random_projection"], "mean_enriched_terms_per_feature"),
            "random_go_median_odds": _mean(model_results["random_projection"], "median_best_odds_ratio"),
            "random_go_frac_3plus": _mean(model_results["random_projection"], "frac_with_3plus_terms"),
            "random_auroc_mean": float(np.mean(rand_aurocs)),
            "random_auroc_std": float(np.std(rand_aurocs)),
            "shuffled_go_mean_terms": _mean(model_results["shuffled"], "mean_enriched_terms_per_feature"),
            "shuffled_go_median_odds": _mean(model_results["shuffled"], "median_best_odds_ratio"),
            "shuffled_go_frac_3plus": _mean(model_results["shuffled"], "frac_with_3plus_terms"),
            "shuffled_auroc_mean": float(np.mean(shuf_aurocs)),
            "shuffled_auroc_std": float(np.std(shuf_aurocs)),
        }

        results[model_name] = model_results
        _save_incremental()

    log.info(f"\nAll done. Final results at {out_path}")


if __name__ == "__main__":
    main()
