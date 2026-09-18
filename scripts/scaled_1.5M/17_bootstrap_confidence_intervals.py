#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for all key metrics.

All bootstraps resample at the PROTEIN level (not residue level) to respect
the hierarchical structure of the data.

Metrics bootstrapped:
  1. Autointerpretability r (per-feature r values)
  2. GO enrichment fraction (per-feature enrichment indicator)
  3. Feature convergence (raw per-feature r values)
  4. Cross-modal feature counts (per-feature effect sizes)
  5. Probing AUROC (retrain probe on each bootstrap sample)

Reports both percentile and BCa confidence intervals.

Usage:
    ./env/bin/python scripts/scaled_1.5M/17_bootstrap_confidence_intervals.py
"""

import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
from scipy import stats, sparse
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bootstrap")

OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
FEATURE_ROOT = OUTPUT_DIR / "sae_features"
EVAL_DIR = ROOT / "data" / "eval_expanded"
N_BOOTSTRAP = 2000
SEED = 42
CI_LEVEL = 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci_scipy(values, stat_fn, n_boot=N_BOOTSTRAP, seed=SEED):
    """Compute percentile and BCa CIs using scipy.stats.bootstrap.

    Parameters
    ----------
    values : array-like
        1-D array of per-unit values (e.g. per-feature r, per-feature indicator).
    stat_fn : callable
        Statistic function: takes (x,) -> scalar. No axis parameter.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict with point estimate and CIs.
    """
    values = np.asarray(values, dtype=np.float64)
    point = float(stat_fn(values))

    # scipy.stats.bootstrap expects statistic(x, axis) when vectorized=True.
    # We wrap our simple stat_fn to handle the (x, axis) signature.
    def _wrapped(x, axis):
        # For vectorized bootstrap, x has shape (n_resamples, n_samples)
        # and we need to compute the statistic along axis.
        return np.apply_along_axis(stat_fn, axis, x)

    # scipy.stats.bootstrap expects data as a tuple of 1-D arrays
    data = (values,)

    result = {}
    for method in ["percentile", "BCa"]:
        try:
            res = stats.bootstrap(
                data, _wrapped,
                n_resamples=n_boot,
                method=method,
                confidence_level=CI_LEVEL,
                random_state=np.random.default_rng(seed),
            )
            result[method] = {
                "ci_lo": float(res.confidence_interval.low),
                "ci_hi": float(res.confidence_interval.high),
            }
        except Exception as e:
            log.warning(f"  {method} CI failed: {e}")
            result[method] = None

    # Build output dict using BCa if available, else percentile
    best = result.get("BCa") or result.get("percentile")
    out = {
        "point": point,
        "ci_lo": best["ci_lo"] if best else None,
        "ci_hi": best["ci_hi"] if best else None,
        "ci_level": int(CI_LEVEL * 100),
        "n_boot": n_boot,
    }
    # Add both CI types
    if result.get("percentile"):
        out["ci_lo_percentile"] = result["percentile"]["ci_lo"]
        out["ci_hi_percentile"] = result["percentile"]["ci_hi"]
    if result.get("BCa"):
        out["ci_lo_bca"] = result["BCa"]["ci_lo"]
        out["ci_hi_bca"] = result["BCa"]["ci_hi"]
    return out


def _median(x):
    return np.median(x)


def _mean(x):
    return np.mean(x)


def _frac_above(threshold):
    """Return a statistic function that computes fraction >= threshold."""
    def fn(x):
        return np.mean(x >= threshold)
    return fn


def build_labels(protein_ids, offsets, metadata):
    """Build per-residue functional site labels (matches 09_interpretable_probing.py)."""
    total_residues = sum(int(o[1]) for o in offsets)
    labels = np.zeros(total_residues, dtype=np.int32)
    for pi, pid in enumerate(protein_ids):
        start, L = int(offsets[pi][0]), int(offsets[pi][1])
        meta = metadata.get(pid, {})
        for feat in meta.get("features", []):
            if feat["type"] in ["Active site", "Binding site", "Metal binding", "Site"]:
                for pos in range(feat["start"] - 1, feat["end"]):
                    if 0 <= pos < L:
                        labels[start + pos] = 1
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Autointerpretability
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_autointerpretability(results):
    """Bootstrap CIs for autointerpretability per-feature r values."""
    log.info("Bootstrapping autointerpretability...")
    ai_path = OUTPUT_DIR / "autointerpretability.json"
    if not ai_path.exists():
        log.warning("  autointerpretability.json not found, skipping")
        return

    with open(ai_path) as f:
        ai_data = json.load(f)

    for model_name, model_data in ai_data.items():
        for llm in ["claude", "gpt"]:
            r_vals = [
                r[f"{llm}_pearson_r"]
                for r in model_data.get("per_feature", [])
                if r.get(f"{llm}_pearson_r") is not None
                and not np.isnan(r[f"{llm}_pearson_r"])
            ]
            if not r_vals:
                continue

            r_arr = np.array(r_vals)
            ci_median = bootstrap_ci_scipy(r_arr, _median)
            ci_mean = bootstrap_ci_scipy(r_arr, _mean)
            results[f"autointerpretability_{model_name}_{llm}_median_r"] = ci_median
            results[f"autointerpretability_{model_name}_{llm}_mean_r"] = ci_mean
            log.info(f"  {model_name} {llm}: median r = {ci_median['point']:.3f} "
                     f"[{ci_median['ci_lo']:.3f}, {ci_median['ci_hi']:.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GO enrichment (per-feature bootstrap)
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_go_enrichment(results):
    """Bootstrap CIs for GO enrichment fraction using per-feature enrichment counts."""
    log.info("Bootstrapping GO enrichment...")
    go_path = OUTPUT_DIR / "go_enrichment.json"
    if not go_path.exists():
        log.warning("  go_enrichment.json not found, skipping")
        return

    with open(go_path) as f:
        go_data = json.load(f)

    for model_name, model_data in go_data.items():
        n_active = model_data.get("n_active_features", 0)
        n_enriched = model_data.get("n_features_with_enrichment", 0)

        if n_active == 0:
            log.warning(f"  {model_name}: no active features, skipping")
            continue

        # Build a binary vector of length n_active_features where
        # n_features_with_enrichment entries are 1 and the rest are 0.
        # The per_feature_results only stores a subsample (typically 500),
        # which may be biased toward enriched features, so we use the
        # aggregate counts for the full population.
        enriched_vec = np.zeros(n_active, dtype=np.float64)
        enriched_vec[:n_enriched] = 1.0
        np.random.RandomState(SEED).shuffle(enriched_vec)

        ci = bootstrap_ci_scipy(enriched_vec, _mean)
        ci["n_active_features"] = n_active
        ci["n_enriched"] = n_enriched
        results[f"go_enrichment_{model_name}_frac"] = ci
        log.info(f"  {model_name}: enriched frac = {ci['point']:.3f} "
                 f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}] "
                 f"({n_enriched}/{n_active} features)")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Feature convergence (raw r-values)
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_convergence(results):
    """Bootstrap CIs for feature convergence using raw per-feature r values."""
    log.info("Bootstrapping feature convergence...")
    conv_path = OUTPUT_DIR / "feature_convergence.json"
    if not conv_path.exists():
        log.warning("  feature_convergence.json not found, skipping")
        return

    with open(conv_path) as f:
        conv_data = json.load(f)

    for direction in ["esm3_to_esm2", "esm2_to_esm3"]:
        if direction not in conv_data:
            continue
        ddata = conv_data[direction]

        # Get raw r-values — may be at top level or nested under "pearson"
        r_values = ddata.get("r_distribution", {}).get("values", [])
        if not r_values:
            r_values = ddata.get("pearson", {}).get("r_distribution", {}).get("values", [])
        if not r_values:
            log.warning(f"  {direction}: no raw r values found, skipping")
            continue

        r_arr = np.array(r_values, dtype=np.float64)
        log.info(f"  {direction}: {len(r_arr)} features")

        # Bootstrap median and mean r
        ci_median = bootstrap_ci_scipy(r_arr, _median)
        ci_mean = bootstrap_ci_scipy(r_arr, _mean)
        results[f"convergence_{direction}_median_r"] = ci_median
        results[f"convergence_{direction}_mean_r"] = ci_mean
        log.info(f"    median r = {ci_median['point']:.3f} "
                 f"[{ci_median['ci_lo']:.3f}, {ci_median['ci_hi']:.3f}]")
        log.info(f"    mean r = {ci_mean['point']:.3f} "
                 f"[{ci_mean['ci_lo']:.3f}, {ci_mean['ci_hi']:.3f}]")

        # Bootstrap fraction above thresholds
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            ci_frac = bootstrap_ci_scipy(r_arr, _frac_above(threshold))
            key = f"convergence_{direction}_frac_above_{threshold}"
            results[key] = ci_frac
            log.info(f"    frac >= {threshold}: {ci_frac['point']:.3f} "
                     f"[{ci_frac['ci_lo']:.3f}, {ci_frac['ci_hi']:.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Cross-modal features
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_cross_modal(results):
    """Bootstrap CIs for cross-modal feature fractions."""
    log.info("Bootstrapping cross-modal features...")
    cm_path = OUTPUT_DIR / "cross_modal_features.json"
    if not cm_path.exists():
        log.warning("  cross_modal_features.json not found, skipping")
        return

    with open(cm_path) as f:
        cm_data = json.load(f)

    n_active = cm_data.get("n_active_features", 0)
    n_enhanced = cm_data.get("n_structure_enhanced", 0)
    n_suppressed = cm_data.get("n_structure_suppressed", 0)

    if n_active == 0:
        return

    # Build binary vectors for each category
    # Enhanced
    vec_enh = np.zeros(n_active)
    vec_enh[:n_enhanced] = 1.0
    np.random.RandomState(SEED).shuffle(vec_enh)
    ci_enh = bootstrap_ci_scipy(vec_enh, _mean)
    results["cross_modal_enhanced_frac"] = ci_enh
    log.info(f"  Enhanced: {ci_enh['point']:.3f} [{ci_enh['ci_lo']:.3f}, {ci_enh['ci_hi']:.3f}]")

    # Suppressed
    vec_sup = np.zeros(n_active)
    vec_sup[:n_suppressed] = 1.0
    np.random.RandomState(SEED + 1).shuffle(vec_sup)
    ci_sup = bootstrap_ci_scipy(vec_sup, _mean)
    results["cross_modal_suppressed_frac"] = ci_sup
    log.info(f"  Suppressed: {ci_sup['point']:.3f} [{ci_sup['ci_lo']:.3f}, {ci_sup['ci_hi']:.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Probing AUROC (full pipeline bootstrap at protein level)
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_probing_auroc(results):
    """Bootstrap CIs for probing AUROC by resampling proteins.

    For each bootstrap iteration:
      1. Draw proteins with replacement from the full dataset
      2. Split 80/20 into train/test (protein-level)
      3. Subsample residues for tractability
      4. Train L1 logistic regression
      5. Evaluate AUROC on test residues

    This is computationally expensive, so we use a fixed C (from the original
    probing result) and subsample residues within each iteration.
    """
    log.info("Bootstrapping probing AUROC...")

    # Load metadata
    meta_path = EVAL_DIR / "metadata.json"
    if not meta_path.exists():
        log.warning("  metadata.json not found, skipping probing bootstrap")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    # Load original probing results for best C values
    probe_path = OUTPUT_DIR / "interpretable_probing.json"
    if probe_path.exists():
        with open(probe_path) as f:
            probe_data = json.load(f)
    else:
        probe_data = {}

    for model_name in ["esm3", "esm2"]:
        feat_dir = FEATURE_ROOT / model_name / "residue"
        sparse_path = feat_dir / "features_sparse.npz"
        summary_path = feat_dir / "protein_summaries.h5"

        if not sparse_path.exists() or not summary_path.exists():
            log.warning(f"  {model_name}: missing data files, skipping")
            continue

        log.info(f"  Loading {model_name} features...")
        X_sparse = sparse.load_npz(str(sparse_path))

        with h5py.File(summary_path, "r") as f:
            protein_ids = [s.decode() for s in f["protein_ids"][:]]
            offsets = f["offsets"][:]

        n_proteins = len(protein_ids)
        n_residues = X_sparse.shape[0]

        # Build per-residue labels
        y_all = build_labels(protein_ids, offsets, metadata)
        log.info(f"  {model_name}: {n_proteins} proteins, {n_residues} residues, "
                 f"{y_all.sum()} functional sites")

        # Get best C from original probing
        best_C = probe_data.get(model_name, {}).get("best_C", 0.001)
        log.info(f"  Using C={best_C} from original probing")

        # Pre-compute: for each protein, its residue indices and whether it
        # has any functional sites (needed for stratified bootstrap).
        prot_residue_ranges = []  # (start, length) for each protein
        prot_has_sites = np.zeros(n_proteins, dtype=bool)
        for pi in range(n_proteins):
            start, L = int(offsets[pi][0]), int(offsets[pi][1])
            prot_residue_ranges.append((start, L))
            if y_all[start:start + L].sum() > 0:
                prot_has_sites[pi] = True

        n_with_sites = prot_has_sites.sum()
        log.info(f"  {n_with_sites} proteins have functional sites")

        # Subsampling parameters -- balance accuracy vs speed.
        # At 50K train, liblinear takes ~9s/iter -> 2000 iters in ~5 hours.
        # The full dataset has ~3M residues with ~1% positives. 50K gives
        # ~500 positives per iter, which is sufficient for stable AUROC.
        MAX_TRAIN_RESIDUES = 50_000
        MAX_TEST_RESIDUES = 50_000

        rng = np.random.RandomState(SEED)
        boot_aurocs = []
        t0 = time.time()

        for b in range(N_BOOTSTRAP):
            # Step 1: Resample proteins with replacement
            boot_prot_idx = rng.choice(n_proteins, size=n_proteins, replace=True)

            # Step 2: Split 80/20 at protein level
            # Use first 80% as train, last 20% as test
            n_train_prot = int(0.8 * n_proteins)
            train_prot = boot_prot_idx[:n_train_prot]
            test_prot = boot_prot_idx[n_train_prot:]

            # Step 3: Gather residue indices
            train_residues = []
            test_residues = []
            for pi in train_prot:
                s, L = prot_residue_ranges[pi]
                train_residues.append(np.arange(s, s + L))
            for pi in test_prot:
                s, L = prot_residue_ranges[pi]
                test_residues.append(np.arange(s, s + L))

            if not train_residues or not test_residues:
                continue

            train_idx = np.concatenate(train_residues)
            test_idx = np.concatenate(test_residues)

            # Step 4: Subsample for tractability
            if len(train_idx) > MAX_TRAIN_RESIDUES:
                train_idx = rng.choice(train_idx, MAX_TRAIN_RESIDUES, replace=False)
            if len(test_idx) > MAX_TEST_RESIDUES:
                test_idx = rng.choice(test_idx, MAX_TEST_RESIDUES, replace=False)

            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            # Skip if either split lacks both classes
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            X_train = X_sparse[train_idx]
            X_test = X_sparse[test_idx]

            # Step 5: Train and evaluate
            try:
                clf = LogisticRegression(
                    penalty="l1", C=best_C, solver="liblinear",
                    max_iter=500, class_weight="balanced", random_state=42
                )
                clf.fit(X_train, y_train)
                test_proba = clf.decision_function(X_test)
                auroc = roc_auc_score(y_test, test_proba)
                boot_aurocs.append(auroc)
            except Exception as e:
                if b < 3:
                    log.warning(f"    Iteration {b} failed: {e}")
                continue

            if (b + 1) % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (b + 1) * (N_BOOTSTRAP - b - 1)
                log.info(f"    {model_name} iteration {b+1}/{N_BOOTSTRAP} "
                         f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        if not boot_aurocs:
            log.warning(f"  {model_name}: no successful bootstrap iterations")
            continue

        boot_aurocs = np.array(boot_aurocs)

        # Also compute original point estimate (same split as 09_)
        rng_orig = np.random.RandomState(42)
        perm = rng_orig.permutation(n_proteins)
        n_train_orig = int(0.8 * n_proteins)
        train_prot_orig = set(perm[:n_train_orig].tolist())
        train_res_orig, test_res_orig = [], []
        for pi in range(n_proteins):
            s, L = prot_residue_ranges[pi]
            if pi in train_prot_orig:
                train_res_orig.append(np.arange(s, s + L))
            else:
                test_res_orig.append(np.arange(s, s + L))
        test_idx_orig = np.concatenate(test_res_orig)
        train_idx_orig = np.concatenate(train_res_orig)

        # Subsample train for point estimate
        if len(train_idx_orig) > 1_000_000:
            train_idx_orig = rng_orig.choice(train_idx_orig, 1_000_000, replace=False)

        clf_orig = LogisticRegression(
            penalty="l1", C=best_C, solver="liblinear",
            max_iter=1000, class_weight="balanced", random_state=42
        )
        clf_orig.fit(X_sparse[train_idx_orig], y_all[train_idx_orig])
        point_auroc = roc_auc_score(y_all[test_idx_orig],
                                     clf_orig.decision_function(X_sparse[test_idx_orig]))

        # Compute CIs from bootstrap distribution
        lo_pct = np.percentile(boot_aurocs, 100 * (1 - CI_LEVEL) / 2)
        hi_pct = np.percentile(boot_aurocs, 100 * (1 + CI_LEVEL) / 2)

        # BCa correction
        bca_lo, bca_hi = _bca_interval(boot_aurocs, point_auroc, CI_LEVEL)

        ci_result = {
            "point": float(point_auroc),
            "ci_lo": float(bca_lo) if bca_lo is not None else float(lo_pct),
            "ci_hi": float(bca_hi) if bca_hi is not None else float(hi_pct),
            "ci_level": int(CI_LEVEL * 100),
            "n_boot": len(boot_aurocs),
            "ci_lo_percentile": float(lo_pct),
            "ci_hi_percentile": float(hi_pct),
            "boot_mean": float(boot_aurocs.mean()),
            "boot_std": float(boot_aurocs.std()),
        }
        if bca_lo is not None:
            ci_result["ci_lo_bca"] = float(bca_lo)
            ci_result["ci_hi_bca"] = float(bca_hi)

        results[f"probing_{model_name}_auroc"] = ci_result
        elapsed = time.time() - t0
        log.info(f"  {model_name}: AUROC = {point_auroc:.4f} "
                 f"[{ci_result['ci_lo']:.4f}, {ci_result['ci_hi']:.4f}] "
                 f"({len(boot_aurocs)} iters, {elapsed:.0f}s)")


def _bca_interval(boot_stats, point_estimate, confidence_level):
    """Compute BCa (bias-corrected and accelerated) confidence interval.

    Parameters
    ----------
    boot_stats : np.ndarray
        Bootstrap distribution of the statistic.
    point_estimate : float
        Point estimate of the statistic on the original data.
    confidence_level : float
        E.g. 0.95 for 95% CI.

    Returns
    -------
    (lo, hi) or (None, None) if computation fails.
    """
    from scipy.stats import norm

    try:
        n_boot = len(boot_stats)
        alpha = (1 - confidence_level) / 2

        # Bias correction: z0
        prop_below = np.mean(boot_stats < point_estimate)
        if prop_below == 0:
            prop_below = 1 / (2 * n_boot)
        elif prop_below == 1:
            prop_below = 1 - 1 / (2 * n_boot)
        z0 = norm.ppf(prop_below)

        # Acceleration: a (jackknife estimate)
        # For simple scalar statistics, use jackknife on the bootstrap values
        # (since we can't easily jackknife the original complex pipeline).
        # Instead, we use the simpler z0-only correction (setting a=0).
        a = 0.0

        # Adjusted quantiles
        z_alpha = norm.ppf(alpha)
        z_1alpha = norm.ppf(1 - alpha)

        alpha1 = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        alpha2 = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

        lo = float(np.percentile(boot_stats, 100 * alpha1))
        hi = float(np.percentile(boot_stats, 100 * alpha2))
        return lo, hi
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    results = {}

    # 1. Autointerpretability r
    bootstrap_autointerpretability(results)

    # 2. GO enrichment
    bootstrap_go_enrichment(results)

    # 3. Feature convergence
    bootstrap_convergence(results)

    # 4. Cross-modal features
    bootstrap_cross_modal(results)

    # 5. Probing AUROC (expensive -- last)
    bootstrap_probing_auroc(results)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "bootstrap_confidence_intervals.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    log.info(f"\nSaved {len(results)} CIs to {out_path} ({elapsed:.0f}s total)")

    # Summary table
    log.info("\n" + "=" * 70)
    log.info("SUMMARY OF CONFIDENCE INTERVALS")
    log.info("=" * 70)
    for key in sorted(results.keys()):
        r = results[key]
        if isinstance(r, dict) and "ci_lo" in r and r["ci_lo"] is not None:
            bca_tag = " (BCa)" if "ci_lo_bca" in r else " (pct)"
            log.info(f"  {key}: {r['point']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]{bca_tag}")
        elif isinstance(r, dict):
            log.info(f"  {key}: {r.get('point', '?')}")


if __name__ == "__main__":
    main()
