#!/usr/bin/env python3
"""Combined/ensemble variant effect prediction.

Tests whether SAE features provide information orthogonal to MLM scores
using ridge regression with cross-validation.

Usage:
    ./env/bin/python scripts/unified/run_combined_variant_prediction.py
"""

import sys
import os
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("combined_vep")


def evaluate_model(X, y, model_name, n_folds=5, seed=42):
    """Fit RidgeCV and return CV Spearman rho."""
    if X.shape[0] < 30:
        return {"model": model_name, "rho": np.nan, "p": np.nan, "n": X.shape[0],
                "note": "too few samples"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))

    try:
        y_pred = cross_val_predict(ridge, X_scaled, y, cv=kf)
        rho, p = stats.spearmanr(y_pred, y)
        return {"model": model_name, "rho": float(rho), "p": float(p),
                "n": int(X.shape[0]), "n_features": int(X.shape[1])}
    except Exception as e:
        return {"model": model_name, "rho": np.nan, "p": np.nan,
                "n": int(X.shape[0]), "error": str(e)}


def main():
    out_dir = ROOT / "results" / "unified" / "variant_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all protein score files
    csv_files = sorted(out_dir.glob("*_scores.csv"))
    if not csv_files:
        log.error("No score CSVs found in results/unified/variant_prediction/")
        return

    log.info(f"Found {len(csv_files)} protein score files")

    all_protein_results = []

    for csv_path in csv_files:
        protein = csv_path.stem.replace("_scores", "")
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["fitness"])

        log.info(f"\n{'='*50}")
        log.info(f"{protein}: {len(df)} mutations")

        # Raw individual correlations (no fitting)
        raw_scores = {}
        for col in ["mlm_score", "sae_magnitude", "sae_disruption"]:
            valid = df.dropna(subset=[col])
            if len(valid) >= 20:
                rho, p = stats.spearmanr(valid[col], valid["fitness"])
                raw_scores[col] = {"rho": float(rho), "p": float(p), "n": len(valid)}
            else:
                raw_scores[col] = {"rho": np.nan, "p": np.nan, "n": len(valid)}

        # Model comparisons using RidgeCV
        y = df["fitness"].values
        model_results = []

        # 1. MLM only
        valid = df.dropna(subset=["mlm_score"])
        if len(valid) >= 30:
            X_mlm = valid[["mlm_score"]].values
            r = evaluate_model(X_mlm, valid["fitness"].values, "MLM_only")
            model_results.append(r)

        # 2. SAE only (magnitude)
        valid = df.dropna(subset=["sae_magnitude"])
        if len(valid) >= 30:
            X_sae = valid[["sae_magnitude"]].values
            r = evaluate_model(X_sae, valid["fitness"].values, "SAE_magnitude_only")
            model_results.append(r)

        # 3. SAE combined (magnitude + disruption)
        valid = df.dropna(subset=["sae_magnitude", "sae_disruption"])
        if len(valid) >= 30:
            X_sae2 = valid[["sae_magnitude", "sae_disruption"]].values
            r = evaluate_model(X_sae2, valid["fitness"].values, "SAE_combined")
            model_results.append(r)

        # 4. MLM + SAE magnitude
        valid = df.dropna(subset=["mlm_score", "sae_magnitude"])
        if len(valid) >= 30:
            X_comb1 = valid[["mlm_score", "sae_magnitude"]].values
            r = evaluate_model(X_comb1, valid["fitness"].values, "MLM+SAE_magnitude")
            model_results.append(r)

        # 5. MLM + SAE all
        valid = df.dropna(subset=["mlm_score", "sae_magnitude", "sae_disruption"])
        if len(valid) >= 30:
            X_comb2 = valid[["mlm_score", "sae_magnitude", "sae_disruption"]].values
            r = evaluate_model(X_comb2, valid["fitness"].values, "MLM+SAE_all")
            model_results.append(r)

        # Print results
        log.info(f"  Raw correlations:")
        for col, s in raw_scores.items():
            log.info(f"    {col}: ρ={s['rho']:.3f} (n={s['n']})")

        log.info(f"  CV Ridge models:")
        for r in model_results:
            rho_str = f"{r['rho']:.3f}" if not np.isnan(r.get('rho', np.nan)) else "NaN"
            log.info(f"    {r['model']}: ρ={rho_str} (n={r['n']})")

        all_protein_results.append({
            "protein": protein,
            "n_mutations": len(df),
            "raw_scores": raw_scores,
            "cv_models": model_results,
        })

    # Aggregate across proteins
    log.info(f"\n{'='*70}")
    log.info("AGGREGATE RESULTS (mean Spearman ρ across proteins)")
    log.info("=" * 70)

    model_names = set()
    for pr in all_protein_results:
        for mr in pr["cv_models"]:
            model_names.add(mr["model"])

    aggregate = {}
    for mn in sorted(model_names):
        rhos = []
        for pr in all_protein_results:
            for mr in pr["cv_models"]:
                if mr["model"] == mn and not np.isnan(mr.get("rho", np.nan)):
                    rhos.append(mr["rho"])
        if rhos:
            aggregate[mn] = {
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "median_rho": float(np.median(rhos)),
                "n_proteins": len(rhos),
                "per_protein_rhos": rhos,
            }
            log.info(f"  {mn:<25} ρ={np.mean(rhos):.3f} ± {np.std(rhos):.3f} (n={len(rhos)})")

    # Paired statistical tests
    paired_tests = {}
    baseline_key = "MLM_only"
    if baseline_key in aggregate:
        baseline_rhos = []
        for pr in all_protein_results:
            for mr in pr["cv_models"]:
                if mr["model"] == baseline_key and not np.isnan(mr.get("rho", np.nan)):
                    baseline_rhos.append((pr["protein"], mr["rho"]))

        baseline_dict = dict(baseline_rhos)

        for mn in sorted(model_names):
            if mn == baseline_key:
                continue
            paired_baseline = []
            paired_other = []
            for pr in all_protein_results:
                for mr in pr["cv_models"]:
                    if mr["model"] == mn and not np.isnan(mr.get("rho", np.nan)):
                        if pr["protein"] in baseline_dict:
                            paired_baseline.append(baseline_dict[pr["protein"]])
                            paired_other.append(mr["rho"])

            if len(paired_baseline) >= 3:
                try:
                    w_stat, w_p = stats.wilcoxon(paired_other, paired_baseline)
                    mean_diff = float(np.mean(np.array(paired_other) - np.array(paired_baseline)))
                    log.info(f"\n  {mn} vs {baseline_key}: "
                             f"Δρ={mean_diff:+.3f} (Wilcoxon p={w_p:.3g}, n={len(paired_baseline)})")
                    paired_tests[f"{mn}_vs_{baseline_key}"] = {
                        "mean_diff": mean_diff,
                        "W": float(w_stat),
                        "p": float(w_p),
                        "n_pairs": len(paired_baseline),
                    }
                except Exception as e:
                    log.info(f"\n  {mn} vs {baseline_key}: test failed ({e})")

    # Count improvements
    for mn in ["MLM+SAE_magnitude", "MLM+SAE_all"]:
        if mn not in aggregate or baseline_key not in aggregate:
            continue
        n_improved = 0
        n_total = 0
        for pr in all_protein_results:
            bl_rho = None
            mn_rho = None
            for mr in pr["cv_models"]:
                if mr["model"] == baseline_key:
                    bl_rho = mr.get("rho")
                if mr["model"] == mn:
                    mn_rho = mr.get("rho")
            if bl_rho is not None and mn_rho is not None and not np.isnan(bl_rho) and not np.isnan(mn_rho):
                n_total += 1
                if abs(mn_rho) > abs(bl_rho):
                    n_improved += 1
        log.info(f"\n  {mn} beats {baseline_key} in {n_improved}/{n_total} proteins")

    # Save
    output = {
        "per_protein": all_protein_results,
        "aggregate": aggregate,
        "paired_tests": paired_tests,
    }
    out_path = out_dir / "combined_prediction.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
