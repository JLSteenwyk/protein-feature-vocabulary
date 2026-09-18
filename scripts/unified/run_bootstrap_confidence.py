#!/usr/bin/env python3
"""Bootstrap confidence intervals and permutation tests for key findings.

Computes 95% CIs for:
- Feature convergence (87.5%)
- Variant prediction correlations
- Cross-modal percentage (98%)
- Layer ablation KL values

Usage:
    ./env/bin/python scripts/unified/run_bootstrap_confidence.py
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

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bootstrap")


def bootstrap_ci(values, stat_fn=np.mean, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, n, replace=True)
        boot_stats.append(stat_fn(sample))
    boot_stats = np.sort(boot_stats)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return {
        "estimate": float(stat_fn(values)),
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


def permutation_test(group_a, group_b, stat_fn=lambda a, b: np.mean(a) - np.mean(b),
                      n_permutations=10000, seed=42):
    """Two-sample permutation test."""
    rng = np.random.RandomState(seed)
    a = np.array(group_a)
    b = np.array(group_b)
    observed = stat_fn(a, b)
    combined = np.concatenate([a, b])
    n_a = len(a)
    n_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_stat = stat_fn(combined[:n_a], combined[n_a:])
        if abs(perm_stat) >= abs(observed):
            n_extreme += 1
    p = (n_extreme + 1) / (n_permutations + 1)
    return {
        "observed": float(observed),
        "p_value": float(p),
        "n_permutations": n_permutations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ════════════════════════════════════════
    # 1. Feature convergence (87.5% = 175/200)
    # ════════════════════════════════════════
    log.info("1. Feature convergence CI...")
    conv_path = ROOT / "results" / "phase1" / "cross_model_comparison" / "esm2_esm3_comparison.json"
    if conv_path.exists():
        with open(conv_path) as f:
            conv_data = json.load(f)

        # Get per-feature similarities
        convergent = conv_data.get("convergent_features", [])
        top_matches = conv_data.get("top_matches", [])

        # Build full similarity vector (200 features)
        all_sims = []
        if top_matches:
            for m in top_matches:
                all_sims.append(m.get("similarity", 0))
        elif convergent:
            # Only have convergent ones; assume divergent have similarity < threshold
            threshold = 0.95
            for c in convergent:
                all_sims.append(c.get("similarity", 1.0))
            # Fill in divergent features with low similarity
            n_divergent = conv_data.get("n_esm2_features", 200) - len(convergent)
            # Use similarity_stats to estimate
            # For now, use binomial CI
            pass

        if all_sims:
            # Bootstrap the mean similarity
            sim_ci = bootstrap_ci(all_sims, stat_fn=np.mean,
                                  n_bootstrap=args.n_bootstrap, seed=args.seed)
            results["convergence_similarity"] = sim_ci
            log.info(f"  Mean similarity: {sim_ci['estimate']:.3f} "
                     f"[{sim_ci['ci_lower']:.3f}, {sim_ci['ci_upper']:.3f}]")

        # Binomial CI for convergence fraction
        n_total = conv_data.get("n_esm2_features", 200)
        n_conv = len(convergent)
        # Clopper-Pearson exact CI
        from scipy.stats import beta as beta_dist
        alpha = 0.05
        lo = float(beta_dist.ppf(alpha / 2, n_conv, n_total - n_conv + 1))
        hi = float(beta_dist.ppf(1 - alpha / 2, n_conv + 1, n_total - n_conv))
        frac = n_conv / n_total
        results["convergence_fraction"] = {
            "estimate": float(frac),
            "ci_lower": lo,
            "ci_upper": hi,
            "n_convergent": n_conv,
            "n_total": n_total,
            "method": "Clopper-Pearson exact",
        }
        log.info(f"  Convergence: {frac:.3f} [{lo:.3f}, {hi:.3f}] (Clopper-Pearson)")

        # Existing permutation test
        perm = conv_data.get("permutation_test", {})
        if perm:
            results["convergence_permutation"] = {
                "z_score": perm.get("z_score"),
                "p_value": perm.get("p_value"),
                "observed_mean": perm.get("observed_mean"),
                "null_mean": perm.get("null_mean"),
                "n_permutations": perm.get("n_permutations"),
            }
            log.info(f"  Permutation test: z={perm.get('z_score', '?'):.2f}, "
                     f"p={perm.get('p_value', '?')}")

    # ════════════════════════════════════════
    # 2. Variant prediction correlations
    # ════════════════════════════════════════
    log.info("\n2. Variant prediction CIs...")
    vp_dir = ROOT / "results" / "unified" / "variant_prediction"
    csv_files = sorted(vp_dir.glob("*_scores.csv"))

    vp_results = {}
    protein_rhos = {"mlm_score": [], "sae_magnitude": [], "sae_disruption": []}

    for csv_path in csv_files:
        protein = csv_path.stem.replace("_scores", "")
        df = pd.read_csv(csv_path)

        protein_cis = {}
        for col in ["mlm_score", "sae_magnitude", "sae_disruption"]:
            valid = df.dropna(subset=[col, "fitness"])
            if len(valid) < 30:
                continue

            # Bootstrap Spearman rho
            def spearman_rho(sample_idx):
                s = valid.iloc[sample_idx]
                r, _ = stats.spearmanr(s[col], s["fitness"])
                return r

            rng = np.random.RandomState(args.seed)
            boot_rhos = []
            n = len(valid)
            for _ in range(args.n_bootstrap):
                idx = rng.choice(n, n, replace=True)
                try:
                    r, _ = stats.spearmanr(valid[col].values[idx], valid["fitness"].values[idx])
                    boot_rhos.append(r)
                except Exception:
                    continue

            if boot_rhos:
                rho_obs, _ = stats.spearmanr(valid[col], valid["fitness"])
                lo = float(np.percentile(boot_rhos, 2.5))
                hi = float(np.percentile(boot_rhos, 97.5))
                protein_cis[col] = {
                    "rho": float(rho_obs),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n": n,
                }
                protein_rhos[col].append(float(rho_obs))

        if protein_cis:
            vp_results[protein] = protein_cis

    # Bootstrap mean rho across proteins
    vp_aggregate = {}
    for col in ["mlm_score", "sae_magnitude", "sae_disruption"]:
        rhos = protein_rhos[col]
        if len(rhos) >= 3:
            ci = bootstrap_ci(rhos, stat_fn=np.mean,
                              n_bootstrap=args.n_bootstrap, seed=args.seed)
            vp_aggregate[col] = ci
            log.info(f"  {col}: mean ρ={ci['estimate']:.3f} "
                     f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] (n={ci['n_samples']})")

    results["variant_prediction"] = {
        "per_protein": vp_results,
        "aggregate": vp_aggregate,
    }

    # ════════════════════════════════════════
    # 3. Cross-modal percentage
    # ════════════════════════════════════════
    log.info("\n3. Cross-modal percentage CI...")
    cm_path = ROOT / "results" / "unified" / "crossmodal_features" / "crossmodal_features_L33.json"
    if cm_path.exists():
        with open(cm_path) as f:
            cm_data = json.load(f)

        n_cm = cm_data["n_crossmodal"]
        n_total = n_cm + cm_data["n_matched"]

        # Clopper-Pearson
        from scipy.stats import beta as beta_dist
        alpha = 0.05
        lo = float(beta_dist.ppf(alpha / 2, n_cm, n_total - n_cm + 1))
        hi = float(beta_dist.ppf(1 - alpha / 2, n_cm + 1, n_total - n_cm))
        frac = n_cm / n_total

        results["crossmodal_fraction"] = {
            "estimate": float(frac),
            "ci_lower": lo,
            "ci_upper": hi,
            "n_crossmodal": n_cm,
            "n_total": n_total,
            "method": "Clopper-Pearson exact",
        }
        log.info(f"  Cross-modal: {frac:.3f} [{lo:.3f}, {hi:.3f}]")

    # ════════════════════════════════════════
    # 4. Layer ablation KL values
    # ════════════════════════════════════════
    log.info("\n4. Layer ablation CIs...")
    for model in ["esm2", "esm3"]:
        la_path = ROOT / "results" / "unified" / model / "layer_ablation.json"
        if not la_path.exists():
            continue
        with open(la_path) as f:
            la_data = json.load(f)

        # Extract per-layer stats
        layer_cis = {}
        layers = la_data.get("layers", la_data.get("results", {}))
        if isinstance(layers, dict):
            for layer_key, layer_info in layers.items():
                if isinstance(layer_info, dict):
                    mean_kl = layer_info.get("mean_kl", layer_info.get("kl_div_mean"))
                    std_kl = layer_info.get("std_kl", layer_info.get("kl_div_std"))
                    n = layer_info.get("n_proteins", layer_info.get("n", 500))
                    if mean_kl is not None and std_kl is not None and n:
                        se = std_kl / np.sqrt(n)
                        layer_cis[layer_key] = {
                            "mean_kl": float(mean_kl),
                            "ci_lower": float(mean_kl - 1.96 * se),
                            "ci_upper": float(mean_kl + 1.96 * se),
                            "std": float(std_kl),
                            "n": int(n),
                            "method": "normal approximation (mean +/- 1.96*SE)",
                        }
        elif isinstance(layers, list):
            for item in layers:
                if isinstance(item, dict):
                    layer_key = str(item.get("layer", ""))
                    mean_kl = item.get("mean_kl", item.get("kl_div_mean"))
                    std_kl = item.get("std_kl", item.get("kl_div_std"))
                    n = item.get("n_proteins", item.get("n", 500))
                    if mean_kl is not None and std_kl is not None and n:
                        se = std_kl / np.sqrt(n)
                        layer_cis[layer_key] = {
                            "mean_kl": float(mean_kl),
                            "ci_lower": float(mean_kl - 1.96 * se),
                            "ci_upper": float(mean_kl + 1.96 * se),
                            "std": float(std_kl),
                            "n": int(n),
                            "method": "normal approximation",
                        }

        if layer_cis:
            results[f"layer_ablation_{model}"] = layer_cis
            # Report max KL layer
            max_layer = max(layer_cis.items(), key=lambda x: x[1]["mean_kl"])
            log.info(f"  {model}: max KL at layer {max_layer[0]}: "
                     f"{max_layer[1]['mean_kl']:.3f} "
                     f"[{max_layer[1]['ci_lower']:.3f}, {max_layer[1]['ci_upper']:.3f}]")

    # Save
    out_path = out_dir / "bootstrap_confidence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
