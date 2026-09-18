#!/usr/bin/env python3
"""Cross-model comparison of unified analysis results.

Loads results from ESM-2 and ESM-3 and generates side-by-side analysis
with statistical tests.

Run as:
    ./env/bin/python scripts/unified/run_cross_model_comparison.py --model-a esm2 --model-b esm3

Output: results/unified/comparison_{model_a}_vs_{model_b}.json
"""

import sys
import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)


def setup_logging():
    out_dir = ROOT / "results" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "cross_model_comparison_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("comparison"), out_dir


def load_results(model_name, out_dir):
    """Load all available results for a model."""
    model_dir = out_dir / model_name
    results = {}

    # Load every JSON result file
    for fpath in sorted(model_dir.glob("*.json")):
        with open(fpath) as f:
            results[fpath.stem] = json.load(f)

    return results


def compare_layer_ablation(res_a, res_b, name_a, name_b, log):
    """Compare layer ablation profiles."""
    if "layer_ablation" not in res_a or "layer_ablation" not in res_b:
        log.info("Layer ablation: data missing for one or both models")
        return None

    la_a = res_a["layer_ablation"]["per_layer"]
    la_b = res_b["layer_ablation"]["per_layer"]
    n_a = res_a["layer_ablation"]["n_layers"]
    n_b = res_b["layer_ablation"]["n_layers"]

    # Normalize layer indices to [0, 1] for comparison
    kl_a = [(int(k) / (n_a - 1), v["mean_kl"]) for k, v in la_a.items()]
    kl_b = [(int(k) / (n_b - 1), v["mean_kl"]) for k, v in la_b.items()]

    kl_a.sort()
    kl_b.sort()

    # Correlation of ablation profiles at matched relative positions
    # Interpolate both to common grid
    from scipy.interpolate import interp1d

    x_a, y_a = zip(*kl_a)
    x_b, y_b = zip(*kl_b)

    grid = np.linspace(0, 1, 20)
    interp_a = interp1d(x_a, y_a, kind='linear', fill_value='extrapolate')(grid)
    interp_b = interp1d(x_b, y_b, kind='linear', fill_value='extrapolate')(grid)

    corr, p_val = stats.pearsonr(interp_a, interp_b)
    log.info(f"Layer ablation profile correlation: r={corr:.3f}, p={p_val:.2e}")

    return {
        "correlation": float(corr),
        "p_value": float(p_val),
        f"{name_a}_kl_profile": kl_a,
        f"{name_b}_kl_profile": kl_b,
        f"{name_a}_mean_kl": float(np.mean(y_a)),
        f"{name_b}_mean_kl": float(np.mean(y_b)),
        f"{name_a}_max_kl_layer_frac": float(x_a[np.argmax(y_a)]),
        f"{name_b}_max_kl_layer_frac": float(x_b[np.argmax(y_b)]),
    }


def compare_head_ablation(res_a, res_b, name_a, name_b, log):
    """Compare head ablation importance distributions."""
    if "head_ablation" not in res_a or "head_ablation" not in res_b:
        log.info("Head ablation: data missing for one or both models")
        return None

    ha_a = res_a["head_ablation"]["per_head"]
    ha_b = res_b["head_ablation"]["per_head"]

    kl_a = [v["mean_kl"] for v in ha_a.values()]
    kl_b = [v["mean_kl"] for v in ha_b.values()]

    # Summary statistics
    result = {
        f"{name_a}_mean_head_kl": float(np.mean(kl_a)),
        f"{name_b}_mean_head_kl": float(np.mean(kl_b)),
        f"{name_a}_max_head_kl": float(np.max(kl_a)),
        f"{name_b}_max_head_kl": float(np.max(kl_b)),
        f"{name_a}_n_important_heads": int(np.sum(np.array(kl_a) > np.mean(kl_a) + np.std(kl_a))),
        f"{name_b}_n_important_heads": int(np.sum(np.array(kl_b) > np.mean(kl_b) + np.std(kl_b))),
    }

    # Top-5 most important heads for each model
    sorted_a = sorted(ha_a.items(), key=lambda x: x[1]["mean_kl"], reverse=True)[:5]
    sorted_b = sorted(ha_b.items(), key=lambda x: x[1]["mean_kl"], reverse=True)[:5]

    result[f"{name_a}_top5_heads"] = [(k, v["mean_kl"]) for k, v in sorted_a]
    result[f"{name_b}_top5_heads"] = [(k, v["mean_kl"]) for k, v in sorted_b]

    log.info(f"Head ablation: {name_a} mean KL={np.mean(kl_a):.4f}, "
             f"{name_b} mean KL={np.mean(kl_b):.4f}")

    return result


def compare_attention(res_a, res_b, name_a, name_b, log):
    """Compare attention head properties."""
    if "attention_analysis" not in res_a or "attention_analysis" not in res_b:
        log.info("Attention: data missing for one or both models")
        return None

    aa_a = res_a["attention_analysis"]["per_head"]
    aa_b = res_b["attention_analysis"]["per_head"]

    # Entropy distributions
    ent_a = [v["entropy"] for v in aa_a.values()]
    ent_b = [v["entropy"] for v in aa_b.values()]

    # Local fraction distributions
    loc_a = [v["local_fraction"] for v in aa_a.values()]
    loc_b = [v["local_fraction"] for v in aa_b.values()]

    result = {
        f"{name_a}_mean_entropy": float(np.mean(ent_a)),
        f"{name_b}_mean_entropy": float(np.mean(ent_b)),
        f"{name_a}_mean_local_frac": float(np.mean(loc_a)),
        f"{name_b}_mean_local_frac": float(np.mean(loc_b)),
        "entropy_ks_stat": float(stats.ks_2samp(ent_a, ent_b).statistic),
        "entropy_ks_pval": float(stats.ks_2samp(ent_a, ent_b).pvalue),
    }

    log.info(f"Attention: {name_a} entropy={np.mean(ent_a):.3f}, "
             f"{name_b} entropy={np.mean(ent_b):.3f}")

    return result


def compare_activation_patching(res_a, res_b, name_a, name_b, log):
    """Compare activation patching effect curves."""
    if "activation_patching" not in res_a or "activation_patching" not in res_b:
        log.info("Activation patching: data missing for one or both models")
        return None

    ap_a = res_a["activation_patching"]["per_layer"]
    ap_b = res_b["activation_patching"]["per_layer"]
    n_a = max(int(k) for k in ap_a.keys()) + 1
    n_b = max(int(k) for k in ap_b.keys()) + 1

    kl_a = [(int(k) / (n_a - 1), v["mean_kl"]) for k, v in ap_a.items()]
    kl_b = [(int(k) / (n_b - 1), v["mean_kl"]) for k, v in ap_b.items()]
    kl_a.sort()
    kl_b.sort()

    result = {
        f"{name_a}_patching_profile": kl_a,
        f"{name_b}_patching_profile": kl_b,
        f"{name_a}_peak_effect_frac": float(kl_a[np.argmax([v for _, v in kl_a])][0]),
        f"{name_b}_peak_effect_frac": float(kl_b[np.argmax([v for _, v in kl_b])][0]),
    }

    log.info(f"Patching peak: {name_a} at {result[f'{name_a}_peak_effect_frac']:.2f}, "
             f"{name_b} at {result[f'{name_b}_peak_effect_frac']:.2f}")

    return result


def compare_logit_lens(res_a, res_b, name_a, name_b, log):
    """Compare logit lens prediction formation curves."""
    if "logit_lens" not in res_a or "logit_lens" not in res_b:
        log.info("Logit lens: data missing for one or both models")
        return None

    ll_a = res_a["logit_lens"]["per_layer"]
    ll_b = res_b["logit_lens"]["per_layer"]
    n_a = res_a["logit_lens"]["n_layers"]
    n_b = res_b["logit_lens"]["n_layers"]

    # Find where predictions "crystallize" (top-1 > 0.8)
    def find_crystallization(per_layer, n_layers):
        for k in sorted(per_layer.keys(), key=int):
            v = per_layer[k]
            top1 = v.get("top1_agreement", v.get("mean_top1", 0))
            if top1 > 0.8:
                return int(k) / (n_layers - 1)
        return 1.0

    cryst_a = find_crystallization(ll_a, n_a)
    cryst_b = find_crystallization(ll_b, n_b)

    # Final layer KL
    last_a = ll_a.get(str(n_a - 1), {})
    last_b = ll_b.get(str(n_b - 1), {})

    result = {
        f"{name_a}_crystallization_frac": cryst_a,
        f"{name_b}_crystallization_frac": cryst_b,
        f"{name_a}_final_kl": last_a.get("mean_kl", None),
        f"{name_b}_final_kl": last_b.get("mean_kl", None),
    }

    log.info(f"Logit lens: crystallization at {name_a}={cryst_a:.2f}, {name_b}={cryst_b:.2f}")
    return result


def compare_tuned_lens(res_a, res_b, name_a, name_b, log):
    """Compare tuned vs raw lens improvement."""
    if "tuned_lens" not in res_a or "tuned_lens" not in res_b:
        log.info("Tuned lens: data missing for one or both models")
        return None

    tl_a = res_a["tuned_lens"]["per_layer"]
    tl_b = res_b["tuned_lens"]["per_layer"]

    def avg_improvement(per_layer):
        improvements = []
        for k, v in per_layer.items():
            if isinstance(v, dict) and "improvement_kl" in v:
                improvements.append(v["improvement_kl"])
        return float(np.mean(improvements)) if improvements else 0.0

    def avg_tuned_top1(per_layer):
        vals = []
        for k, v in per_layer.items():
            if isinstance(v, dict) and "tuned_top1" in v:
                t1 = v["tuned_top1"]
                vals.append(t1["mean"] if isinstance(t1, dict) else t1)
        return float(np.mean(vals)) if vals else 0.0

    result = {
        f"{name_a}_avg_kl_improvement": avg_improvement(tl_a),
        f"{name_b}_avg_kl_improvement": avg_improvement(tl_b),
        f"{name_a}_avg_tuned_top1": avg_tuned_top1(tl_a),
        f"{name_b}_avg_tuned_top1": avg_tuned_top1(tl_b),
    }

    log.info(f"Tuned lens: avg KL improvement {name_a}={result[f'{name_a}_avg_kl_improvement']:.3f}, "
             f"{name_b}={result[f'{name_b}_avg_kl_improvement']:.3f}")
    return result


def compare_direct_logit_attribution(res_a, res_b, name_a, name_b, log):
    """Compare per-layer logit attribution."""
    if "direct_logit_attribution" not in res_a or "direct_logit_attribution" not in res_b:
        log.info("Direct logit attribution: data missing for one or both models")
        return None

    da = res_a["direct_logit_attribution"]
    db = res_b["direct_logit_attribution"]
    n_a = da["n_layers"]
    n_b = db["n_layers"]

    # Peak logit norm layer
    def peak_layer(data, n_layers):
        norms = data.get("per_layer_logit_norm", {})
        best_k, best_v = None, 0
        for k, v in norms.items():
            val = v.get("mean", v) if isinstance(v, dict) else v
            if val > best_v:
                best_v = val
                best_k = int(k)
        return best_k / (n_layers - 1) if best_k is not None else None

    # Attn vs MLP dominance
    def attn_mlp_ratio(data):
        attn = data.get("attn_logit_norm", {})
        mlp = data.get("mlp_logit_norm", {})
        ratios = []
        for k in attn:
            if k in mlp:
                an = attn[k].get("mean", attn[k]) if isinstance(attn[k], dict) else attn[k]
                mn = mlp[k].get("mean", mlp[k]) if isinstance(mlp[k], dict) else mlp[k]
                if (an + mn) > 0:
                    ratios.append(an / (an + mn))
        return float(np.mean(ratios)) if ratios else None

    result = {
        f"{name_a}_peak_logit_frac": peak_layer(da, n_a),
        f"{name_b}_peak_logit_frac": peak_layer(db, n_b),
        f"{name_a}_attn_fraction": attn_mlp_ratio(da),
        f"{name_b}_attn_fraction": attn_mlp_ratio(db),
    }

    log.info(f"Direct logit attribution: peak at {name_a}={result[f'{name_a}_peak_logit_frac']}, "
             f"{name_b}={result[f'{name_b}_peak_logit_frac']}")
    return result


def compare_residual_decomposition(res_a, res_b, name_a, name_b, log):
    """Compare attention vs MLP contributions."""
    if "residual_decomposition" not in res_a or "residual_decomposition" not in res_b:
        log.info("Residual decomposition: data missing for one or both models")
        return None

    rd_a = res_a["residual_decomposition"]["per_layer"]
    rd_b = res_b["residual_decomposition"]["per_layer"]

    def avg_attn_frac(per_layer):
        fracs = []
        for k, v in per_layer.items():
            af = v.get("attn_fraction", None)
            if af is not None:
                fracs.append(af)
        return float(np.mean(fracs)) if fracs else None

    def avg_cosine(per_layer):
        cosines = []
        for k, v in per_layer.items():
            c = v.get("cosine_attn_mlp", None)
            if c is not None:
                cosines.append(c)
        return float(np.mean(cosines)) if cosines else None

    result = {
        f"{name_a}_avg_attn_fraction": avg_attn_frac(rd_a),
        f"{name_b}_avg_attn_fraction": avg_attn_frac(rd_b),
        f"{name_a}_avg_cosine_attn_mlp": avg_cosine(rd_a),
        f"{name_b}_avg_cosine_attn_mlp": avg_cosine(rd_b),
    }

    log.info(f"Residual decomp: attn fraction {name_a}={result[f'{name_a}_avg_attn_fraction']}, "
             f"{name_b}={result[f'{name_b}_avg_attn_fraction']}")
    return result


def compare_ov_qk(res_a, res_b, name_a, name_b, log):
    """Compare OV/QK circuit properties."""
    if "ov_qk_decomposition" not in res_a or "ov_qk_decomposition" not in res_b:
        log.info("OV/QK decomposition: data missing for one or both models")
        return None

    ov_a = res_a["ov_qk_decomposition"]
    ov_b = res_b["ov_qk_decomposition"]

    result = {
        f"{name_a}_head_classifications": ov_a.get("global_head_classifications", {}),
        f"{name_b}_head_classifications": ov_b.get("global_head_classifications", {}),
    }

    # Average effective ranks across layers
    for name, ov in [(name_a, ov_a), (name_b, ov_b)]:
        ov_ranks, qk_ranks = [], []
        for l, ldata in ov.get("per_layer", {}).items():
            ov_ranks.append(ldata.get("mean_ov_rank", 0))
            qk_ranks.append(ldata.get("mean_qk_rank", 0))
        result[f"{name}_avg_ov_rank"] = float(np.mean(ov_ranks)) if ov_ranks else None
        result[f"{name}_avg_qk_rank"] = float(np.mean(qk_ranks)) if qk_ranks else None

    log.info(f"OV/QK: head types {name_a}={result[f'{name_a}_head_classifications']}, "
             f"{name_b}={result[f'{name_b}_head_classifications']}")
    return result


def compare_attribution_patching(res_a, res_b, name_a, name_b, log):
    """Compare gradient-based attribution patching."""
    if "attribution_patching" not in res_a or "attribution_patching" not in res_b:
        log.info("Attribution patching: data missing for one or both models")
        return None

    ap_a = res_a["attribution_patching"]
    ap_b = res_b["attribution_patching"]

    result = {
        f"{name_a}_peak_layer": ap_a.get("peak_layer"),
        f"{name_b}_peak_layer": ap_b.get("peak_layer"),
        f"{name_a}_peak_effect": ap_a.get("peak_effect"),
        f"{name_b}_peak_effect": ap_b.get("peak_effect"),
        f"{name_a}_peak_frac": (ap_a["peak_layer"] / (ap_a["n_layers"] - 1)
                                 if ap_a.get("peak_layer") is not None else None),
        f"{name_b}_peak_frac": (ap_b["peak_layer"] / (ap_b["n_layers"] - 1)
                                 if ap_b.get("peak_layer") is not None else None),
    }

    log.info(f"Attribution patching: peak {name_a}=L{result[f'{name_a}_peak_layer']}, "
             f"{name_b}=L{result[f'{name_b}_peak_layer']}")
    return result


def compare_mutation_sensitivity(res_a, res_b, name_a, name_b, log):
    """Compare mutation sensitivity at functional vs non-functional sites."""
    if "mutation_sensitivity" not in res_a or "mutation_sensitivity" not in res_b:
        log.info("Mutation sensitivity: data missing for one or both models")
        return None

    ms_a = res_a["mutation_sensitivity"]
    ms_b = res_b["mutation_sensitivity"]

    result = {}
    for name, ms in [(name_a, ms_a), (name_b, ms_b)]:
        fvn = ms.get("functional_vs_nonfunctional", {})
        func = fvn.get("functional", {})
        nonfunc = fvn.get("nonfunctional", {})
        result[f"{name}_functional_kl"] = func.get("mean", None)
        result[f"{name}_nonfunctional_kl"] = nonfunc.get("mean", None)
        result[f"{name}_func_nonfunc_ratio"] = fvn.get("ratio", None)
        result[f"{name}_mw_p_value"] = fvn.get("mannwhitney_p", None)

        # Secondary structure breakdown
        ss = ms.get("secondary_structure", {})
        for ss_type in ["helix", "sheet", "coil"]:
            if ss_type in ss:
                result[f"{name}_{ss_type}_kl"] = ss[ss_type].get("mean", None)

    log.info(f"Mutation sensitivity ratios: {name_a}={result.get(f'{name_a}_func_nonfunc_ratio')}, "
             f"{name_b}={result.get(f'{name_b}_func_nonfunc_ratio')}")
    return result


def compare_causal_probing(res_a, res_b, name_a, name_b, log):
    """Compare interchange intervention causal ratios."""
    if "causal_probing" not in res_a or "causal_probing" not in res_b:
        log.info("Causal probing: data missing for one or both models")
        return None

    cp_a = res_a["causal_probing"]
    cp_b = res_b["causal_probing"]

    result = {}
    for name, cp in [(name_a, cp_a), (name_b, cp_b)]:
        # Top-level causal ratio (single-layer probing)
        cr = cp.get("causal_ratio", None)
        result[f"{name}_causal_ratio"] = cr
        result[f"{name}_probe_accuracy"] = cp.get("probe_accuracy", None)
        result[f"{name}_probe_layer"] = cp.get("layer", None)

        # KL divergences
        h2s = cp.get("helix_to_sheet_kl", {})
        s2h = cp.get("sheet_to_helix_kl", {})
        rand = cp.get("random_direction_kl", {})
        result[f"{name}_helix_to_sheet_kl"] = h2s.get("mean", None)
        result[f"{name}_sheet_to_helix_kl"] = s2h.get("mean", None)
        result[f"{name}_random_direction_kl"] = rand.get("mean", None)

    log.info(f"Causal probing: causal ratio {name_a}={result.get(f'{name_a}_causal_ratio')}, "
             f"{name_b}={result.get(f'{name_b}_causal_ratio')}")
    return result


def compare_steering_vectors(res_a, res_b, name_a, name_b, log):
    """Compare steering vector dose-response."""
    if "steering_vectors" not in res_a or "steering_vectors" not in res_b:
        log.info("Steering vectors: data missing for one or both models")
        return None

    sv_a = res_a["steering_vectors"].get("evaluation", {})
    sv_b = res_b["steering_vectors"].get("evaluation", {})

    result = {}
    for vec_name in ["helix_vs_sheet", "helix_vs_coil", "sheet_vs_coil"]:
        for name, sv in [(name_a, sv_a), (name_b, sv_b)]:
            per_scale = sv.get(vec_name, {}).get("per_scale", {})
            if per_scale:
                # Max KL at scale=5
                kl_5 = per_scale.get("5.0", {}).get("mean_kl", None)
                result[f"{name}_{vec_name}_kl_at_5"] = kl_5

    log.info(f"Steering vectors: helix_vs_sheet KL@5 "
             f"{name_a}={result.get(f'{name_a}_helix_vs_sheet_kl_at_5')}, "
             f"{name_b}={result.get(f'{name_b}_helix_vs_sheet_kl_at_5')}")
    return result


def compare_cka(res_a, res_b, name_a, name_b, log):
    """Compare within-model CKA structure."""
    if "cka_within_model" not in res_a or "cka_within_model" not in res_b:
        log.info("CKA: data missing for one or both models")
        return None

    cka_a = res_a["cka_within_model"]
    cka_b = res_b["cka_within_model"]

    result = {}
    for name, cka in [(name_a, cka_a), (name_b, cka_b)]:
        matrix = cka.get("cka_matrix", [])
        if matrix:
            arr = np.array(matrix)
            # Average off-diagonal CKA (similarity between non-adjacent layers)
            n = arr.shape[0]
            off_diag = []
            for i in range(n):
                for j in range(i + 2, n):
                    off_diag.append(arr[i, j])
            result[f"{name}_avg_off_diag_cka"] = float(np.mean(off_diag)) if off_diag else None
            # Identify block structure: average CKA within first/last halves
            half = n // 2
            early = arr[:half, :half]
            late = arr[half:, half:]
            result[f"{name}_early_block_cka"] = float(np.mean(early[np.triu_indices(half, k=1)]))
            result[f"{name}_late_block_cka"] = float(np.mean(late[np.triu_indices(n - half, k=1)]))

    log.info(f"CKA: off-diag {name_a}={result.get(f'{name_a}_avg_off_diag_cka')}, "
             f"{name_b}={result.get(f'{name_b}_avg_off_diag_cka')}")
    return result


def compare_sae_feature_ablation(res_a, res_b, name_a, name_b, log):
    """Compare SAE feature ablation causal importance."""
    # Find SAE ablation files
    sae_a = {k: v for k, v in res_a.items() if k.startswith("sae_feature_ablation")}
    sae_b = {k: v for k, v in res_b.items() if k.startswith("sae_feature_ablation")}

    if not sae_a or not sae_b:
        log.info("SAE feature ablation: data missing for one or both models")
        return None

    result = {}
    for name, sae_files in [(name_a, sae_a), (name_b, sae_b)]:
        for sae_key, sae_data in sae_files.items():
            fr = sae_data.get("feature_results", {})
            if fr:
                kls = [v["mean_kl"] for v in fr.values() if "mean_kl" in v]
                result[f"{name}_{sae_key}_n_features"] = len(fr)
                result[f"{name}_{sae_key}_mean_kl"] = float(np.mean(kls)) if kls else 0
                result[f"{name}_{sae_key}_max_kl"] = float(np.max(kls)) if kls else 0
                n_significant = sum(1 for k in kls if k > 0.01)
                result[f"{name}_{sae_key}_n_significant"] = n_significant

    log.info(f"SAE feature ablation: {name_a} files={list(sae_a.keys())}, "
             f"{name_b} files={list(sae_b.keys())}")
    return result


def compare_sparse_circuits(res_a, res_b, name_a, name_b, log):
    """Compare sparse feature circuit structure."""
    if "sparse_feature_circuits" not in res_a or "sparse_feature_circuits" not in res_b:
        log.info("Sparse feature circuits: data missing for one or both models")
        return None

    sc_a = res_a["sparse_feature_circuits"]
    sc_b = res_b["sparse_feature_circuits"]

    if not sc_a.get("layer_pairs") or not sc_b.get("layer_pairs"):
        log.info("Sparse feature circuits: empty data")
        return None

    result = {}
    for name, sc in [(name_a, sc_a), (name_b, sc_b)]:
        total_connections = 0
        total_downstream = 0
        for pair_key, pair_data in sc.get("layer_pairs", {}).items():
            n_connected = pair_data.get("n_connected_downstream", 0)
            n_total = pair_data.get("n_downstream_features", 0)
            total_connections += n_connected
            total_downstream += n_total
        result[f"{name}_total_connected"] = total_connections
        result[f"{name}_total_downstream"] = total_downstream
        result[f"{name}_connection_rate"] = (total_connections / total_downstream
                                              if total_downstream > 0 else 0)

    log.info(f"Sparse circuits: connection rate {name_a}={result[f'{name_a}_connection_rate']:.2f}, "
             f"{name_b}={result[f'{name_b}_connection_rate']:.2f}")
    return result


def compare_contact_map(res_a, res_b, name_a, name_b, log):
    """Compare contact map prediction from attention."""
    if "contact_map" not in res_a or "contact_map" not in res_b:
        log.info("Contact map: data missing for one or both models")
        return None

    cm_a = res_a["contact_map"]
    cm_b = res_b["contact_map"]

    if not cm_a.get("per_layer") or not cm_b.get("per_layer"):
        log.info("Contact map: empty per_layer data")
        return None

    result = {}
    for name, cm in [(name_a, cm_a), (name_b, cm_b)]:
        result[f"{name}_n_proteins"] = cm.get("n_proteins", 0)
        result[f"{name}_best_layer"] = cm.get("best_layer", None)
        result[f"{name}_best_precision_L"] = cm.get("best_precision_L", 0)
        # Get best P@L/5
        best_l5 = 0
        for l, v in cm.get("per_layer", {}).items():
            l5 = v.get("L/5", 0)
            if l5 > best_l5:
                best_l5 = l5
        result[f"{name}_best_precision_L5"] = best_l5

    log.info(f"Contact map: best P@L {name_a}={result[f'{name_a}_best_precision_L']:.3f}, "
             f"{name_b}={result[f'{name_b}_best_precision_L']:.3f}")
    return result


def compare_crossmodal_sae(res_a, res_b, name_a, name_b, log):
    """Compare cross-modal SAE feature modulation."""
    if "crossmodal_sae_features" not in res_a or "crossmodal_sae_features" not in res_b:
        log.info("Crossmodal SAE: data missing for one or both models")
        return None

    result = {}
    for name, data in [(name_a, res_a["crossmodal_sae_features"]),
                        (name_b, res_b["crossmodal_sae_features"])]:
        result[f"{name}_n_proteins"] = data.get("n_proteins", 0)
        per_layer = data.get("per_layer", {})
        total_modulated = sum(v.get("n_modulated", 0) for v in per_layer.values())
        total_features = sum(v.get("dict_size", 0) for v in per_layer.values())
        result[f"{name}_total_modulated"] = total_modulated
        result[f"{name}_total_features"] = total_features
        result[f"{name}_fraction_modulated"] = total_modulated / max(total_features, 1)
        result[f"{name}_per_layer"] = {
            k: {"enhanced": v.get("n_structure_enhanced", 0),
                "suppressed": v.get("n_structure_suppressed", 0),
                "fraction": v.get("fraction_modulated", 0)}
            for k, v in per_layer.items()
        }

    log.info(f"Crossmodal SAE: {name_a} modulated={result[f'{name_a}_total_modulated']}, "
             f"{name_b} modulated={result[f'{name_b}_total_modulated']}")
    return result


def compare_evolutionary_correlation(res_a, res_b, name_a, name_b, log):
    """Compare evolutionary conservation correlations."""
    if "evolutionary_correlation" not in res_a or "evolutionary_correlation" not in res_b:
        log.info("Evolutionary correlation: data missing for one or both models")
        return None

    result = {}
    for name, data in [(name_a, res_a["evolutionary_correlation"]),
                        (name_b, res_b["evolutionary_correlation"])]:
        agg = data.get("aggregate", {})
        result[f"{name}_n_proteins"] = data.get("n_proteins", 0)
        for metric in ["entropy_vs_sae", "entropy_vs_mutation_kl"]:
            if metric in agg:
                result[f"{name}_{metric}_mean"] = agg[metric].get("mean", 0)
                result[f"{name}_{metric}_std"] = agg[metric].get("std", 0)
        bd = data.get("breakdown", {})
        for cat in ["buried", "exposed", "functional", "nonfunctional"]:
            if cat in bd:
                result[f"{name}_{cat}_rho"] = bd[cat].get("entropy_vs_sae_spearman", 0)

    rho_a = result.get(f"{name_a}_entropy_vs_sae_mean", 0)
    rho_b = result.get(f"{name_b}_entropy_vs_sae_mean", 0)
    log.info(f"Evolutionary correlation: {name_a} ρ={rho_a:.3f}, {name_b} ρ={rho_b:.3f}")
    return result


def compare_dms_correlation(res_a, res_b, name_a, name_b, log):
    """Compare DMS fitness correlations."""
    if "dms_correlation" not in res_a or "dms_correlation" not in res_b:
        log.info("DMS correlation: data missing for one or both models")
        return None

    result = {}
    for name, data in [(name_a, res_a["dms_correlation"]),
                        (name_b, res_b["dms_correlation"])]:
        result[f"{name}_n_proteins"] = data.get("n_dms_proteins", 0)
        agg = data.get("aggregate", {})
        for metric in ["mutation_kl", "sae_activation", "mlm_entropy"]:
            if metric in agg:
                result[f"{name}_{metric}_mean_rho"] = agg[metric].get("mean_spearman", 0)
                result[f"{name}_{metric}_std_rho"] = agg[metric].get("std_spearman", 0)

    rho_a = result.get(f"{name_a}_mutation_kl_mean_rho", 0)
    rho_b = result.get(f"{name_b}_mutation_kl_mean_rho", 0)
    log.info(f"DMS correlation: {name_a} mean ρ={rho_a:.3f}, {name_b} mean ρ={rho_b:.3f}")
    return result


def compare_phylogenetic(res_a, res_b, name_a, name_b, log):
    """Compare phylogenetic recapitulation."""
    if "phylogenetic_recapitulation" not in res_a or "phylogenetic_recapitulation" not in res_b:
        log.info("Phylogenetic recapitulation: data missing for one or both models")
        return None

    result = {}
    for name, data in [(name_a, res_a["phylogenetic_recapitulation"]),
                        (name_b, res_b["phylogenetic_recapitulation"])]:
        result[f"{name}_n_families"] = data.get("n_families", 0)
        agg = data.get("aggregate_by_layer", {})
        # Find best layer
        best_rho = -1
        best_layer = None
        for l, v in agg.items():
            if v["mean_mantel_rho"] > best_rho:
                best_rho = v["mean_mantel_rho"]
                best_layer = l
        result[f"{name}_best_layer"] = best_layer
        result[f"{name}_best_mantel_rho"] = float(best_rho)

    log.info(f"Phylogenetic: {name_a} best ρ={result[f'{name_a}_best_mantel_rho']:.3f}, "
             f"{name_b} best ρ={result[f'{name_b}_best_mantel_rho']:.3f}")
    return result


def compare_novel_features(res_a, res_b, name_a, name_b, log):
    """Compare novel feature characterization."""
    if "novel_feature_characterization" not in res_a or "novel_feature_characterization" not in res_b:
        log.info("Novel features: data missing for one or both models")
        return None

    result = {}
    for name, data in [(name_a, res_a["novel_feature_characterization"]),
                        (name_b, res_b["novel_feature_characterization"])]:
        summary = data.get("summary", {})
        result[f"{name}_total_features"] = summary.get("total_features", 0)
        result[f"{name}_n_uncharacterized"] = summary.get("n_uncharacterized", 0)
        result[f"{name}_n_novel_characterized"] = summary.get("n_characterized_novel", 0)
        result[f"{name}_hypothesis_counts"] = summary.get("hypothesis_counts", {})

    log.info(f"Novel features: {name_a} uncharacterized={result[f'{name_a}_n_uncharacterized']}, "
             f"{name_b} uncharacterized={result[f'{name_b}_n_uncharacterized']}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    args = parser.parse_args()

    log, out_dir = setup_logging()
    log.info(f"=== Cross-Model Comparison: {args.model_a} vs {args.model_b} ===")

    res_a = load_results(args.model_a, out_dir)
    res_b = load_results(args.model_b, out_dir)
    log.info(f"{args.model_a}: {sorted(res_a.keys())}")
    log.info(f"{args.model_b}: {sorted(res_b.keys())}")

    comparison = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "analyses": {}
    }

    # Run all comparisons
    comparisons = [
        ("layer_ablation", compare_layer_ablation),
        ("head_ablation", compare_head_ablation),
        ("attention", compare_attention),
        ("activation_patching", compare_activation_patching),
        ("logit_lens", compare_logit_lens),
        ("tuned_lens", compare_tuned_lens),
        ("direct_logit_attribution", compare_direct_logit_attribution),
        ("residual_decomposition", compare_residual_decomposition),
        ("ov_qk_decomposition", compare_ov_qk),
        ("attribution_patching", compare_attribution_patching),
        ("mutation_sensitivity", compare_mutation_sensitivity),
        ("causal_probing", compare_causal_probing),
        ("steering_vectors", compare_steering_vectors),
        ("cka_within_model", compare_cka),
        ("sae_feature_ablation", compare_sae_feature_ablation),
        ("sparse_feature_circuits", compare_sparse_circuits),
        ("contact_map", compare_contact_map),
        ("crossmodal_sae", compare_crossmodal_sae),
        ("evolutionary_correlation", compare_evolutionary_correlation),
        ("dms_correlation", compare_dms_correlation),
        ("phylogenetic_recapitulation", compare_phylogenetic),
        ("novel_feature_characterization", compare_novel_features),
    ]

    for name, func in comparisons:
        result = func(res_a, res_b, args.model_a, args.model_b, log)
        if result:
            comparison["analyses"][name] = result

    log.info(f"\n=== Summary: {len(comparison['analyses'])} analyses compared ===")
    for name in comparison["analyses"]:
        log.info(f"  ✓ {name}")

    # Save
    out_path = out_dir / f"comparison_{args.model_a}_vs_{args.model_b}.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
