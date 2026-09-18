"""Figure 4: Causal validation and mechanistic architecture.

Layout:
  Row 1 (a-d): steering dose-response, within-model CKA (ESM-3, ESM-2), sparse circuits
  Row 2 (e): case study protein heatmap (Serralysin)

Panels:
  a — Steering dose-response: activation change and KL vs steering strength
  b — Within-model CKA heatmap (ESM-3): sharp phase transition at L24-30
  c — Within-model CKA heatmap (ESM-2): gradual representational evolution
  d — Sparse feature circuits: significant inter-layer causal connections
  e — Case study: Serralysin SAE feature activations with functional sites
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from style import setup_style
from utils import load_json, save_fig, panel_label

setup_style()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def _panel_a(ax, steer_esm3):
    """Dual-axis: activation change + KL vs steering strength."""
    strengths = steer_esm3["strengths"]
    act_changes = []
    kl_values = []
    for alpha in strengths:
        key = f"alpha_{alpha}"
        act_changes.append(steer_esm3["summary"][key]["mean_target_act_change"])
        kl_values.append(steer_esm3["summary"][key]["mean_kl"])

    strengths = np.array(strengths)
    act_changes = np.array(act_changes)
    kl_values = np.array(kl_values)

    line_act, = ax.plot(strengths, act_changes, '-o', color=C_ESM3,
                        markersize=3.5, linewidth=1.0, zorder=5,
                        label="Activation change")
    ax.set_xlabel("Steering strength (\u03b1)")
    ax.set_ylabel("Target activation change", color=C_ESM3)
    ax.tick_params(axis='y', labelcolor=C_ESM3)

    ax.axhline(0, color="#CCCCCC", linewidth=0.4, linestyle="-", zorder=0)
    ax.axvline(0, color="#CCCCCC", linewidth=0.4, linestyle="-", zorder=0)

    mask = np.abs(strengths) <= 20
    slope, intercept = np.polyfit(strengths[mask], act_changes[mask], 1)
    fit_x = np.linspace(strengths.min(), strengths.max(), 100)
    ax.plot(fit_x, slope * fit_x + intercept, '--', color=C_ESM3,
            linewidth=0.5, alpha=0.4, zorder=3)

    ax2 = ax.twinx()
    line_kl, = ax2.plot(strengths, kl_values, '--s', color=C_ESM2,
                        markersize=2.5, linewidth=0.8, zorder=4,
                        label="KL divergence")
    ax2.set_ylabel("KL div.", fontsize=AXIS_LABEL_SIZE - 1, color=C_ESM2)
    ax2.tick_params(axis='y', labelcolor=C_ESM2)
    ax2.spines['right'].set_visible(True)

    r_val = np.corrcoef(strengths, act_changes)[0, 1]
    ax.text(0.03, 0.97, f"r = {r_val:.3f}", transform=ax.transAxes,
            fontsize=TICK_SIZE, va="top", ha="left", color=C_ESM3)

    # Legend removed — y-axis colors convey which line is which


def _panel_cka(ax, data_path, show_ylabel=True):
    """Within-model CKA heatmap. Returns the image for shared colorbar."""
    cka = load_json(data_path)
    mat = np.array(cka["cka_matrix"])
    layers = cka["layers"]
    cmap = LinearSegmentedColormap.from_list("v", VIRIDIS, N=256)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=TICK_SIZE - 1)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=TICK_SIZE - 1)
    ax.set_xlabel("Layer")
    if show_ylabel:
        ax.set_ylabel("Layer")
    return im


def _panel_circuits(ax, circuits_esm3, circuits_esm2):
    """Sparse feature circuits grouped bar chart."""
    entries = []

    def _extract(circuits, model_name, color):
        for lp_name, lp_data in circuits["layer_pairs"].items():
            total_sig = sum(c["n_significant_upstream"] for c in lp_data["connections"])
            n_ds = lp_data.get("n_connected_downstream", len(lp_data["connections"]))
            pair_label = lp_name.replace("_to_", "\u2192")
            entries.append({
                "model": model_name, "pair": pair_label,
                "total_sig": total_sig, "n_downstream": n_ds, "color": color,
            })

    _extract(circuits_esm3, "ESM-3", C_ESM3)
    _extract(circuits_esm2, "ESM-2", C_ESM2)

    n = len(entries)
    x = np.arange(n)
    bar_width = 0.35

    for i, e in enumerate(entries):
        ax.bar(x[i] - bar_width / 2, e["total_sig"], bar_width,
               color=e["color"], edgecolor="white", linewidth=0.4, alpha=0.85)
    for i, e in enumerate(entries):
        ax.bar(x[i] + bar_width / 2, e["n_downstream"], bar_width,
               color=e["color"], edgecolor="white", linewidth=0.4, alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{e['model']}\n{e['pair']}" for e in entries], fontsize=TICK_SIZE - 1)
    ax.set_ylabel("Count")

    for i, e in enumerate(entries):
        ax.text(x[i] - bar_width / 2, e["total_sig"] + 2,
                f"{e['total_sig']}", ha="center", va="bottom",
                fontsize=TICK_SIZE - 0.5, color="#333333")
        ax.text(x[i] + bar_width / 2, e["n_downstream"] + 2,
                f"{e['n_downstream']}", ha="center", va="bottom",
                fontsize=TICK_SIZE - 0.5, color="#999999")


def _panel_case_study(ax, case_studies, accession="Q03023"):
    """Single case study protein heatmap."""
    study = None
    for cs in case_studies["case_studies"]:
        if cs["accession"] == accession:
            study = cs
            break
    if study is None:
        ax.text(0.5, 0.5, "Case study not found", transform=ax.transAxes, ha="center")
        return

    length = study["length"]
    n_features = 8
    features = study["top_features"][:n_features]

    act_matrix = np.zeros((n_features, length))
    for fi, feat in enumerate(features):
        for pos in feat["active_positions"]:
            if pos < length:
                act_matrix[fi, pos] = 1.0
                for dp in [-1, 1]:
                    if 0 <= pos + dp < length:
                        act_matrix[fi, pos + dp] = max(act_matrix[fi, pos + dp], 0.5)

    cmap = LinearSegmentedColormap.from_list(
        "feature_act", ["#1a1a2e", "#16213e", "#e94560", "#ffd700"], N=256)
    ax.imshow(act_matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)

    # Y-axis: feature labels with cross-modal markers
    feat_labels = []
    feat_colors = []
    for feat in features:
        fid = feat["feature_id"]
        cm_label = feat["cross_modal_label"]
        marker = ""
        if cm_label == "structure-enhanced":
            marker = "*"
            feat_colors.append(C_ENHANCED)
        elif cm_label == "structure-suppressed":
            marker = "\u2020"
            feat_colors.append(C_SUPPRESSED)
        else:
            feat_colors.append("#333333")
        feat_labels.append(f"F{fid}{marker}")

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(feat_labels, fontsize=TICK_SIZE - 0.5)
    for tick_label, color in zip(ax.get_yticklabels(), feat_colors):
        tick_label.set_color(color)

    # X-axis
    tick_step = 100 if length > 200 else 50
    xticks = np.arange(0, length, tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.astype(int), fontsize=TICK_SIZE - 0.5)
    ax.set_xlabel("Residue position")

    # Functional site markers
    try:
        meta = load_json(ROOT / "data" / "eval_expanded" / "metadata.json")
        if accession in meta:
            func_sites = meta[accession].get("features", [])
            site_positions = []
            for fs in func_sites:
                ft = fs.get("type", "")
                if ft in ["Active site", "Binding site", "Metal binding",
                          "Site", "Disulfide bond", "Modified residue"]:
                    ps = fs.get("start", 1) - 1
                    pe = fs.get("end", ps + 1)
                    for p in range(ps, pe):
                        if 0 <= p < length:
                            site_positions.append(p)
            if site_positions:
                ax.scatter(site_positions, [n_features - 0.3] * len(site_positions),
                           marker="^", s=10, color="#CC3311", zorder=10,
                           clip_on=False, linewidths=0)
    except Exception:
        pass

    # Title
    name = study["name"]
    if len(name) > 30:
        name = name[:27] + "..."
    n_func = study["n_functional_sites"]
    site_types = ", ".join(study["functional_site_types"].keys())
    ax.set_title(f"{accession} \u2014 {name} ({length} aa, {n_func} {site_types} sites)",
                 fontsize=TICK_SIZE, pad=3, loc="left", color="#333333")


def _panel_b_controls(ax):
    """Steering controls: real vs random vs shuffled vs orthogonalized."""
    from pypubfigs.palettes import friendly_pal
    ito = friendly_pal("ito_seven")

    try:
        sc = load_json(RESULTS_SCALED / "steering_controls.json")
    except Exception:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    dirs = ["real", "random", "shuffled_decoder", "orthogonalized"]
    dir_labels = ["Real", "Random", "Shuffled", "Orthog."]
    dir_colors = [ito[1], ito[4], ito[5], ito[3]]
    dir_markers = ["-o", "--s", "-^", ":d"]
    alphas = [5.0, 10.0, 20.0, 50.0]

    for di, (dname, dlabel, dcolor) in enumerate(zip(dirs, dir_labels, dir_colors)):
        kls = [sc["per_direction"][dname].get(f"alpha_{a}", {}).get("mean_kl", 0) for a in alphas]
        ax.plot(alphas, kls, dir_markers[di], color=dcolor, markersize=4, lw=1.2, label=dlabel, zorder=5 - di)

    ax.set_xlabel("Steering strength (\u03b1)")
    ax.set_ylabel("KL divergence")
    ax.set_yscale("log")
    ax.legend(fontsize=TICK_SIZE, loc="upper left", frameon=False)


def generate():
    """Create and save Figure 4."""
    steer_esm3 = load_json(RESULTS_SCALED / "steering_vectors.json")
    circuits_esm3 = load_json(RESULTS_UNIFIED_ESM3 / "sparse_feature_circuits.json")
    circuits_esm2 = load_json(RESULTS_UNIFIED_ESM2 / "sparse_feature_circuits.json")

    fig = plt.figure(figsize=(DOUBLE_COL, 2.8))

    gs = gridspec.GridSpec(1, 4, figure=fig,
                           width_ratios=[1.0, 0.12, 1.0, 1.0],
                           left=0.07, right=0.97, top=0.90, bottom=0.20,
                           wspace=0.45)

    # Panel a: steering dose-response
    ax_a = fig.add_subplot(gs[0, 0])
    _panel_a(ax_a, steer_esm3)
    panel_label(ax_a, "a", x=-0.18)

    # Gap column 1 is empty

    # Panel b: steering controls
    ax_b = fig.add_subplot(gs[0, 2])
    _panel_b_controls(ax_b)
    panel_label(ax_b, "b")

    # Panel c: sparse circuits
    ax_c = fig.add_subplot(gs[0, 3])
    _panel_circuits(ax_c, circuits_esm3, circuits_esm2)
    panel_label(ax_c, "c")

    save_fig(fig, "fig4_causal")
    print("Figure 4 complete.")


if __name__ == "__main__":
    generate()
