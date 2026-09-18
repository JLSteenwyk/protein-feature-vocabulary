#!/usr/bin/env python3
"""Figure 3: How ESM-3 integrates structural information.

2x2 layout:
  a) CKA modality dropout — CKA(S, S+St) across layers
  b) Cross-modal feature breakdown — donut chart
  c) Attribution patching — ESM-3 vs ESM-2 by relative depth
  d) Attention heatmaps — ESM-3 JSD and ESM-2 entropy side by side

Usage:
    ./env/bin/python scripts/publication_figures/generate_all.py --fig 3
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from style import setup_style
from utils import load_json, save_fig, panel_label


def _build_viridis_cmap():
    """Build a continuous viridis colormap from the pypubfigs discrete palette."""
    return LinearSegmentedColormap.from_list("viridis_pub", VIRIDIS, N=256)


def _panel_a(ax, cka_data):
    """CKA modality dropout: CKA(S, S+St) across ESM-3 layers."""
    layers = cka_data["layers"]
    cka_vals = [cka_data["cka_s_vs_sst"][str(l)] for l in layers]

    # Shaded integration zone (L28-L38, matching dashed lines)
    iz_start = cka_data["integration_zone_start"]  # 28
    iz_end = cka_data["convergence_point"]          # 38
    ax.axvspan(iz_start, iz_end, alpha=0.12, color=C_ESM3, zorder=0)

    # Vertical dashed lines at zone boundaries
    ax.axvline(iz_start, color=C_ESM3, ls="--", lw=0.6, alpha=0.5, zorder=1)
    ax.axvline(iz_end, color=C_ESM3, ls="--", lw=0.6, alpha=0.5, zorder=1)

    # Main line
    ax.plot(layers, cka_vals, "-o", color=C_ESM3, markersize=3.5,
            markeredgecolor="white", markeredgewidth=0.3, lw=1.2, zorder=3)

    # Integration zone text removed — described in figure legend

    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA(S, S+St)")
    ax.set_xlim(-1, 48)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 8, 16, 24, 33, 42, 47])


def _panel_b(ax, feat_data):
    """Cross-modal feature categories with GO enrichment comparison."""
    import numpy as _np
    n_enh = feat_data["n_structure_enhanced"]
    n_sup = feat_data["n_structure_suppressed"]
    n_inv = feat_data["n_structure_invariant"]
    total = n_enh + n_sup + n_inv

    # Bar chart: category counts + GO enrichment comparison
    categories = ["Enhanced", "Suppressed", "Invariant"]
    counts = [n_enh, n_sup, n_inv]
    fracs = [n / total * 100 for n in counts]
    colors = [C_ENHANCED, C_SUPPRESSED, C_INVARIANT]

    bars = ax.bar(range(3), counts, color=colors, edgecolor="white", linewidth=0.5)
    for i, (bar, frac, cnt) in enumerate(zip(bars, fracs, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt * 1.15,
                f"{cnt:,}\n({frac:.1f}%)", ha="center", fontsize=6, va="bottom")

    ax.set_xticks(range(3))
    ax.set_xticklabels(categories, fontsize=TICK_SIZE)
    ax.set_ylabel("Number of features")
    ax.set_yscale("log")
    ax.set_ylim(100, 20000)
    # Title removed — goes in figure legend

    # Annotate: enhanced features are more GO-enriched and more convergent
    try:
        cmc = load_json(RESULTS_SCALED / "cross_modal_convergence_analysis.json")
        enh_go = cmc["cross_modal_go_enrichment"]["enhanced_median_go_terms"]
        inv_go = cmc["cross_modal_go_enrichment"]["invariant_median_go_terms"]
        enh_r = cmc["convergence_x_crossmodal"]["enhanced_median_r"]
        inv_r = cmc["convergence_x_crossmodal"]["invariant_median_r"]
        ax.text(0.03, 0.97,
                f"Enhanced vs Invariant:\n"
                f"  GO terms: {enh_go:.0f} vs {inv_go:.0f}\n"
                f"  Conv. r: {enh_r:.2f} vs {inv_r:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", linewidth=0.5, alpha=0.9))
    except Exception:
        pass


def _panel_c(ax, esm3_data, esm2_data):
    """Attribution patching effect by relative depth for both models."""
    # ESM-3
    esm3_layers = sorted(int(k) for k in esm3_data["per_layer"])
    esm3_n = esm3_data["n_layers"]
    esm3_depth = [l / (esm3_n - 1) for l in esm3_layers]
    esm3_effect = [esm3_data["per_layer"][str(l)]["mean_effect"] for l in esm3_layers]

    # ESM-2
    esm2_layers = sorted(int(k) for k in esm2_data["per_layer"])
    esm2_n = esm2_data["n_layers"]
    esm2_depth = [l / (esm2_n - 1) for l in esm2_layers]
    esm2_effect = [esm2_data["per_layer"][str(l)]["mean_effect"] for l in esm2_layers]

    ax.semilogy(esm3_depth, esm3_effect, "-o", color=C_ESM3, markersize=2.5,
                markeredgecolor="white", markeredgewidth=0.2, lw=1.0,
                label="ESM-3 cross", zorder=3)
    ax.semilogy(esm2_depth, esm2_effect, "-s", color=C_ESM2, markersize=2.5,
                markeredgecolor="white", markeredgewidth=0.2, lw=1.0,
                label="ESM-2 cross", zorder=3)

    # Add same-protein attribution (ESM-3 only) as validation
    try:
        sp = load_json(RESULTS_SCALED / "same_protein_causal.json")
        sp_attr = sp["same_protein_attribution"]
        sp_layers = sorted(int(k) for k in sp_attr["per_layer"])
        sp_depth = [l / (esm3_n - 1) for l in sp_layers]
        sp_effect = [sp_attr["per_layer"][str(l)]["mean_effect"] for l in sp_layers]
        ax.semilogy(sp_depth, sp_effect, "--^", color=C_ESM3, markersize=2,
                    markeredgecolor="white", markeredgewidth=0.2, lw=0.8,
                    alpha=0.6, label="ESM-3 same", zorder=2)
    except Exception:
        pass

    ax.set_xlabel("Relative depth (layer / N)")
    ax.set_ylabel("Mean attribution effect")
    ax.legend(fontsize=LEGEND_SIZE - 0.5, frameon=False, loc="upper left")
    ax.set_xlim(-0.02, 1.02)


def _panel_d(ax_s, ax_sst):
    """Head ablation heatmaps: ESM-3 S-only and ESM-3 S+St."""
    cmap = _build_viridis_cmap()

    # ESM-3 S-only and S+St from matched experiment
    try:
        ha_sst = load_json(RESULTS_SCALED / "head_ablation_sst.json")
        heatmaps = [
            (ax_s, ha_sst["condition_s"]["per_head"], 24, "ESM-3 S"),
            (ax_sst, ha_sst["condition_sst"]["per_head"], 24, "ESM-3 S+St"),
        ]
    except Exception:
        ha_s = load_json(RESULTS_UNIFIED_ESM3 / "head_ablation.json")
        heatmaps = [
            (ax_s, ha_s["per_head"], 24, "ESM-3 S"),
            (ax_sst, ha_s["per_head"], 24, "ESM-3 S"),
        ]

    # Shared color scale
    all_kls = []
    for ax, ph, nh, title in heatmaps:
        entries = list(ph.values()) if isinstance(ph, dict) else ph
        all_kls.extend([e.get("mean_kl", 0) for e in entries])
    vmax = max(all_kls) if all_kls else 0.3

    for hi, (ax, per_head, n_heads, title) in enumerate(heatmaps):
        entries = list(per_head.values()) if isinstance(per_head, dict) else per_head
        abl_layers = sorted(set(int(e["layer"]) for e in entries))

        mat = np.zeros((len(abl_layers), n_heads))
        lmap = {l: i for i, l in enumerate(abl_layers)}
        for e in entries:
            li = lmap.get(int(e["layer"]))
            if li is not None and int(e["head"]) < n_heads:
                mat[li, int(e["head"])] = e.get("mean_kl", 0)

        im = ax.imshow(mat.T, aspect="auto", cmap=cmap,
                        interpolation="nearest", origin="lower",
                        vmin=0, vmax=vmax)
        ax.set_xlabel("Layer")
        if hi == 0:
            ax.set_ylabel("Head")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=AXIS_LABEL_SIZE, pad=3)
        ax.set_xticks(range(len(abl_layers)))
        ax.set_xticklabels(abl_layers, fontsize=TICK_SIZE - 1)
        ax.set_yticks(range(0, n_heads, 4))

        # Store last image for shared colorbar
        if hi == 1:
            last_im = im

        # Annotate peak
        peak_idx = np.unravel_index(mat.T.argmax(), mat.T.shape)
        peak_head, peak_layer_idx = peak_idx
        peak_kl = mat.T[peak_head, peak_layer_idx]
        peak_layer = abl_layers[peak_layer_idx]
        tx = min(peak_layer_idx + 1, len(abl_layers) - 1)
        ty = min(peak_head + 5, n_heads - 1)
        ax.annotate(f"L{peak_layer}H{peak_head}\n{peak_kl:.2f}",
                    xy=(peak_layer_idx, peak_head),
                    xytext=(tx, ty),
                    fontsize=6, color="white",
                    arrowprops=dict(arrowstyle="->", color="white", lw=0.3))

    return last_im


def _panel_l0h7(ax):
    """L0H7 vs random L0 heads: functional impact comparison."""
    from pypubfigs.palettes import friendly_pal
    ito = friendly_pal("ito_seven")

    try:
        data = load_json(RESULTS_SCALED / "l0h7_vs_random_heads.json")
        h7 = data["l0h7"]
        rs = data["random_summary"]
    except Exception:
        ax.text(0.5, 0.5, "Data not\navailable", transform=ax.transAxes, ha="center")
        return

    measures = ["SS3", "Token", "Enh. SAE"]
    h7_fracs = [
        h7["ss3_agreement"],
        h7["token_agreement"],
        h7["enhanced_sae_ablated"] / (h7["enhanced_sae_normal"] + 1e-10),
    ]
    rand_fracs = [
        rs["mean_ss3_agreement"],
        rs["mean_token_agreement"],
        rs["mean_enh_sae_frac"],
    ]
    rand_errs = [
        rs["std_ss3_agreement"],
        rs["std_token_agreement"],
        rs["std_enh_sae_frac"],
    ]

    x = np.arange(len(measures))
    width = 0.35

    # L0H7 bars
    ax.bar(x - width/2, h7_fracs, width, color=ito[1],
           edgecolor="white", linewidth=0.3, label="L0H7")

    # Random L0 heads (mean ± std)
    ax.bar(x + width/2, rand_fracs, width, color=ito[4],
           edgecolor="white", linewidth=0.3, yerr=rand_errs,
           capsize=2, error_kw={"linewidth": 0.5},
           label="Random L0\n(n=10)")

    # Value labels on L0H7 bars
    for i, frac in enumerate(h7_fracs):
        pct_lost = (1 - frac) * 100
        ax.text(x[i] - width/2, frac + 0.02,
                f"-{pct_lost:.0f}%", ha="center", va="bottom", fontsize=6, color="#CC3311")

    ax.set_xticks(x)
    ax.set_xticklabels(measures, fontsize=TICK_SIZE - 1)
    ax.set_ylabel("Fraction remaining\nafter head ablation")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(1.0, color="#CCCCCC", ls="--", lw=0.5)
    ax.legend(fontsize=6, loc="lower center", frameon=False, ncol=2,
              bbox_to_anchor=(0.5, -0.25))


def generate():
    """Generate Figure 3."""
    setup_style()

    # --- Load data ---
    cka_data = load_json(RESULTS_SCALED / "cka_modality_dropout.json")
    feat_data = load_json(RESULTS_SCALED / "cross_modal_features.json")
    esm3_attrib = load_json(RESULTS_SCALED / "attribution_patching.json")
    esm2_attrib = load_json(RESULTS_UNIFIED_ESM2 / "attribution_patching.json")

    # --- Figure layout: 3 rows ---
    # Row 1: a (CKA line), b (CKA ESM-3 heatmap), c (CKA ESM-2 heatmap)
    # Row 2: d (cross-modal categories), e (attribution patching)
    # Row 3: f (head ablation S / S+St), g (L0H7 impact)
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=(7.2, 8.5))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.45,
                           height_ratios=[1.0, 1.0, 1.0],
                           left=0.08, right=0.97, top=0.97, bottom=0.05)

    # Row 1: CKA story
    ax_a = fig.add_subplot(gs[0, 0])
    _panel_a(ax_a, cka_data)
    panel_label(ax_a, "a")

    # Panel b: Within-model CKA ESM-3
    ax_b = fig.add_subplot(gs[0, 1])
    cka3 = load_json(RESULTS_UNIFIED_ESM3 / "cka_within_model.json")
    mat3 = np.array(cka3["cka_matrix"])
    layers3 = cka3["layers"]
    cmap_cka = LinearSegmentedColormap.from_list("v", VIRIDIS, N=256)
    im_b = ax_b.imshow(mat3, cmap=cmap_cka, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax_b.set_xticks(range(len(layers3))); ax_b.set_xticklabels(layers3, fontsize=TICK_SIZE - 1)
    ax_b.set_yticks(range(len(layers3))); ax_b.set_yticklabels(layers3, fontsize=TICK_SIZE - 1)
    ax_b.set_xlabel("Layer"); ax_b.set_ylabel("Layer")
    panel_label(ax_b, "b")

    # Panel c: Within-model CKA ESM-2
    ax_c = fig.add_subplot(gs[0, 2])
    cka2 = load_json(RESULTS_UNIFIED_ESM2 / "cka_within_model.json")
    mat2 = np.array(cka2["cka_matrix"])
    layers2 = cka2["layers"]
    im_c = ax_c.imshow(mat2, cmap=cmap_cka, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax_c.set_xticks(range(len(layers2))); ax_c.set_xticklabels(layers2, fontsize=TICK_SIZE - 1)
    ax_c.set_yticks(range(len(layers2))); ax_c.set_yticklabels(layers2, fontsize=TICK_SIZE - 1)
    ax_c.set_xlabel("Layer"); ax_c.set_ylabel("")
    panel_label(ax_c, "c", x=-0.08)

    # Shared CKA colorbar to the right of panel c
    pos_c = ax_c.get_position()
    cbar_cka_ax = fig.add_axes([pos_c.x1 + 0.01, pos_c.y0, 0.008, pos_c.height])
    cb_cka = fig.colorbar(im_c, cax=cbar_cka_ax)
    cb_cka.set_label("CKA", fontsize=TICK_SIZE)
    cb_cka.ax.tick_params(labelsize=TICK_SIZE - 1)

    # Row 2: feature story
    ax_d = fig.add_subplot(gs[1, 0])
    _panel_b(ax_d, feat_data)
    panel_label(ax_d, "d", x=-0.05)

    ax_e = fig.add_subplot(gs[1, 1:])
    _panel_c(ax_e, esm3_attrib, esm2_attrib)
    panel_label(ax_e, "e")

    # Row 3: attention story
    gs_f = gs[2, :2].subgridspec(1, 2, wspace=0.15)
    ax_f1 = fig.add_subplot(gs_f[0])
    ax_f2 = fig.add_subplot(gs_f[1])
    f_im = _panel_d(ax_f1, ax_f2)
    panel_label(ax_f1, "f", x=-0.20, y=1.15)

    # Horizontal colorbar below heatmaps
    pos_f1 = ax_f1.get_position()
    pos_f2 = ax_f2.get_position()
    cbar_left_f = pos_f1.x0 + pos_f1.width * 0.3
    cbar_right_f = pos_f2.x0 + pos_f2.width * 0.7
    cbar_f_ax = fig.add_axes([cbar_left_f, pos_f1.y0 - 0.07, cbar_right_f - cbar_left_f, 0.008])
    cb_f = fig.colorbar(f_im, cax=cbar_f_ax, orientation="horizontal")
    cb_f.set_label("KL divergence", fontsize=TICK_SIZE)
    cb_f.ax.tick_params(labelsize=TICK_SIZE - 1)

    ax_g = fig.add_subplot(gs[2, 2])
    _panel_l0h7(ax_g)
    panel_label(ax_g, "g")

    save_fig(fig, "fig3_multimodal")
    print("Figure 3 complete.")


if __name__ == "__main__":
    generate()
