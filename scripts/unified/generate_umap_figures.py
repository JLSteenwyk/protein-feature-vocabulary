#!/usr/bin/env python3
"""Generate UMAP visualizations for the publication.

1. SAE decoder vector UMAP — feature geometry colored by biological type
2. Dataset UMAP — proteins in embedding space colored by function

Usage:
    ./env/bin/python scripts/unified/generate_umap_figures.py
"""

import sys
import os
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("umap_fig")


# ── Consistent color palette ──
CATEGORY_COLORS = {
    # Feature types
    "AA-specific": "#E41A1C",
    "SS-helix": "#377EB8",
    "SS-strand": "#4DAF4A",
    "SS-coil": "#984EA3",
    "Functional": "#FF7F00",
    "Dead": "#CCCCCC",
    "Uncharacterized": "#A0A0A0",
    # Protein categories
    "enzymes": "#E41A1C",
    "signaling": "#377EB8",
    "structural": "#4DAF4A",
    "transport": "#984EA3",
    "immune": "#FF7F00",
    "transcription": "#F781BF",
    "other": "#A0A0A0",
}


def load_sae_decoder(ckpt_path):
    """Load SAE and return decoder weight matrix (dict_size, input_dim)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    # decoder.weight shape: (input_dim, dict_size) -> transpose to (dict_size, input_dim)
    W_dec = sae.decoder.weight.data.T.numpy()  # (dict_size, input_dim)
    return W_dec, ckpt["sae_config"]


def classify_features_esm2(n_features, h5_path, sae_ckpt_path, sequences, offsets, accs,
                            sample_proteins=300):
    """Classify ESM-2 SAE features by type using activation patterns."""
    log.info("Classifying ESM-2 features...")
    ckpt = torch.load(sae_ckpt_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

    rng = np.random.RandomState(42)
    sample = list(rng.choice(accs, min(sample_proteins, len(accs)), replace=False))

    feat_count = np.zeros(n_features)
    feat_aa_counts = np.zeros((n_features, 20))

    with h5py.File(h5_path, "r") as f:
        acts = f["activations"]
        total = acts.shape[0]

        for pi, acc in enumerate(sample):
            start, L = offsets[acc]
            seq = sequences[acc]
            L = min(len(seq), L)
            if start + L > total:
                continue

            raw = torch.tensor(acts[start:start+L], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw).numpy()

            active = z > 0
            feat_count += active.sum(axis=0)

            # AA distribution (vectorized per AA type)
            aa_indices = np.array([aa_to_idx.get(seq[p], -1) for p in range(L)])
            for aa_idx in range(20):
                mask = aa_indices == aa_idx
                if mask.any():
                    feat_aa_counts[:, aa_idx] += (z[mask] * active[mask]).sum(axis=0)

            if (pi + 1) % 100 == 0:
                log.info(f"  Classified {pi+1}/{len(sample)} proteins")

    # Classify based on activation patterns
    labels = []
    bg_aa = feat_aa_counts.sum(axis=0)
    bg_aa = bg_aa / (bg_aa.sum() + 1e-10)

    for fi in range(n_features):
        if feat_count[fi] < 5:
            labels.append("Dead")
            continue

        # AA enrichment
        aa_dist = feat_aa_counts[fi]
        aa_dist_norm = aa_dist / (aa_dist.sum() + 1e-10)
        enrichment = aa_dist_norm / (bg_aa + 1e-10)
        max_enr = enrichment.max()
        top_aa = AA_LIST[int(np.argmax(enrichment))]

        if max_enr > 5.0:
            labels.append("AA-specific")
        else:
            labels.append("Uncharacterized")

    return labels


def classify_features_with_annotations(n_features, func_ann_path, struct_ann_path):
    """Classify features using pre-computed annotation files."""
    labels = ["Uncharacterized"] * n_features

    # Load functional annotations
    func_features = {}
    if func_ann_path and Path(func_ann_path).exists():
        with open(func_ann_path) as f:
            for a in json.load(f):
                func_features[a["feature_idx"]] = a

    # Load structural annotations
    struct_features = {}
    if struct_ann_path and Path(struct_ann_path).exists():
        with open(struct_ann_path) as f:
            data = json.load(f)
        feats = data if isinstance(data, list) else data.get("features", [])
        for a in feats:
            struct_features[a["feature_idx"]] = a

    # Classify from annotations
    for fi, a in func_features.items():
        if fi >= n_features:
            continue
        top_aa = a.get("top_enriched_aa", [])
        func_frac = a.get("functional_site_frac", 0)
        freq = a.get("activation_freq", 1.0)

        if freq < 0.001:
            labels[fi] = "Dead"
        elif top_aa and top_aa[0][1] > 5.0:
            labels[fi] = "AA-specific"
        elif func_frac > 0.05:
            labels[fi] = "Functional"

    for fi, a in struct_features.items():
        if fi >= n_features:
            continue
        if labels[fi] not in ("Dead", "AA-specific", "Functional"):
            ss_enr = a.get("ss_enrichment", {})
            pref = a.get("ss_preference", "")
            max_ss = a.get("max_ss_enrichment", 0)
            if max_ss > 1.3:
                if pref == "H":
                    labels[fi] = "SS-helix"
                elif pref == "E":
                    labels[fi] = "SS-strand"
                elif pref == "C":
                    labels[fi] = "SS-coil"

    return labels


def classify_features_from_novel(novel_path, n_features):
    """Use novel feature characterization results for classification."""
    labels = ["Uncharacterized"] * n_features

    if not Path(novel_path).exists():
        return labels

    with open(novel_path) as f:
        data = json.load(f)

    classified = data.get("summary", {}).get("classified", {})
    # The classified dict has type -> count, but we need per-feature labels
    # Use the novel_features list for detailed per-feature info
    # But we need the full classification. Let's use the summary to set proportions
    # and the novel_features for the uncharacterized subset.

    # Actually, the classified dict tells us counts but not which features.
    # We need to re-derive from the data. Let me check if there's a per-feature list.
    return labels  # Fall back to annotation-based classification


def generate_decoder_umap(model_name, ckpt_path, labels, out_path, title=""):
    """Generate UMAP of SAE decoder vectors colored by feature type."""
    log.info(f"Loading decoder weights from {ckpt_path}...")
    W_dec, config = load_sae_decoder(ckpt_path)
    n_features = W_dec.shape[0]
    log.info(f"Decoder matrix: {W_dec.shape}")

    # Normalize decoder vectors (direction matters more than magnitude)
    norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_dec_norm = W_dec / (norms + 1e-10)

    # Filter out dead features for cleaner UMAP
    alive_mask = np.array([l != "Dead" for l in labels])
    alive_indices = np.where(alive_mask)[0]
    alive_labels = [labels[i] for i in alive_indices]
    W_alive = W_dec_norm[alive_indices]

    log.info(f"Alive features: {len(alive_indices)}/{n_features}")
    log.info(f"Label distribution: {Counter(alive_labels)}")

    # Compute UMAP
    log.info("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                        random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(W_alive)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot uncharacterized first (background), then specific types on top
    plot_order = ["Uncharacterized", "SS-coil", "SS-helix", "SS-strand",
                  "Functional", "AA-specific"]

    for category in plot_order:
        mask = np.array([l == category for l in alive_labels])
        if mask.sum() == 0:
            continue
        color = CATEGORY_COLORS.get(category, "#999999")
        alpha = 0.15 if category == "Uncharacterized" else 0.7
        size = 3 if category == "Uncharacterized" else 12
        zorder = 1 if category == "Uncharacterized" else 2
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=color, s=size, alpha=alpha, zorder=zorder,
                  label=f"{category} ({mask.sum()})", edgecolors="none")

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2, framealpha=0.8, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}.png/pdf")

    return embedding, alive_indices, alive_labels


def generate_dataset_umap(h5_path, sequences, metadata, offsets, accs,
                          out_path, title="", max_proteins=4793):
    """Generate UMAP of proteins in embedding space colored by function."""
    log.info("Computing mean-pooled embeddings...")

    with h5py.File(h5_path, "r") as f:
        acts = f["activations"]
        total = acts.shape[0]

        embeddings = []
        valid_accs = []
        categories = []

        for acc in accs:
            start, L = offsets[acc]
            seq = sequences[acc]
            L = min(len(seq), L)
            if start + L > total:
                continue

            # Mean-pool the activations
            raw = acts[start:start+L]
            mean_emb = np.mean(raw, axis=0)
            embeddings.append(mean_emb)
            valid_accs.append(acc)

            # Categorize protein
            m = metadata.get(acc, {})
            cat = categorize_protein(m)
            categories.append(cat)

        if len(valid_accs) > max_proteins:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(valid_accs), max_proteins, replace=False)
            embeddings = [embeddings[i] for i in idx]
            valid_accs = [valid_accs[i] for i in idx]
            categories = [categories[i] for i in idx]

    embeddings = np.array(embeddings)
    log.info(f"Embedding matrix: {embeddings.shape}")
    log.info(f"Category distribution: {Counter(categories)}")

    # UMAP
    log.info("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                        random_state=42, n_jobs=1)
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    plot_order = ["other", "transport", "structural", "immune",
                  "transcription", "signaling", "enzymes"]

    for category in plot_order:
        mask = np.array([c == category for c in categories])
        if mask.sum() == 0:
            continue
        color = CATEGORY_COLORS.get(category, "#999999")
        alpha = 0.3 if category == "other" else 0.6
        size = 4 if category == "other" else 8
        zorder = 1 if category == "other" else 2
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                  c=color, s=size, alpha=alpha, zorder=zorder,
                  label=f"{category.capitalize()} ({mask.sum()})",
                  edgecolors="none")

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2, framealpha=0.8, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}.png/pdf")

    return embedding_2d, valid_accs, categories


def categorize_protein(meta):
    """Assign a broad functional category to a protein based on metadata."""
    keywords = set(kw.lower() for kw in meta.get("keywords", []))
    go_terms = set(g.get("term", "").lower() for g in meta.get("go_terms", []))
    name = meta.get("protein_name", "").lower()
    ec = meta.get("ec_numbers", [])
    category_field = meta.get("category", "").lower()

    # Use category field if available
    if "enzyme" in category_field:
        return "enzymes"
    if "signal" in category_field or "kinase" in category_field:
        return "signaling"
    if "transport" in category_field or "channel" in category_field:
        return "transport"
    if "immune" in category_field or "immuno" in category_field:
        return "immune"
    if "transcript" in category_field or "dna-binding" in category_field:
        return "transcription"
    if "structural" in category_field or "cytoskel" in category_field:
        return "structural"

    # Fall back to keywords
    if ec:
        return "enzymes"
    if any(k in keywords for k in ["kinase", "phosphatase", "transferase",
                                     "hydrolase", "oxidoreductase", "ligase"]):
        return "enzymes"
    if any(k in keywords for k in ["receptor", "signal transduction", "g-protein"]):
        return "signaling"
    if any(k in keywords for k in ["transport", "ion channel", "membrane transport"]):
        return "transport"
    if any(k in keywords for k in ["immunity", "immunoglobulin", "complement",
                                     "antigen", "mhc"]):
        return "immune"
    if any(k in keywords for k in ["transcription", "dna-binding", "repressor",
                                     "activator", "chromatin"]):
        return "transcription"
    if any(k in keywords for k in ["structural protein", "cytoskeleton", "collagen"]):
        return "structural"

    # Name-based fallback
    if any(w in name for w in ["kinase", "phosphatase", "synthase", "transferase",
                                "dehydrogenase", "oxidase", "reductase", "protease",
                                "lyase", "ligase", "isomerase"]):
        return "enzymes"
    if any(w in name for w in ["receptor", "gtp-binding"]):
        return "signaling"

    return "other"


def main():
    out_dir = ROOT / "results" / "figures" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load shared data
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)
    offsets, accs = build_protein_offsets(sequences)

    # ═══════════════════════════════════════════════════
    # 1. SAE decoder UMAP — ESM-2 Layer 24
    # ═══════════════════════════════════════════════════
    log.info("\n=== ESM-2 Decoder UMAP ===")
    esm2_ckpt = ROOT / "models" / "sae" / "esm2_scaled" / "layer_24_topk" / "best.pt"
    esm2_h5 = ROOT / "data" / "activations" / "esm2_scaled" / "layer_24.h5"

    # Classify features
    esm2_labels = classify_features_esm2(
        10240, str(esm2_h5), str(esm2_ckpt), sequences, offsets, accs,
        sample_proteins=300
    )

    # Also incorporate structural annotations if available
    # (ESM-2 doesn't have phase1 structural annotations, so use novel characterization)
    novel_path = ROOT / "results" / "unified" / "esm2" / "novel_feature_characterization.json"
    if novel_path.exists():
        with open(novel_path) as f:
            novel_data = json.load(f)
        classified = novel_data.get("summary", {}).get("classified", {})
        log.info(f"ESM-2 novel classification: {classified}")
        # Re-classify dead features
        # We'll use the count-based info — dead features have low activation
        # Already handled in classify_features_esm2

    log.info(f"ESM-2 labels: {Counter(esm2_labels)}")

    generate_decoder_umap(
        "esm2", str(esm2_ckpt), esm2_labels,
        str(out_dir / "umap_decoder_esm2_L24"),
        title="ESM-2 Layer 24 — SAE Decoder Feature Space"
    )

    # ═══════════════════════════════════════════════════
    # 2. SAE decoder UMAP — ESM-3 Layer 33 (S-only)
    # ═══════════════════════════════════════════════════
    log.info("\n=== ESM-3 Decoder UMAP ===")
    esm3_ckpt = ROOT / "models" / "sae" / "esm3" / "S_layer_33_topk" / "best.pt"
    esm3_h5 = ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S_layer_33.h5"

    # Use annotation files for ESM-3
    esm3_func_ann = ROOT / "results" / "phase1" / "modality_saes" / "S_layer_33_annotations.json"
    esm3_struct_ann = ROOT / "results" / "phase1" / "modality_saes" / "structural_annotations" / "S_layer_33_structural.json"

    esm3_labels = classify_features_with_annotations(
        12288, str(esm3_func_ann), str(esm3_struct_ann)
    )

    # Mark dead features by checking activation counts
    # Quick scan: encode a batch and check which features fire
    log.info("Scanning for dead ESM-3 features...")
    ckpt = torch.load(str(esm3_ckpt), map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()

    with h5py.File(str(esm3_h5), "r") as f:
        raw = torch.tensor(f["activations"][:20000], dtype=torch.float32)
        with torch.no_grad():
            z = sae.encode(raw).numpy()
        feat_counts = (z > 0).sum(axis=0)
        n_dead = (feat_counts < 5).sum()
        log.info(f"Dead features (active <5 in 20K residues): {n_dead}/{len(feat_counts)}")
        for fi in range(len(esm3_labels)):
            if feat_counts[fi] < 5:
                esm3_labels[fi] = "Dead"

    log.info(f"ESM-3 labels: {Counter(esm3_labels)}")

    generate_decoder_umap(
        "esm3", str(esm3_ckpt), esm3_labels,
        str(out_dir / "umap_decoder_esm3_L33"),
        title="ESM-3 Layer 33 — SAE Decoder Feature Space"
    )

    # ═══════════════════════════════════════════════════
    # 3. Dataset UMAP — proteins in ESM-2 embedding space
    # ═══════════════════════════════════════════════════
    log.info("\n=== Dataset UMAP (ESM-2 embeddings) ===")
    generate_dataset_umap(
        str(esm2_h5), sequences, metadata, offsets, accs,
        str(out_dir / "umap_dataset_esm2"),
        title="Protein Dataset — ESM-2 Layer 24 Embeddings"
    )

    # ═══════════════════════════════════════════════════
    # 4. Combined figure for publication
    # ═══════════════════════════════════════════════════
    log.info("\n=== Generating combined publication figure ===")
    generate_combined_figure(out_dir)

    log.info("\nAll UMAP figures complete!")


def build_protein_offsets(sequences):
    accs = sorted(sequences.keys())
    offsets = {}
    idx = 0
    for acc in accs:
        offsets[acc] = (idx, len(sequences[acc]))
        idx += len(sequences[acc])
    return offsets, accs


def generate_combined_figure(out_dir):
    """Combine all three UMAPs into a single publication figure."""
    # Load the individual PNGs and compose
    from matplotlib.image import imread

    paths = [
        out_dir / "umap_dataset_esm2.png",
        out_dir / "umap_decoder_esm2_L24.png",
        out_dir / "umap_decoder_esm3_L33.png",
    ]

    # Check all exist
    for p in paths:
        if not p.exists():
            log.warning(f"Missing: {p}")
            return

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    for i, (ax, path) in enumerate(zip(axes, paths)):
        img = imread(str(path))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(chr(ord('a') + i), fontsize=14, fontweight="bold",
                    loc="left", x=-0.02, y=1.02)

    fig.suptitle("Protein and Feature Space Organization", fontsize=14,
                fontweight="bold", y=1.02)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(str(out_dir / f"fig_umaps_combined.{ext}"),
                   dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved combined figure")


if __name__ == "__main__":
    main()
