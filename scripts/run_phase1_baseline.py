"""Phase 1.1: ESM-3 baseline — extraction, probing, comparison with ESM-2.

Run as: ./env/bin/python scripts/run_phase1_baseline.py
Logs to: results/phase1/phase1_baseline_log.txt

Steps:
1. Extract ESM-3 activations on pilot set (8 layers)
2. Train linear probes (AA identity + functional sites)
3. Compare with ESM-2 Phase 0 results
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

log_dir = ROOT / "results" / "phase1"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "phase1_baseline_log.txt"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("phase1")

import torch
import numpy as np
import h5py


# ============================================================
# Step 1: Extract ESM-3 activations
# ============================================================
def step1_extract_activations():
    log.info("=" * 60)
    log.info("STEP 1: Extracting ESM-3 activations")
    log.info("=" * 60)

    from models.esm3_hooks import load_esm3, ESM3HookManager, tokenize_sequence

    # Load sequences
    seq_path = ROOT / "data" / "pilot" / "sequences" / "sequences.json"
    with open(seq_path) as f:
        seq_dict = json.load(f)

    accessions = list(seq_dict.keys())
    sequences = list(seq_dict.values())
    log.info(f"Loaded {len(sequences)} sequences")

    # Load model
    device = "cuda:0"
    log.info(f"Loading ESM-3 on {device}...")
    model, tokenizers = load_esm3(device=device)
    vram = torch.cuda.memory_allocated(0) / 1e9
    log.info(f"Model loaded. VRAM: {vram:.2f} GB")

    # Target layers (8 evenly spaced across 48 layers)
    layers = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    log.info(f"Extracting layers: {layers}")

    output_dir = ROOT / "data" / "activations" / "esm3"
    output_dir.mkdir(parents=True, exist_ok=True)

    hook_mgr = ESM3HookManager(model, layers=layers, extract_attention=False)
    max_length = 800

    all_activations = {l: [] for l in layers}
    all_accessions = []
    all_lengths = []

    start_time = time.time()
    skipped = 0

    for i, (acc, seq) in enumerate(zip(accessions, sequences)):
        if len(seq) > max_length:
            skipped += 1
            continue

        inputs = tokenize_sequence(seq, tokenizers, device=device)

        hook_mgr.cache.clear()
        hook_mgr.register()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(**inputs)

        # Extract per-residue activations (skip BOS at 0, skip EOS at end)
        seq_len = len(seq)
        all_accessions.append(acc)
        all_lengths.append(seq_len)

        for layer_idx in layers:
            # Hook captures (1, L+2, 1536) — remove batch, take positions 1:seq_len+1
            act = hook_mgr.cache.residual_stream[layer_idx]
            if act.ndim == 3:
                act = act[0]  # remove batch dim
            act = act[1 : seq_len + 1].float().cpu()  # skip BOS/EOS, cast to f32
            all_activations[layer_idx].append(act)

        hook_mgr.remove()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1 - skipped) / elapsed if elapsed > 0 else 0
            vram_now = torch.cuda.memory_allocated(0) / 1e9
            log.info(f"  Processed {i+1}/{len(sequences)} ({rate:.1f} seq/s, VRAM: {vram_now:.1f} GB)")

    # Save to HDF5 — one file per layer
    for layer_idx in layers:
        layer_acts = torch.cat(all_activations[layer_idx], dim=0)
        h5_path = output_dir / f"layer_{layer_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activations", data=layer_acts.numpy(), compression="gzip", compression_opts=4)
            f.attrs["layer"] = layer_idx
            f.attrs["model"] = "esm3_sm_open_v1"
            f.attrs["num_proteins"] = len(all_accessions)
            f.attrs["num_residues"] = layer_acts.shape[0]
            f.attrs["hidden_dim"] = layer_acts.shape[1]
        log.info(f"  Saved layer {layer_idx}: {layer_acts.shape} -> {h5_path}")

    # Save residue index
    index = []
    offset = 0
    for acc, length in zip(all_accessions, all_lengths):
        index.append({"accession": acc, "start": offset, "end": offset + length, "length": length})
        offset += length
    with open(output_dir / "residue_index.json", "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - start_time
    log.info(f"Step 1 complete: {len(all_accessions)} proteins, {offset} residues in {elapsed:.0f}s (skipped {skipped})")

    del model
    torch.cuda.empty_cache()
    return all_accessions, all_lengths


# ============================================================
# Step 2: Linear Probing
# ============================================================
def step2_linear_probing():
    log.info("=" * 60)
    log.info("STEP 2: Linear probing on ESM-3 activations")
    log.info("=" * 60)

    from probing.linear_probe import train_classification_probe

    act_dir = ROOT / "data" / "activations" / "esm3"
    meta_path = ROOT / "data" / "pilot" / "annotations" / "metadata.json"

    with open(meta_path) as f:
        metadata = json.load(f)

    with open(act_dir / "residue_index.json") as f:
        residue_index = json.load(f)

    with open(ROOT / "data" / "pilot" / "sequences" / "sequences.json") as f:
        seq_dict = json.load(f)

    # Build per-residue labels
    log.info("Building per-residue labels...")

    aa_to_class = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    aa_labels = []
    functional_site_labels = []

    for entry in residue_index:
        acc = entry["accession"]
        seq = seq_dict[acc]
        prot = metadata.get(acc, {})
        for pos in range(entry["length"]):
            aa_labels.append(aa_to_class.get(seq[pos] if pos < len(seq) else "X", -1))

            is_functional = 0
            for feat in prot.get("features", []):
                start = feat.get("start", 0) - 1
                end = feat.get("end", 0)
                if start <= pos < end:
                    is_functional = 1
                    break
            functional_site_labels.append(is_functional)

    aa_labels = np.array(aa_labels)
    functional_site_labels = np.array(functional_site_labels)

    valid_mask = aa_labels >= 0
    aa_labels = aa_labels[valid_mask]
    functional_mask = functional_site_labels[valid_mask]

    log.info(f"Total residues: {len(aa_labels)}")
    log.info(f"Functional sites: {functional_mask.sum()} ({functional_mask.mean()*100:.1f}%)")

    # Train/test split (80/20 by proteins)
    n_proteins = len(residue_index)
    n_train = int(0.8 * n_proteins)
    train_residues = sum(e["length"] for e in residue_index[:n_train])

    results = []
    layers = [0, 6, 12, 18, 24, 30, 36, 42, 47]

    for layer_idx in layers:
        h5_path = act_dir / f"layer_{layer_idx}.h5"
        with h5py.File(h5_path, "r") as f:
            X = f["activations"][:][valid_mask]

        X_train, X_test = X[:train_residues], X[train_residues:]
        y_aa_train, y_aa_test = aa_labels[:train_residues], aa_labels[train_residues:]
        y_func_train, y_func_test = functional_mask[:train_residues], functional_mask[train_residues:]

        result = train_classification_probe(
            X_train, y_aa_train, X_test, y_aa_test,
            property_name="amino_acid_identity",
            layer=layer_idx, model_name="esm3_sm_open_v1",
        )
        results.append(result)
        log.info(f"  Layer {layer_idx} AA identity: acc={result.accuracy:.3f} f1={result.f1:.3f}")

        if y_func_test.sum() >= 10:
            result = train_classification_probe(
                X_train, y_func_train, X_test, y_func_test,
                property_name="functional_site",
                layer=layer_idx, model_name="esm3_sm_open_v1",
            )
            results.append(result)
            log.info(f"  Layer {layer_idx} functional site: acc={result.accuracy:.3f} f1={result.f1:.3f}")

    # Save results
    probe_dir = ROOT / "results" / "phase1" / "probing"
    probe_dir.mkdir(parents=True, exist_ok=True)

    results_data = [
        {"property": r.property_name, "layer": r.layer, "accuracy": r.accuracy, "f1": r.f1, "auroc": r.auroc}
        for r in results
    ]
    with open(probe_dir / "esm3_probing_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    log.info("Step 2 complete.")
    return results


# ============================================================
# Step 3: Compare ESM-2 vs ESM-3
# ============================================================
def step3_compare_models():
    log.info("=" * 60)
    log.info("STEP 3: Comparing ESM-2 vs ESM-3 probing results")
    log.info("=" * 60)

    esm2_path = ROOT / "results" / "phase0" / "probing" / "probing_results.json"
    esm3_path = ROOT / "results" / "phase1" / "probing" / "esm3_probing_results.json"

    with open(esm2_path) as f:
        esm2_results = json.load(f)
    with open(esm3_path) as f:
        esm3_results = json.load(f)

    # Compare functional site AUROC across layers
    log.info("\nFunctional site AUROC comparison:")
    log.info(f"  {'Layer':>8}  {'ESM-2':>8}  {'ESM-3':>8}  {'Diff':>8}")
    log.info(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    esm2_func = {r["layer"]: r for r in esm2_results if r["property"] == "functional_site"}
    esm3_func = {r["layer"]: r for r in esm3_results if r["property"] == "functional_site"}

    # Normalize layer indices to fraction of total depth for comparison
    esm2_depth = 33  # 33 layers
    esm3_depth = 48  # 48 layers

    comparisons = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        esm2_layer = int(frac * 32)  # layer 0-32
        esm3_layer = int(frac * 47)  # layer 0-47

        # Find closest available layer
        esm2_match = min(esm2_func.keys(), key=lambda l: abs(l - esm2_layer)) if esm2_func else None
        esm3_match = min(esm3_func.keys(), key=lambda l: abs(l - esm3_layer)) if esm3_func else None

        if esm2_match is not None and esm3_match is not None:
            e2 = esm2_func[esm2_match]
            e3 = esm3_func[esm3_match]
            diff = (e3.get("auroc", 0) or 0) - (e2.get("auroc", 0) or 0)
            log.info(f"  {frac:.0%} depth  L{esm2_match:>3}/{e2.get('auroc', 0):.3f}  L{esm3_match:>3}/{e3.get('auroc', 0):.3f}  {diff:+.3f}")
            comparisons.append({
                "depth_fraction": frac,
                "esm2_layer": esm2_match,
                "esm3_layer": esm3_match,
                "esm2_auroc": e2.get("auroc"),
                "esm3_auroc": e3.get("auroc"),
            })

    # Best AUROC per model
    best_esm2 = max(esm2_func.values(), key=lambda r: r.get("auroc", 0) or 0) if esm2_func else {}
    best_esm3 = max(esm3_func.values(), key=lambda r: r.get("auroc", 0) or 0) if esm3_func else {}

    log.info(f"\nBest functional site AUROC:")
    log.info(f"  ESM-2: {best_esm2.get('auroc', 'N/A')} (layer {best_esm2.get('layer', 'N/A')})")
    log.info(f"  ESM-3: {best_esm3.get('auroc', 'N/A')} (layer {best_esm3.get('layer', 'N/A')})")

    # AA identity comparison
    log.info("\nAA identity accuracy comparison:")
    esm2_aa = {r["layer"]: r for r in esm2_results if r["property"] == "amino_acid_identity"}
    esm3_aa = {r["layer"]: r for r in esm3_results if r["property"] == "amino_acid_identity"}

    best_esm2_aa = max(esm2_aa.values(), key=lambda r: r["accuracy"]) if esm2_aa else {}
    best_esm3_aa = max(esm3_aa.values(), key=lambda r: r["accuracy"]) if esm3_aa else {}
    log.info(f"  ESM-2 best: {best_esm2_aa.get('accuracy', 'N/A')} (layer {best_esm2_aa.get('layer', 'N/A')})")
    log.info(f"  ESM-3 best: {best_esm3_aa.get('accuracy', 'N/A')} (layer {best_esm3_aa.get('layer', 'N/A')})")

    # Save comparison
    comparison = {
        "depth_comparisons": comparisons,
        "best_functional_auroc": {
            "esm2": {"auroc": best_esm2.get("auroc"), "layer": best_esm2.get("layer")},
            "esm3": {"auroc": best_esm3.get("auroc"), "layer": best_esm3.get("layer")},
        },
        "best_aa_accuracy": {
            "esm2": {"accuracy": best_esm2_aa.get("accuracy"), "layer": best_esm2_aa.get("layer")},
            "esm3": {"accuracy": best_esm3_aa.get("accuracy"), "layer": best_esm3_aa.get("layer")},
        },
    }

    with open(ROOT / "results" / "phase1" / "esm2_vs_esm3_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    log.info("\nComparison saved to results/phase1/esm2_vs_esm3_comparison.json")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PHASE 1.1: ESM-3 Baseline Probing")
    log.info("=" * 60)

    overall_start = time.time()

    try:
        step1_extract_activations()
        step2_linear_probing()
        step3_compare_models()

        elapsed = time.time() - overall_start
        log.info(f"\nPHASE 1.1 COMPLETE in {elapsed/60:.1f} minutes")
        log.info("Results saved to results/phase1/")

    except Exception as e:
        log.error(f"Phase 1.1 failed: {e}", exc_info=True)
        raise
