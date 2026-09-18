"""Phase 1.2: Modality dropout experiment for ESM-3.

Run as: ./env/bin/python scripts/run_phase1_modality_dropout.py
Logs to: results/phase1/modality_dropout_log.txt

The centerpiece novel experiment: run ESM-3 under multiple input conditions
and compute CKA similarity between conditions at each layer to identify
modality integration layers.

Conditions:
  S:     sequence only (baseline)
  S+St:  sequence + structure tokens
  S+F:   sequence + function tokens
  S+St+F: sequence + structure + function (full multimodal)

Output: CKA similarity matrices at each layer, integration layer identification.
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

log_dir = ROOT / "results" / "phase1" / "modality_dropout"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "modality_dropout_log.txt"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("modality_dropout")

import torch
import numpy as np


# ============================================================
# CKA Implementation
# ============================================================
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear Centered Kernel Alignment between two representations.

    Args:
        X: (n, d1) representation matrix
        Y: (n, d2) representation matrix

    Returns:
        CKA similarity in [0, 1]
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    XtX = X.T @ X  # (d1, d1)
    YtY = Y.T @ Y  # (d2, d2)
    XtY = X.T @ Y  # (d1, d2)

    # HSIC(X,Y) = ||X^T Y||_F^2 / (n-1)^2
    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


# ============================================================
# Step 1: Download structures from AlphaFold
# ============================================================
def step1_get_structures(subset_accs: list[str]) -> dict:
    """Download AlphaFold structures for subset of proteins."""
    log.info("=" * 60)
    log.info("STEP 1: Downloading AlphaFold structures")
    log.info("=" * 60)

    import urllib.request

    struct_dir = ROOT / "data" / "pilot" / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    available = {}
    failed = []

    for i, acc in enumerate(subset_accs):
        pdb_path = struct_dir / f"{acc}.pdb"
        if pdb_path.exists():
            available[acc] = str(pdb_path)
            continue

        try:
            # Use AlphaFold API to get correct PDB URL
            api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
            resp = urllib.request.urlopen(api_url, timeout=10)
            data = json.loads(resp.read())
            if data and data[0].get("pdbUrl"):
                pdb_url = data[0]["pdbUrl"]
                urllib.request.urlretrieve(pdb_url, pdb_path)
                available[acc] = str(pdb_path)
            else:
                failed.append(acc)
        except Exception as e:
            failed.append(acc)
            if len(failed) <= 5:
                log.warning(f"  Failed to download {acc}: {e}")

        if (i + 1) % 50 == 0:
            log.info(f"  Downloaded {i+1}/{len(subset_accs)} ({len(available)} available, {len(failed)} failed)")

    log.info(f"Step 1 complete: {len(available)} structures available, {len(failed)} failed")
    return available


# ============================================================
# Step 2: Tokenize multimodal inputs
# ============================================================
def step2_tokenize(
    model, available_structures: dict, metadata: dict, seq_dict: dict
) -> dict:
    """Create tokenized inputs for each condition."""
    log.info("=" * 60)
    log.info("STEP 2: Tokenizing multimodal inputs")
    log.info("=" * 60)

    from esm.sdk.api import ESMProtein
    from esm.utils.types import FunctionAnnotation

    device = next(model.parameters()).device
    tokenized = {}  # acc -> {condition -> ESMProteinTensor}
    successful = 0

    for acc, pdb_path in available_structures.items():
        seq = seq_dict[acc]
        meta = metadata.get(acc, {})
        interpro_ids = meta.get("interpro", [])

        try:
            # Load protein with structure from PDB
            protein_with_struct = ESMProtein.from_pdb(pdb_path)

            # Verify sequence matches (may differ due to PDB vs UniProt)
            if protein_with_struct.sequence is None:
                log.warning(f"  {acc}: No sequence from PDB, skipping")
                continue

            # Use the PDB sequence length as reference
            pdb_seq = protein_with_struct.sequence

            # Build function annotations
            func_annotations = []
            for ipr_id in interpro_ids:
                func_annotations.append(
                    FunctionAnnotation(label=ipr_id, start=1, end=len(pdb_seq))
                )

            # Create ESMProtein for each condition
            conditions = {}

            # S: sequence only
            protein_s = ESMProtein(sequence=pdb_seq)
            conditions["S"] = model.encode(protein_s)

            # S+St: sequence + structure
            conditions["S+St"] = model.encode(protein_with_struct)

            # S+F: sequence + function
            protein_sf = ESMProtein(
                sequence=pdb_seq,
                function_annotations=func_annotations if func_annotations else None,
            )
            conditions["S+F"] = model.encode(protein_sf)

            # S+St+F: full multimodal
            protein_full = ESMProtein(
                sequence=pdb_seq,
                coordinates=protein_with_struct.coordinates,
                function_annotations=func_annotations if func_annotations else None,
            )
            conditions["S+St+F"] = model.encode(protein_full)

            tokenized[acc] = conditions
            successful += 1

            if successful % 25 == 0:
                log.info(f"  Tokenized {successful} proteins")

        except Exception as e:
            if successful < 5:
                log.warning(f"  {acc}: Tokenization failed: {e}")

    log.info(f"Step 2 complete: {successful} proteins tokenized across 4 conditions")
    return tokenized


# ============================================================
# Step 3: Extract activations per condition
# ============================================================
def step3_extract_activations(model, tokenized: dict, layers: list[int]) -> dict:
    """Extract layer activations for each protein × condition."""
    log.info("=" * 60)
    log.info("STEP 3: Extracting activations under each condition")
    log.info("=" * 60)

    from models.esm3_hooks import ESM3HookManager

    device = next(model.parameters()).device
    hook_mgr = ESM3HookManager(model, layers=layers, extract_attention=False)
    conditions = ["S", "S+St", "S+F", "S+St+F"]

    # activations[condition][layer] = list of (n_residues, d_model) tensors
    activations = {cond: {l: [] for l in layers} for cond in conditions}

    start_time = time.time()
    processed = 0

    for acc, cond_tensors in tokenized.items():
        for cond_name in conditions:
            if cond_name not in cond_tensors:
                continue

            protein_tensor = cond_tensors[cond_name]

            # Build forward kwargs from ESMProteinTensor
            kwargs = {}
            if protein_tensor.sequence is not None:
                kwargs["sequence_tokens"] = protein_tensor.sequence.unsqueeze(0).to(device)
            if protein_tensor.structure is not None:
                kwargs["structure_tokens"] = protein_tensor.structure.unsqueeze(0).to(device)
            if protein_tensor.function is not None:
                kwargs["function_tokens"] = protein_tensor.function.unsqueeze(0).to(device)
            if protein_tensor.residue_annotations is not None:
                kwargs["residue_annotation_tokens"] = protein_tensor.residue_annotations.unsqueeze(0).to(device)

            hook_mgr.cache.clear()
            hook_mgr.register()

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**kwargs)

            # Get per-residue activations (skip BOS/EOS: positions 1:-1)
            seq_len = kwargs["sequence_tokens"].shape[1] - 2  # subtract BOS+EOS
            for layer_idx in layers:
                act = hook_mgr.cache.residual_stream[layer_idx]
                if act.ndim == 3:
                    act = act[0]
                act = act[1:seq_len + 1].float().cpu()
                activations[cond_name][layer_idx].append(act)

            hook_mgr.remove()

        processed += 1
        if processed % 25 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            log.info(f"  Processed {processed}/{len(tokenized)} proteins ({rate:.1f} prot/s)")

    # Concatenate per-layer
    for cond in conditions:
        for l in layers:
            if activations[cond][l]:
                activations[cond][l] = torch.cat(activations[cond][l], dim=0).numpy()
                log.info(f"  {cond} layer {l}: {activations[cond][l].shape}")
            else:
                activations[cond][l] = None

    elapsed = time.time() - start_time
    log.info(f"Step 3 complete: {processed} proteins × 4 conditions in {elapsed:.0f}s")
    return activations


# ============================================================
# Step 4: Compute CKA similarity
# ============================================================
def step4_compute_cka(activations: dict, layers: list[int]) -> dict:
    """Compute CKA between all condition pairs at each layer."""
    log.info("=" * 60)
    log.info("STEP 4: Computing CKA similarity")
    log.info("=" * 60)

    conditions = ["S", "S+St", "S+F", "S+St+F"]
    results = {}

    for layer_idx in layers:
        cka_matrix = np.zeros((len(conditions), len(conditions)))

        for i, cond_i in enumerate(conditions):
            for j, cond_j in enumerate(conditions):
                if i > j:
                    cka_matrix[i, j] = cka_matrix[j, i]
                    continue

                X = activations[cond_i][layer_idx]
                Y = activations[cond_j][layer_idx]

                if X is None or Y is None:
                    cka_matrix[i, j] = np.nan
                    continue

                # Subsample if too many residues (CKA is O(n*d^2))
                n = min(X.shape[0], Y.shape[0], 50000)
                if X.shape[0] > n:
                    idx = np.random.choice(X.shape[0], n, replace=False)
                    X_sub = X[idx]
                    Y_sub = Y[idx]
                else:
                    X_sub = X[:n]
                    Y_sub = Y[:n]

                cka_val = linear_cka(X_sub, Y_sub)
                cka_matrix[i, j] = cka_val

        results[layer_idx] = cka_matrix.tolist()

        # Print readable summary
        log.info(f"\n  Layer {layer_idx} CKA matrix:")
        log.info(f"  {'':>10} {'S':>8} {'S+St':>8} {'S+F':>8} {'S+St+F':>8}")
        for i, cond_i in enumerate(conditions):
            row = "  " + f"{cond_i:>10}"
            for j in range(len(conditions)):
                row += f" {cka_matrix[i, j]:8.4f}"
            log.info(row)

    # Compute "modality impact" — how much each modality changes representations
    log.info("\n  Modality impact (1 - CKA vs sequence-only):")
    log.info(f"  {'Layer':>8} {'Structure':>10} {'Function':>10} {'Both':>10}")
    for layer_idx in layers:
        mat = np.array(results[layer_idx])
        struct_impact = 1.0 - mat[0, 1]  # S vs S+St
        func_impact = 1.0 - mat[0, 2]    # S vs S+F
        both_impact = 1.0 - mat[0, 3]    # S vs S+St+F
        log.info(f"  {layer_idx:>8} {struct_impact:10.4f} {func_impact:10.4f} {both_impact:10.4f}")

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PHASE 1.2: Modality Dropout Experiment")
    log.info("=" * 60)

    overall_start = time.time()

    try:
        # Load metadata
        with open(ROOT / "data" / "pilot" / "annotations" / "metadata.json") as f:
            metadata = json.load(f)
        with open(ROOT / "data" / "pilot" / "sequences" / "sequences.json") as f:
            seq_dict = json.load(f)

        # Select subset: 200 proteins with diverse categories and InterPro annotations
        subset_accs = []
        for acc, m in metadata.items():
            if len(m.get("interpro", [])) > 0 and len(m.get("pdb_ids", [])) > 0:
                if len(seq_dict.get(acc, "")) <= 500:  # keep shorter for efficiency
                    subset_accs.append(acc)
            if len(subset_accs) >= 200:
                break

        log.info(f"Selected {len(subset_accs)} proteins for modality dropout")

        # Step 1: Download AlphaFold structures
        available_structures = step1_get_structures(subset_accs)

        if len(available_structures) < 50:
            log.error(f"Too few structures available ({len(available_structures)}). Aborting.")
            sys.exit(1)

        # Load ESM-3 model
        log.info("Loading ESM-3 model...")
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device="cuda:0")
        log.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

        # Step 2: Tokenize multimodal inputs
        tokenized = step2_tokenize(model, available_structures, metadata, seq_dict)

        if len(tokenized) < 30:
            log.error(f"Too few tokenized proteins ({len(tokenized)}). Aborting.")
            sys.exit(1)

        # Step 3: Extract activations at 12 layers
        layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]
        activations = step3_extract_activations(model, tokenized, layers)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        # Step 4: Compute CKA
        cka_results = step4_compute_cka(activations, layers)

        # Save results
        output = {
            "layers": layers,
            "conditions": ["S", "S+St", "S+F", "S+St+F"],
            "cka_matrices": {str(k): v for k, v in cka_results.items()},
            "num_proteins": len(tokenized),
        }
        with open(log_dir / "cka_results.json", "w") as f:
            json.dump(output, f, indent=2)

        elapsed = time.time() - overall_start
        log.info(f"\nPHASE 1.2 COMPLETE in {elapsed/60:.1f} minutes")
        log.info(f"Results saved to {log_dir}/")

    except Exception as e:
        log.error(f"Phase 1.2 failed: {e}", exc_info=True)
        raise
