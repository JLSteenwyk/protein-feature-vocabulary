#!/usr/bin/env python3
"""Cross-modal feature identification: which SAE features respond to structure tokens?

For each eval protein with an AlphaFold structure:
  S-only: ESM-3(sequence) → layer 33 → SAE encode
  S+St:   ESM-3(sequence + structure) → layer 33 → SAE encode
  Compare per-feature activations between conditions.

Runs in stages (--step) for flexibility:
  fetch     - Download AlphaFold structures
  tokenize  - Create S and S+St ESMProteinTensors
  extract   - GPU: extract layer 33 under both conditions (long)
  analyze   - CPU: encode through SAE, compute statistics

Usage:
    # All at once:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/13_cross_modal_features.py --step all

    # Or staged:
    ./env/bin/python scripts/scaled_1.5M/13_cross_modal_features.py --step fetch
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/13_cross_modal_features.py --step extract
    ./env/bin/python scripts/scaled_1.5M/13_cross_modal_features.py --step analyze
"""

import os
import sys
import json
import argparse
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cross_modal")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
CM_DIR = ROOT / "results" / "scaled_1.5M" / "cross_modal"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
SAE_TAG = "esm3_residue_ef8_k64"
CHUNK_SIZE = 500


# ══════════════════════════════════════════════════════
# STEP: Fetch AlphaFold structures
# ══════════════════════════════════════════════════════
def step_fetch():
    """Download AlphaFold structures for eval proteins."""
    log.info("=" * 60)
    log.info("STEP: Fetch AlphaFold structures")
    log.info("=" * 60)

    STRUCT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)
    all_accs = list(sequences.keys())

    available = {}
    failed = []

    def download_one(acc):
        pdb_path = STRUCT_DIR / f"{acc}.pdb"
        if pdb_path.exists():
            return acc, str(pdb_path), None

        try:
            api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
            resp = urllib.request.urlopen(api_url, timeout=15)
            data = json.loads(resp.read())
            if data and data[0].get("pdbUrl"):
                pdb_url = data[0]["pdbUrl"]
                urllib.request.urlretrieve(pdb_url, pdb_path)
                return acc, str(pdb_path), None
            return acc, None, "No PDB URL"
        except Exception as e:
            return acc, None, str(e)

    # Concurrent downloads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_one, acc) for acc in all_accs]
        for i, future in enumerate(futures):
            acc, path, error = future.result()
            if path:
                available[acc] = path
            else:
                failed.append(acc)

            if (i + 1) % 500 == 0:
                log.info(f"  {i+1}/{len(all_accs)}: {len(available)} available, {len(failed)} failed")

    # Save manifest
    manifest = {"available": available, "failed": failed}
    with open(STRUCT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f)

    log.info(f"Done: {len(available)} structures, {len(failed)} failed")
    return available


# ══════════════════════════════════════════════════════
# STEP: Extract activations under S and S+St conditions
# ══════════════════════════════════════════════════════
def step_extract(device="cuda", max_proteins=0):
    """Extract layer 33 activations under S-only and S+St conditions."""
    log.info("=" * 60)
    log.info("STEP: Extract cross-modal activations")
    log.info("=" * 60)

    from models.esm3_hooks import load_esm3, ESM3HookManager
    from esm.sdk.api import ESMProtein

    # Load structure manifest
    manifest_path = STRUCT_DIR / "manifest.json"
    if not manifest_path.exists():
        log.error("No structure manifest. Run --step fetch first.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    available = manifest["available"]
    accs_with_struct = sorted(available.keys())

    if max_proteins > 0:
        accs_with_struct = accs_with_struct[:max_proteins]
    log.info(f"Proteins with structures: {len(accs_with_struct)}")

    # Create output dirs
    s_dir = CM_DIR / "S_only" / "residue_L33"
    sst_dir = CM_DIR / "S_St" / "residue_L33"
    s_dir.mkdir(parents=True, exist_ok=True)
    sst_dir.mkdir(parents=True, exist_ok=True)

    # Load ESM-3
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=device)
    hook_mgr = ESM3HookManager(model, layers=[33], extract_attention=False)
    log.info("ESM-3 loaded")

    # Process in chunks
    n_chunks = (len(accs_with_struct) + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_proteins = 0
    total_residues = 0
    start_time = time.time()

    for ci in range(n_chunks):
        chunk_start = ci * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, len(accs_with_struct))
        chunk_accs = accs_with_struct[chunk_start:chunk_end]

        s_file = s_dir / f"chunk{ci:04d}.h5"
        sst_file = sst_dir / f"chunk{ci:04d}.h5"

        # Resume check
        if s_file.exists() and sst_file.exists():
            log.info(f"  Chunk {ci+1}/{n_chunks}: already done, skipping")
            total_proteins += len(chunk_accs)
            continue

        s_acts_list = []
        sst_acts_list = []
        s_offsets = []
        sst_offsets = []
        valid_ids = []
        s_offset = 0
        sst_offset = 0

        for acc in chunk_accs:
            pdb_path = available[acc]

            try:
                # Load protein with structure
                protein_with_struct = ESMProtein.from_pdb(pdb_path)
                if protein_with_struct.sequence is None:
                    continue

                pdb_seq = protein_with_struct.sequence
                if len(pdb_seq) < 50 or len(pdb_seq) > 1022:
                    continue

                # S-only: sequence only
                protein_s = ESMProtein(sequence=pdb_seq)
                tokens_s = model.encode(protein_s)

                hook_mgr.cache.clear()
                hook_mgr.register()
                with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                    model(
                        sequence_tokens=tokens_s.sequence.to(device).unsqueeze(0),
                    )
                acts_s = hook_mgr.cache.residual_stream[33][0].float().cpu()
                acts_s = acts_s[1:-1]  # remove BOS/EOS
                hook_mgr.remove()

                # S+St: sequence + structure
                tokens_sst = model.encode(protein_with_struct)

                hook_mgr.cache.clear()
                hook_mgr.register()
                with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                    model(
                        sequence_tokens=tokens_sst.sequence.to(device).unsqueeze(0),
                        structure_tokens=tokens_sst.structure.to(device).unsqueeze(0),
                    )
                acts_sst = hook_mgr.cache.residual_stream[33][0].float().cpu()
                acts_sst = acts_sst[1:-1]
                hook_mgr.remove()

                # Verify same length
                L = min(acts_s.shape[0], acts_sst.shape[0])
                acts_s = acts_s[:L]
                acts_sst = acts_sst[:L]

                s_acts_list.append(acts_s.half().numpy())
                sst_acts_list.append(acts_sst.half().numpy())
                s_offsets.append((len(valid_ids), s_offset, L))
                sst_offsets.append((len(valid_ids), sst_offset, L))
                valid_ids.append(acc)
                s_offset += L
                sst_offset += L

                torch.cuda.empty_cache()

            except Exception as e:
                if total_proteins < 5:
                    log.warning(f"  Error on {acc}: {e}")
                continue

        if not valid_ids:
            log.warning(f"  Chunk {ci+1}: no valid proteins")
            continue

        # Save S-only
        s_all = np.concatenate(s_acts_list, axis=0)
        with h5py.File(s_file, "w") as f:
            f.create_dataset("activations", data=s_all, dtype="float16",
                             chunks=(min(4096, s_all.shape[0]), 1536))
            f.create_dataset("offsets", data=np.array(s_offsets, dtype=np.int64))
            f.create_dataset("ids", data=[s.encode() for s in valid_ids])

        # Save S+St
        sst_all = np.concatenate(sst_acts_list, axis=0)
        with h5py.File(sst_file, "w") as f:
            f.create_dataset("activations", data=sst_all, dtype="float16",
                             chunks=(min(4096, sst_all.shape[0]), 1536))
            f.create_dataset("offsets", data=np.array(sst_offsets, dtype=np.int64))
            f.create_dataset("ids", data=[s.encode() for s in valid_ids])

        total_proteins += len(valid_ids)
        total_residues += s_all.shape[0]

        elapsed = time.time() - start_time
        rate = total_proteins / elapsed if elapsed > 0 else 0
        eta_h = (len(accs_with_struct) - total_proteins) / rate / 3600 if rate > 0 else 0

        log.info(f"  Chunk {ci+1}/{n_chunks}: {len(valid_ids)} proteins | "
                 f"Total: {total_proteins}, {total_residues} residues | "
                 f"Rate: {rate:.1f}/s | ETA: {eta_h:.1f}h")

        del s_acts_list, sst_acts_list, s_all, sst_all

    log.info(f"\nExtraction done: {total_proteins} proteins, {total_residues} residues")
    log.info(f"Elapsed: {(time.time() - start_time)/3600:.1f}h")


# ══════════════════════════════════════════════════════
# STEP: Analyze — encode through SAE and compute statistics
# ══════════════════════════════════════════════════════
def step_analyze():
    """Encode through SAE and compute cross-modal feature statistics."""
    log.info("=" * 60)
    log.info("STEP: Analyze cross-modal features")
    log.info("=" * 60)

    from sae.model import build_sae
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    # Load SAE
    sae_path = MODEL_ROOT / SAE_TAG / "best.pt"
    ckpt = torch.load(str(sae_path), map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    d_sae = ckpt["sae_config"].input_dim * ckpt["sae_config"].expansion_factor
    log.info(f"SAE: {SAE_TAG}, d_sae={d_sae}")

    # Load S-only and S+St chunks
    s_dir = CM_DIR / "S_only" / "residue_L33"
    sst_dir = CM_DIR / "S_St" / "residue_L33"
    s_chunks = sorted(s_dir.glob("*.h5"))
    sst_chunks = sorted(sst_dir.glob("*.h5"))
    log.info(f"S chunks: {len(s_chunks)}, S+St chunks: {len(sst_chunks)}")

    # Per-protein, per-feature mean activations
    all_protein_ids = []
    protein_mean_S = []
    protein_mean_SSt = []

    for s_cp, sst_cp in zip(s_chunks, sst_chunks):
        with h5py.File(s_cp, "r") as f:
            s_acts = torch.tensor(f["activations"][:].astype(np.float32))
            s_offsets = f["offsets"][:]
            ids = [s.decode() for s in f["ids"][:]]

        with h5py.File(sst_cp, "r") as f:
            sst_acts = torch.tensor(f["activations"][:].astype(np.float32))
            sst_offsets = f["offsets"][:]

        # Encode through SAE
        batch_size = 4096
        z_s_chunks = []
        for i in range(0, s_acts.shape[0], batch_size):
            with torch.no_grad():
                z_s_chunks.append(sae.encode(s_acts[i:i+batch_size]).numpy())
        Z_s = np.concatenate(z_s_chunks, axis=0)

        z_sst_chunks = []
        for i in range(0, sst_acts.shape[0], batch_size):
            with torch.no_grad():
                z_sst_chunks.append(sae.encode(sst_acts[i:i+batch_size]).numpy())
        Z_sst = np.concatenate(z_sst_chunks, axis=0)

        # Per-protein mean
        for pi, pid in enumerate(ids):
            _, off_s, L_s = s_offsets[pi]
            _, off_sst, L_sst = sst_offsets[pi]
            L = min(L_s, L_sst)

            mean_s = Z_s[off_s:off_s+L].mean(axis=0)
            mean_sst = Z_sst[off_sst:off_sst+L].mean(axis=0)

            all_protein_ids.append(pid)
            protein_mean_S.append(mean_s)
            protein_mean_SSt.append(mean_sst)

    protein_mean_S = np.stack(protein_mean_S)  # (n_proteins, d_sae)
    protein_mean_SSt = np.stack(protein_mean_SSt)
    n_proteins = len(all_protein_ids)
    log.info(f"Encoded {n_proteins} proteins through SAE")

    # Per-feature statistics
    log.info("Computing per-feature cross-modal statistics...")
    delta = protein_mean_SSt - protein_mean_S  # (n_proteins, d_sae)

    # Only test features that are active in at least one condition
    active_s = (protein_mean_S > 0).any(axis=0)
    active_sst = (protein_mean_SSt > 0).any(axis=0)
    active_either = active_s | active_sst
    active_features = np.where(active_either)[0]
    log.info(f"Active features (either condition): {len(active_features)}")

    p_values = np.ones(d_sae)
    effect_sizes = np.zeros(d_sae)
    mean_deltas = np.zeros(d_sae)

    for fid in active_features:
        d = delta[:, fid]
        nonzero = d[d != 0]

        if len(nonzero) < 10:
            continue

        # Paired Wilcoxon signed-rank test
        try:
            stat, p = wilcoxon(d[d != 0])
            p_values[fid] = p
        except Exception:
            pass

        # Cohen's d
        mean_d = d.mean()
        std_d = d.std()
        if std_d > 0:
            effect_sizes[fid] = mean_d / std_d
        mean_deltas[fid] = mean_d

    # BH correction
    tested_mask = p_values < 1.0
    tested_idx = np.where(tested_mask)[0]
    if len(tested_idx) > 0:
        reject, corrected_p, _, _ = multipletests(p_values[tested_idx], method="fdr_bh")
        p_corrected = np.ones(d_sae)
        p_corrected[tested_idx] = corrected_p
    else:
        p_corrected = np.ones(d_sae)

    # Classify features
    sig_mask = p_corrected < 0.05
    enhanced = sig_mask & (effect_sizes > 0.2)
    suppressed = sig_mask & (effect_sizes < -0.2)
    invariant = ~enhanced & ~suppressed & active_either

    n_enhanced = enhanced.sum()
    n_suppressed = suppressed.sum()
    n_invariant = invariant.sum()

    log.info(f"\n{'='*60}")
    log.info("CROSS-MODAL FEATURE CLASSIFICATION")
    log.info(f"{'='*60}")
    log.info(f"  Structure-enhanced:  {n_enhanced} features")
    log.info(f"  Structure-suppressed: {n_suppressed} features")
    log.info(f"  Structure-invariant: {n_invariant} features")
    log.info(f"  Total active: {len(active_features)}")

    # Top enhanced and suppressed features
    enhanced_idx = np.where(enhanced)[0]
    suppressed_idx = np.where(suppressed)[0]

    enhanced_ranked = enhanced_idx[np.argsort(-effect_sizes[enhanced_idx])]
    suppressed_ranked = suppressed_idx[np.argsort(effect_sizes[suppressed_idx])]

    log.info(f"\nTop 10 structure-enhanced features:")
    for fid in enhanced_ranked[:10]:
        log.info(f"  f/{fid}: d={effect_sizes[fid]:.3f}, p_corr={p_corrected[fid]:.2e}, "
                 f"delta={mean_deltas[fid]:.4f}")

    log.info(f"\nTop 10 structure-suppressed features:")
    for fid in suppressed_ranked[:10]:
        log.info(f"  f/{fid}: d={effect_sizes[fid]:.3f}, p_corr={p_corrected[fid]:.2e}, "
                 f"delta={mean_deltas[fid]:.4f}")

    # Save results
    output = {
        "n_proteins": n_proteins,
        "n_active_features": len(active_features),
        "n_structure_enhanced": int(n_enhanced),
        "n_structure_suppressed": int(n_suppressed),
        "n_structure_invariant": int(n_invariant),
        "enhanced_feature_ids": [int(f) for f in enhanced_ranked[:200]],
        "suppressed_feature_ids": [int(f) for f in suppressed_ranked[:200]],
        "top_enhanced": [
            {"feature_id": int(fid), "cohens_d": float(effect_sizes[fid]),
             "p_corrected": float(p_corrected[fid]), "mean_delta": float(mean_deltas[fid])}
            for fid in enhanced_ranked[:50]
        ],
        "top_suppressed": [
            {"feature_id": int(fid), "cohens_d": float(effect_sizes[fid]),
             "p_corrected": float(p_corrected[fid]), "mean_delta": float(mean_deltas[fid])}
            for fid in suppressed_ranked[:50]
        ],
        "effect_size_distribution": {
            "mean": float(effect_sizes[active_features].mean()),
            "std": float(effect_sizes[active_features].std()),
            "percentiles": {str(p): float(np.percentile(effect_sizes[active_features], p))
                            for p in [5, 25, 50, 75, 95]},
        },
    }

    out_path = OUTPUT_DIR / "cross_modal_features.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                        choices=["all", "fetch", "extract", "analyze"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-proteins", type=int, default=0)
    args = parser.parse_args()

    if args.step in ["all", "fetch"]:
        step_fetch()
    if args.step in ["all", "extract"]:
        step_extract(device=args.device, max_proteins=args.max_proteins)
    if args.step in ["all", "analyze"]:
        step_analyze()


if __name__ == "__main__":
    main()
