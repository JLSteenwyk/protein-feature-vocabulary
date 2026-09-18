#!/usr/bin/env python3
"""Extract ESM-3 layer 33 activations for 1.5M UniRef50 proteins.

Saves:
  - Protein-level mean-pooled vectors (float32, ~9 GB)
  - Residue-level activations in chunked HDF5 (float16, ~1.1 TB)

Supports resume: skips already-processed chunks.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/02_extract_esm3_activations.py
    # Or split across GPUs:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/02_extract_esm3_activations.py --shard 0 --n-shards 2
    CUDA_VISIBLE_DEVICES=1 ./env/bin/python scripts/scaled_1.5M/02_extract_esm3_activations.py --shard 1 --n-shards 2
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_esm3")

LAYER = 33
D_MODEL = 1536
CHUNK_SIZE = 2_000  # proteins per chunk file
STORAGE_ROOT = Path("/mnt/85740f55-8e9a-4214-9500-be446866627e/interpretability_1.5M")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="Shard index (for multi-GPU)")
    parser.add_argument("--n-shards", type=int, default=1, help="Total shards")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Output paths
    residue_dir = STORAGE_ROOT / "esm3" / "residue_L33"
    protein_dir = STORAGE_ROOT / "esm3" / "protein_L33"
    residue_dir.mkdir(parents=True, exist_ok=True)
    protein_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    seq_path = ROOT / "data" / "sae_training" / "uniref50_1.5M_sequences.json"
    log.info(f"Loading sequences from {seq_path}...")
    with open(seq_path) as f:
        all_sequences = json.load(f)
    all_ids = list(all_sequences.keys())
    log.info(f"Total proteins: {len(all_ids)}")

    # Shard
    shard_ids = all_ids[args.shard::args.n_shards]
    log.info(f"Shard {args.shard}/{args.n_shards}: {len(shard_ids)} proteins")

    # Load ESM-3
    log.info("Loading ESM-3...")
    from models.esm3_hooks import load_esm3, ESM3HookManager, tokenize_sequence
    model, tokenizers = load_esm3(device=args.device)
    hook_mgr = ESM3HookManager(model, layers=[LAYER], extract_attention=False)
    log.info("ESM-3 loaded")

    # Process in chunks
    n_chunks = (len(shard_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_proteins = 0
    total_residues = 0
    start_time = time.time()

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, len(shard_ids))
        chunk_ids = shard_ids[chunk_start:chunk_end]

        # File naming: shard_X_chunk_Y
        residue_file = residue_dir / f"shard{args.shard}_chunk{chunk_idx:04d}.h5"
        protein_file = protein_dir / f"shard{args.shard}_chunk{chunk_idx:04d}.h5"

        # Resume: skip if both files exist
        if residue_file.exists() and protein_file.exists():
            log.info(f"  Chunk {chunk_idx}/{n_chunks}: already done, skipping")
            total_proteins += len(chunk_ids)
            continue

        # Extract activations for this chunk
        chunk_protein_means = []
        chunk_residue_acts = []
        chunk_residue_offsets = []  # (protein_idx_in_chunk, start_residue, n_residues)
        chunk_valid_ids = []
        residue_offset = 0

        for pi, pid in enumerate(chunk_ids):
            seq = all_sequences[pid]
            if len(seq) < 50 or len(seq) > 1022:
                continue

            try:
                inputs = tokenize_sequence(seq, tokenizers, device=args.device)
                hook_mgr.cache.clear()
                hook_mgr.register()

                with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                    model(**inputs)

                # Get layer 33 activations: (1, L+2, 1536) → (L, 1536) removing BOS/EOS
                acts = hook_mgr.cache.residual_stream[LAYER][0].float()  # already CPU from hook
                acts = acts[1:-1]  # remove BOS/EOS
                L = acts.shape[0]

                # Store
                chunk_protein_means.append(acts.mean(dim=0).numpy())
                chunk_residue_acts.append(acts.half().numpy())  # float16 for storage
                chunk_residue_offsets.append((len(chunk_valid_ids), residue_offset, L))
                chunk_valid_ids.append(pid)
                residue_offset += L

                hook_mgr.remove()

            except Exception as e:
                log.warning(f"  Error on {pid} (len={len(seq)}): {e}")
                continue

        if not chunk_valid_ids:
            log.warning(f"  Chunk {chunk_idx}: no valid proteins, skipping")
            continue

        # Save protein-level means
        protein_means = np.stack(chunk_protein_means)
        with h5py.File(protein_file, "w") as f:
            f.create_dataset("activations", data=protein_means, dtype="float32")
            f.create_dataset("ids", data=[s.encode() for s in chunk_valid_ids])

        # Save residue-level (concatenated, float16)
        all_residues = np.concatenate(chunk_residue_acts, axis=0)
        offsets_arr = np.array(chunk_residue_offsets, dtype=np.int64)
        with h5py.File(residue_file, "w") as f:
            f.create_dataset("activations", data=all_residues, dtype="float16",
                             chunks=(min(4096, all_residues.shape[0]), D_MODEL))
            f.create_dataset("offsets", data=offsets_arr)
            f.create_dataset("ids", data=[s.encode() for s in chunk_valid_ids])

        total_proteins += len(chunk_valid_ids)
        total_residues += all_residues.shape[0]

        elapsed = time.time() - start_time
        rate = total_proteins / elapsed if elapsed > 0 else 0
        eta_hours = (len(shard_ids) - total_proteins) / rate / 3600 if rate > 0 else 0

        log.info(f"  Chunk {chunk_idx+1}/{n_chunks}: {len(chunk_valid_ids)} proteins, "
                 f"{all_residues.shape[0]} residues | "
                 f"Total: {total_proteins} proteins, {total_residues} residues | "
                 f"Rate: {rate:.1f} prot/s | ETA: {eta_hours:.1f}h")

        # Free memory
        del chunk_protein_means, chunk_residue_acts, all_residues, protein_means

    log.info(f"\nDone! Shard {args.shard}: {total_proteins} proteins, {total_residues} residues")
    log.info(f"Elapsed: {(time.time() - start_time)/3600:.1f}h")


if __name__ == "__main__":
    main()
