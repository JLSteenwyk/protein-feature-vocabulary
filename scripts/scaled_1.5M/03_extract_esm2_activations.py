#!/usr/bin/env python3
"""Extract ESM-2 layer 33 activations for 1.5M UniRef50 proteins.

Saves:
  - Protein-level mean-pooled vectors (float32, ~7.7 GB)
  - Residue-level activations in chunked HDF5 (float16, ~0.96 TB)

ESM-2 (650M) is much faster than ESM-3 — can batch 8-16 proteins.

Usage:
    CUDA_VISIBLE_DEVICES=1 ./env/bin/python scripts/scaled_1.5M/03_extract_esm2_activations.py
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
from transformers import AutoModel, AutoTokenizer

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_esm2")

LAYER = 33  # ESM-2 650M has 33 layers (0-32), layer 33 = final
D_MODEL = 1280
CHUNK_SIZE = 2_000
BATCH_SIZE = 8  # ESM-2 is smaller, can batch more
STORAGE_ROOT = Path("/mnt/85740f55-8e9a-4214-9500-be446866627e/interpretability_1.5M")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    residue_dir = STORAGE_ROOT / "esm2" / "residue_L33"
    protein_dir = STORAGE_ROOT / "esm2" / "protein_L33"
    residue_dir.mkdir(parents=True, exist_ok=True)
    protein_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    seq_path = ROOT / "data" / "sae_training" / "uniref50_1.5M_sequences.json"
    log.info(f"Loading sequences from {seq_path}...")
    with open(seq_path) as f:
        all_sequences = json.load(f)
    all_ids = list(all_sequences.keys())
    log.info(f"Total proteins: {len(all_ids)}")

    shard_ids = all_ids[args.shard::args.n_shards]
    log.info(f"Shard {args.shard}/{args.n_shards}: {len(shard_ids)} proteins")

    # Load ESM-2
    log.info("Loading ESM-2 (facebook/esm2_t33_650M_UR50D)...")
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(args.device)
    model.eval()
    log.info("ESM-2 loaded")

    # Process in chunks
    n_chunks = (len(shard_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_proteins = 0
    total_residues = 0
    start_time = time.time()

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, len(shard_ids))
        chunk_ids = shard_ids[chunk_start:chunk_end]

        residue_file = residue_dir / f"shard{args.shard}_chunk{chunk_idx:04d}.h5"
        protein_file = protein_dir / f"shard{args.shard}_chunk{chunk_idx:04d}.h5"

        if residue_file.exists() and protein_file.exists():
            log.info(f"  Chunk {chunk_idx}/{n_chunks}: already done, skipping")
            total_proteins += len(chunk_ids)
            continue

        chunk_protein_means = []
        chunk_residue_acts = []
        chunk_residue_offsets = []
        chunk_valid_ids = []
        residue_offset = 0

        # Sort by length for efficient batching
        id_seq_pairs = [(pid, all_sequences[pid]) for pid in chunk_ids
                        if 50 <= len(all_sequences[pid]) <= 1022]
        id_seq_pairs.sort(key=lambda x: len(x[1]))

        # Process in batches
        for bi in range(0, len(id_seq_pairs), args.batch_size):
            batch_pairs = id_seq_pairs[bi:bi + args.batch_size]
            batch_ids = [p[0] for p in batch_pairs]
            batch_seqs = [p[1] for p in batch_pairs]

            try:
                encoded = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                                    truncation=True, max_length=1024)
                encoded = {k: v.to(args.device) for k, v in encoded.items()}

                with torch.no_grad():
                    outputs = model(**encoded)

                # hidden_states[-1] = last layer output, shape (B, L_padded, 1280)
                hidden = outputs.hidden_states[-1].cpu().float()
                attention_mask = encoded["attention_mask"].cpu()

                for j in range(len(batch_ids)):
                    mask = attention_mask[j].bool()
                    # Remove BOS/EOS (first and last True positions)
                    true_positions = mask.nonzero(as_tuple=True)[0]
                    if len(true_positions) < 3:
                        continue
                    start_pos = true_positions[1]   # skip BOS
                    end_pos = true_positions[-1]     # EOS position
                    acts = hidden[j, start_pos:end_pos]  # (L, 1280)
                    L = acts.shape[0]

                    chunk_protein_means.append(acts.mean(dim=0).numpy())
                    chunk_residue_acts.append(acts.half().numpy())
                    chunk_residue_offsets.append((len(chunk_valid_ids), residue_offset, L))
                    chunk_valid_ids.append(batch_ids[j])
                    residue_offset += L

            except Exception as e:
                log.warning(f"  Batch error at {bi}: {e}")
                continue

        if not chunk_valid_ids:
            continue

        # Save
        protein_means = np.stack(chunk_protein_means)
        with h5py.File(protein_file, "w") as f:
            f.create_dataset("activations", data=protein_means, dtype="float32")
            f.create_dataset("ids", data=[s.encode() for s in chunk_valid_ids])

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
                 f"Total: {total_proteins}, {total_residues} residues | "
                 f"Rate: {rate:.1f} prot/s | ETA: {eta_hours:.1f}h")

        del chunk_protein_means, chunk_residue_acts, all_residues, protein_means

    log.info(f"\nDone! Shard {args.shard}: {total_proteins} proteins, {total_residues} residues")
    log.info(f"Elapsed: {(time.time() - start_time)/3600:.1f}h")


if __name__ == "__main__":
    main()
