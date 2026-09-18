#!/usr/bin/env python3
"""Extract ESM-3 and ESM-2 layer 33 activations for the 12,491 eval proteins.

Saves residue-level (float16) and protein-level (mean-pooled, float32).
Supports --model esm3|esm2 flag.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/07_extract_eval_activations.py --model esm3
    CUDA_VISIBLE_DEVICES=1 ./env/bin/python scripts/scaled_1.5M/07_extract_eval_activations.py --model esm2
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
log = logging.getLogger("extract_eval")

EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_ROOT = ROOT / "results" / "scaled_1.5M" / "activations"
CHUNK_SIZE = 2_000


def extract_esm3(sequences, device="cuda"):
    """Extract ESM-3 layer 33 activations."""
    from models.esm3_hooks import load_esm3, ESM3HookManager, tokenize_sequence

    model, tokenizers = load_esm3(device=device)
    hook_mgr = ESM3HookManager(model, layers=[33], extract_attention=False)

    results = []
    for pid, seq in sequences:
        try:
            inputs = tokenize_sequence(seq, tokenizers, device=device)
            hook_mgr.cache.clear()
            hook_mgr.register()
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                model(**inputs)
            acts = hook_mgr.cache.residual_stream[33][0].float()
            acts = acts[1:-1]  # remove BOS/EOS
            hook_mgr.remove()
            results.append((pid, acts.cpu()))
        except Exception as e:
            log.warning(f"  Error on {pid}: {e}")
            continue
    return results


def extract_esm2(sequences, device="cuda", batch_size=4):
    """Extract ESM-2 layer 33 activations with batching."""
    from transformers import AutoModel, AutoTokenizer

    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    results = []
    # Sort by length for efficient batching
    sorted_seqs = sorted(sequences, key=lambda x: len(x[1]))

    for bi in range(0, len(sorted_seqs), batch_size):
        batch = sorted_seqs[bi:bi + batch_size]
        batch_ids = [b[0] for b in batch]
        batch_seqs = [b[1] for b in batch]

        try:
            encoded = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)

            hidden = outputs.hidden_states[-1].cpu().float()
            attention_mask = encoded["attention_mask"].cpu()

            for j in range(len(batch_ids)):
                mask = attention_mask[j].bool()
                true_pos = mask.nonzero(as_tuple=True)[0]
                if len(true_pos) < 3:
                    continue
                acts = hidden[j, true_pos[1]:true_pos[-1]]  # skip BOS, up to EOS
                results.append((batch_ids[j], acts))
        except Exception as e:
            log.warning(f"  Batch error at {bi}: {e}")
            continue

    return results


def save_chunk(results, residue_dir, protein_dir, chunk_idx, d_model):
    """Save a chunk of extraction results."""
    protein_means = []
    residue_acts = []
    residue_offsets = []
    valid_ids = []
    offset = 0

    for pid, acts in results:
        L = acts.shape[0]
        protein_means.append(acts.mean(dim=0).numpy())
        residue_acts.append(acts.half().numpy())
        residue_offsets.append((len(valid_ids), offset, L))
        valid_ids.append(pid)
        offset += L

    if not valid_ids:
        return 0

    # Save protein-level
    pmeans = np.stack(protein_means)
    with h5py.File(protein_dir / f"chunk{chunk_idx:04d}.h5", "w") as f:
        f.create_dataset("activations", data=pmeans, dtype="float32")
        f.create_dataset("ids", data=[s.encode() for s in valid_ids])

    # Save residue-level
    all_res = np.concatenate(residue_acts, axis=0)
    offs = np.array(residue_offsets, dtype=np.int64)
    with h5py.File(residue_dir / f"chunk{chunk_idx:04d}.h5", "w") as f:
        f.create_dataset("activations", data=all_res, dtype="float16",
                         chunks=(min(4096, all_res.shape[0]), d_model))
        f.create_dataset("offsets", data=offs)
        f.create_dataset("ids", data=[s.encode() for s in valid_ids])

    return len(valid_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["esm3", "esm2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (ESM-2 only)")
    args = parser.parse_args()

    d_model = 1536 if args.model == "esm3" else 1280

    # Load eval sequences
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)
    all_pairs = [(pid, seq) for pid, seq in sequences.items() if 50 <= len(seq) <= 1022]
    log.info(f"Eval proteins: {len(all_pairs)}")

    # Output dirs
    residue_dir = OUTPUT_ROOT / args.model / "residue_L33"
    protein_dir = OUTPUT_ROOT / args.model / "protein_L33"
    residue_dir.mkdir(parents=True, exist_ok=True)
    protein_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(all_pairs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    total = 0
    total_residues = 0
    start_time = time.time()

    for ci in range(n_chunks):
        chunk_start = ci * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, len(all_pairs))
        chunk_pairs = all_pairs[chunk_start:chunk_end]

        # Resume check
        if (residue_dir / f"chunk{ci:04d}.h5").exists():
            log.info(f"  Chunk {ci+1}/{n_chunks}: already done, skipping")
            total += len(chunk_pairs)
            continue

        log.info(f"  Chunk {ci+1}/{n_chunks}: processing {len(chunk_pairs)} proteins...")

        if args.model == "esm3":
            results = extract_esm3(chunk_pairs, device=args.device)
        else:
            results = extract_esm2(chunk_pairs, device=args.device, batch_size=args.batch_size)

        n_saved = save_chunk(results, residue_dir, protein_dir, ci, d_model)
        total += n_saved
        total_residues += sum(r[1].shape[0] for r in results)

        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0
        log.info(f"    Saved {n_saved} proteins, {total_residues} residues total | "
                 f"Rate: {rate:.1f} prot/s")

        del results
        torch.cuda.empty_cache()

    log.info(f"\nDone! {total} proteins, {total_residues} residues")
    log.info(f"Elapsed: {(time.time() - start_time)/60:.1f} min")


if __name__ == "__main__":
    main()
