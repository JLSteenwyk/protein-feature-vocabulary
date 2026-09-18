#!/usr/bin/env python3
"""Tuned lens analysis for ESM-2 and ESM-3.

Unlike raw logit lens (which projects intermediate hidden states directly through
the LM head), tuned lens trains a small learned affine transformation at each layer
to better predict the final output distribution. This corrects for the fact that
intermediate representations may not be in the same basis as the final layer.

Method:
1. Collect (hidden_state_at_layer_L, final_logits) pairs from ~200 proteins
2. Train a linear probe (affine transform) at each layer to predict final logits
3. Evaluate on held-out proteins: KL divergence, top-1/top-5 agreement
4. Compare tuned lens vs raw logit lens — the gap reveals how much the
   representation basis changes through the network

Run as:
    ./env/bin/python scripts/unified/run_tuned_lens.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_tuned_lens.py --model esm3 --device cuda:1

Output: results/unified/{model}/tuned_lens.json
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.interventions import (
    get_layers, cache_residual_stream, logit_lens_project,
)
from models.metrics import kl_divergence


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "tuned_lens_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("tuned_lens"), out_dir


def load_sequences(max_proteins=500, max_length=500):
    seq_file = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    if seq_file.exists():
        with open(seq_file) as f:
            all_seqs = json.load(f)
    else:
        all_seqs = {}
        fasta_file = ROOT / "data" / "scaled" / "sequences" / "sequences.fasta"
        if fasta_file.exists():
            acc, seq = None, []
            with open(fasta_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if acc and seq:
                            all_seqs[acc] = "".join(seq)
                        acc = line[1:].split()[0]
                        seq = []
                    else:
                        seq.append(line)
            if acc and seq:
                all_seqs[acc] = "".join(seq)
    filtered = {k: v for k, v in all_seqs.items() if len(v) <= max_length}
    sorted_accs = sorted(filtered.keys())[:max_proteins]
    return {k: filtered[k] for k in sorted_accs}


class AffineLens(nn.Module):
    """Learned affine transformation for tuned lens."""
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return x @ self.weight.T + self.bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--train-fraction", type=float, default=0.6,
                        help="Fraction of proteins for training the lens")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-residues-per-protein", type=int, default=200,
                        help="Subsample residues per protein for training efficiency")
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Tuned Lens: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        n_layers = len(model.esm.encoder.layer)
        d_model = model.config.hidden_size
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        n_layers = len(model.transformer.blocks)
        d_model = model.transformer.blocks[0].ffn[1].in_features

    log.info(f"{args.model}: {n_layers} layers, d_model={d_model}")

    # Evaluate every other layer + first/last
    eval_layers = sorted(set([0, n_layers - 1] + list(range(0, n_layers, 2))))
    log.info(f"Training tuned lens at {len(eval_layers)} layers")

    # Split proteins into train/eval
    all_accs = list(sequences.keys())
    n_train = int(len(all_accs) * args.train_fraction)
    train_accs = all_accs[:n_train]
    eval_accs = all_accs[n_train:]
    log.info(f"Train: {len(train_accs)}, Eval: {len(eval_accs)}")

    # Phase 1: Collect training data (hidden states + final logits)
    log.info("Collecting training data...")
    train_data = {l: {"hidden": [], "logits": []} for l in eval_layers}
    t0 = time.time()

    for i, acc in enumerate(train_accs):
        seq = sequences[acc]
        if len(seq) < 10:
            continue

        with cache_residual_stream(model, args.model, eval_layers) as cache:
            if args.model == "esm2":
                inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                                   max_length=1024).to(args.device)
                with torch.no_grad():
                    out = model(**inputs)
                final_logits = out.logits[0, 1:-1].float().cpu()  # (L, V)
            else:
                seq_tokens = tokenizers.sequence.encode(seq)
                seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                          device=args.device).unsqueeze(0)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(sequence_tokens=seq_tensor)
                logits_raw = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
                final_logits = logits_raw[0, 1:-1].float().cpu()

        L = final_logits.size(0)
        # Subsample residues
        if L > args.max_residues_per_protein:
            idx = np.random.RandomState(i).choice(L, args.max_residues_per_protein, replace=False)
            idx.sort()
        else:
            idx = np.arange(L)

        final_target = torch.log_softmax(final_logits[idx], dim=-1)  # (n, V)

        for l in eval_layers:
            if l in cache:
                h = cache[l][0, 1:1+L].float().cpu()
                train_data[l]["hidden"].append(h[idx])
                train_data[l]["logits"].append(final_target)

        if (i + 1) % 50 == 0:
            log.info(f"  Collected {i+1}/{len(train_accs)} proteins [{time.time()-t0:.0f}s]")

    # Concatenate training data
    for l in eval_layers:
        if train_data[l]["hidden"]:
            train_data[l]["hidden"] = torch.cat(train_data[l]["hidden"], dim=0)
            train_data[l]["logits"] = torch.cat(train_data[l]["logits"], dim=0)
            log.info(f"  Layer {l}: {train_data[l]['hidden'].shape[0]} residues")

    # Phase 2: Train affine lens at each layer
    log.info("Training tuned lenses...")
    lenses = {}

    from models.interventions import get_final_norm, get_lm_head
    lm_head = get_lm_head(model, args.model)
    final_norm = get_final_norm(model, args.model)
    # Cast model components to float32 so they match AffineLens output dtype
    if lm_head is not None:
        lm_head = lm_head.float()
    if final_norm is not None:
        final_norm = final_norm.float()

    for l in eval_layers:
        if train_data[l]["hidden"].shape[0] == 0:
            continue

        lens = AffineLens(d_model).to(args.device)
        optimizer = optim.Adam(lens.parameters(), lr=args.lr)

        H = train_data[l]["hidden"].to(args.device)
        target_log_p = train_data[l]["logits"].to(args.device)

        # Move LM head components
        best_loss = float("inf")
        for epoch in range(args.epochs):
            # Mini-batch training
            perm = torch.randperm(H.size(0))
            total_loss = 0
            n_batches = 0
            for start in range(0, H.size(0), 512):
                batch_idx = perm[start:start+512]
                h_batch = H[batch_idx]
                target_batch = target_log_p[batch_idx]

                transformed = lens(h_batch)
                # Project through final norm + LM head (both cast to float32)
                # No torch.no_grad() here — gradients must flow through
                # frozen norm/head back to lens. Only lens params are in optimizer.
                if final_norm is not None:
                    normed = final_norm(transformed)
                else:
                    normed = transformed
                pred_logits = lm_head(normed)
                pred_log_p = torch.log_softmax(pred_logits.float(), dim=-1)

                loss = torch.nn.functional.kl_div(
                    pred_log_p, target_batch.exp(), reduction="batchmean", log_target=False
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss

        lenses[l] = lens.cpu()
        log.info(f"  Layer {l}: trained, final KL={best_loss:.4f}")

    # Phase 3: Evaluate on held-out proteins
    log.info("Evaluating tuned vs raw lens...")
    raw_kl = {l: [] for l in eval_layers}
    tuned_kl = {l: [] for l in eval_layers}
    raw_top1 = {l: [] for l in eval_layers}
    tuned_top1 = {l: [] for l in eval_layers}
    raw_top5 = {l: [] for l in eval_layers}
    tuned_top5 = {l: [] for l in eval_layers}

    for i, acc in enumerate(eval_accs):
        seq = sequences[acc]
        if len(seq) < 10:
            continue

        with cache_residual_stream(model, args.model, eval_layers) as cache:
            if args.model == "esm2":
                inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                                   max_length=1024).to(args.device)
                with torch.no_grad():
                    out = model(**inputs)
                final_logits = out.logits[0, 1:-1].float().cpu()
            else:
                seq_tokens = tokenizers.sequence.encode(seq)
                seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                          device=args.device).unsqueeze(0)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(sequence_tokens=seq_tensor)
                logits_raw = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
                final_logits = logits_raw[0, 1:-1].float().cpu()

        L = final_logits.size(0)
        final_preds = final_logits.argmax(dim=-1)  # (L,)
        final_top5 = final_logits.topk(5, dim=-1).indices  # (L, 5)

        for l in eval_layers:
            if l not in cache or l not in lenses:
                continue
            h = cache[l][0, 1:1+L].float().cpu()

            # Raw logit lens
            with torch.no_grad():
                raw_logits = logit_lens_project(
                    h.unsqueeze(0).to(args.device), model, args.model
                )[0].float().cpu()

            # Tuned logit lens
            with torch.no_grad():
                transformed = lenses[l](h)
                tuned_logits_proj = logit_lens_project(
                    transformed.unsqueeze(0).to(args.device), model, args.model
                )[0].float().cpu()

            # KL divergence
            r_kl = kl_divergence(final_logits.unsqueeze(0), raw_logits.unsqueeze(0))
            t_kl = kl_divergence(final_logits.unsqueeze(0), tuned_logits_proj.unsqueeze(0))
            raw_kl[l].append(r_kl.mean().item())
            tuned_kl[l].append(t_kl.mean().item())

            # Top-1 agreement
            raw_preds = raw_logits.argmax(dim=-1)
            tuned_preds = tuned_logits_proj.argmax(dim=-1)
            raw_top1[l].append(float((raw_preds == final_preds).float().mean()))
            tuned_top1[l].append(float((tuned_preds == final_preds).float().mean()))

            # Top-5 agreement
            raw_t5 = raw_logits.topk(5, dim=-1).indices
            tuned_t5 = tuned_logits_proj.topk(5, dim=-1).indices
            raw_overlap = sum(
                len(set(raw_t5[j].tolist()) & set(final_top5[j].tolist())) / 5
                for j in range(L)
            ) / L
            tuned_overlap = sum(
                len(set(tuned_t5[j].tolist()) & set(final_top5[j].tolist())) / 5
                for j in range(L)
            ) / L
            raw_top5[l].append(raw_overlap)
            tuned_top5[l].append(tuned_overlap)

        if (i + 1) % 50 == 0:
            log.info(f"  Evaluated {i+1}/{len(eval_accs)} proteins")

    log.info("=== Results ===")
    output = {
        "model": args.model,
        "n_layers": n_layers,
        "d_model": d_model,
        "eval_layers": eval_layers,
        "n_train": len(train_accs),
        "n_eval": len(eval_accs),
        "epochs": args.epochs,
        "per_layer": {},
    }

    for l in eval_layers:
        if not raw_kl[l]:
            continue
        entry = {
            "raw_kl": {"mean": float(np.mean(raw_kl[l])), "std": float(np.std(raw_kl[l]))},
            "tuned_kl": {"mean": float(np.mean(tuned_kl[l])), "std": float(np.std(tuned_kl[l]))},
            "raw_top1": {"mean": float(np.mean(raw_top1[l]))},
            "tuned_top1": {"mean": float(np.mean(tuned_top1[l]))},
            "raw_top5": {"mean": float(np.mean(raw_top5[l]))},
            "tuned_top5": {"mean": float(np.mean(tuned_top5[l]))},
            "improvement_kl": float(np.mean(raw_kl[l]) - np.mean(tuned_kl[l])),
            "improvement_top1": float(np.mean(tuned_top1[l]) - np.mean(raw_top1[l])),
        }
        output["per_layer"][str(l)] = entry
        log.info(f"  Layer {l}: raw_KL={entry['raw_kl']['mean']:.4f} → tuned_KL={entry['tuned_kl']['mean']:.4f} "
                 f"(Δ={entry['improvement_kl']:.4f}), "
                 f"top1: {entry['raw_top1']['mean']:.3f} → {entry['tuned_top1']['mean']:.3f}")

    out_path = out_dir / "tuned_lens.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")

    # Save trained lenses
    lens_path = out_dir / "tuned_lens_weights.pt"
    torch.save({l: {"weight": lenses[l].weight.data, "bias": lenses[l].bias.data}
                for l in lenses}, lens_path)
    log.info(f"Saved lens weights to {lens_path}")


if __name__ == "__main__":
    main()
