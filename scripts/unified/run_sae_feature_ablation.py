#!/usr/bin/env python3
"""SAE feature ablation analysis for ESM-2 and ESM-3.

For each SAE feature in top-200 (by mean activation):
1. Hook target layer: encode -> zero feature -> decode -> substitute
2. Run forward from that layer
3. Measure prediction change (KL divergence)
4. Rank features by causal importance

Run as:
    ./env/bin/python scripts/unified/run_sae_feature_ablation.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_sae_feature_ablation.py --model esm3 --device cuda:1
    ./env/bin/python scripts/unified/run_sae_feature_ablation.py --model esm3 --condition S_St --device cuda:1

Output: results/unified/{model}/sae_feature_ablation_L{layer}.json
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
import numpy as np

from models.metrics import mean_kl_divergence
from models.interventions import sae_feature_ablation


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "sae_feature_ablation_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("sae_ablation"), out_dir


def load_sequences(max_proteins=200, max_length=500):
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


def load_sae(model_name, layer, condition="S"):
    """Load trained SAE checkpoint."""
    from sae.model import build_sae, SAEConfig

    if model_name == "esm2":
        # Try scaled first, then original
        for prefix in ["esm2_scaled", "esm2"]:
            ckpt_path = ROOT / "models" / "sae" / prefix / f"layer_{layer}_topk" / "best.pt"
            if ckpt_path.exists():
                break
    else:
        for prefix in ["esm3_scaled", "esm3"]:
            ckpt_path = ROOT / "models" / "sae" / prefix / f"{condition}_layer_{layer}_topk" / "best.pt"
            if ckpt_path.exists():
                break

    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["sae_config"]
    sae = build_sae(config)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae, config


def find_top_features(sae, model, model_type, tokenizer_or_tokenizers,
                      sequences, layer, device, top_k=200, condition="S"):
    """Find top-K features by mean activation magnitude."""
    from models.interventions import cache_residual_stream

    n_features = sae.config.dict_size
    feature_sums = torch.zeros(n_features)
    total_tokens = 0

    eval_seqs = list(sequences.values())[:100]  # Use subset for feature selection

    for seq in eval_seqs:
        if model_type == "esm2":
            inputs = tokenizer_or_tokenizers(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad():
                    model(**inputs)
        else:
            seq_tokens = tokenizer_or_tokenizers.sequence.encode(seq)
            seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(sequence_tokens=seq_tensor)

        hidden = cache[layer].float()  # (1, L, D)
        flat = hidden.reshape(-1, hidden.size(-1))
        with torch.no_grad():
            z = sae.encode(flat.to(sae.encoder.weight.device))
        feature_sums += z.sum(dim=0).cpu()
        total_tokens += flat.size(0)

    mean_acts = feature_sums / total_tokens
    top_indices = mean_acts.argsort(descending=True)[:top_k].tolist()
    return top_indices, mean_acts


def run_feature_ablation(model, model_type, tokenizer_or_tokenizers,
                         sae, layer, feature_indices, sequences, device, log,
                         condition="S"):
    """Ablate each feature individually and measure KL divergence."""

    sae_device = next(sae.parameters()).device
    eval_seqs = list(sequences.items())

    # Pre-compute baseline logits
    log.info("Computing baseline logits...")
    baseline_logits = []
    for acc, seq in eval_seqs:
        if model_type == "esm2":
            inputs = tokenizer_or_tokenizers(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                out = model(**inputs)
            baseline_logits.append(out.logits.cpu())
        else:
            seq_tokens = tokenizer_or_tokenizers.sequence.encode(seq)
            seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(sequence_tokens=seq_tensor)
            logits = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
            baseline_logits.append(logits.float().cpu())

    # Ablate each feature
    results = {}
    for feat_rank, feat_idx in enumerate(feature_indices):
        t0 = time.time()
        kl_divs = []

        for seq_idx, (acc, seq) in enumerate(eval_seqs):
            with sae_feature_ablation(model, model_type, sae.to(device), layer, [feat_idx]):
                if model_type == "esm2":
                    inputs = tokenizer_or_tokenizers(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
                    with torch.no_grad():
                        out_abl = model(**inputs)
                    logits_abl = out_abl.logits.cpu()
                else:
                    seq_tokens = tokenizer_or_tokenizers.sequence.encode(seq)
                    seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out_abl = model(sequence_tokens=seq_tensor)
                    logits_abl = (out_abl.sequence_logits if hasattr(out_abl, 'sequence_logits') else out_abl.logits).float().cpu()

            kl = mean_kl_divergence(baseline_logits[seq_idx], logits_abl)
            kl_divs.append(kl)
            sae.to("cpu")  # Move SAE back to CPU between uses

        results[str(feat_idx)] = {
            "mean_kl": float(np.mean(kl_divs)),
            "std_kl": float(np.std(kl_divs)),
            "rank": feat_rank,
            "feature_idx": feat_idx,
        }
        dt = time.time() - t0
        if (feat_rank + 1) % 20 == 0 or feat_rank == 0:
            log.info(f"  Feature {feat_idx} (rank {feat_rank}): KL={np.mean(kl_divs):.6f} ({dt:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--top-features", type=int, default=200)
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer for SAE (default: 24 for ESM-2, 33 for ESM-3)")
    parser.add_argument("--condition", default="S",
                        help="Condition for ESM-3 SAE: S or S_St")
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)

    # Default layers
    if args.layer is None:
        args.layer = 24 if args.model == "esm2" else 33

    log.info(f"=== SAE Feature Ablation: {args.model}, layer {args.layer}, condition {args.condition} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    # Load SAE
    sae, sae_config = load_sae(args.model, args.layer, args.condition)
    log.info(f"SAE: {sae_config.architecture}, dict_size={sae_config.dict_size}, k={sae_config.k}")

    # Load model
    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        tok = tokenizer
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        tok = tokenizers

    # Find top features
    log.info(f"Finding top-{args.top_features} features by activation magnitude...")
    top_features, mean_acts = find_top_features(
        sae, model, args.model, tok, sequences, args.layer, args.device,
        top_k=args.top_features, condition=args.condition
    )
    log.info(f"Top features found: {top_features[:10]}...")

    # Run ablation
    results = run_feature_ablation(
        model, args.model, tok, sae, args.layer, top_features,
        sequences, args.device, log, condition=args.condition
    )

    # Save
    output = {
        "model": args.model,
        "layer": args.layer,
        "condition": args.condition,
        "sae_architecture": sae_config.architecture,
        "dict_size": sae_config.dict_size,
        "n_proteins": len(sequences),
        "n_features_tested": len(top_features),
        "feature_results": results,
        "mean_activations": {str(i): float(mean_acts[i]) for i in top_features},
    }

    out_path = out_dir / f"sae_feature_ablation_L{args.layer}_{args.condition}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
