"""Reconstruction figure separating historical metrics from matched-residue R2."""
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT, RESULTS
from probe_checkpoint import atomic_json, file_identity

sys.path.insert(0, str(ROOT/'src'))


def main():
    fresh_path = OUT/'reconstruction_resampled.json'
    fresh = json.loads(fresh_path.read_text())
    if set(fresh['models']) != {'esm3', 'esm2'}:
        raise ValueError('Both matched-residue reconstructions must finish before plotting')
    old_path = RESULTS/'reconstruction_quality.json'
    sparse_path = RESULTS/'feature_sparsity_analysis.json'
    old, sparsity = [json.loads(p.read_text()) for p in [old_path, sparse_path]]
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    colors = ['#0072B2', '#D55E00']
    checkpoint_records = {}
    for i, (model, color) in enumerate(zip(['esm3', 'esm2'], colors)):
        path = ROOT/f'models/sae_1.5M/{model}_residue_ef8_k64/best.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        history = checkpoint['log_history']
        ax = axes[0, i]
        ax.plot([r['step'] for r in history], [r['loss'] for r in history], color=color)
        ax.set(xlabel='Recorded optimizer step', ylabel='Recorded training loss', title=f'{"AB"[i]}  {model.upper()} principal SAE')
        checkpoint_records[model] = file_identity(path, content=True)
        histogram = sparsity[model]['residue']['l0_distribution']['histogram']
        edges, counts = np.array(histogram['bin_edges']), np.array(histogram['counts'])
        axes[0, 2].step((edges[:-1]+edges[1:])/2, counts, where='mid', label=model.upper(), color=color)
        r = old[model]['residue']['cosine_similarity']
        axes[1, 0].bar(i, r['mean'], yerr=r['std'], color=color, capsize=3)
        r = old[model]['residue']['variance_explained']
        axes[1, 1].bar(i, r['mean'], yerr=r['std'], color=color, capsize=3)
        r = fresh['models'][model]
        axes[1, 2].bar(i, r['global_centered_r2'], color=color)
        axes[1, 2].vlines(i, *r['r2_cluster_range95'], color='black', linewidth=2)
        axes[1, 2].text(i, r['global_centered_r2']+.04, f"{r['global_centered_r2']:.3f}", ha='center')
    axes[0, 2].set(xlabel='Number of positive SAE activations', ylabel='Residues (log scale)',
                    yscale='log', title='C  TopK is an upper bound')
    axes[0, 2].legend(fontsize=8)
    for ax in axes[1]:
        ax.set(xticks=[0, 1], xticklabels=['ESM-3', 'ESM-2'], ylim=(0, 1.1))
    axes[1, 0].set(title='D  Historical residue cosine', ylabel='Mean cosine +/- SD (not CI)')
    axes[1, 1].set(title='E  Historical within-vector ratio', ylabel='Mean [1 - Var(error)/Var(input)] +/- SD')
    axes[1, 2].set(title='F  Matched full-cohort residue sample', ylabel='Global centered R2 + cluster range')
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT/f'revision/figures/reconstruction_corrected.{ext}', dpi=220)
    plt.close(fig)
    atomic_json(OUT/'reconstruction_figure_audit.json', {
        'inputs': [file_identity(p, content=True) for p in [fresh_path, old_path, sparse_path]],
        'checkpoints': checkpoint_records,
        'historical_sampling': 'First activation chunks until >=500000 rows, then seeded subsample within that prefix, independently by model; not a uniform sample of the full evaluation cohort.',
        'historical_variance_definition': 'Average per-vector [1 - Var_dimension(error)/(Var_dimension(input)+1e-10)]; not global centered R2.',
        'historical_error_bars': 'Across-residue SD, not confidence intervals or independent proteins.',
        'ESM3_l0_below64': int(sum(sparsity['esm3']['residue']['l0_distribution']['histogram']['counts'][:-1])),
        'ESM2_l0_range': [sparsity['esm2']['residue']['l0_distribution'][k] for k in ['min', 'max']],
        'new_sampling': fresh['sampling'], 'new_n': fresh['n'], 'new_uncertainty_scope': fresh['scope'],
        'interpretation': 'No attribution of MSE/loss differences to dimensionality, no claim of exact 64 nonzero features or retention of all biological information; training step provenance limitations remain.'})
    print('Reconstruction figure saved from complete matched-sample results.')


if __name__ == '__main__':
    main()
