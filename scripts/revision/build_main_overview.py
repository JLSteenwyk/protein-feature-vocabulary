"""Build Figure 1 solely from corrected, provenance-tracked revision outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity
from build_probe_detail import validate_control_set


def main():
    sources = {key: OUT/name for key, name in {
        'go': 'go_audit.json', 'reconstruction': 'reconstruction_resampled.json',
        'probes': 'combined_probe_controls.json', 'probe_check': 'probe_figure_verification.json',
        'reconstruction_check': 'reconstruction_verification.json'}.items()}
    data = {key: json.loads(path.read_text()) for key, path in sources.items()}
    validate_control_set(data['probes'])
    if data['probe_check']['input']['sha256'] != file_identity(sources['probes'], content=True)['sha256']:
        raise ValueError('Probe verification refers to a different snapshot')
    if not data['probe_check']['prediction_hashes_and_metrics_verified']:
        raise ValueError('Probe metrics unverified')
    if data['reconstruction_check']['source']['sha256'] != file_identity(sources['reconstruction'], content=True)['sha256']:
        raise ValueError('Reconstruction verification refers to a different snapshot')
    if not all(r['paired_sample_verified'] for r in data['reconstruction_check']['checks'].values()):
        raise ValueError('Reconstruction sample pairing unverified')
    models = ['esm3', 'esm2']
    colors = ['#0072B2', '#D55E00']
    widths = [12288, 10240]
    active, enriched = [], []
    for model in models:
        go = data['go']['models'][model]
        records = go['per_feature']
        count = sum(r['n_terms_all_background_bh'] > 0 for r in records)
        if len(records) != go['n_active'] or count != go['n_enriched_revised']:
            raise ValueError('GO denominator or enriched-count mismatch')
        active.append(len(records)); enriched.append(count)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    x = np.arange(2)
    ax = axes[0, 0]
    ax.bar(x, widths, color='#DDDDDD', label='Dictionary width')
    ax.bar(x, active, color=colors, label='Sequence-evaluation active')
    for i in range(2):
        ax.text(i, widths[i]+250, f'{widths[i]:,}', ha='center')
        ax.text(i, active[i]/2, f'{active[i]:,} active', ha='center', color='white')
    ax.set(title='A  Dictionary size is not TopK sparsity', ylabel='SAE coordinates',
           xticks=x, xticklabels=['ESM-3', 'ESM-2'], ylim=(0, 15800))
    ax.text(.5, .96, 'TopK: 64 selected entries per residue', ha='center', va='top', transform=ax.transAxes)
    ax = axes[0, 1]
    for i, model in enumerate(models):
        r = data['reconstruction']['models'][model]
        point = r['global_centered_r2']; lo, hi = r['r2_cluster_range95']
        ax.bar(i, point, color=colors[i])
        ax.errorbar(i, point, yerr=[[point-lo], [hi-point]], color='#222222', capsize=4)
        ax.text(i, hi+.025, f'{point:.3f}', ha='center')
    ax.set(title='B  Matched-residue reconstruction', ylabel='Global centered reconstruction R2',
           xticks=x, xticklabels=['ESM-3', 'ESM-2'], ylim=(0, 1))
    ax = axes[1, 0]
    ax.bar(x, np.array(enriched)/active, color=colors)
    for i in range(2):
        ax.text(i, enriched[i]/active[i]+.025, f'{enriched[i]:,}/{active[i]:,}', ha='center')
    ax.set(title='C  Complete residue-feature GO tests', ylabel='Fraction with detected enrichment',
           xticks=x, xticklabels=['ESM-3', 'ESM-2'], ylim=(0, 1.05))
    ax = axes[1, 1]
    for i, model in enumerate(models):
        r = data['probes']['models'][model]['intact']
        ys = [r['auroc'], r['macro_within_protein_auroc']]
        ax.plot(x, ys, 'o-', color=colors[i], label=model.upper().replace('ESM', 'ESM-'))
        ci = r['ci95_auroc']
        ax.errorbar(0, ys[0], yerr=[[ys[0]-ci['low']], [ci['high']-ys[0]]], color=colors[i], capsize=4)
    ax.set(title='D  Cluster-held-out intact probes', ylabel='AUROC', xticks=x,
           xticklabels=['Pooled residues', 'Within protein'], xlim=(-.3, 1.3), ylim=(.5, 1))
    ax.legend(loc='lower left')
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT/f'revision/figures/main_overview_corrected.{ext}', dpi=220)
    plt.close(fig)
    atomic_json(OUT/'main_overview_provenance.json', {
        'inputs': [file_identity(p, content=True) for p in sources.values()],
        'script': file_identity(Path(__file__), content=True),
        'historical_generator': file_identity(ROOT/'scripts/publication_figures/fig1_interpretability.py', content=True),
        'go_active': active, 'go_enriched': enriched, 'dictionary_widths': widths,
        'scope': 'Different readouts have distinct populations and uncertainty; no intrinsic biological-richness or raw/PCA superiority inference.',
        'withdrawals': ['Historical PCA value reused for both models, with hard-coded fallback in generator.',
                        'Metadata scores are separate protein dictionaries, now confined to qualified ED2.',
                        'Random baseline is sparse noise rather than a projection of original hidden states.',
                        'Historical within-vector variance ratio is not global centered R2.']})
    print('Corrected Figure 1 and input provenance exported')


if __name__ == '__main__':
    main()
