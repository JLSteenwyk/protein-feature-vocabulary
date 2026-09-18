"""Offline provenance/selection audit of archived metadata-based LLM scores."""
import json
import numpy as np
import h5py
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT, RESULTS
from probe_checkpoint import atomic_json, file_identity


def validation_indices(values, rng, start=9000):
    order = np.argsort(-values[start:], kind='stable')+start
    n = len(order)
    extra = rng.choice(range(15, n-15), min(7, max(0, n-30)), replace=False) if n > 30 else []
    candidates = list(order[:8])+list(order[n//3:n//3+7])+list(order[-8:])+list(order[extra])
    return np.array(list(dict.fromkeys(candidates))[:30], dtype=int)


def score_summary(records, key):
    values = np.array([np.nan if r.get(key) is None else r[key] for r in records], dtype=float)
    finite = np.isfinite(values)
    null = np.array([r.get(key) is None for r in records])
    constant = np.array([np.ptp(r['actual_values']) == 0 for r in records])
    if np.any(finite & constant):
        raise ValueError('Finite correlation for constant actual values')
    return {'n_finite': int(finite.sum()), 'n_null': int(null.sum()),
            'n_nonfinite_nonnull': int((~finite & ~null).sum()),
            'n_nonfinite_with_constant_actual': int((~finite & ~null & constant).sum()),
            'median': float(np.median(values[finite])), 'mean': float(values[finite].mean())}


def main():
    source = RESULTS/'autointerpretability.json'
    archive = json.loads(source.read_text())
    clusters = json.loads((OUT/'clusters.json').read_text())
    report = {'readout': 'Protein-level SAE activation predicted from names, accessions, organism and Pfam metadata; not residue activation from sequence.',
              'description': 'First 9000 proteins by storage order; 17 high and 15 low examples. Active examples include GO/Pfam metadata. Same Claude family generates descriptions and scores them.',
              'validation': '29 or 30 activation-selected proteins per feature from remaining 3491, ordered top/middle/bottom/random; not all residues or all held-out proteins. Order is not randomized after activation-based selection.',
              'normalization': 'Global maximum uses both description and validation pools; scaling alone does not change Pearson r but protocol is not strictly discovery-only.',
              'missingness': 'Null may reflect parsing/API failure. Nonfinite correlations may reflect constant actual values or predictions. Archive omits raw predictions and responses, so missingness cannot all be labeled invalid JSON.',
              'imputation': 'Malformed or short numeric lists are padded with 0.5; there is no guarantee that this moves correlation toward zero.',
              'scope': 'Selected metadata-based protein-level task; not direct validation of principal residue dictionaries or proof of semantic interpretability. No API calls made.',
              'models': {}, 'inputs': [file_identity(source, content=True), file_identity(OUT/'clusters.json', content=True),
                                        file_identity(OUT/'go_audit.json', content=True)]}
    provenance_sources = ['scripts/scaled_1.5M/11_protein_autointerpretability.py',
                          'scripts/scaled_1.5M/08_encode_eval_through_saes.py',
                          'scripts/publication_figures/supplementary/ed03_autointerpretability_go.py']
    report['inputs'] += [file_identity(ROOT/p, content=True) for p in provenance_sources]
    selections = {}
    for model in ['esm3', 'esm2']:
        path = RESULTS/f'sae_features/{model}/protein/features.h5'
        with h5py.File(path) as f:
            z = f['features'][:]
            ids = np.array([p.decode() for p in f['protein_ids'][:]])
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate protein IDs')
        active = np.flatnonzero((z > 0).any(0))
        rng = np.random.RandomState(42)
        chosen = rng.choice(active, 300, replace=False) if len(active) > 300 else active
        records = archive[model]['per_feature']
        if not np.array_equal(chosen, [r['feature_id'] for r in records]):
            raise ValueError('Archived feature selection disagrees with protein-level source')
        selections[model] = {'description_ids': ids[:9000].tolist(), 'validation_pool_ids': ids[9000:].tolist(), 'features': []}
        for r in records:
            values = z[:, r['feature_id']]
            selected = validation_indices(values, rng)
            actual = values[selected].astype(float)/max(float(values.max()), 1e-8)
            if len(selected) != r['n_validate'] or not np.allclose(actual, r['actual_values'], atol=1e-7, rtol=1e-6):
                raise ValueError('Reconstructed validation actual values disagree')
            selections[model]['features'].append({'feature_id': r['feature_id'], 'validation_ids': ids[selected].tolist()})
        summaries = {}
        for judge in ['claude', 'gpt']:
            summaries[judge] = score_summary(records, f'{judge}_pearson_r')
            old = archive[model]['summary'][judge]
            if old['n_valid'] != summaries[judge]['n_finite'] or not np.isclose(old['median_r'], summaries[judge]['median']):
                raise ValueError('Score summary mismatch')
        paired = np.array([[r.get('claude_pearson_r') if r.get('claude_pearson_r') is not None else np.nan,
                            r.get('gpt_pearson_r') if r.get('gpt_pearson_r') is not None else np.nan] for r in records])
        paired = paired[np.isfinite(paired).all(1)]
        shared_clusters = set(clusters[p] for p in ids[:9000]) & set(clusters[p] for p in ids[9000:])
        report['models'][model] = {'dictionary_width': z.shape[1], 'n_active_protein_features': len(active),
            'n_features_tested': len(records), 'scores': summaries, 'n_shared_split_clusters': len(shared_clusters),
            'n_features_with_constant_actual': sum(np.ptp(r['actual_values']) == 0 for r in records),
            'judge_correlation': float(pearsonr(paired[:, 0], paired[:, 1]).statistic),
            'judge_paired_n': len(paired), 'validation_actual_values_reproduced': True}
        report['models'][model]['n_features_with_constant_actual'] = int(report['models'][model]['n_features_with_constant_actual'])
        report['inputs'].append(file_identity(path, content=True))
        del z
    atomic_json(OUT/'autointerpretability_selection.json', selections)
    atomic_json(OUT/'autointerpretability_audit.json', report)
    go = json.loads((OUT/'go_audit.json').read_text())
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, model, letter in zip(axes[0], ['esm3', 'esm2'], ['A', 'B']):
        for judge, color in [('claude', '#0072B2'), ('gpt', '#D55E00')]:
            values = np.array([r[f'{judge}_pearson_r'] for r in archive[model]['per_feature'] if r.get(f'{judge}_pearson_r') is not None])
            values = np.sort(values[np.isfinite(values)])
            ax.step(values, np.arange(1,len(values)+1)/len(values), where='post', color=color,
                    label=f'{judge}: {len(values)}/{len(archive[model]["per_feature"])}; median {np.median(values):.3f}')
        ax.set(title=f'{letter}  {model.upper()} protein-metadata task', xlabel='Archived finite Pearson r', ylabel='Cumulative fraction', xlim=(-1,1))
        ax.legend(fontsize=8, loc='upper left')
    ax = axes[1, 0]
    labels, bottom = [], np.zeros(4)
    conditions = [(m, j) for m in ['esm3', 'esm2'] for j in ['claude', 'gpt']]
    for key, label, color in [('n_finite','Finite score','#0072B2'), ('n_null','Null result','#D55E00'), ('n_nonfinite_nonnull','Nonfinite correlation','#999999')]:
        counts = [report['models'][m]['scores'][j][key] for m,j in conditions]
        ax.bar(range(4), counts, bottom=bottom, label=label, color=color)
        bottom += counts
    ax.set(xticks=range(4), xticklabels=[f'{m}\n{j}' for m,j in conditions], ylabel='Tested protein-level features', title='C  Score availability is not JSON success')
    ax.set_ylim(0, 400)
    ax.legend(fontsize=8, loc='upper right')
    for model, color in [('esm3','#0072B2'), ('esm2','#D55E00')]:
        values = np.sort([r['n_terms_all_background_bh'] for r in go['models'][model]['per_feature']])
        axes[1, 1].step(values, np.arange(1,len(values)+1)/len(values), where='post', color=color,
                        label=f'{model}: {np.count_nonzero(values):,}/{len(values):,} enriched')
    axes[1, 1].set(xscale='symlog', xlabel='Enriched GO terms (zeros retained)', ylabel='Cumulative fraction', title='D  Separate complete residue-feature GO task')
    axes[1, 1].legend(fontsize=8)
    for ext in ['png','pdf']:
        fig.savefig(ROOT/f'revision/figures/autointerpretability_corrected.{ext}', dpi=220)
    plt.close(fig)
    print(json.dumps(report['models'], indent=2))


if __name__ == '__main__':
    main()
