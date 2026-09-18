"""Complete-population GO figure with explicit denominators and matching checks."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity


def validate_go_categories(model, balance):
    rows = model['per_feature']
    indexed = {r['feature_id']: r for r in rows}
    if len(indexed) != len(rows) or len(rows) != model['n_active']:
        raise ValueError('Duplicate or missing feature records')
    if sum(c['n_features'] for c in model['categories'].values()) != len(rows):
        raise ValueError('Category populations do not partition the active set')
    for category, summary in model['categories'].items():
        values = np.array([r['n_terms_all_background_bh'] for r in rows if r['category'] == category])
        if len(values) != summary['n_features'] or np.count_nonzero(values) != summary['n_enriched']:
            raise ValueError('Category denominator or enriched count mismatch')
        if not np.isclose(np.count_nonzero(values)/len(values), summary['frac_enriched']):
            raise ValueError('Enriched fraction mismatch')
        if np.median(values) != summary['median_all_features']:
            raise ValueError('Category median mismatch')
        positive = values[values > 0]
        median = float(np.median(positive)) if len(positive) else None
        if median != summary['median_enriched_only']:
            raise ValueError('Enriched-only median mismatch')
    matched = [[], []]
    used = [set(), set()]
    for pair in balance['pairs']:
        for i, category in enumerate(['enhanced', 'unclassified']):
            fid = pair[f'{category}_feature_id']
            if fid in used[i]:
                raise ValueError('Matching reuses a feature')
            used[i].add(fid)
            row = indexed[fid]
            expected = 'invariant' if category == 'unclassified' else category
            if row['category'] != expected or row['n_study_proteins'] != pair[f'{category}_study_size']:
                raise ValueError('Matching category or study size mismatch')
            matched[i].append(row['n_terms_all_background_bh'])
    if len(matched[0]) != balance['n_pairs']:
        raise ValueError('Matching denominator mismatch')
    for category, values in zip(['enhanced', 'unclassified'], matched):
        if np.median(values) != balance['outcome'][f'median_{category}']:
            raise ValueError('Matched median mismatch')
    return matched


def main():
    inputs = [OUT/'go_audit.json', OUT/'go_category_balance.json']
    go, balance = [json.loads(path.read_text()) for path in inputs]
    model = go['models']['esm3']
    matched = validate_go_categories(model, balance)
    names = ['enhanced', 'invariant', 'suppressed']
    labels = ['enhanced', 'unclassified', 'suppressed']
    colors = ['#D55E00', '#777777', '#0072B2']
    summaries = [model['categories'][name] for name in names]
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    for i, (record, color) in enumerate(zip(summaries, colors)):
        ax.bar(i, record['frac_enriched'], color=color)
        ax.vlines(i, *record['feature_bootstrap_fraction_ci95'], color='black', lw=1.5)
        ax.text(i, record['frac_enriched']+.035, f"{record['n_enriched']:,}/{record['n_features']:,}", ha='center')
    ax.set(xticks=range(3), xticklabels=labels, ylim=(0, 1.17), ylabel='Fraction with at least one enriched term',
           title='A  All sequence-active category members')
    ax = axes[0, 1]
    for offset, key, label, hatch in [(-.2, 'median_all_features', 'All members', None),
                                     (.2, 'median_enriched_only', 'Enriched members only', '//')]:
        values = [r[key] for r in summaries]
        ax.bar(np.arange(3)+offset, values, .38, color=colors, hatch=hatch,
               edgecolor='black', linewidth=.4, label=label)
        for i, value in enumerate(values):
            ax.text(i+offset, value+2, f'{value:g}', ha='center', fontsize=8)
    ax.set(xticks=range(3), xticklabels=labels, ylim=(0, 120), ylabel='Median enriched GO terms',
           title='B  Two different summary denominators')
    ax.legend(fontsize=8, loc='upper left')
    ax = axes[1, 0]
    for name, label, color in zip(names, labels, colors):
        rows = [r for r in model['per_feature'] if r['category'] == name]
        ax.scatter([r['n_study_proteins'] for r in rows], [r['n_terms_all_background_bh'] for r in rows],
                   s=3, alpha=.3, color=color, label=label, rasterized=True)
    ax.set(xscale='symlog', yscale='symlog', xlabel='GO study proteins (symlog; zeros retained)',
           ylabel='Enriched terms (symlog; zeros retained)', title='C  Study size and detected enrichment')
    ax = axes[1, 1]
    for values, label, color in zip(matched, labels[:2], colors[:2]):
        ordered = np.sort(values)
        ax.step(ordered, np.arange(1, len(ordered)+1)/len(ordered), where='post', color=color,
                label=f'{label}: median {np.median(ordered):g}')
    ax.set(xscale='symlog', xlabel='Enriched terms (symlog; zeros retained)',
           ylabel='Cumulative fraction of matched features', title=f'D  Study-size matching ({balance["n_pairs"]:,} pairs)')
    ax.legend(fontsize=8, loc='lower right')
    destination = ROOT/'revision/figures'
    for extension in ['png', 'pdf']:
        fig.savefig(destination/f'go_detail_corrected.{extension}', dpi=220)
    plt.close(fig)
    atomic_json(destination/'go_detail_provenance.json', {
        'inputs': [file_identity(path, content=True) for path in inputs],
        'historical_panel_sources': [file_identity(ROOT/path, content=True) for path in [
            'scripts/publication_figures/supplementary/ed09_decoder_dms_phylo.py',
            'scripts/scaled_1.5M/15_coactivation_analysis.py',
            'scripts/unified/run_dms_correlation.py',
            'results/scaled_1.5M/coactivation_analysis.json',
            'results/unified/esm3/dms_correlation.json', 'results/unified/esm2/dms_correlation.json']],
        'categories': dict(zip(labels, summaries)),
        'n_sequence_active': model['n_active'], 'n_inactive_in_paired_set_not_in_three_bars': 2,
        'matched_pairs_verified': len(matched[0]), 'matching_is_exploratory': True,
        'fraction_uncertainty': 'Descriptive feature bootstrap, not independent biological sampling or category selection uncertainty.',
        'old_panels_removed': {
            'GO_and_convergence': 'Truncated GO exports and shifted identifiers; replace, not reinterpret missing records.',
            'coactivation': 'Frequency-selected 2000-feature subset, Ward linkage on Jaccard dissimilarity, not k-nearest-neighbor clustering of all features; no evidence that clusters are functional modules.',
            'DMS': 'Different unified SAE checkpoints; figure displays absolute mean signed correlation, not mean absolute correlation; activation metric is baseline magnitude, not mutation-induced change. Optional exploratory benchmark omitted rather than treated as validation of principal dictionaries.'},
        'historical_inputs_unchanged': True})
    print('Validated complete GO category summaries and all matched pairs; replacement ED9 saved.')


if __name__ == '__main__':
    main()
