"""Replace selected-subset cross-modal panels with complete-population summaries."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT, ROOT
from probe_checkpoint import atomic_json, file_identity


def category_summary(records):
    categories = ['enhanced', 'unclassified', 'suppressed']
    result = {}
    for category in categories:
        rows = [r for r in records if r['category'] == category]
        if not rows:
            raise ValueError(f'Empty category: {category}')
        rows.sort(key=lambda r: (r['paired_standardized_difference_ddof0'], r['feature_id']))
        effects = np.array([r['paired_standardized_difference_ddof0'] for r in rows])
        if not np.isfinite(effects).all():
            raise ValueError('Nonfinite effects')
        # Fixed within-category rank quantiles, not the largest or most certain effects.
        ranks = np.rint(np.linspace(.1, .9, 5)*(len(rows)-1)).astype(int)
        result[category] = {'n': len(rows), 'median_effect': float(np.median(effects)),
                            'effect_quantiles_10_25_50_75_90': np.quantile(effects, [.1, .25, .5, .75, .9]).tolist(),
                            'illustrated_records': [rows[i] for i in ranks]}
    return result


def main():
    source = OUT/'cross_modal_complete.json'
    data = json.loads(source.read_text())
    records = data['per_feature']
    if len(records) != data['dictionary_width'] or [r['feature_id'] for r in records] != list(range(len(records))):
        raise ValueError('Dictionary IDs or width inconsistent')
    summary = category_summary(records)
    if sum(r['n'] for r in summary.values()) != data['n_active']:
        raise ValueError('Active population mismatch')
    for category, record in summary.items():
        if record['n'] != data['category_counts'][category]:
            raise ValueError('Category count mismatch')
    colors = ['#D55E00', '#777777', '#0072B2']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    for index, (category, color) in enumerate(zip(summary, colors)):
        rows = [r for r in records if r['category'] == category]
        values = np.sort([r['paired_standardized_difference_ddof0'] for r in rows])
        axes[0].plot(values, np.arange(1, len(values)+1)/len(values), color=color,
                     label=f'{category} (n={len(values):,})')
        axes[1].scatter([r['mean_delta'] for r in rows],
                        [r['paired_standardized_difference_ddof0'] for r in rows],
                        s=3, alpha=.35, color=color, rasterized=True)
        for j, record in enumerate(summary[category]['illustrated_records']):
            y = index*6+j
            low, high = record['effect_cluster_percentile_range95']
            axes[2].hlines(y, low, high, color=color, lw=1.3)
            axes[2].plot(record['paired_standardized_difference_ddof0'], y, 'o', color=color, ms=3)
    axes[0].set(title='A  All active feature effects', xlabel='Paired standardized difference',
                ylabel='Within-category cumulative fraction')
    axes[0].legend(fontsize=7, loc='lower right')
    axes[1].set(title='B  Magnitude and standardized effect', xlabel='Mean activation change (S+St minus S)',
                ylabel='Paired standardized difference')
    axes[2].set(title='C  Fixed-rank feature examples', xlabel='Effect and 95% cluster-resampling range',
                yticks=[i*6+j for i in range(3) for j in range(5)],
                yticklabels=[f'{category[:3]} / {r["feature_id"]}' for category in summary
                             for r in summary[category]['illustrated_records']])
    axes[2].invert_yaxis()
    for ax in [axes[0], axes[2]]:
        for threshold in [-.2, .2]:
            ax.axvline(threshold, color='black', lw=.6, ls=':')
    for threshold in [-.2, .2]:
        axes[1].axhline(threshold, color='black', lw=.6, ls=':')
    destination = ROOT/'revision/figures'
    for extension in ['png', 'pdf']:
        fig.savefig(destination/f'cross_modal_detail.{extension}', dpi=220)
    plt.close(fig)
    atomic_json(destination/'cross_modal_detail_provenance.json', {
        'input': file_identity(source, content=True), 'categories': summary,
        'illustration_rule': 'Five nearest ranks at 10,30,50,70,90 percent within each category, ties by zero-based ID.',
        'uncertainty': data['uncertainty'], 'all_active_features_in_panels_a_b': data['n_active'],
        'not_claimed': ['Equivalence of unclassified features', 'Independent validation of threshold-defined categories',
                        'Simultaneous intervals', 'Feature independence']})
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'illustrated_records'}
                      for key, value in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
