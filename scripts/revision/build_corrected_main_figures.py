"""Main-figure replacements using validated revision outputs only."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity


def main():
    destination = ROOT/'revision/figures'
    destination.mkdir(exist_ok=True)
    inputs = {name: OUT/name for name in ['cross_modal_complete.json', 'go_audit.json',
                                         'go_category_balance.json', 'archived_intervention_uncertainty.json']}
    cross, go, balance, archived = [json.loads(p.read_text()) for p in inputs.values()]
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 220})
    categories = ['enhanced', 'unclassified', 'suppressed']
    colors = ['#D55E00', '#777777', '#0072B2']
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    counts = [cross['category_counts'][c] for c in categories]
    assert sum(counts) == cross['n_active']
    ax = axes[0, 0]
    ax.bar(categories, counts, color=colors)
    for i, n in enumerate(counts):
        ax.text(i, n+160, f'{n:,}\n({n/cross["n_active"]:.1%})', ha='center', fontsize=9)
    ax.set(ylabel='Active paired-set coordinates', ylim=(0, 9200), title='A  Paired modality categories')
    ax = axes[0, 1]
    active_effects = [r['paired_standardized_difference_ddof0'] for r in cross['per_feature'] if r['active']]
    bins = np.linspace(min(-.3, min(active_effects))-.02, max(.3, max(active_effects))+.02, 120)
    clipped = {}
    for category, color in zip(categories, colors):
        values = np.array([r['paired_standardized_difference_ddof0'] for r in cross['per_feature']
                           if r['category'] == category])
        clipped[category] = int(np.sum((values < bins[0]) | (values > bins[-1])))
        ax.hist(values, bins=bins, histtype='step', linewidth=1.5, color=color, label=category)
    if any(clipped.values()):
        raise ValueError('Histogram would omit effects; expand bins instead of silently clipping.')
    ax.axvline(-.2, color='black', ls=':', lw=.8); ax.axvline(.2, color='black', ls=':', lw=.8)
    ax.set(yscale='log', xlabel='Mean paired difference / population SD', ylabel='Coordinates (log scale)',
           title='B  Paired effects across 11,704 proteins')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    cat_go = go['models']['esm3']['categories']
    keys = ['enhanced', 'invariant', 'suppressed']
    medians = [cat_go[k]['median_all_features'] for k in keys]
    ax.bar(categories, medians, color=colors)
    for i, (value, key) in enumerate(zip(medians, keys)):
        ax.text(i, value+2, f'{value:g}\nn={cat_go[key]["n_features"]:,}', ha='center', fontsize=8)
    ax.set(ylabel='Median enriched GO terms', ylim=(0, 110), title='C  Sequence-active category members')
    ax = axes[1, 1]
    matched = balance['outcome']
    values = [matched['median_enhanced'], matched['median_unclassified']]
    ax.bar(categories[:2], values, color=colors[:2])
    for i, value in enumerate(values):
        ax.text(i, value+1, f'{value:g}', ha='center')
    ax.set(ylabel='Median enriched GO terms', ylim=(0, 40),
           title=f'D  Study-size matched ({balance["n_pairs"]:,} pairs)')
    for extension in ['png', 'pdf']:
        fig.savefig(destination/f'cross_modal_corrected.{extension}')
    plt.close(fig)

    steering = archived['steering']
    subset = steering['archived_subset']
    keys = sorted(subset, key=lambda k: float(k.removeprefix('alpha_')))
    strengths = np.array([float(k.removeprefix('alpha_')) for k in keys])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for ax, metric, full_key, title, ylabel in zip(
            axes, ['target_act_change', 'mean_kl', 'token_change_rate'],
            ['mean_target_act_change', 'mean_kl', 'mean_token_change_rate'],
            ['A  Target activation response', 'B  Output distribution change', 'C  Token-prediction disagreement'],
            ['Mean target activation change', 'Mean KL divergence', 'Fraction changed from baseline']):
        records = [subset[k]['metrics'][metric] for k in keys]
        means = np.array([r['subset_mean'] for r in records])
        bounds = np.array([r['cluster_percentile_ci95'] for r in records])
        whole = [steering['full_aggregate_summary'][k][full_key] for k in keys]
        ax.plot(strengths, whole, 's--', color='#777777', ms=4, label='500-protein aggregate')
        # Explicit interval segments do not assume the point estimate lies inside every percentile range.
        ax.vlines(strengths, bounds[:, 0], bounds[:, 1], color='#0072B2', lw=1.5)
        ax.plot(strengths, means, 'o-', color='#0072B2', ms=4, label='100 retained records + 95% range')
        ax.set(xlabel='Perturbation scale (archived settings)', ylabel=ylabel, title=title)
        if metric == 'mean_kl':
            ax.set_yscale('log')
        if metric == 'target_act_change':
            ax.axhline(0, color='black', ls=':', lw=.6)
    axes[0].legend(fontsize=7, loc='upper left')
    for extension in ['png', 'pdf']:
        fig.savefig(destination/f'steering_archived_corrected.{extension}')
    plt.close(fig)
    atomic_json(destination/'corrected_main_figure_provenance.json', {
        'inputs': {name: file_identity(path, content=True) for name, path in inputs.items()},
        'fig3_active_counts': dict(zip(categories, counts)), 'fig3_effects_outside_plot': clipped,
        'fig3_GO_scope': 'Sequence-active category members, including zeros; different population from paired union.',
        'fig4_subset_n': steering['archived_n'], 'fig4_aggregate_n': steering['reported_full_n'],
        'fig4_uncertainty': steering['scope'],
        'withdrawn_panels': ['Invalid SS3 biological readout', 'Reconstruction-confounded circuit edge counts'],
        'historical_inputs_unchanged': True})


if __name__ == '__main__':
    main()
