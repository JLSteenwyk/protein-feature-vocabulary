"""Plot full-strength paired steering controls and export every descriptive contrast."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT
from probe_checkpoint import file_identity, atomic_json
from build_circuit_report import check_digest
from steering_controls import STRENGTHS, condition_order
from summarize_steering_controls import METRICS


def validate_summary(summary, identity, scores):
    expected = [list(row) for row in condition_order(identity['directions'])]
    if (not summary.get('verified') or summary['n_proteins'] != 100
            or summary['n_evaluation_clusters'] != 100 or summary['bootstrap_resamples'] != 1000
            or len(identity['directions']) != 16 or identity['conditions'] != expected
            or len(expected) != 146 or scores.shape != (100, 146, 3)
            or not np.isfinite(scores).all()):
        raise ValueError('Expected verified full-cohort steering records')
    names = [row[0] for row in expected]
    if set(summary['conditions']) != set(names):
        raise ValueError('Incomplete condition summary')
    for i, (name, direction, alpha) in enumerate(expected):
        if alpha == 0 and np.any(scores[:, i] != 0):
            raise ValueError('Nonzero no-op readouts')
        for j, metric in enumerate(METRICS):
            if not np.isclose(scores[:, i, j].mean(), summary['conditions'][name][metric]['mean'], rtol=1e-12, atol=1e-14):
                raise ValueError('Summary/score mismatch')
    return names


def main():
    directory = OUT/'steering_controls'
    source = directory/'verified_summary.json'
    summary = json.loads(source.read_text())
    for key in ('identity', 'verifier', 'scores_export'):
        check_digest(summary[key])
    identity = json.loads(Path(summary['identity']['path']).read_text())
    with np.load(summary['scores_export']['path'], allow_pickle=False) as data:
        scores = data['scores']
    names = validate_summary(summary, identity, scores)
    counts = np.random.default_rng(summary['bootstrap_seed']).multinomial(100, np.full(100, .01), size=1000)/100
    colors = {'target': '#202020', 'random': '#0072B2', 'orthogonal': '#009E73', 'decoder': '#D55E00'}
    groups, records = {}, []

    def append(kind, series, alpha, metrics):
        for metric, value in metrics.items():
            lo, hi = value['conditional_95_percentile_interval']
            records.append({'record_type': kind, 'series': series, 'strength': alpha,
                            'metric': metric, 'mean': value['mean'], 'conditional_95_low': lo,
                            'conditional_95_high': hi})

    for name, direction, alpha in identity['conditions']:
        append('individual_direction', name, alpha, summary['conditions'][name])
    for kind in colors:
        groups[kind] = {}
        for alpha in STRENGTHS:
            selected = [names.index(f'{name}_alpha_{alpha:g}') for name, metadata in identity['directions'].items()
                        if metadata['kind'] == kind]
            if len(selected) != (1 if kind == 'target' else 5):
                raise ValueError('Unexpected number of direction seeds')
            values = scores[:, selected].mean(1)
            means = values.mean(0)
            lo, hi = np.quantile(counts @ values, [.025, .975], axis=0)
            metrics = {metric: {'mean': float(means[j]),
                               'conditional_95_percentile_interval': [float(lo[j]), float(hi[j])]}
                       for j, metric in enumerate(METRICS)}
            groups[kind][f'{alpha:g}'] = metrics
            append('within_protein_seed_mean', kind, alpha, metrics)
    for kind, strengths in summary['paired_target_minus_control_mean'].items():
        for alpha, metrics in strengths.items():
            append('paired_target_minus_seed_mean', kind, float(alpha), metrics)
    csv_path = directory/'all_strength_results.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    x = np.asarray(STRENGTHS)

    def curve(ax, strengths, metric, scale, color, label):
        rows = [strengths[f'{a:g}'][metric] for a in x]
        y = np.array([r['mean'] for r in rows])*scale
        interval = np.array([r['conditional_95_percentile_interval'] for r in rows])*scale
        ax.plot(x, y, 'o-', color=color, label=label, markersize=3, linewidth=1.2)
        ax.fill_between(x, interval[:, 0], interval[:, 1], color=color, alpha=.12, linewidth=0)
        ax.axhline(0, color='.6', linewidth=.5)
        ax.set_xlabel('Perturbation strength')
        ax.grid(alpha=.15)

    upper = [('target_activation_change', 1, 'A  Target SAE activation', 'Mean activation change'),
             ('sequence_kl_normal_to_perturbed', 1, 'B  Output distribution', 'Mean residue KL'),
             ('sequence_disagreement', 100, 'C  Prediction disagreement', 'Changed residues (%)')]
    for ax, (metric, scale, title, ylabel) in zip(axes[0], upper):
        for kind, color in colors.items():
            curve(ax, groups[kind], metric, scale, color, kind.title())
        ax.set(title=title, ylabel=ylabel)
    axes[0, 0].legend(fontsize=8)
    for ax, metric, scale, title, ylabel in [
            (axes[1, 0], 'sequence_kl_normal_to_perturbed', 1, 'D  Paired output contrasts', 'Target - control mean KL'),
            (axes[1, 1], 'sequence_disagreement', 100, 'E  Paired disagreement contrasts', 'Target - control (percentage points)')]:
        for kind in ('random', 'orthogonal', 'decoder'):
            curve(ax, summary['paired_target_minus_control_mean'][kind], metric, scale, colors[kind], kind.title())
        ax.set(title=title, ylabel=ylabel)
    prevalence = summary['baseline_residue_prevalence']
    labels = list(prevalence)
    axes[1, 2].bar(range(len(labels)), [prevalence[k]*100 for k in labels],
                   color=[colors['target']]+[colors['decoder']]*5)
    axes[1, 2].set(xticks=range(len(labels)), xticklabels=['Target']+[f'D{i+1}' for i in range(5)],
                   title='F  Baseline activation mismatch', ylabel='Mean positive residues (%)')
    axes[1, 2].text(.98, .97, 'Decoder sets are not\nprevalence matched', transform=axes[1, 2].transAxes,
                    ha='right', va='top', fontsize=8)
    outputs = [csv_path]
    for extension in ('png', 'pdf'):
        path = ROOT/'revision/figures'/f'steering_paired_corrected.{extension}'
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    report = ROOT/'revision/steering_results.md'
    lines = ['# Paired Steering Results', '',
        'Verified 100 human proteins in distinct detected clusters, 146 conditions each.',
        'All zero-strength and unhooked restoration readouts equal normal inference.',
        'No failures are recorded. Three control types use five fixed seeds each;',
        'control seeds are averaged within proteins before cluster resampling.', '',
        '![Paired steering controls](figures/steering_paired_corrected.png)', '',
        '**Alt text:** Target and control perturbations change model outputs. Paired',
        'differences vary with strength and control type; some intervals cross zero.',
        'Target features activate much more frequently than sampled decoder controls.', '',
        '## All-Strength Paired KL Contrasts', '',
        '| Strength | Control type | Target minus seed-mean control | Conditional 95% interval |',
        '| ---: | --- | ---: | --- |']
    for alpha in STRENGTHS:
        for kind in ('random', 'orthogonal', 'decoder'):
            r = summary['paired_target_minus_control_mean'][kind][f'{alpha:g}']['sequence_kl_normal_to_perturbed']
            lo, hi = r['conditional_95_percentile_interval']
            lines.append(f'| {alpha:g} | {kind} | {r["mean"]:.8f} | {lo:.8f}, {hi:.8f} |')
    lines += ['', '## Interpretation Limits', '',
        'The target is the archived selected direction, not independently validated.',
        f'Mean baseline residue-positive prevalence is {prevalence["target"]*100:.3f}% for targets versus',
        f'{min(v for k, v in prevalence.items() if k != "target")*100:.4f}--{max(v for k, v in prevalence.items() if k != "target")*100:.4f}% for the five decoder-control sets. Equal norm and corrected',
        'category exclusion do not match activation prevalence or dictionary geometry.',
        'The negative endpoint target-minus-decoder KL interval includes zero.',
        'All intervals use 1,000 cluster resamples conditional on fixed models, SAEs',
        'and sampled directions. They are descriptive, not simultaneous tests or',
        'uncertainty over training and direction selection. Neither KL nor disagreement',
        'measures biological function or accuracy against ground truth.', '',
        'Every condition/metric, control-group mean and paired contrast is exported',
        'at full precision in `analyses/steering_controls/all_strength_results.csv`.',
        'Raw residue records, per-protein scores and direction cosines remain in',
        '`analyses/steering_controls/`. Historical cohorts and outputs are unchanged.', '']
    report.write_text('\n'.join(lines))
    outputs.append(report)
    atomic_json(directory/'figure_manifest.json', {'source': file_identity(source, content=True),
        'scores': summary['scores_export'], 'generator': file_identity(Path(__file__), content=True),
        'rows': len(records), 'outputs': [file_identity(p, content=True) for p in outputs],
        'group_means': groups, 'scope': 'Complete strengths, conditional descriptive uncertainty; not biological specificity.'})
    print(f'Exported {len(records)} rows and six-panel paired figure')


if __name__ == '__main__':
    main()
