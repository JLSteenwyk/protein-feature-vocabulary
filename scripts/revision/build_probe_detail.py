"""Validate committed predictions and plot complete probe controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, OUT, metrics
from probe_checkpoint import atomic_json, file_identity


GROUPS = [
    ('intact', 'Intact SAE'), ('binary_support', 'Binary support'),
    ('legacy_support_preserving', 'Support-preserving shuffle'),
    ('global_column_shuffle', 'Global column shuffle'),
    ('within_protein_shuffle', 'Within-protein shuffle'),
    ('protein_mean_only', 'Protein mean'),
    ('protein_composition_length', 'Composition + length'),
    ('residue_identity_plus_composition', 'Residue identity + composition')]
METRICS = ['auroc', 'macro_within_protein_auroc', 'average_precision']


def group_index(name):
    for i, (prefix, _) in enumerate(GROUPS):
        if name == prefix or name in [f'{prefix}_{seed}' for seed in [2288, 2289, 2290]]:
            return i
    raise ValueError(f'Unknown probe control: {name}')


def validate_control_set(report):
    expected = {'intact', 'binary_support', 'protein_mean_only'}
    expected.update(f'{prefix}_{seed}' for prefix in
                    ['legacy_support_preserving', 'global_column_shuffle', 'within_protein_shuffle']
                    for seed in [2288, 2289, 2290])
    models = report['models']
    if set(models) != {'esm3', 'esm2', 'composition'}:
        raise ValueError('Unexpected model set')
    for model in ['esm3', 'esm2']:
        if set(models[model]) != expected:
            raise ValueError(f'Incomplete or unexpected controls: {model}')
    if set(models['composition']) != {'protein_composition_length', 'residue_identity_plus_composition'}:
        raise ValueError('Incomplete composition controls')
    if report['completed_controls'] != 26:
        raise ValueError('Incorrect completed-control count')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=OUT)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    source = directory/'combined_probe_controls.json'
    report = json.loads(source.read_text())
    validate_control_set(report)
    rows, reference = [], None
    for model, records in report['models'].items():
        for name, record in records.items():
            identity = report['prediction_files'][model][name]
            path = ROOT/identity['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != identity['sha256']:
                raise ValueError(f'Prediction digest changed: {path}')
            with np.load(path) as prediction:
                if reference is None:
                    reference = {key: prediction[key].copy() for key in ['y', 'protein', 'cluster']}
                if any(not np.array_equal(reference[key], prediction[key]) for key in reference):
                    raise ValueError('Prediction alignment mismatch')
                computed = metrics(prediction['y'], prediction['score'], prediction['protein'])
            counts = report['sample_counts']['test']
            if computed['n_residues'] != counts['residues'] or computed['n_positive'] != counts['positive']:
                raise ValueError('Test counts disagree with cohort summary')
            for key in METRICS + ['prevalence', 'n_positive', 'n_residues', 'n_proteins_with_both_classes']:
                if not np.isclose(computed[key], record[key], atol=1e-12, rtol=1e-10):
                    raise ValueError(f'Saved metric mismatch: {model}/{name}/{key}')
            rows.append({'model': model, 'control': name,
                         **{key: record[key] for key in METRICS},
                         'auroc_ci_low': record['ci95_auroc']['low'],
                         'auroc_ci_high': record['ci95_auroc']['high'],
                         'selected_C': record['selected_C'],
                         'convergence_warning': record['convergence_warning']})
    with (directory/'probe_controls_verified.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True, constrained_layout=True)
    colors = {'esm3': '#0072B2', 'esm2': '#D55E00', 'composition': '#333333'}
    for row in rows:
        name, model = row['control'], row['model']
        seed_shift = (int(name[-4:])-2289)*.06 if name[-4:].isdigit() else 0
        y = group_index(name) + {'esm3': -.16, 'esm2': .16, 'composition': 0}[model] + seed_shift
        for ax, key in zip(axes, METRICS):
            point = row[key]
            if key == 'auroc':
                ax.plot([row['auroc_ci_low'], row['auroc_ci_high']], [y, y], color=colors[model], alpha=.5)
            ax.plot(point, y, 'o', color=colors[model], markersize=4)
    for ax, key, title in zip(axes, METRICS, ['A  Pooled AUROC', 'B  Within-protein AUROC', 'C  Average precision']):
        ax.set(title=title, xlabel=key.replace('_', ' '))
        ax.set_yticks(range(len(GROUPS)), [label for _, label in GROUPS])
        ax.grid(axis='x', alpha=.2)
        ax.axvline(.5 if key != 'average_precision' else report['sample_counts']['test']['prevalence'],
                   color='#777777', linestyle=':', linewidth=1)
    axes[0].set_ylim(len(GROUPS)-.5, -.7)
    for model, color in colors.items():
        axes[0].plot([], [], 'o', color=color, label=model.upper())
    axes[0].legend(loc='lower left', fontsize=8)
    figure_directory = ROOT/'revision/figures' if directory == OUT.resolve() else directory
    for ext in ['png', 'pdf']:
        fig.savefig(figure_directory/f'probe_controls_corrected.{ext}', dpi=220)
    plt.close(fig)
    atomic_json(directory/'probe_figure_verification.json', {
        'input': file_identity(source, content=True), 'n_controls': len(rows),
        'script': file_identity(Path(__file__), content=True),
        'prediction_hashes_and_metrics_verified': True,
        'sample_counts': report['sample_counts'],
        'n_mixed_label_test_proteins': computed['n_proteins_with_both_classes'],
        'uncertainty': 'Pooled AUROC bars: archived 1000 test-cluster percentile resamples; other panels point estimates. Repeated shuffle seeds are distinct dots, not biological replicates.',
        'scope': 'Conditional on fixed dictionaries, fitted probes, sampled residues and one cluster split; no claim of SAE-training or pretraining independence.'})
    print(f'Validated prediction hashes, alignment and metrics for {len(rows)} controls')


if __name__ == '__main__':
    main()
