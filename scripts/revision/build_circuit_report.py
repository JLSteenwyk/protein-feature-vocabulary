"""Export comparable protocols, not cross-dictionary effect magnitudes, for circuits."""
import csv
import json
from pathlib import Path
import numpy as np
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity
from capture_reproducibility import stable_digest


CONTRASTS = ('reconstruction_shift', 'unmatched_removal', 'matched_removal',
             'residual_preserving_removal')
PAIRS = {'esm2': {'16_24'}, 'esm3': {'16_33', '33_42'}}


def check_digest(record):
    observed = stable_digest(record['path'])
    if any(observed[key] != record[key] for key in ('bytes', 'sha256')):
        raise ValueError(f'Changed verified artifact: {record["path"]}')


def validate_protocol(summary, identity, model):
    if (not summary.get('verified') or summary['n_discovery_clusters'] != 50
            or summary['n_evaluation_clusters'] != 100
            or summary['bootstrap_resamples'] != 1000
            or identity['model'] != model or set(summary['pairs']) != PAIRS[model]
            or identity['upstream_count'] != 100 or identity['downstream_count'] != 50):
        raise ValueError('Expected complete full-cohort circuit protocol')
    for record in summary['pairs'].values():
        if record['tested_edges'] != 5000 or record['decomposition_max_abs_error'] > 1e-9:
            raise ValueError('Invalid tested family or effect decomposition')
        if set(record['readout_summary']) != set(CONTRASTS):
            raise ValueError('Incomplete contrasts')
        for value in record['readout_summary'].values():
            estimate = value['mean_absolute_change']
            interval = value['conditional_95_percentile_interval']
            if (len(interval) != 2 or not np.isfinite([estimate, *interval]).all()
                    or min(estimate, *interval) < 0 or interval[0] > interval[1]):
                raise ValueError('Invalid estimate or interval')


def main():
    sources, rows, cohort = [], [], None
    for model in ('esm2', 'esm3'):
        directory = OUT/'circuit_controls'/model
        source = directory/'verified_summary.json'
        summary = json.loads(source.read_text())
        for key in ('identity', 'selection', 'verifier'):
            check_digest(summary[key])
        identity = json.loads(Path(summary['identity']['path']).read_text())
        validate_protocol(summary, identity, model)
        if cohort is not None and identity['cohort'] != cohort:
            raise ValueError('Models must use identical discovery/evaluation cohorts')
        cohort = identity['cohort']
        for record in identity['inputs'] + identity['code']:
            check_digest(record)
        for pair, record in summary['pairs'].items():
            check_digest(record['effects'])
            if set(record['raw_hashes']) != set(cohort['evaluation']):
                raise ValueError('Missing evaluation records')
            for protein, digest in record['raw_hashes'].items():
                if stable_digest(directory/pair/f'{protein}.npz')['sha256'] != digest:
                    raise ValueError('Changed raw circuit record')
            for contrast in CONTRASTS:
                value = record['readout_summary'][contrast]
                lo, hi = value['conditional_95_percentile_interval']
                rows.append({'model': model, 'layer_pair': pair, 'contrast': contrast,
                             'mean_absolute_change': value['mean_absolute_change'],
                             'conditional_95_low': lo, 'conditional_95_high': hi,
                             'evaluation_clusters': 100, 'tested_edges': 5000})
        sources.append(file_identity(source, content=True))
    target = OUT/'circuit_controls/combined_summary.csv'
    with target.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ROOT/'revision/circuit_results.md'
    lines = ['# Corrected Circuit-Control Results', '',
             'Both models use the same 50 discovery and 100 evaluation clusters,',
             'one human protein per cluster. Features are selected on discovery only.',
             'Every evaluation protein receives all 203 conditions per layer pair.',
             'All 5,000 tested pairs per layer pair are exported, including zero effects.', '',
             'The statistic averages absolute changes in residue-mean downstream SAE',
             'activation over tested pairs within proteins, then over proteins.',
             'Reconstruction alone averages only over the 50 downstream features.',
             'Intervals use 1,000 cluster resamples, conditional on fixed SAEs and',
             'discovery selection. Units differ across dictionaries: do not rank models',
             'or layer pairs by these raw magnitudes. No edge significance or biological',
             'function is established by these descriptive intervals.', '',
             '| Model | Layers | Contrast | Mean absolute change | Conditional 95% interval |',
             '| --- | --- | --- | ---: | --- |']
    for row in rows:
        lines.append(f'| {row["model"].upper()} | {row["layer_pair"].replace("_", "->")} | '
                     f'{row["contrast"].replace("_", " ")} | {row["mean_absolute_change"]:.3f} | '
                     f'{row["conditional_95_low"]:.3f}, {row["conditional_95_high"]:.3f} |')
    lines += ['', 'Unmatched removal compares reconstruction-with-removal to the original',
              'model. Matched removal uses reconstruction-without-removal as baseline;',
              'residual-preserving removal subtracts only the decoded feature contribution.',
              'Signed protein-level effects decompose exactly into reconstruction shift',
              'plus matched removal; absolute averages above need not add. Empty',
              'residual-preserving interventions leave saved readouts exactly unchanged.',
              'These results do not recover the historical 235/113/224 selected-edge graphs.',
              '', 'Full-precision values: `analyses/circuit_controls/combined_summary.csv`.',
              'Per-edge effects, intervals, prevalences and raw records remain under',
              '`analyses/circuit_controls/{esm2,esm3}/`. See `circuit_control_protocol.md`',
              'for sampling, intervention positions and verification details.', '']
    report.write_text('\n'.join(lines))
    atomic_json(OUT/'circuit_controls/combined_report_manifest.json', {
        'sources': sources, 'generator': file_identity(Path(__file__), content=True),
        'outputs': [file_identity(p, content=True) for p in (target, report)],
        'scope': 'Verified full-cohort descriptive summaries; no cross-dictionary magnitude inference'})
    print(f'Exported {len(rows)} contrasts for three layer pairs')


if __name__ == '__main__':
    main()
