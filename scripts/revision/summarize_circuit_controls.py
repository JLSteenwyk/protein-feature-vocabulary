"""Verify complete circuit-control records and export paired descriptive intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import OUT
from probe_checkpoint import atomic_json, file_identity
from capture_reproducibility import stable_digest
from run_circuit_controls import committed, select_cohort, conditions, rank_features


def paired_effects(means):
    """Input: proteins x (normal,noop,reconstruction,paired removals) x features."""
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 3 or means.shape[1] < 5 or (means.shape[1]-3) % 2 or not np.isfinite(means).all():
        raise ValueError('Invalid complete paired measurement array')
    if not np.array_equal(means[:, 0], means[:, 1]):
        raise ValueError('No-op mismatch')
    return {'reconstruction_shift': means[:, 0]-means[:, 2],
            'unmatched_removal': means[:, 0, None]-means[:, 3::2],
            'matched_removal': means[:, 2, None]-means[:, 3::2],
            'residual_preserving_removal': means[:, 0, None]-means[:, 4::2]}


def validate_readout(means, fractions, shape, length):
    means, fractions = np.asarray(means), np.asarray(fractions)
    if means.shape != shape or fractions.shape != shape:
        raise ValueError('Readout shape mismatch')
    if not np.isfinite(means).all() or (means < 0).any():
        raise ValueError('Invalid nonnegative SAE mean')
    if not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError('Invalid residue prevalence')
    if length < 1 or np.max(np.abs(fractions.astype(np.float64)*length-np.rint(fractions.astype(np.float64)*length))) > 5e-5:
        raise ValueError('Prevalence incompatible with residue count')
    if np.any((means == 0) != (fractions == 0)):
        raise ValueError('Mean/prevalence zero-support mismatch')


def validate_metadata(record, protein, clusters, sequences, pair=None, protocol=None):
    if record is None or record.get('protein') != protein or record.get('cluster') != clusters[protein] or record.get('length') != len(sequences[protein]):
        raise ValueError('Missing or misaligned protein metadata')
    if pair is not None and (record.get('pair') != pair or record.get('conditions') != protocol):
        raise ValueError('Misaligned layer-pair or condition metadata')


def summarize(directory, resamples=1000):
    if resamples < 100:
        raise ValueError('Require at least 100 resamples')
    identity = json.loads((directory/'identity.json').read_text())
    selection = json.loads((directory/'selection.json').read_text())
    progress = json.loads((directory/'progress.json').read_text())
    cohort = identity['cohort']
    expected = len(identity['pairs'])*len(cohort['evaluation'])
    if not progress['terminal'] or progress['complete_pair_proteins'] != expected:
        raise ValueError('Run is incomplete')
    for record in identity['inputs'] + identity['code'] + list(identity['checkpoints'].values()) + identity['weights']:
        current = stable_digest(record['path'])
        if current['sha256'] != record['sha256'] or current['bytes'] != record['bytes']:
            raise ValueError(f'Changed input: {record["path"]}')
    sequences, metadata, clusters = [json.loads(Path(r['path']).read_text()) for r in identity['inputs']]
    reproduced = select_cohort(sequences, metadata, clusters, len(cohort['discovery']), len(cohort['evaluation']), identity['seed'])
    if reproduced != cohort:
        raise ValueError('Cohort sampling mismatch')
    if len(set(cohort['clusters'].values())) != len(cohort['clusters']):
        raise ValueError('Expected one protein per distinct cluster across both partitions')
    frequencies = {}
    for layer in sorted({layer for pair in identity['pairs'] for layer in pair}):
        rows = []
        for p in cohort['discovery']:
            path = directory/'discovery'/f'{p}.npz'
            record = committed(path)
            validate_metadata(record, p, clusters, sequences)
            if record is None or record['sha256'] != selection['discovery_hashes'][p]:
                raise ValueError('Discovery digest mismatch')
            with np.load(path, allow_pickle=False) as data:
                values = data[f'L{layer}_positive_fraction']
                if values.ndim != 1:
                    raise ValueError('Invalid discovery dictionary shape')
                validate_readout(data[f'L{layer}_mean'], values, values.shape, len(sequences[p]))
                rows.append(values)
        frequencies[layer] = np.mean(rows, axis=0, dtype=np.float64)
    n = len(cohort['evaluation'])
    rng = np.random.default_rng(identity['seed'])
    # Each evaluation protein belongs to its own detected cluster by design.
    counts = rng.multinomial(n, np.full(n, 1/n), size=resamples)/n
    report = {'scope': 'Fixed-SAE, discovery-selected, human-only descriptive paired contrasts. Per-edge percentile intervals are not simultaneous, multiplicity-adjusted edge tests or biological validation.',
              'identity': file_identity(directory/'identity.json', content=True),
              'selection': file_identity(directory/'selection.json', content=True),
              'verifier': file_identity(Path(__file__), content=True),
              'n_discovery_clusters': len(cohort['discovery']), 'n_evaluation_clusters': n,
              'bootstrap_seed': identity['seed'], 'bootstrap_resamples': resamples,
              'units': 'Raw SAE activation, not commensurate across layers/dictionaries/models',
              'pairs': {}}
    for up, down in identity['pairs']:
        pair = f'{up}_{down}'
        chosen = selection['pairs'][pair]
        if chosen != {'upstream': rank_features(frequencies[up], identity['upstream_count']),
                       'downstream': rank_features(frequencies[down], identity['downstream_count'])}:
            raise ValueError('Feature selection mismatch')
        protocol = [row[0] for row in conditions(chosen['upstream'])]
        rows, hashes = [], {}
        up_freq, down_freq = [], []
        for p in cohort['evaluation']:
            path = directory/pair/f'{p}.npz'
            record = committed(path)
            validate_metadata(record, p, clusters, sequences, pair, protocol)
            hashes[p] = record['sha256']
            with np.load(path, allow_pickle=False) as data:
                means = data['downstream_mean']
                fractions = data['downstream_positive_fraction']
                validate_readout(means, fractions, (len(protocol), len(chosen['downstream'])), len(sequences[p]))
                validate_readout(data['upstream_mean'], data['upstream_positive_fraction'],
                                 (len(chosen['upstream']),), len(sequences[p]))
                if not np.array_equal(fractions[0], fractions[1]):
                    raise ValueError('No-op prevalence mismatch')
                rows.append(means)
                up_freq.append(data['upstream_positive_fraction'])
                down_freq.append(fractions[0])
        effects = paired_effects(np.stack(rows))
        arrays = {'upstream_ids': np.array(chosen['upstream']), 'downstream_ids': np.array(chosen['downstream']),
                  'upstream_positive_protein_fraction': (np.asarray(up_freq) > 0).mean(0),
                  'downstream_positive_protein_fraction': (np.asarray(down_freq) > 0).mean(0)}
        scalar = {}
        for name, effect in effects.items():
            means = effect.mean(0)
            sampled = (counts @ effect.reshape(n, -1)).reshape((resamples,)+means.shape)
            arrays[name+'_mean'] = means
            arrays[name+'_low'], arrays[name+'_high'] = np.quantile(sampled, [.025, .975], axis=0)
            per_protein_abs = np.abs(effect).reshape(n, -1).mean(1)
            scalar[name] = {'mean_absolute_change': float(per_protein_abs.mean()),
                            'conditional_95_percentile_interval': np.quantile(counts @ per_protein_abs, [.025, .975]).tolist()}
        decomposition_error = np.max(np.abs(effects['unmatched_removal'] - effects['matched_removal'] - effects['reconstruction_shift'][:, None]))
        if decomposition_error > 1e-9:
            raise ValueError('Paired reconstruction decomposition failed')
        target = directory/f'{pair}_paired_effects.npz'
        with target.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        target.with_suffix('.tmp').replace(target)
        report['pairs'][pair] = {'tested_edges': len(chosen['upstream'])*len(chosen['downstream']),
                                 'raw_hashes': hashes, 'readout_summary': scalar,
                                 'decomposition_max_abs_error': float(decomposition_error),
                                 'effects': file_identity(target, content=True)}
    report['verified'] = True
    atomic_json(directory/'verified_summary.json', report)
    print(f'Verified {len(report["pairs"])} pairs, {n} evaluation clusters, {resamples} resamples', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=1000)
    args = parser.parse_args()
    summarize(args.run_dir, args.resamples)
