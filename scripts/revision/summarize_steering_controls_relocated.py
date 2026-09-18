"""Verify complete paired steering records and recompute descriptive readouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import log_softmax
from common import OUT
from probe_checkpoint import atomic_json, file_identity
from build_circuit_report import check_digest
from run_circuit_controls import select_cohort, committed, save_arrays
from steering_controls import make_directions, condition_order, STRENGTHS
from relocated_inputs import mapped_identity, parse_mappings


METRICS = ('sequence_kl_normal_to_perturbed', 'sequence_disagreement', 'target_activation_change')


def readouts(normal_logits, logits, normal_activations, activations, target_columns):
    if (normal_logits.shape != logits.shape or logits.ndim != 2
            or normal_activations.shape != activations.shape or activations.ndim != 2
            or activations.shape[0] != logits.shape[0] or not len(logits)
            or any(not np.isfinite(x).all() for x in (normal_logits, logits, normal_activations, activations))
            or (normal_activations < 0).any() or (activations < 0).any()):
        raise ValueError('Invalid aligned residue readouts')
    columns = np.asarray(target_columns)
    if (columns.ndim != 1 or len(columns) == 0 or not np.issubdtype(columns.dtype, np.integer)
            or min(columns) < 0 or max(columns) >= activations.shape[1]
            or len(set(columns.tolist())) != len(columns)):
        raise ValueError('Invalid target columns')
    p, q = [log_softmax(x.astype(np.float64), axis=-1) for x in (normal_logits, logits)]
    return np.array([(np.exp(p)*(p-q)).sum(-1).mean(),
                     np.mean(normal_logits.argmax(-1) != logits.argmax(-1)),
                     (activations[:, columns].astype(np.float64)-normal_activations[:, columns]).mean()])


def summarize(directory, resamples=1000, mappings=()):
    if resamples < 100:
        raise ValueError('Require at least 100 resamples')
    identity = json.loads((directory/'identity.json').read_text())
    identity = mapped_identity(identity, mappings, Path.cwd())
    progress = json.loads((directory/'progress.json').read_text())
    cohort, protocol = identity['cohort'], identity['conditions']
    n = len(cohort['evaluation'])
    if not progress['terminal'] or progress['complete_conditions'] != n*len(protocol):
        raise ValueError('Run is incomplete')
    for record in list(identity['inputs'].values()) + identity['code'] + [identity['weights']]:
        check_digest(record)
    loaded = {key: json.loads(Path(identity['inputs'][key]['path']).read_text())
              for key in ('sequences', 'metadata', 'clusters', 'categories', 'archived_categories')}
    sequences, clusters = loaded['sequences'], loaded['clusters']
    expected_cohort = select_cohort(sequences, loaded['metadata'], clusters,
        len(cohort['discovery']), n, identity['seed'])
    if expected_cohort != cohort or len(set(cohort['clusters'].values())) != len(cohort['clusters']):
        raise ValueError('Invalid cluster-disjoint cohort')
    torch.set_num_threads(2)
    checkpoint = torch.load(identity['inputs']['checkpoint']['path'], map_location='cpu', weights_only=False)
    target_ids = loaded['archived_categories']['enhanced_feature_ids'][:20]
    categories = {r['feature_id']: r['category'] for r in loaded['categories']['per_feature']}
    vectors, metadata = make_directions(checkpoint['model_state_dict']['decoder.weight'].numpy(), target_ids, categories)
    monitored = identity['monitored_feature_ids']
    if (metadata != identity['directions'] or [list(r) for r in condition_order(vectors)] != protocol
            or monitored != sorted({i for row in metadata.values() for i in row['feature_ids']})):
        raise ValueError('Direction/condition/feature selection mismatch')
    direction_record = committed(directory/'directions.npz')
    if direction_record is None:
        raise ValueError('Missing direction export')
    with np.load(directory/'directions.npz', allow_pickle=False) as data:
        if set(data.files) != set(vectors) or any(not np.array_equal(data[k], v) for k, v in vectors.items()):
            raise ValueError('Direction vectors changed')
    target_columns = [monitored.index(i) for i in target_ids]
    scores, prevalence, hashes = [], [], {}
    for protein in cohort['evaluation']:
        rows = []
        normal_logits = normal_z = None
        for condition in protocol:
            name, direction, alpha = condition
            path = directory/'records'/protein/f'{name}.npz'
            record = committed(path)
            if (record is None or record['protein'] != protein or record['cluster'] != clusters[protein]
                    or record['length'] != len(sequences[protein]) or record['condition'] != condition
                    or record['monitored_feature_ids'] != monitored):
                raise ValueError('Misaligned protein/condition metadata')
            hashes[f'{protein}/{name}'] = record['sha256']
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {'sequence_logits', 'monitored_activations'}:
                    raise ValueError('Unexpected readout inventory')
                logits, z = data['sequence_logits'], data['monitored_activations']
            if (logits.shape != (len(sequences[protein]), 64)
                    or z.shape != (len(sequences[protein]), len(monitored))):
                raise ValueError('Readout dimensions differ from protocol')
            if name == 'normal':
                normal_logits, normal_z = logits, z
                prevalence.append((z > 0).mean(0))
            if alpha == 0 and (not np.array_equal(logits, normal_logits) or not np.array_equal(z, normal_z)):
                raise ValueError('No-op/restoration differs from normal')
            rows.append(readouts(normal_logits, logits, normal_z, z, target_columns))
        scores.append(rows)
    scores = np.asarray(scores)
    rng = np.random.default_rng(identity['seed'])
    counts = rng.multinomial(n, np.full(n, 1/n), size=resamples)/n

    def estimates(values):
        mean = values.mean(0)
        lo, hi = np.quantile(counts @ values, [.025, .975], axis=0)
        return {key: {'mean': float(mean[i]), 'conditional_95_percentile_interval': [float(lo[i]), float(hi[i])]}
                for i, key in enumerate(METRICS)}

    names = [r[0] for r in protocol]
    report = {'scope': 'Human-only model-output sensitivity; no accuracy, biological function or independent validation of target selection.',
              'identity': file_identity(directory/'identity.json', content=True),
              'verifier': file_identity(Path(__file__), content=True),
              'n_proteins': n, 'n_evaluation_clusters': n, 'raw_hashes': hashes,
              'bootstrap_resamples': resamples, 'bootstrap_seed': identity['seed'],
              'uncertainty': 'Fixed model, SAE, target and sampled control vectors; cluster resampling of proteins. Control seeds are averaged within proteins, not treated as independent biological replicates.',
              'conditions': {name: estimates(scores[:, i]) for i, name in enumerate(names)},
              'paired_target_minus_control_mean': {}}
    for kind in ('random', 'orthogonal', 'decoder'):
        report['paired_target_minus_control_mean'][kind] = {}
        for alpha in STRENGTHS:
            target = names.index(f'target_alpha_{alpha:g}')
            controls = [names.index(f'{direction}_alpha_{alpha:g}') for direction, row in metadata.items() if row['kind'] == kind]
            difference = scores[:, target]-scores[:, controls].mean(1)
            report['paired_target_minus_control_mean'][kind][f'{alpha:g}'] = estimates(difference)
    prevalence = np.asarray(prevalence)
    report['baseline_residue_prevalence'] = {
        name: float(prevalence[:, [monitored.index(i) for i in row['feature_ids']]].mean())
        for name, row in metadata.items() if row['feature_ids']}
    report['direction_cosines'] = {a: {b: float(np.dot(x, y)) for b, y in vectors.items()} for a, x in vectors.items()}
    target = directory/'paired_readouts.npz'
    save_arrays(target, {'scores': scores, 'baseline_prevalence': prevalence},
        {'protein_ids': cohort['evaluation'], 'condition_names': names,
         'metric_names': METRICS, 'monitored_feature_ids': monitored})
    report['scores_export'] = file_identity(target, content=True)
    report['failure_history'] = [str(p.relative_to(directory)) for p in sorted(directory.glob('records/*/*.failure.json'))]
    report['verified'] = True
    if mappings:
        report['path_relocations'] = [[str(old), str(new)] for old, new in mappings]
    atomic_json(directory/'verified_summary.json', report)
    print(f'Verified {n} proteins, {len(protocol)} conditions, {resamples} cluster resamples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=OUT/'steering_controls')
    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('--map', action='append', default=[], metavar='OLD=NEW',
                        help='Explicit absolute prefix mapping; all identities retain hash checks')
    args = parser.parse_args()
    summarize(args.run_dir, args.resamples, parse_mappings(args.map))
