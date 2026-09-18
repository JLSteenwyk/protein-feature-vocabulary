"""Verify all raw head-evaluation logits and summarize paired cluster uncertainty."""
import json
import numpy as np
from common import OUT
from probe_checkpoint import atomic_json, file_identity
from capture_reproducibility import stable_digest
from run_corrected_head_evaluation import prediction_changes

METRICS = ['ss3_disagreement', 'sequence_disagreement', 'mean_sequence_kl_normal_to_perturbed']


def cluster_mean_interval(values, clusters, seed=2288, n=1000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(clusters) or not np.isfinite(values).all():
        raise ValueError('Invalid paired protein values')
    names, index = np.unique(clusters, return_inverse=True)
    if not len(names):
        raise ValueError('Empty cohort')
    sums = np.bincount(index, weights=values)
    counts = np.bincount(index)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(n):
        weights = rng.multinomial(len(names), np.full(len(names), 1/len(names)))
        bootstrap.append(float(weights@sums/(weights@counts)))
    return {'mean': float(values.mean()), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'n_proteins': len(values), 'n_clusters': len(names), 'n_resamples': n,
            'unit': 'sequence_cluster', 'estimand': 'unweighted protein mean'}


def main():
    directory = OUT/'corrected_head_evaluation'
    progress = json.loads((directory/'progress.json').read_text())
    identity = json.loads((directory/'identity.json').read_text())
    if not progress.get('terminal') or progress['processed'] != progress['selected']:
        raise ValueError('Inference is not complete; do not summarize partial results')
    if set(progress['records']) != set(identity['selected_ids']):
        raise ValueError('Selected and processed IDs differ')
    names = [c[0] for c in identity['conditions']]
    prefixes = [f'{m}__{c}' for m in ['S', 'S_St'] for c in names]
    records = [progress['records'][p] for p in identity['selected_ids'] if progress['records'][p]['status'] == 'complete']
    for record in records:
        path = directory/f'{record["accession"]}.npz'
        if stable_digest(path)['sha256'] != record['raw_sha256']:
            raise ValueError('Raw-logit digest mismatch')
        if set(record['conditions']) != set(prefixes):
            raise ValueError('Missing intervention condition')
        with np.load(path) as arrays:
            for modality in ['S', 'S_St']:
                seq = arrays[f'{modality}__normal__sequence_logits']
                ss = arrays[f'{modality}__normal__ss8_logits']
                if len(seq) != record['length'] or ss.shape != (record['length'], 11):
                    raise ValueError('Residue count mismatch')
                for name in names:
                    prefix = f'{modality}__{name}'
                    metrics = prediction_changes(seq, arrays[prefix+'__sequence_logits'], ss, arrays[prefix+'__ss8_logits'])
                    if any(not np.isclose(metrics[k], record['conditions'][prefix][k], atol=1e-12, rtol=1e-10) for k in METRICS):
                        raise ValueError('Metrics do not reproduce from logits')
        print(f'Verified {record["accession"]}', flush=True)
    clusters = [r['cluster'] for r in records]
    result = {'selected': progress['selected'], 'successful': len(records),
              'failures': [r for r in progress['records'].values() if r['status'] != 'complete'],
              'scope': 'New human-only cohort, fixed model and sampled heads. Prediction changes, not ground-truth accuracy or functional damage. Cluster intervals condition on cohort and controls; no multiple-comparison correction.',
              'inputs': [file_identity(directory/p, content=True) for p in ['identity.json', 'progress.json']],
              'raw_logits_verified': True, 'conditions': {}, 'paired_contrasts': {}}
    values = {prefix: {metric: np.array([r['conditions'][prefix][metric] for r in records]) for metric in METRICS} for prefix in prefixes}
    for prefix in prefixes:
        result['conditions'][prefix] = {metric: cluster_mean_interval(v, clusters) for metric, v in values[prefix].items()}
    for modality in ['S', 'S_St']:
        for metric in METRICS:
            target = values[f'{modality}__target_L0H7'][metric]
            random = np.mean([values[f'{modality}__{name}'][metric] for name in names if name.startswith('random_')], axis=0)
            same_layer = values[f'{modality}__same_layer_L0H0'][metric]
            for control, v in [('mean_ten_random_heads', random), ('same_layer_L0H0', same_layer)]:
                key = f'{modality}__target_minus_{control}__{metric}'
                result['paired_contrasts'][key] = cluster_mean_interval(target-v, clusters)
    for metric in METRICS:
        delta = values['S_St__target_L0H7'][metric]-values['S__target_L0H7'][metric]
        result['paired_contrasts'][f'target_S_St_minus_S__{metric}'] = cluster_mean_interval(delta, clusters)
    atomic_json(directory/'verified_summary.json', result)
    print(json.dumps(result['paired_contrasts'], indent=2))


if __name__ == '__main__':
    main()
