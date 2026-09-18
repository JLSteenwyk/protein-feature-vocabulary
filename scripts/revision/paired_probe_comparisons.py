"""Paired, cluster-resampled comparisons of saved held-out predictions."""
import argparse
import json
import fcntl
import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score
from common import OUT, ROOT, SEED
from probe_checkpoint import atomic_json


def score_tables(y, score, cluster_index, n_clusters):
    _, rank = np.unique(score, return_inverse=True)
    shape = (rank.max()+1, n_clusters)
    return (sparse.csr_matrix((y.astype(float), (rank, cluster_index)), shape=shape),
            sparse.csr_matrix(((1-y).astype(float), (rank, cluster_index)), shape=shape))


def weighted_metrics(tables, counts):
    p, q = [table @ counts for table in tables]
    if not p.sum() or not q.sum():
        return np.array([np.nan, np.nan])
    auc = np.sum(p * (np.cumsum(q) - .5*q)) / (p.sum()*q.sum())
    p, q = p[::-1], q[::-1]
    cumulative = np.cumsum(p+q)
    precision = np.divide(np.cumsum(p), cumulative, out=np.zeros_like(p), where=cumulative > 0)
    return np.array([auc, np.sum(p*precision)/p.sum()])


def within_protein_tables(y, scores, proteins, cluster_index, n_clusters):
    _, pi = np.unique(proteins, return_inverse=True)
    order = np.argsort(pi, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(pi[order]))+1)
    sums = np.zeros((n_clusters, len(scores)))
    counts = np.zeros(n_clusters)
    for rows in groups:
        assert len(np.unique(cluster_index[rows])) == 1
        if np.unique(y[rows]).size != 2:
            continue
        c = cluster_index[rows[0]]
        sums[c] += [roc_auc_score(y[rows], score[rows]) for score in scores]
        counts[c] += 1
    return sums, counts


def compare(reference, other, n_resamples=500):
    for key in ['y', 'protein', 'cluster']:
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f'Paired prediction alignment mismatch: {key}')
    y = reference['y']
    cluster_names, cluster_index = np.unique(reference['cluster'], return_inverse=True)
    ng = len(cluster_names)
    scores = [reference['score'], other['score']]
    tables = [score_tables(y, score, cluster_index, ng) for score in scores]
    within, counts = within_protein_tables(y, scores, reference['protein'], cluster_index, ng)
    point = np.array([[roc_auc_score(y, s), average_precision_score(y, s)] for s in scores])
    point = np.r_[point[0]-point[1], (within.sum(0)[0]-within.sum(0)[1])/counts.sum()]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(n_resamples):
        w = rng.multinomial(ng, np.full(ng, 1/ng))
        difference = weighted_metrics(tables[0], w) - weighted_metrics(tables[1], w)
        within_diff = (w @ (within[:, 0]-within[:, 1]))/(w @ counts) if w @ counts else np.nan
        draws.append(np.r_[difference, within_diff])
    draws = np.asarray(draws)
    result = {'direction': 'reference minus control', 'n_resamples': n_resamples,
              'n_test_clusters': ng, 'n_mixed_label_proteins': int(counts.sum()),
              'unit': 'test_sequence_cluster', 'method': 'paired percentile bootstrap',
              'scope': 'Conditional on fitted probes, saved test residues, and one cluster split. '
                       'Exploratory comparisons, not multiplicity-adjusted significance tests.', 'metrics': {}}
    for i, key in enumerate(['auroc', 'average_precision', 'macro_within_protein_auroc']):
        finite = np.isfinite(draws[:, i])
        result['metrics'][key] = {'difference': float(point[i]),
                                  'ci95': np.quantile(draws[finite, i], [.025, .975]).tolist(),
                                  'valid_resamples': int(finite.sum())}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resamples', type=int, default=500)
    parser.add_argument('--run-dir', type=Path, default=OUT)
    parser.add_argument('--combined', action='store_true', help='Use the validated combined-run snapshot')
    args = parser.parse_args()
    lock = (args.run_dir / '.paired_comparisons.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = json.loads((args.run_dir / ('combined_probe_controls.json' if args.combined else 'probe_controls.json')).read_text())
    completed = snapshot['models']
    composition_path = args.run_dir / 'composition_controls.json'
    if composition_path.exists() and not args.combined:
        completed.setdefault('composition', {}).update(json.loads(composition_path.read_text()))
    # Use committed controls, not arbitrary files which may still be being written.
    predictions = {model: [args.run_dir / f'probe_predictions_{model}_{name}.npz' for name in records]
                   for model, records in completed.items()}
    if args.combined:
        predictions = {}
        for model, records in snapshot['prediction_files'].items():
            predictions[model] = []
            for record in records.values():
                path = ROOT / record['path']
                if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                    raise ValueError(f'Combined snapshot prediction changed: {path}')
                predictions[model].append(path)
    output_path = args.run_dir / ('combined_paired_probe_comparisons.json' if args.combined else 'paired_probe_comparisons.json')
    output = {}
    for model in ['esm3', 'esm2']:
        reference_path = args.run_dir / f'probe_predictions_{model}_intact.npz'
        if 'intact' not in completed.get(model, {}):
            continue
        if args.combined:
            reference_path = ROOT / snapshot['prediction_files'][model]['intact']['path']
        with np.load(reference_path) as reference:
            paths = sorted(predictions.get(model, [])) + sorted(predictions.get('composition', []))
            if model == 'esm3' and 'intact' in completed.get('esm2', {}):
                paths += [path for path in predictions['esm2'] if path.name == 'probe_predictions_esm2_intact.npz']
            for path in paths:
                if path == reference_path:
                    continue
                with np.load(path) as other:
                    key = f'{model}_intact_vs_{path.stem.removeprefix("probe_predictions_")}'
                    output[key] = compare(reference, other, args.resamples)
                    atomic_json(output_path, output)
                    print(key, output[key]['metrics'], flush=True)
    atomic_json(output_path, output)
    lock.close()


if __name__ == '__main__':
    main()
