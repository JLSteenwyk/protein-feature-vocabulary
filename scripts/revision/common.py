"""Reproducible statistical utilities for BIOINF-2026-2288 revision."""
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/scaled_1.5M'
OUT = ROOT / 'revision/analyses'
SEED = 2288


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def summaries(model, field='protein_max_activations'):
    with h5py.File(RESULTS / 'sae_features' / model / 'residue/protein_summaries.h5') as f:
        ids = np.array([x.decode() for x in f['protein_ids'][:]])
        return ids, f[field][:], f['offsets'][:]


def standardize(x):
    x = np.asarray(x, dtype=np.float32).copy()
    x -= x.mean(0, dtype=np.float64).astype(np.float32)
    norm = np.sqrt(np.sum(x.astype(np.float64)**2, axis=0))
    valid = norm > 0
    x /= np.where(valid, norm, 1).astype(np.float32)
    return x, valid


def full_column_shuffle(x, seed):
    """Permute every column over ALL rows, including its zero entries."""
    rng = np.random.default_rng(seed)
    x = x.tocsc(copy=True)
    for j in range(x.shape[1]):
        a, b = x.indptr[j:j+2]
        x.indices[a:b] = rng.choice(x.shape[0], b-a, replace=False)
    x.has_sorted_indices = False
    x.sort_indices()
    return x.tocsr()


def within_protein_shuffle(x, groups, seed):
    rng = np.random.default_rng(seed)
    perm = np.arange(x.shape[0])
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        perm[rows] = rng.permutation(rows)
    return x[perm]


def labels(ids, offsets, metadata, broad=False):
    types = {'Active site', 'Binding site', 'Metal binding', 'Site'}
    if broad:
        types |= {'Disulfide bond', 'Modified residue'}
    n = int(np.max(offsets[:, 0] + offsets[:, 1]))
    y = np.zeros(n, dtype=np.int8)
    for pid, (start, length) in zip(ids, offsets):
        for feat in metadata.get(pid, {}).get('features', []):
            if feat.get('type') not in types:
                continue
            a = feat.get('start', feat.get('location', {}).get('start', {}).get('value'))
            b = feat.get('end', feat.get('location', {}).get('end', {}).get('value'))
            if a is None or b is None:
                continue
            a, b = max(0, int(a)-1), min(int(length), int(b))
            # Disulfide bond annotations link endpoints; the interval is not a site.
            if feat.get('type') == 'Disulfide bond':
                for pos in {a, b-1}:
                    if 0 <= pos < length:
                        y[int(start)+pos] = 1
            else:
                y[int(start)+a:int(start)+b] = 1
    return y


def metrics(y, score, groups):
    aucs = []
    for g in np.unique(groups):
        ix = groups == g
        if np.unique(y[ix]).size == 2:
            aucs.append(roc_auc_score(y[ix], score[ix]))
    return {'auroc': float(roc_auc_score(y, score)),
            'average_precision': float(average_precision_score(y, score)),
            'prevalence': float(y.mean()),
            'macro_within_protein_auroc': float(np.mean(aucs)) if aucs else None,
            'n_proteins_with_both_classes': len(aucs),
            'n_residues': len(y), 'n_positive': int(y.sum())}


def auc_bootstrap(y, score, groups, seed=SEED, n=1000):
    """Cluster bootstrap AUROC via precomputed score ties (no repeated sorting)."""
    _, g = np.unique(groups, return_inverse=True)
    _, ranks = np.unique(score, return_inverse=True)
    ng, nr = g.max()+1, ranks.max()+1
    pos = sparse.csr_matrix((y.astype(float), (ranks, g)), shape=(nr, ng))
    neg = sparse.csr_matrix(((1-y).astype(float), (ranks, g)), shape=(nr, ng))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        counts = rng.multinomial(ng, np.full(ng, 1/ng))
        p, q = pos @ counts, neg @ counts
        if p.sum() and q.sum():
            values.append(float(np.sum(p*(np.cumsum(q)-0.5*q))/(p.sum()*q.sum())))
    return {'low': float(np.quantile(values, .025)), 'high': float(np.quantile(values, .975)),
            'n_resamples': n, 'unit': 'sequence_cluster', 'method': 'percentile'}
