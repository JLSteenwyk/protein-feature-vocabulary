"""Refit one model's controls on saved canonical selections with explicit ID joins."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from common import ROOT, OUT, RESULTS, SEED, summaries, labels, metrics, auc_bootstrap
from common import full_column_shuffle, within_protein_shuffle
from probe_checkpoint import ProbeCheckpoint, atomic_json, file_identity
from run_probe_controls import fit_select, legacy_shuffle
from run_aligned_esm2_probes import align_residue_rows


PARTITIONS = ['train', 'validation', 'test']


def validate_split(ids, offsets, rows, split, clusters):
    if split['seed'] != SEED or len(rows) != 3 or len(set(ids)) != len(ids):
        raise ValueError('Invalid canonical split identity')
    if not np.array_equal(offsets[:, 0], np.r_[0, np.cumsum(offsets[:-1, 1])]):
        raise ValueError('Noncontiguous canonical offsets')
    known, excluded = set(ids), set(split.get('excluded_training_homologs', []))
    previous, previous_clusters, groups = set(), set(), []
    for name, selected in zip(PARTITIONS, rows):
        proteins = set(split[name])
        if len(proteins) != len(split[name]) or not proteins <= known or proteins & excluded:
            raise ValueError('Unknown, duplicate or excluded split protein')
        current_clusters = {clusters[p] for p in proteins}
        if proteins & previous or current_clusters & previous_clusters:
            raise ValueError('Protein or cluster leakage between partitions')
        previous |= proteins
        previous_clusters |= current_clusters
        if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer) or
                len(selected) == 0 or np.any(selected < 0) or
                np.any(selected >= offsets[:, 1].sum()) or np.any(np.diff(selected) <= 0)):
            raise ValueError('Invalid sampled residue rows')
        group = np.searchsorted(offsets[:, 0], selected, side='right')-1
        if not set(ids[group]) <= proteins:
            raise ValueError('Residue selection outside its protein partition')
        groups.append(group)
    if previous != known-excluded:
        raise ValueError('Split does not cover retained canonical proteins')
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run-dir', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--model', required=True, choices=['esm3', 'esm2', 'composition'])
    args = parser.parse_args()
    if args.base_run_dir.resolve() == args.run_dir.resolve():
        parser.error('Use separate outputs; preserve the saved base run.')
    inputs = [ROOT/'data/eval_expanded'/name for name in ['metadata.json', 'sequences.json']]
    inputs += [OUT/'clusters.json', args.base_run_dir/'probe_rows.npz', args.base_run_dir/'probe_split.json']
    code = [Path(__file__)] + [Path(__file__).with_name(name) for name in
            ['common.py', 'probe_checkpoint.py', 'run_probe_controls.py', 'run_aligned_esm2_probes.py']]
    large = [RESULTS/'sae_features/esm3/residue/protein_summaries.h5']
    if args.model != 'composition':
        large += [RESULTS/f'sae_features/{args.model}/residue'/name for name in
                  ['protein_summaries.h5', 'features_sparse.npz']]
    identity = {'protocol': 'canonical-split portable refit v1', 'model': args.model,
                'seed': SEED, 'small_inputs': [file_identity(p, content=True) for p in inputs+code],
                'large_inputs': [file_identity(p) for p in sorted(set(large))]}
    checkpoint = ProbeCheckpoint(args.run_dir, identity)
    try:
        ids, _, offsets = summaries('esm3')
        split = json.loads((args.base_run_dir/'probe_split.json').read_text())
        with np.load(args.base_run_dir/'probe_rows.npz', allow_pickle=False) as saved:
            rows = [saved[k] for k in PARTITIONS]
        clusters = json.loads((OUT/'clusters.json').read_text())
        groups = validate_split(ids, offsets, rows, split, clusters)
        metadata = json.loads(inputs[0].read_text())
        y = labels(ids, offsets, metadata)
        ys = [y[r] for r in rows]
        test_clusters = np.array([clusters[p] for p in ids[groups[2]]])
        expected = {'y': ys[2], 'protein': ids[groups[2]], 'cluster': test_clusters}
        output = checkpoint.load({'seed': SEED, 'scope': 'Refits conditional on saved canonical residue selections; not new sampling or feature extraction.',
                                  'models': {args.model: {}},
                                  'sample_counts': {name: {'proteins': len(split[name]), 'residues': len(r),
                                                          'positive': int(v.sum()), 'prevalence': float(v.mean())}
                                                    for name, r, v in zip(PARTITIONS, rows, ys)}})
        atomic_json(args.run_dir/'probe_split.json', split)
        np.savez_compressed(args.run_dir/'probe_rows.npz', **dict(zip(PARTITIONS, rows)))

        def evaluate(name, features, yy=ys, weights=None):
            if checkpoint.reusable(output, args.model, name, expected):
                print(args.model, name, 'verified checkpoint', flush=True)
                return
            print(args.model, name, flush=True)
            xx = features() if callable(features) else features
            fitted, selection = fit_select(xx, yy, weights)
            scores = fitted.decision_function(xx[2])
            output['models'][args.model][name] = {**metrics(ys[2], scores, groups[2]), **selection,
                'ci95_auroc': auc_bootstrap(ys[2], scores, test_clusters, n=1000)}
            checkpoint.commit(output, args.model, name, {**expected, 'score': scores},
                              {'coef': fitted.coef_, 'intercept': fitted.intercept_, 'classes': fitted.classes_})
            print(output['models'][args.model][name], flush=True)

        if args.model == 'composition':
            sequences = json.loads(inputs[1].read_text())
            if any(len(sequences[p]) != length for p, length in zip(ids, offsets[:, 1])):
                raise ValueError('Sequence lengths differ from canonical residue offsets')
            aa = 'ACDEFGHIKLMNPQRSTVWY'
            index = {letter: i for i, letter in enumerate(aa)}
            composition = np.array([[sequences[p].count(a)/len(sequences[p]) for a in aa]
                                    + [np.log(len(sequences[p]))] for p in ids], dtype=np.float32)
            comp = [sparse.csr_matrix(composition[g]) for g in groups]
            evaluate('protein_composition_length', comp)
            flattened = np.concatenate([np.array([index.get(a, 20) for a in sequences[p]], dtype=np.int8) for p in ids])
            identity_features = [sparse.csr_matrix((np.ones(len(r)), (np.arange(len(r)), flattened[r])),
                                                  shape=(len(r), 21)) for r in rows]
            evaluate('residue_identity_plus_composition', [sparse.hstack([a, b], format='csr')
                                                          for a, b in zip(identity_features, comp)])
        else:
            model_ids, means, model_offsets = summaries(args.model, 'protein_mean_activations')
            mapped = []
            for selection in rows:
                target, order = align_residue_rows(ids, offsets, model_ids, model_offsets, selection)
                mapped.append(target)
            means = means[order]
            full = sparse.load_npz(RESULTS/f'sae_features/{args.model}/residue/features_sparse.npz')
            if full.shape != (int(model_offsets[:, 1].sum()), means.shape[1]):
                raise ValueError('Sparse feature dimensions differ from summaries')
            xs = [full[r].tocsr() for r in mapped]
            del full
            evaluate('intact', xs)
            binary = [x.copy() for x in xs]
            for x in binary:
                x.data[:] = 1
            evaluate('binary_support', binary)
            del binary
            for seed in [2288, 2289, 2290]:
                evaluate(f'legacy_support_preserving_{seed}', lambda: [legacy_shuffle(x, seed+i*100) for i, x in enumerate(xs)])
                evaluate(f'global_column_shuffle_{seed}', lambda: [full_column_shuffle(x, seed+i*100) for i, x in enumerate(xs)])
                evaluate(f'within_protein_shuffle_{seed}', lambda: [within_protein_shuffle(x, g, seed+i*100)
                                                                 for i, (x, g) in enumerate(zip(xs, groups))])
            train_p = np.flatnonzero(np.isin(ids, split['train']))
            positive = np.bincount(groups[0], weights=ys[0], minlength=len(ids))[train_p]
            total = np.bincount(groups[0], minlength=len(ids))[train_p]
            weights = np.r_[positive/(ys[0].mean()*2), (total-positive)/((1-ys[0].mean())*2)]
            xx = [sparse.csr_matrix(np.concatenate([means[train_p], means[train_p]])),
                  sparse.csr_matrix(means[groups[1]]), sparse.csr_matrix(means[groups[2]])]
            yy = [np.r_[np.ones(len(train_p)), np.zeros(len(train_p))], ys[1], ys[2]]
            evaluate('protein_mean_only', xx, yy, weights)
    finally:
        checkpoint.close()


if __name__ == '__main__':
    main()
