"""Complete ESM-2 and composition controls using saved canonical residue selections."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from common import ROOT, OUT, RESULTS, SEED, summaries, labels, metrics, auc_bootstrap, full_column_shuffle, within_protein_shuffle
from run_probe_controls import fit_select, legacy_shuffle
from probe_checkpoint import ProbeCheckpoint, atomic_json, file_identity


def align_residue_rows(reference_ids, reference_offsets, target_ids, target_offsets, rows):
    if len(set(reference_ids)) != len(reference_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError('Protein IDs must be unique')
    if set(reference_ids) != set(target_ids):
        raise ValueError('Models contain different protein sets')
    index = {pid: i for i, pid in enumerate(target_ids)}
    order = np.array([index[pid] for pid in reference_ids])
    if not np.array_equal(reference_offsets[:, 1], target_offsets[order, 1]):
        raise ValueError('Matched protein lengths differ')
    for offsets in [reference_offsets, target_offsets]:
        if not np.array_equal(offsets[:, 0], np.r_[0, np.cumsum(offsets[:-1, 1])]):
            raise ValueError('Offsets must be contiguous')
    rows = np.asarray(rows)
    if np.any(rows < 0) or np.any(rows >= reference_offsets[:, 1].sum()):
        raise ValueError('Reference row outside residue matrix')
    protein = np.searchsorted(reference_offsets[:, 0], rows, side='right')-1
    position = rows-reference_offsets[protein, 0]
    mapped = target_offsets[order[protein], 0]+position
    return mapped, order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run-dir', type=Path, default=OUT)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.base_run_dir.resolve() == args.run_dir.resolve():
        parser.error('Use an isolated output directory; completed base fits must be preserved.')
    metadata_path = ROOT/'data/eval_expanded/metadata.json'
    sequences_path = ROOT/'data/eval_expanded/sequences.json'
    identity = {'protocol': 'protein-ID/residue-position-aligned ESM-2 continuation v1',
                'small_inputs': [file_identity(p, content=True) for p in [metadata_path, sequences_path, OUT/'clusters.json',
                    args.base_run_dir/'probe_rows.npz', args.base_run_dir/'probe_split.json', Path(__file__),
                    Path(__file__).with_name('run_probe_controls.py'), Path(__file__).with_name('common.py'),
                    Path(__file__).with_name('probe_checkpoint.py')]],
                'large_inputs': [file_identity(RESULTS/'sae_features'/m/'residue'/name)
                                 for m in ['esm2', 'esm3'] for name in ['protein_summaries.h5', 'features_sparse.npz']]}
    checkpoint = ProbeCheckpoint(args.run_dir, identity)
    ids, _, offsets = summaries('esm3')
    model_ids, means, model_offsets = summaries('esm2', 'protein_mean_activations')
    split = json.loads((args.base_run_dir/'probe_split.json').read_text())
    with np.load(args.base_run_dir/'probe_rows.npz') as saved:
        rows = [saved[k] for k in ['train', 'validation', 'test']]
    mapped, order = [], None
    for selection in rows:
        target_rows, order = align_residue_rows(ids, offsets, model_ids, model_offsets, selection)
        mapped.append(target_rows)
    means = means[order]
    groups = np.repeat(np.arange(len(ids)), offsets[:, 1])
    protein_groups = [groups[r] for r in rows]
    for name, group in zip(['train', 'validation', 'test'], protein_groups):
        if not set(ids[group]).issubset(set(split[name])):
            raise ValueError('Sampled residues differ from saved protein partitions')
    clusters = json.loads((OUT/'clusters.json').read_text())
    test_clusters = np.array([clusters[pid] for pid in ids[protein_groups[2]]])
    metadata = json.loads(metadata_path.read_text())
    y = labels(ids, offsets, metadata); ys = [y[r] for r in rows]
    expected = {'y': ys[2], 'protein': ids[protein_groups[2]], 'cluster': test_clusters}
    output = checkpoint.load({'seed': SEED, 'base_run_dir': str(args.base_run_dir),
                              'alignment': 'Exact protein-ID join, verified lengths, within-protein residue offsets preserved.',
                              'models': {}, 'sample_counts': {name: {'proteins': len(split[name]),
                                  'residues': len(r), 'positive': int(v.sum()), 'prevalence': float(v.mean())}
                                  for name, r, v in zip(['train', 'validation', 'test'], rows, ys)}})
    atomic_json(args.run_dir/'probe_split.json', split)
    np.savez_compressed(args.run_dir/'probe_rows.npz', **dict(zip(['train', 'validation', 'test'], rows)))
    np.savez_compressed(args.run_dir/'esm2_aligned_rows.npz', **dict(zip(['train', 'validation', 'test'], mapped)))

    def evaluate(model, name, features, yy=ys, weights=None):
        output['models'].setdefault(model, {})
        if checkpoint.reusable(output, model, name, expected):
            print(model, name, 'verified checkpoint', flush=True)
            return
        print(model, name, flush=True)
        xx = features() if callable(features) else features
        fitted, selection = fit_select(xx, yy, weights)
        scores = fitted.decision_function(xx[2])
        output['models'][model][name] = {**metrics(ys[2], scores, protein_groups[2]), **selection,
                                        'ci95_auroc': auc_bootstrap(ys[2], scores, test_clusters, n=1000)}
        checkpoint.commit(output, model, name, {**expected, 'score': scores},
                          {'coef': fitted.coef_, 'intercept': fitted.intercept_, 'classes': fitted.classes_})
        print(output['models'][model][name], flush=True)

    print('Loading ESM-2 sparse matrix with explicit row mapping', flush=True)
    full = sparse.load_npz(RESULTS/'sae_features/esm2/residue/features_sparse.npz')
    assert full.shape[0] == int(model_offsets[:, 1].sum())
    xs = [full[r].tocsr() for r in mapped]; del full
    evaluate('esm2', 'intact', xs)
    binary = [x.copy() for x in xs]
    for x in binary:
        x.data[:] = 1
    evaluate('esm2', 'binary_support', binary); del binary
    for seed in [2288, 2289, 2290]:
        evaluate('esm2', f'legacy_support_preserving_{seed}', lambda: [legacy_shuffle(x, seed+i*100) for i, x in enumerate(xs)])
        evaluate('esm2', f'global_column_shuffle_{seed}', lambda: [full_column_shuffle(x, seed+i*100) for i, x in enumerate(xs)])
        evaluate('esm2', f'within_protein_shuffle_{seed}', lambda: [within_protein_shuffle(x, g, seed+i*100)
                 for i, (x, g) in enumerate(zip(xs, protein_groups))])
    train_p = np.flatnonzero(np.isin(ids, split['train']))
    positives = np.bincount(protein_groups[0], weights=ys[0], minlength=len(ids))[train_p]
    total = np.bincount(protein_groups[0], minlength=len(ids))[train_p]
    weights = np.r_[positives/(ys[0].mean()*2), (total-positives)/((1-ys[0].mean())*2)]
    xx = [sparse.csr_matrix(np.concatenate([means[train_p], means[train_p]])),
          sparse.csr_matrix(means[protein_groups[1]]), sparse.csr_matrix(means[protein_groups[2]])]
    yy = [np.r_[np.ones(len(train_p)), np.zeros(len(train_p))], ys[1], ys[2]]
    evaluate('esm2', 'protein_mean_only', xx, yy, weights)
    del xx, means, xs
    sequences = json.loads(sequences_path.read_text())
    aa = 'ACDEFGHIKLMNPQRSTVWY'; aa_index = {a: i for i, a in enumerate(aa)}
    composition = np.array([[sequences[p].count(a)/len(sequences[p]) for a in aa]
                            + [np.log(len(sequences[p]))] for p in ids], dtype=np.float32)
    comp = [sparse.csr_matrix(composition[g]) for g in protein_groups]
    evaluate('composition', 'protein_composition_length', comp)
    flattened = np.concatenate([np.array([aa_index.get(a, 20) for a in sequences[p]], dtype=np.int8) for p in ids])
    aa_rows = [sparse.csr_matrix((np.ones(len(r)), (np.arange(len(r)), flattened[r])), shape=(len(r), 21)) for r in rows]
    evaluate('composition', 'residue_identity_plus_composition', [sparse.hstack([a, b], format='csr') for a, b in zip(aa_rows, comp)])
    checkpoint.close()


if __name__ == '__main__':
    main()
