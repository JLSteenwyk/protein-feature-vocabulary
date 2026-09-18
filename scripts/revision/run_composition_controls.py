"""Run inexpensive composition controls on the exact saved probe partitions."""
import json
import numpy as np
from scipy import sparse
from common import ROOT, OUT, summaries, labels, metrics, auc_bootstrap, save
from run_probe_controls import fit_select


def main():
    ids, _, offsets = summaries('esm3')
    sequences = json.loads((ROOT / 'data/eval_expanded/sequences.json').read_text())
    metadata = json.loads((ROOT / 'data/eval_expanded/metadata.json').read_text())
    clusters = json.loads((OUT / 'clusters.json').read_text())
    saved = np.load(OUT / 'probe_rows.npz')
    rows = [saved[k] for k in ['train', 'validation', 'test']]
    groups = np.repeat(np.arange(len(ids)), offsets[:, 1])
    partitions = [groups[r] for r in rows]
    y = labels(ids, offsets, metadata)
    ys = [y[r] for r in rows]
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    index = {a: i for i, a in enumerate(aa)}
    composition = np.array([[sequences[p].count(a) / len(sequences[p]) for a in aa]
                            + [np.log(len(sequences[p]))] for p in ids], dtype=np.float32)
    comp = [sparse.csr_matrix(composition[g]) for g in partitions]
    flattened = np.concatenate([np.array([index.get(a, 20) for a in sequences[p]], dtype=np.int8) for p in ids])
    identity = [sparse.csr_matrix((np.ones(len(r)), (np.arange(len(r)), flattened[r])),
                                  shape=(len(r), 21)) for r in rows]
    controls = {'protein_composition_length': comp,
                'residue_identity_plus_composition': [sparse.hstack([a, b], format='csr') for a, b in zip(identity, comp)]}
    result = {}
    test_clusters = np.array([clusters[p] for p in ids[partitions[2]]])
    for name, xs in controls.items():
        model, selection = fit_select(xs, ys)
        score = model.decision_function(xs[2])
        result[name] = {**metrics(ys[2], score, partitions[2]), **selection,
                        'ci95_auroc': auc_bootstrap(ys[2], score, test_clusters)}
        np.savez_compressed(OUT / f'probe_predictions_composition_{name}.npz', y=ys[2], score=score,
                            protein=ids[partitions[2]], cluster=test_clusters)
        save('composition_controls.json', result)
        print(name, result[name], flush=True)


if __name__ == '__main__':
    main()
