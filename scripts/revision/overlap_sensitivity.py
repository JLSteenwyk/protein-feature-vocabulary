"""Conservative overlap exclusion with frozen matches and existing fitted probes."""
import json
import numpy as np
from common import OUT, summaries, save, metrics, auc_bootstrap
from run_matching import matched_r, describe


def main():
    audit = json.loads((OUT / 'training_homology_audit.json').read_text())
    excluded = audit['evaluation_ids_identity50']
    ids3, x, _ = summaries('esm3'); ids2, y, _ = summaries('esm2')
    ids, a, b = np.intersect1d(ids3, ids2, return_indices=True)
    keep = ~np.isin(ids, excluded)
    x, y = x[a][keep], y[b][keep]
    output = {'excluded_rule': 'Any detected >=50%-identity training-candidate hit, without coverage restriction',
              'n_excluded_evaluation_proteins': int((~keep).sum()), 'n_retained_evaluation_proteins': int(keep.sum()),
              'scope': 'Frozen existing feature selections and fitted probes. Not retraining of SAEs or probes. '
                       'Full-data match choices can have used excluded proteins; held-out matches remain frozen discovery choices.',
              'matching': {}, 'probes': {}}
    with np.load(OUT / 'matching_pairs.npz') as pairs:
        for name, source, target in [('full_data_best_frozen', pairs['source_ids'], pairs['best_target']),
                                    ('full_data_one_to_one_frozen', pairs['one_to_one_source'], pairs['one_to_one_target'])]:
            output['matching'][name] = describe(matched_r(x, y, source, target))
    with np.load(OUT / 'matching_discovery_pairs.npz') as pairs:
        valid = np.isin(ids[keep], pairs['validation_protein_ids'])
        output['matching']['heldout_best_frozen'] = describe(matched_r(x[valid], y[valid], pairs['source_ids'], pairs['target_ids']))
        output['matching']['n_retained_validation_proteins'] = int(valid.sum())
    save('overlap_excluded_sensitivity.json', output)
    for path in sorted(OUT.glob('probe_predictions_*.npz')):
        with np.load(path) as data:
            mask = ~np.isin(data['protein'], excluded)
            ytest, score, protein, clusters = [data[k][mask] for k in ['y', 'score', 'protein', 'cluster']]
            result = {**metrics(ytest, score, protein),
                      'ci95_auroc': auc_bootstrap(ytest, score, clusters, n=500),
                      'n_excluded_test_residues': int((~mask).sum()),
                      'n_retained_test_proteins': len(np.unique(protein))}
            output['probes'][path.stem.removeprefix('probe_predictions_')] = result
            save('overlap_excluded_sensitivity.json', output)
    print('Overlap sensitivity complete for', len(output['probes']), 'probe prediction files', flush=True)


if __name__ == '__main__':
    main()
