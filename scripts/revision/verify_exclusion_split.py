"""Check exclusion-before-sampling against saved primary partitions and rows."""
import json
import numpy as np
from common import OUT, summaries
from probe_checkpoint import atomic_json, file_identity


def validate_membership(primary, excluded_split, excluded, clusters):
    partitions = ['train', 'validation', 'test']
    cluster_sets = []
    for key in partitions:
        actual = excluded_split[key]
        expected = [p for p in primary[key] if p not in excluded]
        if actual != expected or len(set(actual)) != len(actual):
            raise ValueError(f'Exclusion partition mismatch: {key}')
        cluster_sets.append({clusters[p] for p in actual})
    if any(cluster_sets[i] & cluster_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Sequence cluster crosses partitions')
    if set(excluded_split['excluded_training_homologs']) != excluded:
        raise ValueError('Exclusion list mismatch')
    return {key: len(values) for key, values in zip(partitions, cluster_sets)}


def main():
    directory = OUT/'probes_overlap_excluded'
    paths = [OUT/'probe_split.json', directory/'probe_split.json',
             OUT/'training_homology_audit.json', OUT/'clusters.json']
    primary, split, audit, clusters = [json.loads(p.read_text()) for p in paths]
    excluded = set(audit['evaluation_ids_identity50'])
    counts = validate_membership(primary, split, excluded, clusters)
    ids, _, offsets = summaries('esm3')
    row_to_protein = np.repeat(np.arange(len(ids)), offsets[:, 1])
    rng = np.random.default_rng(split['seed'])
    unique = np.unique([clusters[p] for p in ids])
    rng.shuffle(unique)  # Advance exactly as the partitioning stage did.
    report = {'excluded_proteins': len(excluded), 'cluster_counts': counts,
              'inputs': [file_identity(p, content=True) for p in paths], 'partitions': {}}
    with np.load(directory/'probe_rows.npz') as rows:
        for key, cap in zip(['train', 'validation', 'test'], split['caps']):
            proteins = np.flatnonzero(np.isin(ids, split[key]))
            eligible = np.concatenate([np.arange(offsets[p, 0], offsets[p].sum()) for p in proteins])
            expected = np.sort(rng.choice(eligible, min(cap, len(eligible)), replace=False))
            actual = rows[key]
            if not np.array_equal(expected, actual):
                raise ValueError(f'Sampled rows do not reproduce: {key}')
            sampled_ids = ids[row_to_protein[actual]]
            if set(sampled_ids) & excluded or not set(sampled_ids) <= set(split[key]):
                raise ValueError(f'Excluded or wrong-partition sampled protein: {key}')
            report['partitions'][key] = {'proteins': len(proteins), 'sampled_residues': len(actual),
                                         'sampled_proteins': len(set(sampled_ids))}
    report['inputs'].append(file_identity(directory/'probe_rows.npz', content=True))
    report['passed'] = True
    report['scope'] = 'Detected homologs removed before sampling and probe fitting; fixed SAEs, not retrained or certified pretraining-independent.'
    atomic_json(directory/'exclusion_split_verification.json', report)
    print(json.dumps(report['partitions'], indent=2))


if __name__ == '__main__':
    main()
