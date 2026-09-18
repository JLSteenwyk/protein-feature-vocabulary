"""Verify saved probe partitions and exact SAE-training/evaluation overlap."""
import hashlib
import json
from collections import defaultdict
from common import ROOT, OUT, save, summaries
import numpy as np


def fasta_records(path):
    name = None
    sequence = []
    with path.open() as handle:
        for line in handle:
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(sequence)
                name, sequence = line[1:].split()[0], []
            else:
                sequence.append(line.strip())
    if name is not None:
        yield name, ''.join(sequence)


def main():
    ids, _, offsets = summaries('esm3')
    seqs = json.loads((ROOT / 'data/eval_expanded/sequences.json').read_text())
    split = json.loads((OUT / 'probe_split.json').read_text())
    clusters = json.loads((OUT / 'clusters.json').read_text())
    names = ['train', 'validation', 'test']
    sets = {name: set(split[name]) for name in names}
    assert set.union(*sets.values()) == set(ids)
    assert sum(map(len, sets.values())) == len(ids)
    cluster_sets = {name: {clusters[p] for p in sets[name]} for name in names}
    sequence_sets = {name: {seqs[p] for p in sets[name]} for name in names}
    comparisons = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            comparisons[f'{a}_vs_{b}'] = {
                'shared_protein_ids': len(sets[a] & sets[b]),
                'shared_detected_clusters': len(cluster_sets[a] & cluster_sets[b]),
                'shared_exact_sequences': len(sequence_sets[a] & sequence_sets[b])}
            assert not any(comparisons[f'{a}_vs_{b}'].values())
    row_groups = np.repeat(ids, offsets[:, 1])
    rows = np.load(OUT / 'probe_rows.npz')
    for name in names:
        assert len(np.unique(rows[name])) == len(rows[name])
        assert set(row_groups[rows[name]]) <= sets[name]
    by_hash = defaultdict(list)
    for pid in ids:
        by_hash[hashlib.sha256(seqs[pid].encode()).hexdigest()].append(str(pid))
    train_path = ROOT / 'data/sae_training/uniref50_1.5M.fasta'
    matches = []
    n = 0
    for name, seq in fasta_records(train_path):
        n += 1
        for pid in by_hash.get(hashlib.sha256(seq.encode()).hexdigest(), []):
            assert seq == seqs[pid]
            matches.append({'evaluation_id': pid, 'training_id': name})
    result = {'n_evaluation_proteins': len(ids), 'n_training_fasta_records': n,
              'split_checks': comparisons, 'sampled_residue_membership_verified': True,
              'exact_training_sequence_matches': matches,
              'n_evaluation_with_exact_training_match': len({r['evaluation_id'] for r in matches}),
              'scope': 'Exact sequence equality against the archived candidate SAE-training FASTA. '
                       'Does not establish which training entries yielded activations or exclude nonidentical homologs. '
                       'No claim about foundation-model pretraining overlap.'}
    save('dataset_audit.json', result)
    print({k: v for k, v in result.items() if k != 'exact_training_sequence_matches'}, flush=True)


if __name__ == '__main__':
    main()
