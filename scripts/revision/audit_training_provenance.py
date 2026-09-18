"""Reconstruct archived training-row selection without loading activation tensors."""
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
import h5py
import numpy as np
import torch
from common import ROOT, OUT, save

sys.path.insert(0, str(ROOT / 'src'))

STORAGE = Path('/mnt/85740f55-8e9a-4214-9500-be446866627e/interpretability_1.5M')


def main():
    hits = {}
    with (OUT / 'evaluation_vs_training.tsv').open() as handle:
        for line in handle:
            query, target, identity, *_ = line.rstrip().split('\t')
            if float(identity) >= 50:
                hits.setdefault(target, set()).add(query)
    output = {'scope': 'Reconstruction from current archived chunks and scaled training code. '
                       'Not a contemporaneous saved training-ID manifest; modified chunks/code would invalidate reconstruction.',
              'models': {}}
    for model in ['esm3', 'esm2']:
        paths = sorted((STORAGE / model / 'residue_L33').glob('*.h5'))
        assert paths
        sizes = []
        for path in paths:
            with h5py.File(path) as f:
                sizes.append(f['activations'].shape[0])
        fraction = min(1., 20_000_000/sum(sizes))
        samples = [max(1, int(n*fraction)) if fraction < 1 else n for n in sizes]
        n_total = sum(samples)
        assert n_total <= 20_000_000
        validation_n = max(int(.05*n_total), 1000)
        permutation = np.random.RandomState(42).permutation(n_total)
        training = np.ones(n_total, dtype=bool)
        training[permutation[:validation_n]] = False
        del permutation
        rng = np.random.RandomState(42)
        cursor = 0
        records = []
        n_ids = 0
        for path, n, k in zip(paths, sizes, samples):
            selected = rng.choice(n, k, replace=False) if fraction < 1 else np.arange(n)
            with h5py.File(path) as f:
                ids = [v.decode() for v in f['ids'][:]]
                offsets = f['offsets'][:]
            n_ids += len(ids)
            assert np.array_equal(offsets[:, 0], np.arange(len(ids)))
            assert offsets[-1, 1]+offsets[-1, 2] == n
            pindex = np.searchsorted(offsets[:, 1], selected, side='right')-1
            mask = training[cursor:cursor+k]
            train_counts = np.bincount(pindex[mask], minlength=len(ids))
            val_counts = np.bincount(pindex[~mask], minlength=len(ids))
            for i, pid in enumerate(ids):
                if pid in hits:
                    records.append({'training_id': pid, 'chunk': path.name,
                                    'reconstructed_training_rows': int(train_counts[i]),
                                    'reconstructed_validation_rows': int(val_counts[i]),
                                    'homologous_evaluation_ids': sorted(hits[pid])})
            cursor += k
        checkpoint = torch.load(ROOT / f'models/sae_1.5M/{model}_residue_ef8_k64/best.pt',
                                map_location='cpu', weights_only=False)
        metadata = {k: asdict(v) if is_dataclass(v) else v for k, v in checkpoint.items()
                    if k not in ['model_state_dict', 'optimizer_state_dict']
                    and (is_dataclass(v) or isinstance(v, (str, int, float, bool, dict)))}
        # Only configuration/scalar metadata belongs in the small provenance report.
        metadata = {k: v for k, v in metadata.items() if not isinstance(v, dict) or k.endswith('config')}
        config = metadata['train_config']
        expected_steps = ((n_total-validation_n)//config['batch_size'])*config['num_epochs']
        eval_ids = set().union(*(set(r['homologous_evaluation_ids']) for r in records if r['reconstructed_training_rows']))
        output['models'][model] = {'n_chunks': len(paths), 'n_extracted_protein_records': n_ids,
                                  'total_archived_residues': int(sum(sizes)), 'sampled_residues': n_total,
                                  'reconstructed_training_residues': n_total-validation_n,
                                  'reconstructed_validation_residues': validation_n,
                                  'checkpoint_metadata': metadata,
                                  'expected_steps_from_current_sampler': expected_steps,
                                  'checkpoint_step_matches_current_sampler': metadata['step'] == expected_steps,
                                  'historical_selection_verified': False,
                                  'verification_note': 'Agreement of aggregate steps is a consistency check, not an archived row manifest. '
                                                       'Disagreement means the current sampler cannot establish historical row selection.',
                                  'n_hit_targets_in_archived_chunks': len(records),
                                  'n_evaluation_homologous_to_reconstructed_training': len(eval_ids),
                                  'evaluation_ids_homologous_to_reconstructed_training': sorted(eval_ids),
                                  'hit_target_records': records}
        save('training_provenance_audit.json', output)
        print(model, {k: v for k, v in output['models'][model].items()
                      if k not in ['hit_target_records', 'evaluation_ids_homologous_to_reconstructed_training']}, flush=True)
    save('training_provenance_audit.json', output)


if __name__ == '__main__':
    main()
