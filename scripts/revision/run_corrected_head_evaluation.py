"""Resumable generic human-protein head perturbation; no biological accuracy claim."""
import argparse
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.special import logsumexp
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity
from capture_reproducibility import stable_digest
from secondary_structure import ss3_predictions, SS8_VOCAB

sys.path.insert(0, str(ROOT/'src'))
from models.esm3_hooks import load_esm3
from models.interventions import head_ablation_esm3


def head_conditions(seed=2288):
    candidates = [(layer, head) for layer in range(48) for head in range(24)
                  if (layer, head) not in [(0, 7), (0, 0)]]
    rng = np.random.default_rng(seed)
    random = [candidates[i] for i in rng.choice(len(candidates), 10, replace=False)]
    return [('normal', None), ('noop', None), ('target_L0H7', (0, 7)),
            ('same_layer_L0H0', (0, 0))] + [(f'random_L{l}H{h}', (l, h)) for l, h in random]


def prediction_changes(reference_sequence, sequence, reference_ss, ss):
    arrays = [reference_sequence, sequence, reference_ss, ss]
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite logits')
    if reference_sequence.shape != sequence.shape or reference_ss.shape != ss.shape or len(ss) != len(sequence):
        raise ValueError('Unpaired logit shapes')
    logp = reference_sequence-logsumexp(reference_sequence, axis=-1, keepdims=True)
    logq = sequence-logsumexp(sequence, axis=-1, keepdims=True)
    return {'sequence_disagreement': float(np.mean(reference_sequence.argmax(-1) != sequence.argmax(-1))),
            'ss3_disagreement': float(np.mean(ss3_predictions(reference_ss) != ss3_predictions(ss))),
            'mean_sequence_kl_normal_to_perturbed': float(np.mean(np.sum(np.exp(logp)*(logp-logq), axis=-1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=OUT/'corrected_head_evaluation')
    parser.add_argument('--n-proteins', type=int, default=100)
    args = parser.parse_args()
    if args.n_proteins < 1:
        parser.error('--n-proteins must be positive')
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'.writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata_path = ROOT/'data/eval_expanded/metadata.json'
    sequences_path = ROOT/'data/eval_expanded/sequences.json'
    structures_path = ROOT/'data/eval_expanded/structures/manifest.json'
    metadata, sequences, manifest = [json.loads(p.read_text()) for p in
                                     [metadata_path, sequences_path, structures_path]]
    available = manifest['available']
    candidates = [p for p in sorted(available) if p in sequences and
                  'Homo sapiens' in metadata.get(p, {}).get('organism', '') and 50 <= len(sequences[p]) <= 300]
    selected = np.random.default_rng(2288).choice(candidates, min(args.n_proteins, len(candidates)), replace=False).tolist()
    conditions = head_conditions()
    from esm.utils.constants.esm3 import data_root
    from esm.sdk.api import ESMProtein
    model_root = data_root('esm3')
    weights = [model_root/'data/weights'/name for name in
               ['esm3_sm_open_v1.pth', 'esm3_structure_encoder_v0.pth']]
    print('Hashing cached model and selected structure inputs', flush=True)
    identity = {'protocol': 1, 'seed': 2288, 'n_candidates': len(candidates), 'selected_ids': selected,
                'conditions': [[name, list(head) if head else None] for name, head in conditions],
                'inputs': [file_identity(p, content=True) for p in
                           [metadata_path, sequences_path, structures_path, OUT/'clusters.json']],
                'code': [file_identity(p, content=True) for p in
                         [Path(__file__), ROOT/'scripts/revision/secondary_structure.py',
                          ROOT/'src/models/esm3_hooks.py', ROOT/'src/models/interventions.py']],
                'structures': {p: file_identity(available[p], content=True) for p in selected},
                'weights': [stable_digest(p) for p in weights],
                'torch': torch.__version__, 'dtype': 'bfloat16 autocast',
                'scope': 'Human-only new cohort; unmasked output disagreement, not ground-truth accuracy. No direct coordinate inputs to model; S+St uses encoded structure tokens only.'}
    identity_path = directory/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Inputs or protocol changed; use a new run directory')
    atomic_json(identity_path, identity)
    clusters = json.loads((OUT/'clusters.json').read_text())
    torch.set_num_threads(2)
    torch.manual_seed(2288)
    model, tokenizers = load_esm3(device='cuda')
    if list(tokenizers.secondary_structure.vocab) != list(SS8_VOCAB):
        raise ValueError('Unexpected runtime secondary-structure vocabulary')
    records = {}
    for index, accession in enumerate(selected):
        record_path = directory/f'{accession}.json'
        data_path = directory/f'{accession}.npz'
        if record_path.exists():
            old = json.loads(record_path.read_text())
            if old['status'] == 'complete':
                if not data_path.exists() or stable_digest(data_path)['sha256'] != old['raw_sha256']:
                    raise ValueError('Committed raw output missing or changed')
            records[accession] = old
            continue
        try:
            protein = ESMProtein.from_pdb(available[accession])
            if protein.sequence != sequences[accession]:
                raise ValueError('Structure-derived sequence differs from archived evaluation sequence')
            encoded = model.encode(protein)
            if encoded.structure is None:
                raise ValueError('Structure encoder produced no structure tokens')
            seq = encoded.sequence.to('cuda').unsqueeze(0)
            struct = encoded.structure.to('cuda').unsqueeze(0)
            length = len(protein.sequence)
            if seq.shape[1] != length+2 or struct.shape != seq.shape:
                raise ValueError('Unexpected token/residue alignment')
            arrays = {'sequence_tokens': seq.cpu().numpy(), 'structure_tokens': struct.cpu().numpy()}
            record = {'status': 'complete', 'accession': accession, 'cluster': clusters[accession],
                      'length': length, 'conditions': {}}
            for modality in ['S', 'S_St']:
                reference_sequence = reference_ss = None
                for name, head in conditions:
                    manager = head_ablation_esm3(model, *head) if head is not None else nullcontext()
                    with manager, torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(sequence_tokens=seq, structure_tokens=struct if modality == 'S_St' else None)
                    sequence_logits = output.sequence_logits[0, 1:-1].float().cpu().numpy()
                    ss_logits = output.secondary_structure_logits[0, 1:-1].float().cpu().numpy()
                    del output
                    if len(sequence_logits) != length or ss_logits.shape != (length, 11):
                        raise ValueError('Output/residue alignment mismatch')
                    prefix = f'{modality}__{name}'
                    arrays[prefix+'__sequence_logits'] = sequence_logits
                    arrays[prefix+'__ss8_logits'] = ss_logits
                    if name == 'normal':
                        reference_sequence, reference_ss = sequence_logits, ss_logits
                    changes = prediction_changes(reference_sequence, sequence_logits, reference_ss, ss_logits)
                    if name == 'noop' and (changes['sequence_disagreement'] or changes['ss3_disagreement']):
                        raise RuntimeError('No-op prediction disagreement; inference is not repeatable')
                    record['conditions'][prefix] = changes
            temporary = data_path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **arrays)
            temporary.replace(data_path)
            record['raw_sha256'] = stable_digest(data_path)['sha256']
            atomic_json(record_path, record)
        except torch.cuda.OutOfMemoryError:
            raise
        except (ValueError, FileNotFoundError) as error:
            record = {'status': 'excluded', 'accession': accession, 'reason': str(error)}
            atomic_json(record_path, record)
        records[accession] = record
        atomic_json(directory/'progress.json', {'selected': len(selected), 'processed': len(records),
                    'complete': sum(r['status'] == 'complete' for r in records.values()),
                    'records': records})
        print(f'{index+1}/{len(selected)} {accession} {record["status"]}', flush=True)
    atomic_json(directory/'progress.json', {'selected': len(selected), 'processed': len(records),
                'complete': sum(r['status'] == 'complete' for r in records.values()), 'records': records,
                'terminal': True})
    lock.close()


if __name__ == '__main__':
    main()
