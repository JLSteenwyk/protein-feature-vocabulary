"""Resumable human-only paired steering controls with full residue readouts."""
import argparse
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
import torch
from common import ROOT, OUT, RESULTS
from probe_checkpoint import atomic_json, file_identity
from capture_reproducibility import stable_digest
from run_circuit_controls import select_cohort, committed, save_arrays
from steering_controls import make_directions, condition_order, steering_intervention

sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=OUT/'steering_controls')
    parser.add_argument('--discovery', type=int, default=50)
    parser.add_argument('--evaluation', type=int, default=100)
    args = parser.parse_args()
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'.writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = {'sequences': ROOT/'data/eval_expanded/sequences.json',
             'metadata': ROOT/'data/eval_expanded/metadata.json', 'clusters': OUT/'clusters.json',
             'categories': OUT/'cross_modal_complete.json',
             'archived_categories': RESULTS/'cross_modal_features.json',
             'checkpoint': ROOT/'models/sae_1.5M/esm3_residue_ef8_k64/best.pt'}
    inputs = {name: file_identity(path, content=True) for name, path in paths.items()}
    sequences, metadata, clusters, categories, archived = [json.loads(paths[key].read_text())
        for key in ('sequences', 'metadata', 'clusters', 'categories', 'archived_categories')]
    if categories['checkpoint_sha256'] != inputs['checkpoint']['sha256']:
        raise ValueError('Categories and SAE checkpoint differ')
    cohort = select_cohort(sequences, metadata, clusters, args.discovery, args.evaluation)
    torch.set_num_threads(2)
    torch.manual_seed(2288)
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
    target_ids = archived['enhanced_feature_ids'][:20]
    category_map = {row['feature_id']: row['category'] for row in categories['per_feature']}
    vectors, vector_metadata = make_directions(saved['model_state_dict']['decoder.weight'].numpy(),
                                               target_ids, category_map)
    monitored = sorted({i for row in vector_metadata.values() for i in row['feature_ids']})
    protocol = condition_order(vectors)
    from esm.utils.constants.esm3 import data_root
    weights = data_root('esm3')/'data/weights/esm3_sm_open_v1.pth'
    identity = {'protocol': 1, 'cohort': cohort, 'seed': 2288, 'layer': 33,
                'scope': 'Fixed archived direction; generic human-protein model readouts only',
                'inputs': inputs, 'weights': {'path': str(weights), **stable_digest(weights)},
                'code': [file_identity(p, content=True) for p in [Path(__file__),
                    ROOT/'scripts/revision/steering_controls.py', ROOT/'scripts/revision/run_circuit_controls.py',
                    ROOT/'src/sae/model.py', ROOT/'src/models/esm3_hooks.py']],
                'directions': vector_metadata, 'conditions': protocol, 'monitored_feature_ids': monitored,
                'decoder_pool': 'Corrected unclassified coordinates excluding all target IDs; not prevalence matched',
                'positions': 'Intervene all tokens including BOS/EOS; export residues only',
                'precision': 'bfloat16 model autocast; float32 direction addition and SAE encoding',
                'torch': torch.__version__}
    identity = json.loads(json.dumps(identity))
    identity_path = directory/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Input/protocol identity changed; use a new directory')
    atomic_json(identity_path, identity)
    direction_path = directory/'directions.npz'
    if committed(direction_path) is None:
        save_arrays(direction_path, vectors, {'direction_names': list(vectors)})
    with np.load(direction_path, allow_pickle=False) as data:
        if set(data.files) != set(vectors) or any(not np.array_equal(data[k], v) for k, v in vectors.items()):
            raise ValueError('Direction reconstruction mismatch')
    from models.esm3_hooks import load_esm3, tokenize_sequence
    from sae.model import build_sae
    model, tokenizer = load_esm3(device='cuda')
    sae = build_sae(saved['sae_config']).cuda().float().eval()
    sae.load_state_dict(saved['model_state_dict'])
    block = model.transformer.blocks[33]
    gpu_vectors = {name: torch.as_tensor(value, device='cuda') for name, value in vectors.items()}

    def forward(sequence, direction, alpha):
        cache = {}
        context = steering_intervention(block, gpu_vectors[direction], alpha) if direction else nullcontext()
        with context:
            def capture(module, args, output):
                cache['hidden'] = (output[0] if isinstance(output, tuple) else output).detach()
            handle = block.register_forward_hook(capture)
            try:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(**tokenize_sequence(sequence, tokenizer))
                with torch.no_grad():
                    h = cache['hidden'][0, 1:-1].float()
                    logits = output.sequence_logits[0, 1:-1].float()
                    if h.shape[0] != len(sequence) or logits.shape[0] != len(sequence):
                        raise ValueError('Token/residue alignment mismatch')
                    z = sae.encode(h)[:, monitored]
                return {'sequence_logits': logits.cpu().numpy(), 'monitored_activations': z.cpu().numpy()}
            finally:
                handle.remove()

    expected = len(cohort['evaluation'])*len(protocol)
    complete = 0
    for protein in cohort['evaluation']:
        folder = directory/'records'/protein
        folder.mkdir(parents=True, exist_ok=True)
        for name, direction, alpha in protocol:
            path = folder/f'{name}.npz'
            if committed(path) is None:
                try:
                    arrays = forward(sequences[protein], direction, alpha)
                    if name != 'normal' and alpha == 0:
                        with np.load(folder/'normal.npz', allow_pickle=False) as baseline:
                            if any(not np.array_equal(value, baseline[k]) for k, value in arrays.items()):
                                raise ValueError('No-op/restoration mismatch')
                    save_arrays(path, arrays, {'protein': protein, 'cluster': clusters[protein],
                        'length': len(sequences[protein]), 'condition': [name, direction, alpha],
                        'monitored_feature_ids': monitored})
                except Exception as error:
                    atomic_json(folder/f'{name}.failure.json', {'protein': protein, 'condition': name,
                        'error': repr(error), 'committed': False, 'policy': 'Fail-fast; no silent exclusion'})
                    raise
            complete += 1
            atomic_json(directory/'progress.json', {'complete_conditions': complete,
                        'expected_conditions': expected, 'terminal': False})
        print(f'evaluation {complete//len(protocol)}/{len(cohort["evaluation"])} {protein}', flush=True)
    atomic_json(directory/'progress.json', {'complete_conditions': complete,
                'expected_conditions': expected, 'terminal': True})


if __name__ == '__main__':
    main()
