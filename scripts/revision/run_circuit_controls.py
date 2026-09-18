"""Identity-bound generic human-cohort SAE controls, not recovery of historical edges."""
import argparse
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
import torch
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity
from capture_reproducibility import stable_digest
from feature_interventions import feature_intervention

sys.path.insert(0, str(ROOT/'src'))
from sae.model import TopKSAE
from models.interventions import get_layers


def select_cohort(sequences, metadata, clusters, discovery=50, evaluation=100, seed=2288):
    groups = {}
    for p in sorted(sequences):
        if p in clusters and str(metadata.get(p, {}).get('taxid')) == '9606' and 50 <= len(sequences[p]) <= 300:
            groups.setdefault(clusters[p], []).append(p)
    if min(discovery, evaluation) < 1 or discovery+evaluation > len(groups):
        raise ValueError('Insufficient human clusters or nonpositive cohort size')
    rng = np.random.default_rng(seed)
    selected = rng.choice(sorted(groups), discovery+evaluation, replace=False)
    proteins = [str(rng.choice(groups[c])) for c in selected]
    return {'discovery': proteins[:discovery], 'evaluation': proteins[discovery:],
            'clusters': {p: clusters[p] for p in proteins},
            'eligible_proteins': sum(map(len, groups.values())), 'eligible_clusters': len(groups)}


def rank_features(frequencies, count):
    values = np.asarray(frequencies)
    if values.ndim != 1 or not np.isfinite(values).all() or count < 1 or count > len(values):
        raise ValueError('Invalid feature selection')
    selected = np.lexsort((np.arange(len(values)), -values))[:count]
    if (values[selected] <= 0).any():
        raise ValueError('Requested more features than discovery-active dictionary')
    return selected.tolist()


def conditions(features):
    return [('normal', None, []), ('noop', 'residual_preserving', []),
            ('reconstruction', 'reconstruction', [])] + [
            (f'{mode}_{feature}', mode, [feature]) for feature in features
            for mode in ('reconstruction', 'residual_preserving')]


def committed(path):
    meta = path.with_suffix('.json')
    if not meta.exists():
        return None
    record = json.loads(meta.read_text())
    if not path.exists() or stable_digest(path)['sha256'] != record['sha256']:
        raise ValueError(f'Corrupt committed record: {path}')
    return record


def save_arrays(path, arrays, record):
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite measurement')
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    atomic_json(path.with_suffix('.json'), {**record, 'sha256': stable_digest(path)['sha256']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['esm2', 'esm3'], required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--discovery', type=int, default=50)
    parser.add_argument('--evaluation', type=int, default=100)
    parser.add_argument('--upstream', type=int, default=100)
    parser.add_argument('--downstream', type=int, default=50)
    args = parser.parse_args()
    if min(args.upstream, args.downstream) < 1:
        parser.error('Feature counts must be positive')
    directory = args.run_dir or OUT/'circuit_controls'/args.model
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'.writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = [ROOT/'data/eval_expanded/sequences.json', ROOT/'data/eval_expanded/metadata.json', OUT/'clusters.json']
    sequences, metadata, clusters = [json.loads(p.read_text()) for p in paths]
    cohort = select_cohort(sequences, metadata, clusters, args.discovery, args.evaluation)
    pairs = [(16, 24)] if args.model == 'esm2' else [(16, 33), (33, 42)]
    layers = sorted({layer for pair in pairs for layer in pair})
    checkpoint_paths = {layer: ROOT/'models/sae'/f'{args.model}_scaled'/
                        (f'layer_{layer}_topk' if args.model == 'esm2' else f'S_layer_{layer}_topk')/'best.pt'
                        for layer in layers}
    if args.model == 'esm3':
        from esm.utils.constants.esm3 import data_root
        weights = [data_root('esm3')/'data/weights/esm3_sm_open_v1.pth']
    else:
        from huggingface_hub import snapshot_download
        cache = Path(snapshot_download('facebook/esm2_t33_650M_UR50D', local_files_only=True))
        weights = sorted(p for p in cache.iterdir() if p.is_file())
    identity = {'protocol': 1, 'model': args.model, 'seed': 2288, 'cohort': cohort, 'pairs': pairs,
                'upstream_count': args.upstream, 'downstream_count': args.downstream,
                'selection': 'Unweighted mean per-protein residue-positive fraction on discovery only; ties by feature ID',
                'intervention_positions': 'All positions including BOS/EOS, matching historical scope',
                'readout_positions': 'Residues only; unweighted protein mean and positive fraction',
                'scope': 'New human-only cluster-separated controls; no ground-truth biology or recovered historical graph',
                'dtype': 'float32 SAE encode/decode; bfloat16 model autocast for ESM3 only',
                'inputs': [file_identity(p, content=True) for p in paths],
                'checkpoints': {str(k): file_identity(v, content=True) for k, v in checkpoint_paths.items()},
                'weights': [{'path': str(p), **stable_digest(p)} for p in weights],
                'code': [file_identity(p, content=True) for p in [Path(__file__),
                         ROOT/'scripts/revision/feature_interventions.py', ROOT/'src/sae/model.py',
                         ROOT/'src/models/interventions.py', ROOT/f'src/models/{args.model}_hooks.py']],
                'torch': torch.__version__}
    identity = json.loads(json.dumps(identity))
    manifest = directory/'identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Protocol/input identity changed; use a new run directory')
    atomic_json(manifest, identity)
    torch.set_num_threads(2)
    torch.manual_seed(2288)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.model == 'esm3':
        from models.esm3_hooks import load_esm3, tokenize_sequence
        model, tokenizer = load_esm3(device='cuda')
        tokenize = lambda seq: tokenize_sequence(seq, tokenizer)
    else:
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device='cuda')
        tokenize = lambda seq: tokenizer(seq, return_tensors='pt').to('cuda')
    blocks = get_layers(model, args.model)
    saes = {}
    for layer, path in checkpoint_paths.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        sae = TopKSAE(saved['sae_config']).cuda().float().eval()
        sae.load_state_dict(saved['model_state_dict'])
        saes[layer] = sae

    def forward(sequence, read_layers, intervention=None):
        cache, handles = {}, []
        context = intervention if intervention is not None else nullcontext()
        with context:
            try:
                for layer in read_layers:
                    def capture(module, args, output, layer=layer):
                        cache[layer] = (output[0] if isinstance(output, tuple) else output).detach()
                    handles.append(blocks[layer].register_forward_hook(capture))
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.model == 'esm3'):
                    model(**tokenize(sequence))
                result = {}
                with torch.no_grad():
                    for layer in read_layers:
                        h = cache[layer][0, 1:-1].float()
                        if len(h) != len(sequence):
                            raise ValueError('Residue/token alignment mismatch')
                        z = saes[layer].encode(h)
                        result[layer] = (z.mean(0).cpu().numpy(), (z > 0).float().mean(0).cpu().numpy())
                return result
            finally:
                for handle in handles:
                    handle.remove()

    discovery_dir = directory/'discovery'
    discovery_dir.mkdir(exist_ok=True)
    for i, p in enumerate(cohort['discovery']):
        path = discovery_dir/f'{p}.npz'
        if committed(path) is None:
            result = forward(sequences[p], layers)
            arrays = {f'L{layer}_{kind}': result[layer][j] for layer in layers
                      for j, kind in enumerate(['mean', 'positive_fraction'])}
            save_arrays(path, arrays, {'protein': p, 'cluster': clusters[p], 'length': len(sequences[p])})
        print(f'discovery {i+1}/{args.discovery} {p}', flush=True)
    frequency = {}
    for layer in layers:
        rows = []
        for p in cohort['discovery']:
            with np.load(discovery_dir/f'{p}.npz', allow_pickle=False) as data:
                rows.append(data[f'L{layer}_positive_fraction'])
        frequency[layer] = np.mean(rows, axis=0, dtype=np.float64)
    selection = {'discovery_hashes': {p: committed(discovery_dir/f'{p}.npz')['sha256'] for p in cohort['discovery']},
                 'pairs': {f'{up}_{down}': {'upstream': rank_features(frequency[up], args.upstream),
                                          'downstream': rank_features(frequency[down], args.downstream)} for up, down in pairs}}
    selection_path = directory/'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Discovery selection changed')
    atomic_json(selection_path, selection)
    completed = 0
    for up, down in pairs:
        pair = f'{up}_{down}'
        pair_dir = directory/pair
        pair_dir.mkdir(exist_ok=True)
        chosen = selection['pairs'][pair]
        protocol = conditions(chosen['upstream'])
        for p in cohort['evaluation']:
            path = pair_dir/f'{p}.npz'
            if committed(path) is None:
                values, fractions = [], []
                for name, mode, feature in protocol:
                    intervention = feature_intervention(blocks[up], saes[up], feature, mode) if mode else None
                    result = forward(sequences[p], [down], intervention)[down]
                    values.append(result[0][chosen['downstream']])
                    fractions.append(result[1][chosen['downstream']])
                means, fractions = np.asarray(values), np.asarray(fractions)
                if not np.array_equal(means[0], means[1]) or not np.array_equal(fractions[0], fractions[1]):
                    raise ValueError('No-op changed downstream readouts')
                upstream = forward(sequences[p], [up])[up]
                save_arrays(path, {'downstream_mean': means, 'downstream_positive_fraction': fractions,
                                   'upstream_mean': upstream[0][chosen['upstream']],
                                   'upstream_positive_fraction': upstream[1][chosen['upstream']]},
                            {'protein': p, 'cluster': clusters[p], 'length': len(sequences[p]),
                             'pair': pair, 'conditions': [row[0] for row in protocol]})
            completed += 1
            atomic_json(directory/'progress.json', {'complete_pair_proteins': completed,
                        'expected_pair_proteins': len(pairs)*args.evaluation, 'terminal': False})
            print(f'evaluation {pair} {completed}/{len(pairs)*args.evaluation} {p}', flush=True)
    atomic_json(directory/'progress.json', {'complete_pair_proteins': completed,
                'expected_pair_proteins': len(pairs)*args.evaluation, 'terminal': True})


if __name__ == '__main__':
    main()
