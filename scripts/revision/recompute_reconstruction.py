"""Matched-residue reconstruction check with global centered R2 and cluster ranges."""
import argparse
import fcntl
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy import sparse
import torch
from common import ROOT, OUT, RESULTS, SEED, summaries
from probe_checkpoint import atomic_json, file_identity

sys.path.insert(0, str(ROOT/'src'))
from sae.model import build_sae


def reconstruction_r2(sse, sum_x, sum_x2, count):
    denominator = float(sum_x2 - np.square(sum_x).sum()/count)
    if denominator <= 0:
        raise ValueError('Centered reconstruction denominator is nonpositive')
    return float(1-sse/denominator)


def read_sample(paths, ids, lengths, sample_protein, positions, width):
    wanted = {pid: np.flatnonzero(sample_protein == i) for i, pid in enumerate(ids)}
    expected_lengths = dict(zip(ids, lengths))
    x = np.empty((len(positions), width), dtype=np.float32)
    seen, covered = set(), np.zeros(len(positions), dtype=bool)
    for path in paths:
        with h5py.File(path) as handle:
            pids = [v.decode() for v in handle['ids'][:]]
            offsets = handle['offsets'][:]
            if (len(offsets) != len(pids) or
                    not np.array_equal(offsets[:, 0], np.arange(len(pids))) or
                    not np.array_equal(offsets[:, 1], np.r_[0, np.cumsum(offsets[:-1, 2])]) or
                    offsets[:, 2].sum() != handle['activations'].shape[0]):
                raise ValueError('Noncontiguous activation offsets')
            local, target = [], []
            for pid, (index, start, length) in zip(pids, offsets):
                if pid in seen or pid not in expected_lengths or length != expected_lengths[pid]:
                    raise ValueError('Activation protein/length mismatch')
                seen.add(pid)
                rows = wanted[pid]
                local.extend((start+positions[rows]).tolist()); target.extend(rows.tolist())
            order = np.argsort(local)
            local, target = np.asarray(local, dtype=int)[order], np.asarray(target, dtype=int)[order]
            for begin in range(0, len(local), 512):
                idx = target[begin:begin+512]
                x[idx] = handle['activations'][local[begin:begin+512]]
                covered[idx] = True
    if seen != set(ids) or not covered.all() or not np.isfinite(x).all():
        raise ValueError('Incomplete activation sample')
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=50000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    directory = OUT/'reconstruction_resampled'
    directory.mkdir(exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = {model: sorted((RESULTS/f'activations/{model}/residue_L33').glob('chunk*.h5')) for model in ['esm3', 'esm2']}
        checkpoints = {model: ROOT/f'models/sae_1.5M/{model}_residue_ef8_k64/best.pt' for model in paths}
        identity = {'n': args.n, 'seed': SEED, 'bootstrap': args.bootstrap,
                    'files': [file_identity(p, content=True) for p in [Path(__file__), Path(__file__).with_name('common.py'), ROOT/'src/sae/model.py', OUT/'clusters.json', *checkpoints.values()]],
                    'activation_files': [file_identity(p) for group in paths.values() for p in group],
                    'canonical_summary': file_identity(RESULTS/'sae_features/esm3/residue/protein_summaries.h5')}
        identity_path = directory/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Reconstruction inputs/protocol changed')
        atomic_json(identity_path, identity)
        ids, _, offsets = summaries('esm3')
        rows = np.sort(np.random.default_rng(SEED).choice(int(offsets[:, 1].sum()), args.n, replace=False))
        sample_protein = np.searchsorted(offsets[:, 0], rows, side='right')-1
        positions = rows-offsets[sample_protein, 0]
        clusters = json.loads((OUT/'clusters.json').read_text())
        group_names, gi = np.unique([clusters[pid] for pid in ids[sample_protein]], return_inverse=True)
        membership = sparse.csr_matrix((np.ones(args.n), (gi, np.arange(args.n))), shape=(len(group_names), args.n))
        counts = np.bincount(gi)
        output_path = OUT/'reconstruction_resampled.json'
        result = json.loads(output_path.read_text()) if output_path.exists() else {
            'sampling': 'Uniform without replacement over all canonical evaluation residues; exact protein-ID/position joins across models.',
            'n': args.n, 'seed': SEED, 'n_sampled_proteins': len(np.unique(sample_protein)), 'n_clusters': len(group_names),
            'scope': 'Fixed fitted SAE dictionaries; residue-weighted sample; cluster percentile ranges conditional on sampled residues, not SAE-training variability or proof of training independence.',
            'models': {}}
        for model in paths:
            if model in result['models']:
                record = result['models'][model]
                if file_identity(ROOT/record['records']['path'], content=True)['sha256'] != record['records']['sha256']:
                    raise ValueError('Saved reconstruction records changed')
                print(model, 'verified completed result', flush=True)
                continue
            checkpoint = torch.load(checkpoints[model], map_location='cpu', weights_only=False)
            sae = build_sae(checkpoint['sae_config']); sae.load_state_dict(checkpoint['model_state_dict']); sae.eval()
            print(model, 'reading matched sample', flush=True)
            x = read_sample(paths[model], ids, offsets[:, 1], sample_protein, positions, sae.encoder.weight.shape[1])
            cosine, sse, within_ratio, l0 = [np.empty(args.n) for _ in range(4)]
            for start in range(0, args.n, 256):
                end = min(start+256, args.n)
                batch = torch.from_numpy(x[start:end])
                with torch.no_grad():
                    z = sae.encode(batch); predicted = sae.decode(z)
                error = (batch-predicted).double()
                sse[start:end] = error.square().sum(-1).numpy()
                cosine[start:end] = torch.nn.functional.cosine_similarity(batch, predicted, dim=-1).numpy()
                within_ratio[start:end] = (1-error.var(-1)/(batch.double().var(-1)+1e-10)).numpy()
                l0[start:end] = (z > 0).sum(-1).numpy()
                if start % 10000 < 256:
                    print(model, end, '/', args.n, flush=True)
            xd = x.astype(np.float64)
            sums = membership @ xd
            squares = membership @ np.square(xd).sum(1)
            errors = membership @ sse
            cos_sums = membership @ cosine
            r2 = reconstruction_r2(errors.sum(), sums.sum(0), squares.sum(), args.n)
            boot = []
            rng = np.random.default_rng(SEED)
            for _ in range(args.bootstrap):
                w = rng.multinomial(len(counts), np.full(len(counts), 1/len(counts)))
                n = w @ counts
                boot.append([reconstruction_r2(w @ errors, w @ sums, w @ squares, n), w @ cos_sums/n])
            bounds = np.quantile(boot, [.025, .975], axis=0)
            target = directory/f'{model}.npz'
            temporary = target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, canonical_rows=rows, protein=ids[sample_protein], position=positions,
                                cluster_index=gi, cluster_names=group_names, cosine=cosine, sse=sse,
                                within_vector_variance_ratio=within_ratio, l0=l0,
                                cluster_counts=counts, cluster_sum_x=sums, cluster_sum_x2=squares,
                                cluster_sse=errors, cluster_sum_cosine=cos_sums)
            temporary.replace(target)
            record = file_identity(target, content=True); record['path'] = str(target.relative_to(ROOT))
            result['models'][model] = {'global_centered_r2': r2, 'r2_cluster_range95': bounds[:, 0].tolist(),
                'mean_cosine': float(cosine.mean()), 'cosine_cluster_range95': bounds[:, 1].tolist(),
                'mean_within_vector_variance_ratio': float(within_ratio.mean()),
                'mean_mse': float(sse.mean()/x.shape[1]), 'n_below_k': int((l0 < 64).sum()),
                'min_l0': int(l0.min()), 'max_l0': int(l0.max()), 'records': record}
            atomic_json(output_path, result)
            print(model, result['models'][model], flush=True)
            del x, xd, sae, checkpoint, sums


if __name__ == '__main__':
    main()
