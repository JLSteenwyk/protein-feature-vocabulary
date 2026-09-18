"""Resumable CPU re-encoding and complete paired cross-modal feature statistics."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy import sparse
from scipy.stats import wilcoxon
import torch
from common import ROOT, RESULTS, OUT, SEED, save
from run_go_audit import bh

sys.path.insert(0, str(ROOT / 'src'))
from sae.model import build_sae


def aligned_chunk_metadata(left, right):
    ids_left = np.array([v.decode() for v in left['ids'][:]])
    ids_right = np.array([v.decode() for v in right['ids'][:]])
    offsets_left, offsets_right = left['offsets'][:], right['offsets'][:]
    if not np.array_equal(ids_left, ids_right):
        raise ValueError('Paired chunk protein IDs/order differ')
    if not np.array_equal(offsets_left, offsets_right):
        raise ValueError('Paired chunk residue offsets differ; no silent truncation allowed')
    if len(np.unique(ids_left)) != len(ids_left):
        raise ValueError('Duplicate protein IDs in chunk')
    if not np.array_equal(offsets_left[:, 0], np.arange(len(ids_left))):
        raise ValueError('Offset protein indices are not contiguous')
    if not np.array_equal(offsets_left[:, 1], np.r_[0, np.cumsum(offsets_left[:-1, 2])]):
        raise ValueError('Residue offsets are not contiguous')
    nrows = int(offsets_left[:, 2].sum())
    if any(f['activations'].shape[0] != nrows for f in [left, right]):
        raise ValueError('Offset lengths do not cover activations')
    return ids_left, offsets_left


def protein_means(handle, offsets, sae, batch_size):
    width = sae.encoder.weight.shape[0]
    total = torch.zeros((len(offsets), width), dtype=torch.float64)
    dataset = handle['activations']
    for start in range(0, dataset.shape[0], batch_size):
        end = min(start+batch_size, dataset.shape[0])
        positions = np.searchsorted(offsets[:, 1], np.arange(start, end), side='right')-1
        with torch.no_grad():
            z = sae.encode(torch.from_numpy(dataset[start:end].astype(np.float32)))
        total.index_add_(0, torch.from_numpy(positions), z.double())
    return (total.numpy()/offsets[:, 2, None]).astype(np.float32)


def classify(delta, active):
    width = delta.shape[1]
    n_nonzero = np.count_nonzero(delta, axis=0)
    mean, sd = delta.mean(0), delta.std(0, ddof=0)
    effect = np.divide(mean, sd, out=np.zeros(width), where=sd > 0)
    p = np.ones(width)
    eligible = active & (n_nonzero >= 10)
    for fid in np.flatnonzero(eligible):
        values = delta[:, fid]
        p[fid] = wilcoxon(values[values != 0], zero_method='wilcox', alternative='two-sided', method='auto').pvalue
    q = np.ones(width)
    q[active] = bh(p[active])
    category = np.full(width, 'inactive', dtype='<U16')
    category[active] = 'unclassified'
    category[active & (q < .05) & (effect > .2)] = 'enhanced'
    category[active & (q < .05) & (effect < -.2)] = 'suppressed'
    return n_nonzero, mean, effect, p, q, category, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cache = OUT / 'cross_modal_cache'
    cache.mkdir(exist_ok=True)
    checkpoint_path = ROOT / 'models/sae_1.5M/esm3_residue_ef8_k64/best.pt'
    checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sae = build_sae(checkpoint['sae_config'])
    sae.load_state_dict(checkpoint['model_state_dict']); sae.eval()
    source = RESULTS / 'cross_modal'
    left = {p.name: p for p in (source/'S_only/residue_L33').glob('*.h5')}
    right = {p.name: p for p in (source/'S_St/residue_L33').glob('*.h5')}
    if set(left) != set(right):
        raise ValueError('Paired conditions have different chunk filenames')
    ids, means_s, means_sst = [], [], []
    for name in sorted(left):
        destination = cache / (Path(name).stem+'.npz')
        fingerprint = json.dumps({'checkpoint_sha256': checksum, 'batch_size': args.batch_size,
                                  'encoding': 'float32; float64 protein sums; float32 stored means',
                                  'files': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in [left[name], right[name]]]}, sort_keys=True)
        if destination.exists():
            with np.load(destination) as cached:
                if str(cached['fingerprint']) != fingerprint:
                    raise ValueError(f'Stale cache {destination}; investigate before regenerating')
                ids.append(cached['ids']); means_s.append(cached['s']); means_sst.append(cached['sst'])
            print('Cached', name, flush=True)
            continue
        with h5py.File(left[name]) as a, h5py.File(right[name]) as b:
            pids, offsets = aligned_chunk_metadata(a, b)
            s, sst = protein_means(a, offsets, sae, args.batch_size), protein_means(b, offsets, sae, args.batch_size)
        temporary = destination.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, ids=pids, s=s, sst=sst, fingerprint=fingerprint)
        temporary.replace(destination)
        ids.append(pids); means_s.append(s); means_sst.append(sst)
        print('Encoded', name, len(pids), flush=True)
    ids = np.concatenate(ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate IDs across chunks')
    s, sst = np.concatenate(means_s), np.concatenate(means_sst)
    delta = sst.astype(float)-s.astype(float)
    active = (s > 0).any(0) | (sst > 0).any(0)
    nz, mean, effect, p, q, category, eligible = classify(delta, active)
    clusters = json.loads((OUT/'clusters.json').read_text())
    ug, gi = np.unique([clusters[pid] for pid in ids], return_inverse=True)
    member = sparse.csr_matrix((np.ones(len(gi)), (gi, np.arange(len(gi)))), shape=(len(ug), len(gi)))
    sums, squares = member @ delta, member @ (delta*delta)
    counts = np.bincount(gi)
    rng = np.random.default_rng(SEED)
    boot = np.empty((args.bootstrap, delta.shape[1]))
    for i in range(args.bootstrap):
        w = rng.multinomial(len(ug), np.full(len(ug), 1/len(ug)))
        n = w @ counts
        mu = w @ sums / n
        sd = np.sqrt(np.maximum(w @ squares/n-mu*mu, 0))
        boot[i] = np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)
    low, high = np.quantile(boot, [.025, .975], axis=0)
    records = [{'feature_id': int(fid), 'active': bool(active[fid]), 'test_eligible': bool(eligible[fid]),
                'n_nonzero_differences': int(nz[fid]), 'mean_delta': float(mean[fid]),
                'paired_standardized_difference_ddof0': float(effect[fid]), 'p': float(p[fid]), 'q': float(q[fid]),
                'category': str(category[fid]), 'effect_cluster_percentile_range95': [float(low[fid]), float(high[fid])]}
               for fid in range(delta.shape[1])]
    output = {'n_proteins': len(ids), 'n_clusters': len(ug), 'dictionary_width': delta.shape[1],
              'n_active': int(active.sum()), 'n_test_eligible': int(eligible.sum()),
              'category_counts': {str(c): int((category == c).sum()) for c in np.unique(category)},
              'feature_id_convention': 'zero based', 'aggregation': 'mean of residue SAE activations per protein',
              'test': 'two-sided Wilcoxon on nonzero paired differences, >=10 required',
              'correction': 'BH across all active-in-either-condition features; ineligible tests assigned p=1',
              'uncertainty': {'n_resamples': args.bootstrap, 'unit': 'sequence_cluster', 'seed': SEED,
                              'scope': 'conditional descriptive percentile ranges of standardized differences; '
                                       'zero bootstrap variance assigned effect 0; no category equivalence claim'},
              'checkpoint_sha256': checksum, 'per_feature': records}
    save('cross_modal_complete.json', output)
    np.savez_compressed(OUT/'cross_modal_protein_means.npz', ids=ids, s=s, sst=sst)
    print('Complete', output['category_counts'], flush=True)


if __name__ == '__main__':
    main()
