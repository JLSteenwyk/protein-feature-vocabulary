"""Check paired samples, saved sufficient statistics, and reconstruction summaries."""
import json
import numpy as np
from common import ROOT, OUT, SEED, summaries
from probe_checkpoint import atomic_json, file_identity
from recompute_reconstruction import reconstruction_r2


def main():
    result_path = OUT/'reconstruction_resampled.json'
    result = json.loads(result_path.read_text())
    if set(result['models']) != {'esm3', 'esm2'}:
        raise ValueError('Reconstruction run incomplete')
    ids, _, offsets = summaries('esm3')
    expected = np.sort(np.random.default_rng(SEED).choice(int(offsets[:, 1].sum()), result['n'], replace=False))
    protein = np.searchsorted(offsets[:, 0], expected, side='right')-1
    clusters = json.loads((OUT/'clusters.json').read_text())
    reference = None
    checks = {}
    for model, record in result['models'].items():
        path = ROOT/record['records']['path']
        assert file_identity(path, content=True)['sha256'] == record['records']['sha256']
        with np.load(path) as saved:
            assert np.array_equal(saved['canonical_rows'], expected)
            assert np.array_equal(saved['protein'], ids[protein])
            assert np.array_equal(saved['position'], expected-offsets[protein, 0])
            assert np.array_equal(saved['cluster_names'][saved['cluster_index']], [clusters[p] for p in ids[protein]])
            if reference is None:
                reference = {k: saved[k] for k in ['protein', 'position', 'cluster_index', 'cluster_names']}
            for key, value in reference.items():
                assert np.array_equal(value, saved[key])
            gi = saved['cluster_index']
            assert np.array_equal(np.bincount(gi), saved['cluster_counts'])
            assert np.allclose(np.bincount(gi, weights=saved['sse']), saved['cluster_sse'])
            assert np.allclose(np.bincount(gi, weights=saved['cosine']), saved['cluster_sum_cosine'])
            r2 = reconstruction_r2(saved['sse'].sum(), saved['cluster_sum_x'].sum(0), saved['cluster_sum_x2'].sum(), len(expected))
            assert np.isclose(r2, record['global_centered_r2'], atol=1e-12, rtol=0)
            assert np.isclose(saved['cosine'].mean(), record['mean_cosine'], atol=1e-12, rtol=0)
            assert np.isclose(saved['within_vector_variance_ratio'].mean(), record['mean_within_vector_variance_ratio'])
            assert int((saved['l0'] < 64).sum()) == record['n_below_k']
            assert len(np.unique(protein)) == result['n_sampled_proteins']
            checks[model] = {'n': len(expected), 'r2_reproduced': r2, 'paired_sample_verified': True}
    atomic_json(OUT/'reconstruction_verification.json', {'source': file_identity(result_path, content=True),
        'checks': checks, 'scope': 'Sample/cluster identity and exported-statistic arithmetic; not an independent re-encoding or training-provenance certificate.'})
    print('Matched reconstruction sampling and sufficient-statistic checks passed.')


if __name__ == '__main__':
    main()
