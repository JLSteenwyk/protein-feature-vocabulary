"""Diagnose selection-fixed bootstrap bias and rare-feature support loss."""
import json
import numpy as np
from scipy import sparse
from common import OUT, SEED, summaries, standardize, save


def correlation_from_sums(sums, counts, weights):
    n = weights @ counts
    sx, sy, xx, yy, xy = [weights @ z for z in sums]
    denominator = np.sqrt(np.maximum(xx-sx*sx/n, 0) * np.maximum(yy-sy*sy/n, 0))
    return np.divide(xy-sx*sy/n, denominator, out=np.zeros_like(xy), where=denominator > 1e-12), denominator > 1e-12


def main():
    i3, x, _ = summaries('esm3'); i2, y, _ = summaries('esm2')
    ids, a, b = np.intersect1d(i3, i2, return_indices=True)
    x, y = x[a], y[b]
    f3, f2 = np.flatnonzero((x > 0).any(0)), np.flatnonzero((y > 0).any(0))
    x, y = x[:, f3], y[:, f2]
    clusters = json.loads((OUT / 'clusters.json').read_text())
    group = np.array([clusters[p] for p in ids])
    rng = np.random.default_rng(SEED)
    unique = np.unique(group); rng.shuffle(unique)
    discovery = np.isin(group, unique[:len(unique)//2]); validation = ~discovery
    dx, validx = standardize(x[discovery]); dy, validy = standardize(y[discovery])
    sources, targets = np.flatnonzero(validx), np.flatnonzero(validy)
    best = targets[(dx[:, sources].T @ dy[:, targets]).argmax(1)]
    vx, vy = x[validation][:, sources].astype(float), y[validation][:, best].astype(float)
    groups, gi = np.unique(group[validation], return_inverse=True)
    membership = sparse.csr_matrix((np.ones(len(gi)), (gi, np.arange(len(gi)))), shape=(len(groups), len(gi)))
    sums = [membership @ z for z in [vx, vy, vx*vx, vy*vy, vx*vy]]
    counts = np.bincount(gi)
    original, valid = correlation_from_sums(sums, counts, np.ones(len(groups)))
    original_pass = original > .3
    support = np.minimum((membership @ (vx > 0)).astype(bool).sum(0),
                         (membership @ (vy > 0)).astype(bool).sum(0))
    supported = support >= 10
    draws, losses, flips_down, flips_up, supported_draws = [], [], [], [], []
    for _ in range(500):
        w = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        r, rv = correlation_from_sums(sums, counts, w)
        passed = r > .3
        draws.append(float(passed.mean()))
        losses.append(int((original_pass & ~rv).sum()))
        flips_down.append(int((original_pass & rv & ~passed).sum()))
        flips_up.append(int((~original_pass & passed).sum()))
        supported_draws.append(float(passed[supported].mean()))
    old = json.loads((OUT / 'matching.json').read_text())['heldout']
    assert abs(original_pass.mean() - old['many_to_one']['fractions']['0.3']) < 1 / len(sources)
    np.savez_compressed(OUT / 'matching_discovery_pairs.npz', source_ids=f3[sources], target_ids=f2[best],
                        discovery_protein_ids=ids[discovery], validation_protein_ids=ids[validation],
                        validation_r=original, validation_variable=valid, minimum_active_clusters=support)
    result = {'seed': SEED, 'n_resamples': 500, 'original_fraction': float(original_pass.mean()),
              'bootstrap_mean': float(np.mean(draws)), 'bootstrap_percentiles': np.quantile(draws, [.025, .975]).tolist(),
              'original_n_pass': int(original_pass.sum()),
              'mean_original_pass_lost_to_constant': float(np.mean(losses)),
              'mean_original_pass_crossing_below_threshold_while_variable': float(np.mean(flips_down)),
              'mean_original_fail_crossing_above_threshold': float(np.mean(flips_up)),
              'n_pairs_minimum_ten_active_validation_clusters': int(supported.sum()),
              'supported_original_fraction': float(original_pass[supported].mean()),
              'supported_bootstrap_percentiles': np.quantile(supported_draws, [.025, .975]).tolist(),
              'minimum_active_cluster_distribution': {str(k): int((support == k).sum()) for k in [0, 1, 2, 3, 4, 5]},
              'interpretation': 'Direct decomposition of resampling changes into undefined correlations and threshold crossings. '
                                'The >=10-active-cluster restriction is post hoc diagnostic, not a replacement primary population. '
                                'No bias-corrected interval is claimed.'}
    save('matching_bootstrap_diagnostic.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
