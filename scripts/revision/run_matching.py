"""Full one-to-one matching, discovery/validation split, and search-adjusted null."""
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import OUT, SEED, save, summaries, standardize


def describe(r, denominator=None):
    r = np.asarray(r)
    finite = np.isfinite(r)
    denom = len(r) if denominator is None else denominator
    return {'n_pairs': len(r), 'n_valid': int(finite.sum()),
            'median_r': float(np.median(r[finite])) if finite.any() else None,
            'fractions': {str(t): float(np.sum(r[finite] > t)/denom)
                          for t in [.1, .2, .3, .4, .5, .6, .7, .8, .9]}}


def matched_r(x, y, rows, cols):
    a, av = standardize(x[:, rows]); b, bv = standardize(y[:, cols])
    r = np.einsum('ij,ij->j', a, b)
    r[~(av & bv)] = np.nan
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--permutations', type=int, default=99)
    parser.add_argument('--null-features', type=int, default=512)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    i3, x, _ = summaries('esm3'); i2, y, _ = summaries('esm2')
    ids, a, b = np.intersect1d(i3, i2, return_indices=True)
    x, y = x[a], y[b]
    f3 = np.flatnonzero((x > 0).any(0)); f2 = np.flatnonzero((y > 0).any(0))
    x, y = x[:, f3], y[:, f2]
    xn, _ = standardize(x); yn, _ = standardize(y)
    print('Full correlation matrix', x.shape, y.shape, flush=True)
    corr = xn.T @ yn
    best = corr.argmax(1); reverse = corr.argmax(0)
    rows, cols = linear_sum_assignment(corr, maximize=True)
    mutual = reverse[best] == np.arange(len(best))
    out = {'seed': SEED, 'n_proteins': len(ids), 'n_esm3': len(f3), 'n_esm2': len(f2),
           'threshold_rule': 'strictly greater than',
           'many_to_one': describe(corr.max(1)), 'reverse_many_to_one': describe(corr.max(0)),
           'one_to_one': describe(corr[rows, cols], len(f3)),
           'mutual_nearest': describe(corr[np.flatnonzero(mutual), best[mutual]], len(f3)),
           'n_distinct_targets_many_to_one': int(len(np.unique(best)))}
    np.savez_compressed(OUT/'matching_pairs.npz', source_ids=f3, target_ids=f2,
                        best_target=f2[best], one_to_one_source=f3[rows],
                        one_to_one_target=f2[cols], one_to_one_r=corr[rows, cols])
    save('matching.json', out)
    print('Full matching', out, flush=True)
    # Hold out sequence clusters if available; otherwise fail rather than silently downgrade.
    clusters = __import__('json').loads((OUT/'clusters.json').read_text())
    group = np.array([clusters[p] for p in ids])
    u = np.unique(group); rng.shuffle(u)
    discovery = np.isin(group, u[:len(u)//2]); validation = ~discovery
    dn3, v3 = standardize(x[discovery]); dn2, v2 = standardize(y[discovery])
    dr = np.flatnonzero(v3); dc = np.flatnonzero(v2)
    c = dn3[:, dr].T @ dn2[:, dc]
    mb = dc[c.argmax(1)]
    ar, ac = linear_sum_assignment(c, maximize=True)
    vr = matched_r(x[validation], y[validation], dr, mb)
    vo = matched_r(x[validation], y[validation], dr[ar], dc[ac])
    out['heldout'] = {'n_discovery_proteins': int(discovery.sum()),
                      'n_validation_proteins': int(validation.sum()),
                      'n_discovery_clusters': len(u)//2,
                      'n_validation_clusters': len(u)-len(u)//2,
                      'n_source_discovery_variable': len(dr), 'n_target_discovery_variable': len(dc),
                      'many_to_one': describe(vr), 'one_to_one': describe(vo),
                      'interpretation': 'Pairs selected on discovery clusters, frozen on validation; constant validation pairs count as failures in fractions.'}
    # Cluster bootstrap of held-out correlations for frozen pairs, retaining all source features.
    vx = x[validation][:, dr].astype(np.float64)
    vy = y[validation][:, mb].astype(np.float64)
    ug, gi = np.unique(group[validation], return_inverse=True)
    from scipy import sparse
    membership = sparse.csr_matrix((np.ones(len(gi)), (gi, np.arange(len(gi)))), shape=(len(ug), len(gi)))
    sums = [membership @ z for z in [vx, vy, vx*vx, vy*vy, vx*vy]]
    counts = np.bincount(gi)
    bs = []
    for _ in range(500):
        w = rng.multinomial(len(ug), np.full(len(ug), 1/len(ug)))
        n = w @ counts
        sx, sy, xx, yy, xy = [w @ z for z in sums]
        denom = np.sqrt(np.maximum(xx-sx*sx/n, 0)*np.maximum(yy-sy*sy/n, 0))
        r = np.divide(xy-sx*sy/n, denom, out=np.zeros_like(xy), where=denom > 1e-12)
        bs.append(np.mean(r > .3))
    out['heldout']['many_to_one_fraction_gt_0.3_ci95'] = {
        'low': float(np.quantile(bs,.025)), 'high': float(np.quantile(bs,.975)),
        'n_resamples':500, 'unit':'validation_sequence_cluster',
        'scope':'conditional on frozen discovery-selected pairs and fitted SAEs'}
    save('matching.json', out)
    print('Held-out matching complete', out['heldout'], flush=True)
    del vx, vy, sums, c, dn3, dn2
    sample = np.sort(rng.choice(len(f3), min(args.null_features,len(f3)),replace=False))
    obs = describe(corr[sample].max(1))
    null = []
    for k in range(args.permutations):
        p = rng.permutation(len(ids))
        nr = (xn[p][:,sample].T @ yn).max(1)
        null.append(describe(nr))
        if (k+1)%10 == 0: print('Permutation',k+1,flush=True)
    observed = obs['fractions']['0.3']
    stats = [z['fractions']['0.3'] for z in null]
    out['search_adjusted_null'] = {
        'n_permutations':args.permutations, 'n_source_features':len(sample),
        'n_target_features':len(f2), 'source_feature_ids':f3[sample].tolist(),
        'observed_same_subset':obs, 'null_per_permutation':null,
        'mean_null_fraction_gt_0.3':float(np.mean(stats)),
        'p_empirical':float((1+np.sum(np.array(stats)>=observed))/(1+len(stats))),
        'caveat':'Whole protein rows permuted; inter-feature dependence retained, homology exchangeability not assumed. Descriptive sensitivity, not family-conditioned test.'}
    save('matching.json',out)
    print('Matching complete',flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    main()
