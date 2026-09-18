"""Discovery-only matching sensitivities for aggregation, breadth and training overlap."""
import json
import fcntl
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import OUT, RESULTS, summaries, standardize
from run_matching import describe, matched_r
from probe_checkpoint import atomic_json, file_identity


def breadth_best(correlation, source_counts, target_counts, minimum=10, ratio=2.):
    """Return -1 when no variable target has comparable discovery activation breadth."""
    source_counts, target_counts = np.asarray(source_counts), np.asarray(target_counts)
    allowed = ((source_counts[:, None] >= minimum) & (target_counts[None, :] >= minimum)
               & (target_counts[None, :] >= source_counts[:, None]/ratio)
               & (target_counts[None, :] <= source_counts[:, None]*ratio))
    count = allowed.sum(1)
    best = np.where(allowed, correlation, -np.inf).argmax(1)
    best[count == 0] = -1
    return best, count


def union_support_r(x, y, source, target):
    """Conditional sensitivity excluding jointly zero validation proteins per pair."""
    result = np.full(len(source), np.nan)
    support = np.zeros(len(source), dtype=int)
    for start in range(0, len(source), 128):
        a = x[:, source[start:start+128]].astype(float)
        b = y[:, target[start:start+128]].astype(float)
        keep = (a != 0) | (b != 0)
        n = keep.sum(0)
        support[start:start+128] = n
        denom_n = np.maximum(n, 1)
        # Excluded positions are zero in both vectors by construction.
        sx, sy = a.sum(0), b.sum(0)
        vx = np.maximum((a*a).sum(0)-sx*sx/denom_n, 0)
        vy = np.maximum((b*b).sum(0)-sy*sy/denom_n, 0)
        denominator = np.sqrt(vx*vy)
        good = (n >= 3) & (denominator > 0)
        result[start:start+128] = np.divide((a*b).sum(0)-sx*sy/denom_n,
                                           denominator, out=np.full(len(n), np.nan), where=good)
    return result, support


def main():
    directory = OUT/'matching_sensitivity'
    directory.mkdir(exist_ok=True)
    lock = (directory/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_paths = [RESULTS/'sae_features'/m/'residue/protein_summaries.h5' for m in ['esm3', 'esm2']]
    small_paths = [OUT/'clusters.json', OUT/'training_homology_audit.json', Path(__file__),
                   Path(__file__).with_name('common.py'), Path(__file__).with_name('run_matching.py')]
    identity = {'protocol': 1, 'seeds': [2288, 2289, 2290], 'breadth_minimum': 10, 'breadth_ratio': 2.,
                'large_inputs': [file_identity(p) for p in source_paths],
                'small_inputs': [file_identity(p, content=True) for p in small_paths]}
    manifest = directory/'identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Inputs or implementation changed; do not reuse stale sensitivities.')
    atomic_json(manifest, identity)
    clusters = json.loads((OUT/'clusters.json').read_text())
    excluded = json.loads((OUT/'training_homology_audit.json').read_text())['evaluation_ids_identity50']
    output = {'scope': 'Fixed SAE dictionaries; discovery-only selection; three splits are sensitivity repeats, not independent replicates.',
              'exclusion': 'All detected >=50%-identity training-candidate hits, without coverage restriction',
              'breadth_rule': 'Both features active on >=10 discovery proteins; target count within factor two of source count.',
              'validation': 'Frozen pairs; constant or unavailable pairs count as failures. Union-support correlations exclude jointly zero validation proteins and change the estimand.',
              'runs': {}}
    for field in ['protein_max_activations', 'protein_mean_activations']:
        i3, x, _ = summaries('esm3', field); i2, y, _ = summaries('esm2', field)
        ids, a, b = np.intersect1d(i3, i2, return_indices=True)
        x, y = x[a], y[b]
        groups = np.array([clusters[p] for p in ids])
        for cohort in ['all', 'training_homologs_excluded']:
            keep = np.ones(len(ids), dtype=bool) if cohort == 'all' else ~np.isin(ids, excluded)
            for seed in identity['seeds']:
                name = f'{field}_{cohort}_{seed}'
                target_path = directory/f'{name}.json'
                pairs_path = directory/f'{name}.npz'
                if target_path.exists() and pairs_path.exists():
                    output['runs'][name] = json.loads(target_path.read_text())
                    print('Cached', name, flush=True)
                    continue
                unique = np.unique(groups)
                np.random.default_rng(seed).shuffle(unique)
                partition = np.isin(groups, unique[:len(unique)//2])
                discovery, validation = keep & partition, keep & ~partition
                dx, vx = standardize(x[discovery]); dy, vy = standardize(y[discovery])
                source, target = np.flatnonzero(vx), np.flatnonzero(vy)
                print('Matching', name, len(source), len(target), flush=True)
                correlation = dx[:, source].T @ dy[:, target]
                best = correlation.argmax(1)
                one_source, one_target = linear_sum_assignment(correlation, maximize=True)
                scount = (x[discovery][:, source] > 0).sum(0)
                tcount = (y[discovery][:, target] > 0).sum(0)
                restricted, n_candidates = breadth_best(correlation, scount, tcount)
                eligible = restricted >= 0
                r = matched_r(x[validation], y[validation], source, target[best])
                one_r = matched_r(x[validation], y[validation], source[one_source], target[one_target])
                restricted_r = np.full(len(source), np.nan)
                restricted_r[eligible] = matched_r(x[validation], y[validation], source[eligible], target[restricted[eligible]])
                union_r, union_count = union_support_r(x[validation], y[validation], source, target[best])
                record = {'aggregation': field, 'cohort': cohort, 'seed': seed,
                          'n_discovery_proteins': int(discovery.sum()), 'n_validation_proteins': int(validation.sum()),
                          'n_discovery_clusters': len(np.unique(groups[discovery])),
                          'n_validation_clusters': len(np.unique(groups[validation])),
                          'n_source_discovery_variable': len(source), 'n_target_discovery_variable': len(target),
                          'best': describe(r), 'one_to_one_assigned_denominator': describe(one_r),
                          'one_to_one_all_source_denominator': describe(one_r, len(source)),
                          'breadth_all_source_denominator': describe(restricted_r),
                          'breadth_eligible_source_denominator': describe(restricted_r[eligible]) if eligible.any() else None,
                          'n_breadth_eligible_sources': int(eligible.sum()),
                          'median_breadth_candidates_eligible': float(np.median(n_candidates[eligible])) if eligible.any() else None,
                          'best_union_support': describe(union_r),
                          'breadth_strata': {}}
                for low, high in [(1, 10), (10, 100), (100, 1000), (1000, len(ids)+1)]:
                    mask = (scount >= low) & (scount < high)
                    if mask.any():
                        record['breadth_strata'][f'[{low},{high})'] = describe(r[mask])
                temporary = pairs_path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, discovery_ids=ids[discovery], validation_ids=ids[validation],
                                    source_ids=source, target_ids=target, best_target_ids=target[best],
                                    best_validation_r=r, source_discovery_active_counts=scount,
                                    target_discovery_active_counts=tcount,
                                    one_to_one_source_ids=source[one_source], one_to_one_target_ids=target[one_target],
                                    one_to_one_validation_r=one_r,
                                    breadth_target_ids=np.where(eligible, target[np.maximum(restricted, 0)], -1),
                                    breadth_n_candidates=n_candidates, breadth_validation_r=restricted_r,
                                    union_validation_r=union_r, union_validation_support=union_count)
                temporary.replace(pairs_path)
                atomic_json(target_path, record)
                output['runs'][name] = record
                atomic_json(directory/'summary.json', output)
                print('Complete', name, 'best', record['best']['fractions']['0.3'], flush=True)
                del dx, dy, correlation
        del x, y
    atomic_json(directory/'summary.json', output)
    lock.close()


if __name__ == '__main__':
    main()
