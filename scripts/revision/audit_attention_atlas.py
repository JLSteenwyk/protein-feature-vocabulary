"""Audit archived query-marginal attention JSD and distinguish head-ablation KL."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT, RESULTS
from probe_checkpoint import atomic_json, file_identity


def query_marginal_jsd(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 2 or left.shape != right.shape or not left.size:
        raise ValueError('Attention matrices require aligned nonempty query/key dimensions')
    if any(not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0 for a in [left, right]):
        raise ValueError('Attention weights must be finite nonnegative distributions')
    p, q = [np.clip(a.mean(axis=0), 1e-10, None) for a in [left, right]]
    p, q = p/p.sum(), q/q.sum()
    mixture = (p+q)/2
    return float((np.sum(p*np.log(p/mixture))+np.sum(q*np.log(q/mixture)))/2)


def main():
    atlas_path = RESULTS/'attention_atlas.json'
    ablation_path = RESULTS/'head_ablation_sst.json'
    atlas, ablation = [json.loads(p.read_text()) for p in [atlas_path, ablation_path]]
    matrix = np.asarray(atlas['mean_jsd_matrix'])
    assert matrix.shape == (48, 24) and np.isfinite(matrix).all()
    assert (matrix >= -1e-8).all() and (matrix <= np.log(2)+1e-7).all()
    responsive = matrix > atlas['jsd_threshold']
    assert int(responsive.sum()) == atlas['n_responsive_heads']
    for layer in range(48):
        saved = atlas['per_layer'][str(layer)]
        assert saved['n_responsive_heads'] == int(responsive[layer].sum())
        assert np.isclose(saved['mean_jsd'], matrix[layer].mean())
        assert np.isclose(saved['max_jsd'], matrix[layer].max())
    peak = np.unravel_index(matrix.argmax(), matrix.shape)
    head_rows = [{'block': l, 'head': h, 'mean_query_marginal_jsd': float(matrix[l, h]),
                  'above_threshold': bool(responsive[l, h])} for l in range(48) for h in range(24)]
    records = []
    left, right = ablation['condition_s']['per_head'], ablation['condition_sst']['per_head']
    assert set(left) == set(right)
    for key in sorted(left):
        a, b = left[key], right[key]
        l, h = a['layer'], a['head']
        assert (l, h) == (b['layer'], b['head'])
        records.append({'id': key, 'block': l, 'head': h, 'mean_jsd': float(matrix[l, h]),
                        'mean_kl_s': a['mean_kl'], 'sd_kl_s': a['std_kl'],
                        'mean_kl_sst': b['mean_kl'], 'sd_kl_sst': b['std_kl']})
    assert len(records) == len(ablation['ablation_layers'])*24
    output = {'inputs': [file_identity(p, content=True) for p in [atlas_path, ablation_path]],
              'indexing': 'zero-based blocks 0-47 and heads 0-23 of standard block.attn, not block.geom_attn',
              'jsd_definition': 'Natural-log JSD after averaging attention across query positions within protein, then arithmetic mean over proteins. Includes special-token positions.',
              'jsd_limitation': 'Query marginal can remain unchanged when query-specific patterns differ. Historical extraction silently truncates unequal lengths; raw paired matrices and per-head counts were not retained.',
              'inference_scope': 'Archived aggregate reconciliation, not fresh attention extraction or native-versus-patched numerical equivalence validation.',
              'n_atlas_proteins_reported': atlas['n_proteins'], 'n_atlas_heads': matrix.size,
              'n_ablation_proteins_reported': ablation['n_proteins'], 'n_ablation_heads': len(records),
              'tested_ablation_blocks': ablation['ablation_layers'],
              'n_responsive': int(responsive.sum()), 'n_responsive_blocks_2_to_9': int(responsive[2:10].sum()),
              'peak_jsd_head': {'block': int(peak[0]), 'head': int(peak[1]), 'value': float(matrix[peak])},
              'l0h7_jsd': float(matrix[0, 7]), 'atlas_heads': head_rows, 'ablation_heads': records,
              'uncertainty': 'No reconstructed paired protein intervals; stored ablation SD is dispersion, not an interval.'}
    atomic_json(OUT/'attention_atlas_audit.json', output)
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    ax = axes[0, 0]
    artist = ax.imshow(matrix, aspect='auto', cmap='cividis', vmin=0, vmax=np.log(2))
    ax.set(title='A  Query-marginal JSD (494 proteins)', xlabel='Head (zero-based)', ylabel='Block (zero-based)')
    figure.colorbar(artist, ax=ax, label='Mean JSD (natural log)')
    ax = axes[0, 1]
    ax.bar(np.arange(48), responsive.sum(1), color=['#D55E00' if 2 <= l <= 9 else '#777777' for l in range(48)])
    ax.set(title='B  Heads with mean JSD > 0.1', xlabel='Block (zero-based)', ylabel='Head count')
    ax = axes[1, 0]
    for condition, color in [('s', '#0072B2'), ('sst', '#D55E00')]:
        ax.scatter([r['mean_jsd'] for r in records], [r[f'mean_kl_{condition}'] for r in records],
                   s=12, alpha=.65, color=color, label='S' if condition == 's' else 'S+St')
    target = next(r for r in records if r['id'] == 'L0_H7')
    ax.annotate('L0H7', (target['mean_jsd'], target['mean_kl_sst']), xytext=(.10, .12),
                arrowprops={'arrowstyle': '->'}, fontsize=8)
    ax.set(yscale='log', title='C  Distinct metrics; 144 scanned heads',
           xlabel='Mean query-marginal JSD (atlas)', ylabel='Mean output KL after ablation')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.bar(['S', 'S+St'], [target['mean_kl_s'], target['mean_kl_sst']],
           yerr=[target['sd_kl_s'], target['sd_kl_sst']], capsize=4, color=['#0072B2', '#D55E00'])
    ax.set(title='D  L0H7 ablation (97 proteins)', ylabel='Mean output KL +/- archived SD')
    for extension in ['png', 'pdf']:
        figure.savefig(ROOT/'revision/figures'/f'attention_audit.{extension}', dpi=220)
    plt.close(figure)
    print('Verified', output['n_responsive'], 'responsive heads;', output['n_responsive_blocks_2_to_9'],
          'in blocks 2-9; maximum', output['peak_jsd_head'], '; L0H7', output['l0h7_jsd'])


if __name__ == '__main__':
    main()
