"""Correct labels and validate archived lens/component summary arithmetic."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity


def validate_lens_records(data):
    layers = data['layers']
    if len(set(layers)) != len(layers) or set(map(str, layers)) != set(data['per_layer']):
        raise ValueError('Layer records differ from declared sample')
    for layer in layers:
        row = data['per_layer'][str(layer)]
        values = [row[k] for k in ['accuracy_S', 'accuracy_SSt', 'kl_S', 'kl_SSt', 'accuracy_diff', 'kl_diff']]
        if not np.isfinite(values).all() or min(values[:2]) < 0 or max(values[:2]) > 1 or min(values[2:4]) < 0:
            raise ValueError('Invalid agreement or KL')
        for prefix in ['accuracy', 'kl']:
            if not np.isclose(row[f'{prefix}_SSt']-row[f'{prefix}_S'], row[f'{prefix}_diff'], atol=1e-12, rtol=0):
                raise ValueError('Condition difference inconsistent')
    return {
        'SSt_higher_agreement_blocks': [l for l in layers if data['per_layer'][str(l)]['accuracy_diff'] > 0],
        'SSt_lower_KL_blocks': [l for l in layers if data['per_layer'][str(l)]['kl_diff'] < 0]}


def main():
    paths = [ROOT/'results/scaled_1.5M/logit_lens.json', ROOT/'results/unified/esm3/tuned_lens.json',
             ROOT/'results/unified/esm3/residual_decomposition.json', ROOT/'results/unified/esm2/residual_decomposition.json']
    lens, tuned, rd3, rd2 = [json.loads(p.read_text()) for p in paths]
    contrasts = validate_lens_records(lens)
    for row in tuned['per_layer'].values():
        for metric in ['top1', 'kl']:
            raw, fit = row[f'raw_{metric}']['mean'], row[f'tuned_{metric}']['mean']
            expected = fit-raw if metric == 'top1' else raw-fit
            if not np.isclose(expected, row[f'improvement_{metric}'], atol=1e-12, rtol=0):
                raise ValueError('Tuned-lens improvement mismatch')
    sources = [ROOT/'scripts/scaled_1.5M/24_logit_lens.py',
               ROOT/'scripts/unified/run_tuned_lens.py', ROOT/'scripts/unified/run_residual_decomposition.py',
               ROOT/'src/models/interventions.py',
               ROOT/'scripts/publication_figures/supplementary/sfig09_lens_decomposition.py',
               ROOT/'env/lib/python3.11/site-packages/esm/layers/blocks.py']
    report = {
        'inputs': [file_identity(p, content=True) for p in paths+sources],
        'scaled_lens_reported_successful_proteins': lens['n_proteins'],
        'scaled_lens_blocks': lens['layers'], **contrasts,
        'readout': 'Agreement with each condition\'s own final same-position top-1 prediction; not next-token or ground-truth accuracy. KL(final || intermediate). BOS/EOS excluded.',
        'paired_limit': 'Condition summaries appended independently before success counter increments; raw records and per-condition/layer counts unavailable, so exact pairing cannot be certified.',
        'tuned_lens_train_eval': [tuned['n_train'], tuned['n_eval']],
        'tuned_lens_condition': 'Sequence only, separate unified cohort; accession-order 60/40 split, not homology-cluster split.',
        'tuned_lens_worse_agreement_blocks': [int(l) for l, r in tuned['per_layer'].items() if r['improvement_top1'] < 0],
        'tuned_last_block': tuned['per_layer']['47'],
        'component_reported_proteins': {'esm3': rd3['n_proteins'], 'esm2': rd2['n_proteins']},
        'component_scope': 'Separate 500-protein sequence-only cohort selected in accession order; includes special positions. ESM-3 hooks capture raw standard-attention and FFN outputs before residual scaling, omit geometric attention. ESM-2 captures incremental residual contributions. Not commensurate causal importance or a complete ESM-3 residual decomposition.',
        'uncertainty': 'No raw per-protein records for these plots; retained curves are descriptive, without fabricated confidence intervals.',
        'resolution': 'Replace ED5 labels/caption; withdraw next-token accuracy, consistent SSt advantage, CKA-integration-zone and model-dimensionality explanations.'}
    atomic_json(OUT/'lens_summary_audit.json', report)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    layers = lens['layers']
    for condition, label, color in [('S', 'S', '#0072B2'), ('SSt', 'S+St', '#D55E00')]:
        for ax, field in [(axes[0, 0], 'accuracy'), (axes[0, 1], 'kl')]:
            ax.plot(layers, [lens['per_layer'][str(l)][f'{field}_{condition}'] for l in layers],
                    'o-', ms=3, color=color, label=label)
    axes[0, 0].set(title='A  Agreement, not label accuracy', ylabel='Top-1 agreement with own final output', ylim=(0, 1))
    axes[0, 1].set(title='B  Final-to-intermediate divergence', ylabel='Mean KL(final || intermediate)')
    for ax in axes[0, :2]:
        ax.legend(fontsize=8)
    tl = tuned['eval_layers']
    for name, color in [('raw', '#0072B2'), ('tuned', '#D55E00')]:
        axes[0, 2].plot(tl, [tuned['per_layer'][str(l)][f'{name}_top1']['mean'] for l in tl],
                        'o-', ms=3, label=name, color=color)
    axes[0, 2].set(title='C  Separate sequence-only tuned lens', ylabel='Top-1 agreement with final output', ylim=(0, 1))
    axes[0, 2].legend(fontsize=8)
    axes[1, 0].bar(layers, [lens['per_layer'][str(l)]['accuracy_diff'] for l in layers], color='#777777', width=1.5)
    axes[1, 0].axhline(0, color='black', lw=.7)
    axes[1, 0].set(title='D  Difference of condition means', ylabel='Agreement difference (S+St minus S)')
    for ax, data, title in [(axes[1, 1], rd3, 'E  ESM-3 raw component outputs'), (axes[1, 2], rd2, 'F  ESM-2 residual increments')]:
        ls = sorted(map(int, data['per_layer']))
        for component, style, color in [('attn', '-', '#0072B2'), ('mlp', '--', '#D55E00')]:
            values = [data['per_layer'][str(l)][f'{component}_l2_norm'] for l in ls]
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError('Invalid component norms')
            ax.plot(ls, values, style, color=color, label=component)
        ax.set(title=title, ylabel='Mean unnormalized L2 norm')
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Zero-based transformer block')
    for extension in ['png', 'pdf']:
        fig.savefig(ROOT/f'revision/figures/lens_descriptive.{extension}', dpi=220)
    plt.close(fig)
    print(json.dumps(contrasts))


if __name__ == '__main__':
    main()
