"""Plot the verified new head-evaluation cohort separately from historical atlas."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity


def validate_summary(summary):
    if not summary.get('raw_logits_verified') or summary['successful'] != 100 or summary['failures']:
        raise ValueError('Expected complete verified 100-protein cohort')
    for identity in summary['inputs']:
        if file_identity(identity['path'], content=True)['sha256'] != identity['sha256']:
            raise ValueError('Verified input snapshot changed')
    names = [key.removeprefix('S__') for key in summary['conditions'] if key.startswith('S__')]
    if len(names) != 14 or set(summary['conditions']) != {f'{m}__{n}' for m in ['S', 'S_St'] for n in names}:
        raise ValueError('Missing modality or intervention')
    return [n for n in names if n != 'normal']


def main():
    source = OUT/'corrected_head_evaluation/verified_summary.json'
    summary = json.loads(source.read_text())
    names = validate_summary(summary)
    labels = [n.replace('target_', '').replace('same_layer_', '').replace('random_', '') for n in names]
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    metrics = [('ss3_disagreement', 100, 'A  Corrected SS3 prediction disagreement', 'Changed residues (%)'),
               ('sequence_disagreement', 100, 'B  Sequence prediction disagreement', 'Changed residues (%)'),
               ('mean_sequence_kl_normal_to_perturbed', 1, 'C  Sequence-output distribution change', 'Mean residue KL(normal || ablated)')]
    for ax, (metric, scale, title, xlabel) in zip(axes.flat, metrics):
        for modality, shift, color in [('S', -.16, '#0072B2'), ('S_St', .16, '#D55E00')]:
            for i, name in enumerate(names):
                record = summary['conditions'][f'{modality}__{name}'][metric]
                lo, hi = np.array(record['ci95'])*scale
                ax.plot([lo, hi], [i+shift]*2, color=color, linewidth=1)
                ax.plot(record['mean']*scale, i+shift, 'o', color=color, markersize=4)
            ax.plot([], [], 'o', color=color, label='S' if modality == 'S' else 'S+St')
        ax.set(yticks=range(len(names)), yticklabels=labels, title=title, xlabel=xlabel)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    axes[0, 0].legend(loc='lower right')
    ax = axes[1, 1]
    contrasts = [
        ('S__target_minus_mean_ten_random_heads__ss3_disagreement', 'S: target - mean random'),
        ('S_St__target_minus_mean_ten_random_heads__ss3_disagreement', 'S+St: target - mean random'),
        ('S_St__target_minus_same_layer_L0H0__ss3_disagreement', 'S+St: target - L0H0'),
        ('target_S_St_minus_S__ss3_disagreement', 'Target: S+St - S')]
    for i, (key, _) in enumerate(contrasts):
        record = summary['paired_contrasts'][key]
        ax.plot(np.array(record['ci95'])*100, [i, i], color='#333333')
        ax.plot(record['mean']*100, i, 'o', color='#333333')
    ax.set(yticks=range(len(contrasts)), yticklabels=[label for _, label in contrasts],
           title='D  Paired SS3 contrasts', xlabel='Difference in changed residues (percentage points)')
    ax.invert_yaxis()
    ax.axvline(0, color='#777777', linestyle=':')
    ax.grid(axis='x', alpha=.2)
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT/f'revision/figures/corrected_head_evaluation.{ext}', dpi=220)
    plt.close(fig)
    atomic_json(OUT/'corrected_head_figure_provenance.json', {
        'input': file_identity(source, content=True), 'script': file_identity(Path(__file__), content=True),
        'conditions': names, 'scope': summary['scope'],
        'uncertainty': '1000 sequence-cluster resamples of protein means; fixed sampled control heads, not independent biological replicates.'})
    print('Corrected head figure exported with all 13 non-normal conditions in both modalities')


if __name__ == '__main__':
    main()
