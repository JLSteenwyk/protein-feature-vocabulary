"""Check printed intervention effects and intervals against verified summaries."""
import json
from pathlib import Path

from common import ROOT, OUT
from audit_core_reported_numbers import paragraph, require
from probe_checkpoint import atomic_json, file_identity


def interval_text(row, key='ci95', scale=1, precision=2, separator='--'):
    low, high = row[key]
    if low > high:
        raise ValueError('Reversed interval endpoints')
    return separator.join(f'{value*scale:.{precision}f}' for value in [low, high])


def main():
    documents = [ROOT/'revision/manuscript/sn-article.tex', ROOT/'revision/response_to_reviewers.md']
    manuscript, response = [p.read_text() for p in documents]
    paths = {'head': OUT/'corrected_head_evaluation/verified_summary.json',
             'steering': OUT/'steering_controls/verified_summary.json',
             'esm2': OUT/'circuit_controls/esm2/verified_summary.json',
             'esm3': OUT/'circuit_controls/esm3/verified_summary.json'}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if not data['head']['raw_logits_verified'] or not all(data[k]['verified'] for k in ['steering', 'esm2', 'esm3']):
        raise ValueError('Intervention summary not verified')
    checks = []

    def check(label, text, snippet):
        require(text, snippet)
        checks.append({'claim': label, 'verified_text': snippet})

    h = data['head']
    main_head = paragraph(manuscript, 'A new evaluation uses')
    for label, text, separator in [('Main', main_head, '--'),
                                   ('Response', paragraph(response, '**Secondary-structure decoding.**'), '-')]:
        for modality, supplied in [('S', 'S'), ('S_St', 'S+St')]:
            row = h['conditions'][modality+'__target_L0H7']['ss3_disagreement']
            prefix = '95% cluster interval ' if modality == 'S' else ''
            snippet = (f"{100*row['mean']:.2f}% under {supplied} ("+prefix+
                       interval_text(row, scale=100, separator=separator)+'%)')
            check(label+' head '+modality+' mean/interval', text, snippet)
        row = h['paired_contrasts']['target_S_St_minus_S__ss3_disagreement']
        check(label+' paired head difference/interval', text,
              f"paired difference is {100*row['mean']:.2f} percentage points ("+
              interval_text(row, scale=100, separator=separator)+')')
    row = h['conditions']['S__target_L0H7']['ss3_disagreement']
    check('Main head proteins', main_head, f"{row['n_proteins']} sequence/structure-aligned human proteins")
    check('Main head clusters', main_head, f"{row['n_clusters']} detected clusters")
    row = h['paired_contrasts']['S_St__target_minus_mean_ten_random_heads__ss3_disagreement']
    check('Main target-minus-random head contrast/interval', main_head,
          f"{100*row['mean']:.2f} percentage points ("+interval_text(row, scale=100)+')')
    row = h['conditions']['S_St__same_layer_L0H0']['ss3_disagreement']
    check('Main same-layer head', main_head, f"L0H0 changes {100*row['mean']:.3f}%")

    s = data['steering']
    main_steer = paragraph(manuscript, 'A new paired steering evaluation')
    check('Main steering proteins', main_steer, f"{s['n_proteins']} human proteins")
    check('Main steering evaluations', main_steer, f"{s['n_proteins']*len(s['conditions']):,} evaluations")
    interval_key = 'conditional_95_percentile_interval'
    row = s['conditions']['target_alpha_50']['sequence_disagreement']
    check('Main steering sequence disagreement/interval', main_steer,
          f"{100*row['mean']:.2f}% (conditional 95% cluster interval "+
          interval_text(row, interval_key, scale=100)+'%)')
    for alpha in [50, -50]:
        row = s['conditions'][f'target_alpha_{alpha}']['target_activation_change']
        value = f"{row['mean']:+.3f}" if alpha > 0 else f"{row['mean']:.3f}"
        sep = '--' if alpha > 0 else ' to '
        check('Main target activation '+str(alpha), main_steer,
              value+' ('+interval_text(row, interval_key, precision=3, separator=sep)+')')
    main_kl = paragraph(manuscript, 'Target-minus-seed-mean decoder-control KL')
    reply_steer = paragraph(response, '**Paired steering.**')
    for label, text, sep in [('Main', main_kl, '--'), ('Response', reply_steer, '-')]:
        for alpha in [50, -50]:
            row = s['paired_target_minus_control_mean']['decoder'][str(alpha)]['sequence_kl_normal_to_perturbed']
            check(label+' paired steering KL '+str(alpha), text,
                  f"{row['mean']:.6f} ("+interval_text(row, interval_key, precision=6, separator=sep)+')')
        prevalence = s['baseline_residue_prevalence']
        controls = [v for name, v in prevalence.items() if name.startswith('decoder_')]
        check(label+' target prevalence', text, f"{100*prevalence['target']:.2f}%")
        check(label+' decoder prevalence range', text,
              f'{100*min(controls):.4f}'+sep+f'{100*max(controls):.4f}%')

    main_circuit = paragraph(manuscript, 'New matched-baseline evaluations')
    if any(data[m]['n_discovery_clusters'] != 50 or data[m]['n_evaluation_clusters'] != 100 for m in ['esm2', 'esm3']):
        raise ValueError('Circuit cohort sizes differ from printed shared cohort')
    check('Main circuit cohorts', main_circuit, '50 discovery and 100 evaluation clusters')
    for model, pair in [('esm2', '16_24'), ('esm3', '16_33'), ('esm3', '33_42')]:
        record = data[model]['pairs'][pair]
        check('Main tested pairs '+model+'/'+pair, main_circuit, f"{record['tested_edges']:,} tested feature pairs")
        values = [record['readout_summary'][name]['mean_absolute_change'] for name in
                  ['reconstruction_shift', 'matched_removal', 'residual_preserving_removal']]
        if model == 'esm2':
            snippet = f'{values[0]:.3f} versus {values[1]:.3f} (matched reconstruction) and {values[2]:.3f} (residual-preserving)'
        else:
            snippet = f'{values[0]:.3f} versus {values[1]:.3f} and {values[2]:.3f} ('+pair.replace('_', '--')+')'
        check('Main circuit effects '+model+'/'+pair, main_circuit, snippet)
    atomic_json(OUT/'intervention_reported_numbers_audit.json', {
        'scope': 'Selected printed head/steering/circuit numbers in scoped paragraphs; consistency with verified summaries, not a new raw-record audit or all manuscript numbers.',
        'inputs': [file_identity(p, content=True) for p in documents+list(paths.values())],
        'script': file_identity(Path(__file__), content=True), 'checks': checks, 'verified': True})
    print(f'Verified {len(checks)} intervention numeric claims against four verified summaries.')


if __name__ == '__main__':
    main()
