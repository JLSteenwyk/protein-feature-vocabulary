"""Check selected core manuscript/response numbers against their result exports."""
import json
import re
from pathlib import Path

from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity


def paragraph(text, marker):
    found = [p for p in text.split('\n\n') if marker in p]
    if len(found) != 1:
        raise ValueError(f'Expected one paragraph containing {marker!r}, found {len(found)}')
    return re.sub(r'\s+', ' ', found[0].replace(r'\%', '%'))


def require(text, snippet):
    snippet = re.sub(r'\s+', ' ', snippet)
    pattern = re.escape(snippet)
    if snippet and snippet[0].isdigit():
        pattern = r'(?<![\d.,+\-])'+pattern
    if snippet and snippet[-1].isdigit():
        pattern += r'(?![\d])'
    if re.search(pattern, re.sub(r'\s+', ' ', text)) is None:
        raise ValueError(f'Reported value/context does not match export: {snippet}')


def main():
    sources = [ROOT/'revision/manuscript/sn-article.tex', ROOT/'revision/response_to_reviewers.md']
    manuscript, response = [p.read_text() for p in sources]
    paths = {name: OUT/f'{name}.json' for name in ['matching', 'go_audit', 'cross_modal_complete', 'combined_probe_controls']}
    data = {name: json.loads(p.read_text()) for name, p in paths.items()}
    checks = []

    def check(label, text, snippet):
        require(text, snippet)
        checks.append({'claim': label, 'verified_text': snippet})

    m = data['matching']
    rows = [('Full-data ESM-3-to-ESM-2 best matching', m['many_to_one'], f"{m['n_esm3']:,} source features"),
            ('Reverse best matching', m['reverse_many_to_one'], f"{m['n_esm2']:,} source features"),
            ('Maximum-weight one-to-one assignment', m['one_to_one'], f"{m['n_esm3']:,} assigned sources"),
            ('Mutual nearest neighbors', m['mutual_nearest'], f"All {m['n_esm3']:,} sources"),
            ('Discovery-frozen best matches on validation clusters', m['heldout']['many_to_one'],
             f"{m['heldout']['many_to_one']['n_pairs']:,} discovery-variable sources"),
            ('Discovery-frozen one-to-one pairs on validation clusters', m['heldout']['one_to_one'],
             f"{m['heldout']['one_to_one']['n_pairs']:,} assigned pairs")]
    for label, row, denominator in rows:
        check('Response matching: '+label, response,
              f"| {label} | {row['fractions']['0.3']*100:.2f}% | {denominator} |")
    main = paragraph(manuscript, 'Under many-to-one matching')
    for kind in ['many_to_one', 'one_to_one', 'mutual_nearest']:
        check('Main matching '+kind, main, f"{m[kind]['fractions']['0.3']*100:.2f}%")
    check('Main matching distinct targets', main, f"{m['n_distinct_targets_many_to_one']:,} distinct target features")
    main = paragraph(manuscript, 'For held-out evaluation')
    for kind in ['many_to_one', 'one_to_one']:
        check('Main held-out '+kind, main, f"{m['heldout'][kind]['fractions']['0.3']*100:.2f}%")
        check('Main held-out denominator '+kind, main, f"{m['heldout'][kind]['n_pairs']:,}")
    main = paragraph(manuscript, 'Complete GO recomputation')
    for model, label in [('esm3', 'ESM-3'), ('esm2', 'ESM-2')]:
        row = data['go_audit']['models'][model]
        check('Main enriched count '+model, main,
              f"{row['n_enriched_revised']:,} of {row['n_active']:,}")
    c = data['cross_modal_complete']
    main = paragraph(manuscript, 'Re-encoding paired S and S+St')
    check('Main paired protein count', main, f"{c['n_proteins']:,} proteins")
    check('Main paired active count', main, f"{c['n_active']:,} active coordinates")
    for name in ['enhanced', 'suppressed', 'unclassified']:
        check('Main paired category '+name, main, f"{c['category_counts'][name]:,} {name}")
    p = data['combined_probe_controls']
    main = paragraph(manuscript, 'New functional-site probes')
    reply = response.split('### Major Comment 2:', 1)[1].split('### Major Comment 3:', 1)[0]
    for metric in ['auroc', 'average_precision', 'macro_within_protein_auroc']:
        value = '/'.join(f"{p['models'][model]['intact'][metric]:.4f}" for model in ['esm3', 'esm2'])
        check('Main intact probe '+metric, main, value)
        check('Response intact probe '+metric, reply, value)
    for key, label in [('residues', 'test residues'), ('positive', 'annotated positives')]:
        check('Main primary test '+key, main, f"{p['sample_counts']['test'][key]:,} {label}")
    check('Main mixed-label proteins', main, f"{p['models']['esm3']['intact']['n_proteins_with_both_classes']} proteins")
    report = {'scope': 'Selected core numeric claims in specific manuscript paragraphs and response rows/section only. Not all numerical claims, semantics, references, or full scientific validation.',
              'inputs': [file_identity(p, content=True) for p in sources+list(paths.values())],
              'script': file_identity(Path(__file__), content=True), 'checks': checks, 'verified': True}
    atomic_json(OUT/'core_reported_numbers_audit.json', report)
    print(f'Verified {len(checks)} selected core numeric claims against four result exports')


if __name__ == '__main__':
    main()
