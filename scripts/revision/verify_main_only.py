"""Verify source separation, rendered main figure labels and build diagnostics."""
import json
import re
import subprocess
from pathlib import Path
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity
from build_main_only import make_main_source


def main():
    path = OUT/'main_only_manifest.json'
    manifest = json.loads(path.read_text())
    for key in ['source', 'generator', 'generated_source', 'marked_wrapper']:
        expected = manifest[key]
        current = file_identity(expected['path'], content=True)
        if any(current[k] != expected[k] for k in ['sha256', 'bytes']):
            raise ValueError(f'Stale main-only input: {key}')
    regenerated, refs = make_main_source(Path(manifest['source']['path']).read_text())
    if regenerated != Path(manifest['generated_source']['path']).read_text() or refs != manifest['references']:
        raise ValueError('Generated source/reference map mismatch')
    supplement = json.loads((OUT/'supplement_manifest.json').read_text())
    if supplement['source']['sha256'] != manifest['source']['sha256']:
        raise ValueError('Main and supplement derive from different manuscript versions')
    if {row['label']: str(i+1) for i, row in enumerate(supplement['figures'])} != refs['supplement_figure_numbers']:
        raise ValueError('Supplementary figure number mismatch')
    report = {'scope': 'Exact generated-source agreement, main/supplement numbering, PDF text and TeX diagnostics; not complete visual review or published-page certification.',
              'manifest': file_identity(path, content=True),
              'verifier': file_identity(Path(__file__), content=True), 'builds': {}}
    for key in ['generated_source', 'marked_wrapper']:
        tex = Path(manifest[key]['path'])
        pdf, log, aux = [tex.with_suffix(suffix) for suffix in ['.pdf', '.log', '.aux']]
        warnings = re.findall(r'^.*(?:Overfull|undefined|Missing character|error:|too large).*$', log.read_text(), re.M)
        if warnings:
            raise ValueError(f'Render warnings: {warnings}')
        aux_text = aux.read_text()
        for number in range(1, 5):
            if f'\\newlabel{{fig{number}}}{{{{{number}}}' not in aux_text:
                raise ValueError(f'Missing/misnumbered main figure {number}')
        if re.search(r'\\newlabel\{ed\d+\}', aux_text):
            raise ValueError('Supplementary figure labels leaked into main-only build')
        text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-']).decode()
        if '??' in text or re.search(r'Extended Data Fig\.\s*\d+\s*:', text):
            raise ValueError('Unresolved reference or supplementary caption in main PDF')
        if text.count('Alt text:') != 4:
            raise ValueError('Expected exactly four main-figure alt texts')
        report['builds'][key] = {'pdf': file_identity(pdf, content=True),
                                'log': file_identity(log, content=True), 'aux': file_identity(aux, content=True),
                                'pages': len([p for p in text.split('\f') if p.strip()])}
    report['verified'] = True
    atomic_json(OUT/'main_only_verification.json', report)
    print({key: value['pages'] for key, value in report['builds'].items()})


if __name__ == '__main__':
    main()
