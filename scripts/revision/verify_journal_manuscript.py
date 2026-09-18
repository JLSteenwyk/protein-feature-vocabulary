"""Verify OUP source/render consistency; report length separately from build success."""
import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity
from build_journal_manuscript import journal_source


def check_pdf_text(xml):
    pages = ET.fromstring(xml).findall('.//{*}page')
    if not pages:
        raise ValueError('No rendered pages')
    texts = []
    for page in pages:
        width, height = float(page.attrib['width']), float(page.attrib['height'])
        words = page.findall('.//{*}word')
        for word in words:
            x0, y0, x1, y1 = (float(word.attrib[k]) for k in ('xMin', 'yMin', 'xMax', 'yMax'))
            if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
                raise ValueError('Text outside page bounds')
        texts.append(' '.join(''.join(w.itertext()) for w in words))
    text = '\n'.join(texts)
    if '??' in text or text.count('Alt text:') != 4:
        raise ValueError('Unresolved reference or missing main figure alt text')
    return len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-logs', nargs=2, required=True, type=Path,
                        help='Captured clean then marked Tectonic stdout/stderr logs')
    parser.add_argument('--require-length', action='store_true')
    args = parser.parse_args()
    manifest_path = OUT/'journal_manuscript_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for expected in [manifest['source'], manifest['generator'], *manifest['assets'], *manifest['generated']]:
        actual = file_identity(expected['path'], content=True)
        if any(actual[k] != expected[k] for k in ('sha256', 'bytes')):
            raise ValueError(f'Stale journal artifact: {expected["path"]}')
    source, refs = journal_source(Path(manifest['source']['path']).read_text())
    if source != Path(manifest['generated'][0]['path']).read_text() or refs != manifest['references']:
        raise ValueError('Journal regeneration mismatch')
    for name in ('oup-authoring-template.cls', 'oup-abbrvnat.bst'):
        if (ROOT/'revision/journal'/name).read_bytes() != (ROOT/'revision/vendor/oup-authoring-template'/name).read_bytes():
            raise ValueError('Publisher asset modified')
    report = {'scope': 'Source, text bounds, references, engine diagnostics and page counts; not complete visual or scientific validation.',
              'manifest': file_identity(manifest_path, content=True),
              'verifier': file_identity(Path(__file__), content=True), 'builds': []}
    for record, engine_log in zip(manifest['generated'], args.engine_logs):
        tex = Path(record['path'])
        pdf, log, aux = (tex.with_suffix(ext) for ext in ('.pdf', '.log', '.aux'))
        diagnostics = log.read_text()+'\n'+engine_log.read_text()
        if re.search(r'Overfull|undefined|Missing character|error:|too large|pdfmark|Unknown token', diagnostics):
            raise ValueError(f'Unresolved rendering diagnostic for {tex.name}')
        for n in range(1, 5):
            if f'\\newlabel{{fig{n}}}{{{{{n}}}' not in aux.read_text():
                raise ValueError('Missing or misnumbered main figure')
        xml = subprocess.check_output(['pdftotext', '-bbox', str(pdf), '-'])
        pages = check_pdf_text(xml)
        report['builds'].append({'pdf': file_identity(pdf, content=True),
                                 'engine_log': file_identity(engine_log, content=True),
                                 'pages': pages, 'within_seven_page_target': pages <= 7})
    report['render_checks_pass'] = True
    report['length_target_met'] = all(b['within_seven_page_target'] for b in report['builds'])
    atomic_json(OUT/'journal_manuscript_verification.json', report)
    print({'render_checks_pass': True, 'pages': [b['pages'] for b in report['builds']],
           'length_target_met': report['length_target_met']})
    if args.require_length and not report['length_target_met']:
        raise SystemExit('Seven-page target not met')


if __name__ == '__main__':
    main()
