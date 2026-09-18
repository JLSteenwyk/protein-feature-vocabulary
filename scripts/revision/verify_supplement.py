"""Check standalone supplement provenance and rendered text, not scientific validity."""
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity
from build_supplement import extract_extended_figures


def check_rendered_pages(xml, expect_methods=False):
    root = ET.fromstring(xml)
    pages = root.findall('.//{*}page')
    if (not expect_methods and len(pages) != 12) or (expect_methods and len(pages) < 14):
        raise ValueError('Expected title page plus eleven figure pages')
    records = []
    methods_found, references_found = False, False
    for index, page in enumerate(pages):
        width, height = float(page.attrib['width']), float(page.attrib['height'])
        words = page.findall('.//{*}word')
        text = ' '.join(''.join(word.itertext()) for word in words)
        if '??' in text:
            raise ValueError(f'Unresolved reference on PDF page {index+1}')
        captions = re.findall(r'Extended Data Fig\.\s*(\d+)\s*:', text)
        if 1 <= index <= 11 and (captions != [str(index)] or 'Alt text:' not in text):
            raise ValueError(f'Missing or misordered caption/alt text on PDF page {index+1}')
        if index == 12:
            methods_found = 'Supplementary Methods' in text
        if index > 12:
            references_found |= 'References' in text
        if not words:
            raise ValueError(f'Empty PDF page {index+1}')
        for word in words:
            a = word.attrib
            if not (0 <= float(a['xMin']) <= float(a['xMax']) <= width
                    and 0 <= float(a['yMin']) <= float(a['yMax']) <= height):
                raise ValueError(f'Text outside page bounds on PDF page {index+1}')
        records.append({'pdf_page': index+1, 'figure': f'ed{index}' if 1 <= index <= 11 else None,
                        'words': len(words), 'text_within_page': True})
    if expect_methods and not (methods_found and references_found):
        raise ValueError('Missing supplementary Methods or References heading')
    return records


def main():
    manifest_path = OUT/'supplement_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    identities = [manifest[key] for key in ('source', 'generator', 'generated_source', 'methods_source', 'bibliography')]
    identities += [identity for figure in manifest['figures'] for identity in figure['images']]
    for identity in identities:
        current = file_identity(Path(identity['path']), content=True)
        if any(current[key] != identity[key] for key in ('sha256', 'bytes')):
            raise ValueError(f'Stale supplement input: {identity["path"]}')
    figures = extract_extended_figures(Path(manifest['source']['path']).read_text())
    if len(figures) != len(manifest['figures']):
        raise ValueError('Figure manifest length mismatch')
    generated = Path(manifest['generated_source']['path']).read_text()
    if generated.count(Path(manifest['methods_source']['path']).read_text()) != 1:
        raise ValueError('Detailed Methods not copied exactly once')
    for (label, body), record in zip(figures, manifest['figures']):
        if label != record['label'] or hashlib.sha256(body.encode()).hexdigest() != record['body_sha256']:
            raise ValueError(f'Figure body mismatch: {label}')
        if generated.count(body) != 1:
            raise ValueError(f'Figure body not copied exactly once: {label}')
    if r'\newcommand{\rev}[1]{#1}' not in generated:
        raise ValueError('Clean revision macro missing')
    pdf = ROOT/'revision/supplement/supplement-clean.pdf'
    xml = subprocess.check_output(['pdftotext', '-bbox', str(pdf), '-'])
    pages = check_rendered_pages(xml, expect_methods=True)
    log = pdf.with_suffix('.log').read_text()
    warnings = re.findall(r'^.*(?:Overfull|undefined|Missing character|error:|too large).*$', log, re.M)
    if warnings:
        raise ValueError(f'Render warnings: {warnings}')
    result = {'scope': 'Provenance, exact figure extraction, rendered text order/bounds and TeX diagnostics; not image clipping, accessibility tagging or scientific certification.',
              'manifest': file_identity(manifest_path, content=True),
              'verifier': file_identity(Path(__file__), content=True),
              'pdf': file_identity(pdf, content=True), 'log': file_identity(pdf.with_suffix('.log'), content=True),
              'pages': pages, 'verified': True}
    atomic_json(OUT/'supplement_verification.json', result)
    print(f'Verified {len(pages)} pages and {len(figures)} exact figure blocks')


if __name__ == '__main__':
    main()
