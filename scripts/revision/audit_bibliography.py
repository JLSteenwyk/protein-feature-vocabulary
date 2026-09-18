"""Check bibliography keys and stray markup; not a full metadata/reference audit."""
import re
from pathlib import Path
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity


def validate_bibliography(bibliography, sources):
    if '<?' in bibliography or '?>' in bibliography:
        raise ValueError('Processing-instruction markup in bibliography')
    keys = re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,', bibliography)
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate bibliography keys')
    cited = set()
    for source in sources:
        for group in re.findall(r'\\cite[a-zA-Z]*\*?(?:\[[^]]*\])*\{([^}]+)\}', source):
            cited.update(key.strip() for key in group.split(','))
    missing = sorted(cited-set(keys))
    if missing:
        raise ValueError(f'Unresolved citation keys: {missing}')
    return {'entries': len(keys), 'cited_keys': sorted(cited), 'uncited_keys': sorted(set(keys)-cited)}


def main():
    directory = ROOT/'revision/manuscript'
    bib = directory/'sn-bibliography.bib'
    sources = [directory/'sn-article.tex', directory/'supplementary-methods.tex']
    report = validate_bibliography(bib.read_text(), [p.read_text() for p in sources])
    report.update({'scope': 'Citation-key completeness, duplicate keys and stray processing instructions only; not external verification of every reference.',
                   'inputs': [file_identity(p, content=True) for p in [bib]+sources],
                   'script': file_identity(Path(__file__), content=True), 'verified': True})
    atomic_json(OUT/'bibliography_audit.json', report)
    print(f'Checked {report["entries"]} entries and {len(report["cited_keys"])} cited keys')


if __name__ == '__main__':
    main()
