"""Check main-text heading revision markers against the immutable submitted source."""
import re
from pathlib import Path

from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity


def headings(text):
    records = []
    for match in re.finditer(r'^\\(subsection|section)(\*?)\{([^\n]+)\}\s*$', text, re.M):
        title = match.group(3)
        marked = title.startswith(r'\rev{') and title.endswith('}')
        if marked:
            title = title[5:-1]
        if any(char in title for char in '{}\\'):
            raise ValueError('Review unsupported heading syntax')
        records.append({'level': match.group(1), 'starred': bool(match.group(2)),
                        'title': title, 'marked': marked})
    return records


def check_headings(submitted, revised):
    baseline = {(r['level'], r['starred'], r['title']) for r in headings(submitted)}
    rows = headings(revised)
    for row in rows:
        row['new_or_changed'] = (row['level'], row['starred'], row['title']) not in baseline
        if row['new_or_changed'] and not row['marked']:
            raise ValueError(f'Unmarked revised heading: {row["title"]}')
    return rows


def main():
    submitted = ROOT/'revision/submitted/sn-article.tex'
    revised = ROOT/'revision/manuscript/sn-article.tex'
    rows = check_headings(submitted.read_text(), revised.read_text())
    report = {'scope': 'Heading-marker coverage only, not a complete semantic diff or rendered-color audit.',
              'submitted': file_identity(submitted, content=True),
              'revised': file_identity(revised, content=True),
              'auditor': file_identity(Path(__file__), content=True),
              'headings': rows, 'verified': True}
    atomic_json(OUT/'revision_heading_audit.json', report)
    print(f'Verified {sum(r["new_or_changed"] for r in rows)} revised headings are marked')


if __name__ == '__main__':
    main()
