"""Approximate editable-source word counts; not a journal typesetting estimate."""
import re
import subprocess
from pathlib import Path
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity


def plain(text):
    source = r'\newcommand{\rev}[1]{#1}' + '\n' + text
    return subprocess.check_output(['pandoc', '-f', 'latex', '-t', 'plain', '--wrap=none'],
                                   input=source.encode()).decode()


def main():
    source = ROOT/'revision/manuscript/sn-article.tex'
    text = source.read_text()
    abstract = text.split(r'\abstract{', 1)[1].split(r'\keywords', 1)[0].strip()
    if not abstract.endswith('}'):
        raise ValueError('Unrecognized abstract boundary')
    abstract = abstract[:-1]
    headings = re.findall(r'\\textbf\{([^}]+):\}', abstract)
    expected = ['Motivation', 'Results', 'Availability and Implementation', 'Contact', 'Supplementary Information']
    if headings != expected:
        raise ValueError('Abstract headings missing, duplicated or out of order')
    body = text.split(r'\section{Introduction}', 1)[1].split(r'\bibliography{', 1)[0]
    sections = re.split(r'\\section\*?\{([^}]+)\}', body)
    pieces = {'Introduction': sections[0]}
    for index in range(1, len(sections), 2):
        pieces[sections[index]] = sections[index+1]
    counts = {name: len(plain(content).split()) for name, content in pieces.items()}
    abstract_plain = plain(abstract)
    report = {'scope': 'Approximate whitespace word counts after Pandoc LaTeX-to-plain conversion; abstract includes headings; body includes subsection headings/citation tokens but excludes bibliography and all figure legends. Not published-page compliance.',
              'source': file_identity(source, content=True), 'audit_script': file_identity(Path(__file__), content=True),
              'pandoc': subprocess.check_output(['pandoc', '--version']).decode().splitlines()[0],
              'abstract_headings': headings, 'abstract_words': len(abstract_plain.split()),
              'body_section_words': counts, 'body_words': sum(counts.values()),
              'abstract_plus_body_words': len(abstract_plain.split())+sum(counts.values()),
              'current_class': re.search(r'\\documentclass(?:\[[^]]*\])?\{([^}]+)\}', text).group(1),
              'journal_template_verified': False, 'published_page_limit_verified': False}
    atomic_json(OUT/'manuscript_length_audit.json', report)
    print(f'Abstract: {report["abstract_words"]} words; body: {report["body_words"]}; total: {report["abstract_plus_body_words"]}')
    print(counts)


if __name__ == '__main__':
    main()
