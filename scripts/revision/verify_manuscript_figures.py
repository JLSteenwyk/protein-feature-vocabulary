"""Verify corrected main-figure wiring and list remaining accessibility/source gaps."""
import re
from common import ROOT, OUT
from probe_checkpoint import atomic_json


def main():
    path = ROOT/'revision/manuscript/sn-article.tex'
    text = path.read_text()
    original = (ROOT/'revision/submitted/sn-article.tex').read_text()
    declarations = [
        'JLS is a Howard Hughes Medical Institute Awardee of the Life Sciences Research Foundation',
        'JLS is an advisor to ForensisGroup Inc and a scientific consultant to Anthropic PBC and Sift Biosciences Inc.']
    assert all(value in original and value in text for value in declarations)
    figures = []
    for body in re.findall(r'\\begin\{figure\}(.*?)\\end\{figure\}', text, flags=re.S):
        labels = re.findall(r'\\label\{([^}]+)\}', body)
        images = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', body)
        figures.append({'labels': labels, 'images': images, 'has_alt_text': 'Alt text:' in body})
    for label, filename in [('fig1', 'main_overview_corrected.png'),
                            ('fig3', 'cross_modal_corrected.png'), ('fig4', 'steering_paired_corrected.png'),
                            ('ed6', 'cross_modal_detail.png'), ('ed7', 'attention_audit.png'),
                            ('ed9', 'go_detail_corrected.png'), ('ed8', 'contact_descriptive.png'),
                            ('ed5', 'lens_descriptive.png'), ('ed1', 'reconstruction_corrected.png'),
                            ('ed2', 'autointerpretability_corrected.png'),
                            ('ed3', 'probe_controls_corrected.png'),
                            ('ed4', 'feature_biology_corrected.png'),
                            ('ed10', '../analyses/probes_overlap_excluded/probe_controls_corrected.png'),
                            ('ed11', 'corrected_head_evaluation.png')]:
        found = [f for f in figures if label in f['labels']]
        assert len(found) == 1 and found[0]['images'] == [filename] and found[0]['has_alt_text']
        assert (ROOT/'revision/figures'/filename).is_file()
    assert '{fig3_multimodal.png}' not in text and '{fig4_causal.png}' not in text
    assert '{ed06_cross_modal.png}' not in text and 'median 2.10' not in text
    assert '{ed09_decoder_dms_phylo.png}' not in text and '50/1{,}348' not in text
    assert '{ed08_contact_circuits.png}' not in text and 'significant causal connections' not in text
    assert '{ed05_lens_decomposition.png}' not in text and 'next-token prediction accuracy' not in text
    assert '{ed01_training_reconstruction.png}' not in text and 'exactly $k = 64$ active' not in text
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    references = re.findall(r'\\ref\{([^}]+)\}', text)
    missing = sorted(set(references)-set(labels))
    assert not missing
    report = {'scope': 'Source wiring, literal references and author declarations; not numerical validation of every surviving panel or full journal readiness.',
              'corrected_main_figures_wired': True, 'submitted_declarations_preserved': True,
              'unresolved_literal_references': missing, 'figures': figures,
              'figures_missing_alt_text': [f['labels'] for f in figures if not f['has_alt_text']],
              'historical_image_figures': [f['labels'] for f in figures if any((ROOT/'revision/submitted'/name).exists() for name in f['images'])]}
    atomic_json(OUT/'manuscript_figure_verification.json', report)
    print('Corrected main figures and declarations verified.')
    print('Remaining figures without alt text:', report['figures_missing_alt_text'])


if __name__ == '__main__':
    main()
