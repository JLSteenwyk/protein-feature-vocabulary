import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'author_uploads', Path(__file__).resolve().parents[1]/'scripts/revision/build_author_uploads.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_public_inputs_exclude_private_results_and_external_weights():
    for path in ('INTERPRETABILITY.zip', 'revision/response_to_reviewers.md',
                 'results/scaled_1.5M/go_enrichment.json', 'external_inputs/model.pth',
                 'models/esm3_sm_open_v1.pth', 'revision/figures/fig1.png'):
        assert MODULE.input_kind(path) is None
    assert MODULE.input_kind('models/sae/esm2_scaled/layer_16_topk/best.pt') == 'sae'
    assert MODULE.input_kind('results/scaled_1.5M/activations/a.h5') == 'data'


def test_upload_source_rejects_escape_and_symlink(tmp_path):
    (tmp_path/'ok').write_text('test')
    (tmp_path/'link').symlink_to(tmp_path/'ok')
    for name in ('../ok', '/ok', 'link'):
        with pytest.raises(ValueError):
            MODULE.safe_source(tmp_path, name)
    assert MODULE.safe_source(tmp_path, 'ok') == tmp_path/'ok'
