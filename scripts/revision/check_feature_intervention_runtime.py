"""Synthetic full-model prerequisite for corrected SAE interventions, not a circuit map."""
from contextlib import nullcontext
import json
import sys
from pathlib import Path
import torch
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity
from capture_reproducibility import stable_digest
from feature_interventions import feature_intervention, intervene_hidden

sys.path.insert(0, str(ROOT/'src'))
from models.esm3_hooks import load_esm3, tokenize_sequence
from models.interventions import cache_residual_stream, sae_feature_ablation
from sae.model import TopKSAE


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2288)
    torch.backends.cuda.matmul.allow_tf32 = False
    from esm.utils.constants.esm3 import data_root
    weight = data_root('esm3')/'data/weights/esm3_sm_open_v1.pth'
    report = {'scope': 'Synthetic software fixture only; no biological circuit validation or corrected edge counts.',
              'seed': 2288, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'model_weight': stable_digest(weight), 'code': [file_identity(p, content=True) for p in
              [Path(__file__), ROOT/'scripts/revision/feature_interventions.py',
               ROOT/'src/models/interventions.py', ROOT/'src/models/esm3_hooks.py', ROOT/'src/sae/model.py']],
              'cases': [], 'complete': False}
    path = OUT/'feature_intervention_runtime_check.json'
    atomic_json(path, report)
    model, tokenizers = load_esm3(device='cuda')
    # Alphabet repetition is a software fixture, not a designed protein.
    inputs = tokenize_sequence('ACDEFGHIKLMNPQRSTVWY'*3, tokenizers)
    for layer in [16, 33]:
        checkpoint = ROOT/f'models/sae/esm3_scaled/S_layer_{layer}_topk/best.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        sae = TopKSAE(saved['sae_config']).cuda().float().eval()
        sae.load_state_dict(saved['model_state_dict'])
        block = model.transformer.blocks[layer]

        def forward(context):
            with context, cache_residual_stream(model, 'esm3', [layer, 42]) as cache:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(**inputs)
            logits = output.sequence_logits.detach().float().cpu()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite model output')
            return logits, cache

        normal, cache = forward(nullcontext())
        hidden = cache[layer].cuda()
        with torch.no_grad():
            z = sae.encode(hidden[0, 1:-1].float())
        feature = int(z.mean(0).argmax())
        mask = torch.ones(hidden.shape[:-1], dtype=torch.bool, device='cuda')
        mask[:, [0, -1]] = False
        case = {'layer': layer, 'checkpoint': file_identity(checkpoint, content=True),
                'feature_selection': 'Maximum mean activation on synthetic residue positions, software test only',
                'feature': feature, 'conditions': {}}
        conditions = [('residual_empty', [], 'residual_preserving', None),
                      ('reconstruction_empty', [], 'reconstruction', None),
                      ('residual_single', [feature], 'residual_preserving', None),
                      ('reconstruction_single', [feature], 'reconstruction', None),
                      ('residue_only_single', [feature], 'residual_preserving', mask)]
        for name, indices, mode, token_mask in conditions:
            logits, altered = forward(feature_intervention(block, sae, indices, mode, token_mask))
            with torch.no_grad():
                expected = intervene_hidden(hidden, sae, indices, mode, token_mask).cpu()
            direct_error = float((altered[layer]-expected).abs().max())
            case['conditions'][name] = {
                'hook_vs_direct_max_abs': direct_error,
                'sequence_logits_max_abs_change': float((normal-logits).abs().max()),
                'downstream_hidden_relative_l2': float((altered[42].float()-cache[42].float()).norm()/cache[42].float().norm()),
                'special_positions_unchanged': bool(torch.equal(altered[layer][:, [0, -1]], cache[layer][:, [0, -1]]))}
            if direct_error != 0:
                raise ValueError('Hook and direct intervention disagree')
        legacy_logits, legacy_cache = forward(sae_feature_ablation(model, 'esm3', sae, layer, []))
        restored, _ = forward(nullcontext())
        case['legacy_empty_logits_max_abs_change'] = float((legacy_logits-normal).abs().max())
        case['legacy_empty_hidden_relative_l2'] = float((legacy_cache[layer].float()-cache[layer].float()).norm()/cache[layer].float().norm())
        case['restored_logits_exact'] = bool(torch.equal(normal, restored))
        case['passed'] = (case['restored_logits_exact']
                          and case['conditions']['residual_empty']['sequence_logits_max_abs_change'] == 0
                          and case['conditions']['residue_only_single']['special_positions_unchanged']
                          and case['legacy_empty_hidden_relative_l2'] > 0)
        report['cases'].append(case)
        atomic_json(path, report)
        print(f'layer={layer} passed={case["passed"]}', flush=True)
        del sae
    report['complete'] = True
    report['passed'] = all(case['passed'] for case in report['cases'])
    atomic_json(path, report)
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases'])}), flush=True)
    if not report['passed']:
        raise RuntimeError('Feature intervention runtime check failed')


if __name__ == '__main__':
    main()
