"""Offline ESM-3 numerical checks on synthetic inputs, not biological evaluation."""
import importlib.util
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity
from secondary_structure import ss3_predictions, SS8_VOCAB

sys.path.insert(0, str(ROOT/'src'))
from models.esm3_hooks import load_esm3, tokenize_sequence
from models.interventions import head_ablation_esm3


def differences(a, b):
    a, b = a.float(), b.float()
    delta = a-b
    return {'max_abs': delta.abs().max().item(),
            'relative_l2': (delta.norm()/a.norm().clamp_min(1e-12)).item(),
            'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback for full model diagnostic')
    torch.set_num_threads(2)
    torch.manual_seed(2288)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = ROOT/'scripts/scaled_1.5M/20_attention_atlas.py'
    spec = importlib.util.spec_from_file_location('historical_attention_atlas', source)
    atlas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atlas)
    model, tokenizers = load_esm3(device='cuda')
    report = {'scope': 'Synthetic software/numerical checks only; not corrected biological ablation results or validation of historical protein-level aggregates.',
              'seed': 2288, 'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'inputs': [file_identity(p, content=True) for p in [source, ROOT/'src/models/interventions.py',
                          ROOT/'src/models/esm3_hooks.py', Path(__file__)]], 'cases': []}
    for block_id in [0, 2, 9, 33, 47]:
        # Cached model parameters are bfloat16; use isolated fp32 copies for
        # controlled fp32 versus autocast comparisons without mutating the model.
        attn = copy.deepcopy(model.transformer.blocks[block_id].attn).float()
        wrapper = SimpleNamespace(transformer=SimpleNamespace(blocks=[SimpleNamespace(attn=attn)]))
        for length in [32, 128]:
            x = torch.randn(1, length, attn.d_model, device='cuda')
            for packed in [False, True]:
                seq_id = torch.arange(length, device='cuda')[None]//(length//2) if packed else None
                for dtype in [torch.float32, torch.bfloat16]:
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                        native = attn(x, seq_id)
                        with atlas.AttentionCapture(wrapper) as capture:
                            captured = attn(x, seq_id)
                        weights = capture.attention_weights[0]
                        head = 7
                        def zero_output_head(module, args):
                            value = args[0].clone()
                            value[..., head*attn.d_head:(head+1)*attn.d_head] = 0
                            return (value,)
                        handle = attn.out_proj.register_forward_pre_hook(zero_output_head)
                        try:
                            reference_ablation = attn(x, seq_id)
                        finally:
                            handle.remove()
                        with head_ablation_esm3(wrapper, 0, head):
                            historical_ablation = attn(x, seq_id)
                        restored = attn(x, seq_id)
                    capture_error = differences(native, captured)
                    ablation_error = differences(reference_ablation, historical_ablation)
                    restore_error = differences(native, restored)
                    tolerance = 1e-5 if dtype == torch.float32 else .03
                    case = {'block': block_id, 'length': length, 'packed_sequence_mask': packed,
                            'dtype': str(dtype), 'capture': capture_error, 'ablation': ablation_error,
                            'restoration': restore_error, 'relative_l2_tolerance': tolerance,
                            'attention_row_sum_max_error': float((weights.sum(-1)-1).abs().max())}
                    case['passed'] = all(v['finite'] and v['relative_l2'] <= tolerance for v in
                                         [capture_error, ablation_error, restore_error])
                    report['cases'].append(case)
                    atomic_json(OUT/'intervention_runtime_check.json', report)
                    print(f'block={block_id} length={length} packed={packed} dtype={dtype} passed={case["passed"]}', flush=True)
    # Balanced alphabet repeated solely as a software fixture, not a designed protein.
    inputs = tokenize_sequence('ACDEFGHIKLMNPQRSTVWY'*3, tokenizers)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(**inputs)
    logits = output.secondary_structure_logits[0, 1:-1].float().cpu().numpy()
    decoded = ss3_predictions(logits)
    vocab = tokenizers.secondary_structure.vocab
    report['decoder_smoke'] = {'residues': len(decoded), 'logit_shape': list(logits.shape),
                               'finite': bool(np.isfinite(logits).all()),
                               'valid_class_ids': bool(np.isin(decoded, [0, 1, 2]).all()),
                               'runtime_vocabulary': vocab, 'expected_vocabulary': list(SS8_VOCAB)}
    report['passed'] = (all(c['passed'] for c in report['cases']) and
                        report['decoder_smoke']['valid_class_ids'] and list(vocab) == list(SS8_VOCAB))
    atomic_json(OUT/'intervention_runtime_check.json', report)
    print(json.dumps(report['decoder_smoke'], indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('Numerical check failed; inspect output before biological reruns')


if __name__ == '__main__':
    main()
