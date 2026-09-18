"""Offline synthetic model/SAE integration check for an isolated GPU environment."""
import argparse
from contextlib import nullcontext
import fcntl
import os
from pathlib import Path
import sys

import torch

from common import ROOT
from probe_checkpoint import atomic_json, file_identity
from capture_reproducibility import stable_digest
from feature_interventions import feature_intervention, intervene_hidden

sys.path.insert(0, str(ROOT/'src'))
from models.esm2_hooks import load_esm2
from models.esm3_hooks import load_esm3, tokenize_sequence
from models.interventions import get_layers, cache_residual_stream
from sae.model import TopKSAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['esm2', 'esm3'], required=True)
    args = parser.parse_args()
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        parser.error('Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1; no model downloads allowed')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this inference check')
    directory = ROOT/'revision/reproducibility/gpu_runtime'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'.writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = directory/f'{args.model}.json'
    source = [Path(__file__), ROOT/'scripts/revision/feature_interventions.py',
              ROOT/'src/models'/f'{args.model}_hooks.py', ROOT/'src/models/interventions.py',
              ROOT/'src/sae/model.py']
    report = {'scope': 'Synthetic cached-model integration test only; not a biological evaluation or full-cohort reproduction.',
              'model': args.model, 'seed': 2288, 'fixture': 'Three repetitions of the canonical amino-acid alphabet',
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'device': torch.cuda.get_device_name(),
              'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'sources': [file_identity(p, content=True) for p in source],
              'complete': False, 'passed': False, 'conditions': {}}
    atomic_json(output, report)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(2288)
        torch.backends.cuda.matmul.allow_tf32 = False
        fixture = 'ACDEFGHIKLMNPQRSTVWY'*3
        if args.model == 'esm3':
            from esm.utils.constants.esm3 import data_root
            weights = [data_root('esm3')/'data/weights/esm3_sm_open_v1.pth']
            model, tokenizers = load_esm3(device='cuda')
            inputs = tokenize_sequence(fixture, tokenizers)
            checkpoint = ROOT/'models/sae/esm3_scaled/S_layer_16_topk/best.pt'
        else:
            from huggingface_hub import snapshot_download
            cache = Path(snapshot_download('facebook/esm2_t33_650M_UR50D', local_files_only=True))
            weights = sorted(p for p in cache.iterdir() if p.is_file())
            model, tokenizer = load_esm2(device='cuda')
            inputs = {k: v.cuda() for k, v in tokenizer(fixture, return_tensors='pt').items()}
            checkpoint = ROOT/'models/sae/esm2_scaled/layer_16_topk/best.pt'
        report['weights'] = [{'path': str(p), **stable_digest(p)} for p in weights]
        report['checkpoint'] = file_identity(checkpoint, content=True)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        sae = TopKSAE(saved['sae_config']).cuda().float().eval()
        sae.load_state_dict(saved['model_state_dict'])
        block = get_layers(model, args.model)[16]

        def forward(context):
            precision = torch.autocast('cuda', dtype=torch.bfloat16) if args.model == 'esm3' else nullcontext()
            with context, cache_residual_stream(model, args.model, [16]) as hidden:
                with torch.no_grad(), precision:
                    result = model(**inputs)
            logits = (result.sequence_logits if args.model == 'esm3' else result.logits).float().detach().cpu()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite output logits')
            return logits, hidden[16]

        normal, hidden_cpu = forward(nullcontext())
        hidden = hidden_cpu.cuda()
        with torch.no_grad():
            feature = int(sae.encode(hidden[0, 1:-1].float()).mean(0).argmax())
        report['selected_fixture_coordinate'] = feature
        report['logit_shape'] = list(normal.shape)
        for mode, indices in [('residual_preserving', []), ('reconstruction', []),
                              ('residual_preserving', [feature]), ('reconstruction', [feature])]:
            actual_logits, actual_hidden = forward(feature_intervention(block, sae, indices, mode))
            with torch.no_grad():
                expected = intervene_hidden(hidden, sae, indices, mode).cpu()
            key = mode + ('_single' if indices else '_empty')
            row = {'hook_matches_direct': bool(torch.equal(actual_hidden, expected)),
                   'max_abs_logit_change': float((normal-actual_logits).abs().max())}
            report['conditions'][key] = row
            if not row['hook_matches_direct']:
                raise ValueError(f'Hook/direct mismatch: {key}')
            if key == 'residual_preserving_empty' and row['max_abs_logit_change'] != 0:
                raise ValueError('Empty residual intervention changed inference')
        restored, _ = forward(nullcontext())
        report['restored_logits_exact'] = bool(torch.equal(normal, restored))
        if not report['restored_logits_exact']:
            raise ValueError('Restoration mismatch')
        report.update(complete=True, passed=True)
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        atomic_json(output, report)
        raise
    atomic_json(output, report)
    print(f'{args.model}: cached inference, four hook/direct checks, no-op and restoration passed')


if __name__ == '__main__':
    main()
