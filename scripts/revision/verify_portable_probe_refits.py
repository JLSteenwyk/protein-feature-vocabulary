"""Compare a model's refitted controls with the committed manuscript snapshot."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT
from probe_checkpoint import atomic_json
from stage_revision_release import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run-dir', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--model', required=True, choices=['esm2', 'esm3', 'composition'])
    args = parser.parse_args()
    report_path = args.run_dir/'refit_verification.json'
    atomic_json(report_path, {'model': args.model, 'complete': False})
    baseline = args.base_run_dir/'combined_probe_controls.json'
    result = args.run_dir/'probe_controls.json'
    expected, actual = json.loads(baseline.read_text()), json.loads(result.read_text())
    identity = {'baseline': digest(baseline), 'result': digest(result),
                'verifier': digest(Path(__file__))}
    atomic_json(report_path, {'model': args.model, 'identity': identity, 'complete': False})
    if expected['sample_counts'] != actual['sample_counts']:
        raise ValueError('Refit sample counts differ')
    if expected['models'][args.model] != actual['models'][args.model]:
        raise ValueError('Refit control inventory or scientific metrics differ')
    checks = {}
    for name, reference in expected['prediction_files'][args.model].items():
        source = ROOT/reference['path']
        if digest(source)['sha256'] != reference['sha256']:
            raise ValueError(f'Archived prediction identity differs: {name}')
        target = args.run_dir/f'probe_predictions_{args.model}_{name}.npz'
        with np.load(source, allow_pickle=False) as a, np.load(target, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError(f'Prediction array inventory differs: {name}')
            for key in a.files:
                left, right = a[key], b[key]
                if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                    raise ValueError(f'Refitted prediction differs: {name}/{key}')
            if not np.isfinite(b['score']).all():
                raise ValueError('Nonfinite refit scores')
        fit = args.run_dir/f'probe_fit_{args.model}_{name}.npz'
        with np.load(fit, allow_pickle=False) as saved:
            if not np.isfinite(saved['coef']).all() or not np.isfinite(saved['intercept']).all():
                raise ValueError('Nonfinite refit coefficients')
        checks[name] = {'metrics_exact': True, 'prediction_arrays_exact': True,
                        'reference': digest(source), 'prediction': digest(target), 'fit': digest(fit)}
    if digest(baseline) != identity['baseline'] or digest(result) != identity['result']:
        raise ValueError('Result snapshot changed during comparison')
    atomic_json(report_path, {
        'scope': 'Metrics, selection grid, intervals and test predictions compared with manuscript snapshot; new coefficients checked finite, not compared where historical coefficients are unavailable.',
        'model': args.model, 'identity': identity, 'checks': checks, 'complete': True})
    print(f'Exact refit metrics and predictions: {len(checks)} controls ({args.model}).')


if __name__ == '__main__':
    main()
