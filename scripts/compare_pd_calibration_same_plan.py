#!/usr/bin/env python3
"""Compare a counterfactual model to another run only after same-plan checks."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_diagnostic_reference_replay import compare_calibrated_predictions


def load_lines(path, prefix=None):
    lines = path.read_text().splitlines()
    return [json.loads(line if prefix is None else line.split(' ', 1)[1])
            for line in lines if line.strip() and (prefix is None or line.startswith(prefix + ' '))]


def one_at(records, stamp, event=None):
    rows = [row for row in records if abs(row.get('utime_us', -1) - stamp) <= 1
            and (event is None or row.get('event') == event)]
    if len(rows) != 1:
        raise ValueError('Need exactly one record for the captured plan')
    return rows[0]


def require_equal(left, right, keys):
    differences = {}
    for key in keys:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'Invalid matched-plan arrays: {key}')
        differences[key] = float(np.max(np.abs(a - b)))
        if differences[key] != 0:
            raise ValueError(f'Cross-run plan differs: {key}')
    return differences


def compare(model, execution):
    ma = load_lines(model / 'effect_audit.jsonl')
    ea = load_lines(execution / 'effect_audit.jsonl')
    starts = [row for row in ea if row['event'] == 'diagnostic_reference_replay_start']
    if len(starts) != 1:
        raise ValueError('Execution must contain one captured reference')
    stamp = starts[0]['plan_utime_us']
    solver_differences = require_equal(one_at(ma, stamp, 'solver_diagnostic'),
                                      one_at(ea, stamp, 'solver_diagnostic'), ['x', 'u', 'lambda', 'z', 'delta'])
    mn = load_lines(model / 'controller_online.log', 'C3_PD_REFERENCE_COMPARISON')
    en = load_lines(execution / 'controller_online.log', 'C3_PD_REFERENCE_COMPARISON')
    native_differences = require_equal(one_at(mn, stamp), one_at(en, stamp),
                                      ['raw_states', 'zoh_states', 'foh_states'])
    mc = load_lines(model / 'controller_online.log', 'C3_PD_CALIBRATION')
    ec = load_lines(execution / 'controller_online.log', 'C3_PD_CALIBRATION')
    frozen_differences = {}
    for mass in [.057, 1.]:
        select = lambda rows: [r for r in rows if r['actor_mass_kg'] == mass
                              and r.get('contact_geometry', 'frozen') == 'frozen']
        frozen_differences[str(mass)] = require_equal(one_at(select(mc), stamp), one_at(select(ec), stamp), ['states'])['states']
    result = json.loads((execution / 'result.json').read_text())
    model_result = json.loads((model / 'result.json').read_text())
    if not model_result.get('diagnostic_pd_contact_relinearization'):
        raise ValueError('Model run does not declare contact relinearization diagnostics')
    # This flag describes the compared counterfactual source, not the physical
    # execution. Keep both source paths explicit; do not change either result.
    comparison = compare_calibrated_predictions(mc, mn, ea, dict(result, diagnostic_pd_contact_relinearization=True))
    return {'schema': 'nonprehensile.cross_run_same_plan_model_comparison.v1',
            'acceptance_eligible': False, 'physical_execution': str(execution),
            'model_diagnostics_source': str(model), 'plan_utime_us': stamp,
            'solver_lcm_max_differences': solver_differences,
            'full_precision_native_max_differences': native_differences,
            'frozen_calibration_max_differences': frozen_differences, 'comparison': comparison,
            'limitation': 'Neighboring overlapping references are not independent held-out scenes.',
            'artifact_sha256': {str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for root in (model, execution)
                               for name in ('result.json', 'effect_audit.jsonl', 'controller_online.log')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--execution-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.model_dir, args.execution_dir)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['comparison']))


if __name__ == '__main__':
    main()
