"""Count recorded nonfinite candidate-cost boundaries without inferring task outcomes."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(scene_directory):
    path = scene_directory / 'effect_audit.jsonl'
    costs, references = {}, {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('channel') == 'SAMPLE_COSTS':
                stamp = row['utime_us']
                if stamp in costs:
                    raise ValueError('Duplicate candidate-cost timestamp')
                costs[stamp] = row['trajectories'][0]['datapoints'][0]
            elif row.get('event') == 'task_reference':
                # A watchdog can hold an older plan while changing command mode.
                # Match the command's measured-state clock, not that old plan ID.
                stamp = row['relay_state_utime_us']
                if stamp in references and references[stamp] != row['c3_mode']:
                    raise ValueError('Conflicting command mode for one plan')
                references[stamp] = row['c3_mode']
    report = dict(total_plans=len(costs), matched_command_modes=0, missing_command_modes=0,
                  invalid_current_finite_alternative=0, invalid_previous_finite_alternative=0,
                  all_costs_invalid=0, invalid_current_c3_selected=0,
                  invalid_previous_reposition_selected=0, examples=[])
    if not costs:
        raise ValueError('No recorded candidate costs')
    for stamp, values in costs.items():
        if len(values) < 2:
            raise ValueError('Missing current and previous candidate costs')
        modes = [references[t] for t in (stamp-1, stamp, stamp+1) if t in references]
        if len(modes) > 1:
            raise ValueError('Ambiguous cost/command timestamp match')
        mode = modes[0] if modes else None
        report['matched_command_modes' if modes else 'missing_command_modes'] += 1
        finite = [i for i, value in enumerate(values[1:], 1) if math.isfinite(value)]
        if not any(map(math.isfinite, values)):
            report['all_costs_invalid'] += 1
        if not finite:
            continue
        if not math.isfinite(values[0]):
            report['invalid_current_finite_alternative'] += 1
            report['invalid_current_c3_selected'] += mode is True
        if not math.isfinite(values[1]):
            report['invalid_previous_finite_alternative'] += 1
            report['invalid_previous_reposition_selected'] += mode is False
            if len(report['examples']) < 5:
                report['examples'].append(dict(utime_us=stamp,
                    current_cost=values[0] if math.isfinite(values[0]) else 'invalid',
                    previous_cost='invalid', best_finite_cost=min(values[i] for i in finite),
                    selected_c3_mode=mode))
    report['interpretation'] = ('Observed boundary cases, not counterfactual closed-loop outcomes. '
                                'Other mode/progress/height gates can affect choices.')
    report['effect_audit_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-directory', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.scene_directory)
    (args.scene_directory / 'nonfinite_cost_boundary_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'examples'}))


if __name__ == '__main__':
    main()
