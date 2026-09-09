import json

import pytest

from scripts.audit_nonfinite_cost_modes import audit


def records():
    return [
        dict(event='planner_diagnostic', channel='SAMPLE_COSTS', utime_us=149999,
             trajectories=[dict(datapoints=[[float('inf'), 20., 10.]])]),
        dict(event='task_reference', plan_utime_us=149999, relay_state_utime_us=150000, c3_mode=True),
        # A watchdog holds this old plan but disables C3 at a later state time.
        dict(event='task_reference', plan_utime_us=149999, relay_state_utime_us=200000, c3_mode=False),
    ]


def test_held_plan_does_not_rewrite_earlier_command_mode(tmp_path):
    (tmp_path/'effect_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records()))
    report = audit(tmp_path)
    assert report['matched_command_modes'] == 1
    assert report['missing_command_modes'] == 0
    assert report['invalid_current_c3_selected'] == 1


def test_conflicting_modes_for_same_measured_state_are_rejected(tmp_path):
    rows = records()
    rows[-1]['relay_state_utime_us'] = 150000
    (tmp_path/'effect_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='Conflicting command mode'):
        audit(tmp_path)
