import json
from pathlib import Path

import pytest

from scripts.analyze_offline_pd_resolution import audit_sampled_references


def run_fixture(tmp_path, change=None):
    f = json.loads((Path(__file__).parent / 'fixtures/offline_pd_sampled_reference_actual.json').read_text())
    if change == 'force':
        f['references'][0]['external_force_n'][0] *= -1
    elif change == 'timestamp':
        for row in f['result']['trace']:
            row['measurement_utime_us'] -= 10000
    elif change == 'duplicate':
        f['result']['trace'].append(f['result']['trace'][0])
    elif change == 'sparse':
        del f['result']['trace'][1]
    path = tmp_path / 'native.log'
    path.write_text(''.join('C3_OFFLINE_PD_REFERENCE ' + json.dumps(r) + '\n' for r in f['references']))
    return audit_sampled_references(path, f['result'], f['capture_us'], f['duration'])


def test_actual_pre_step_commands_match_post_step_measurement_clock(tmp_path):
    report = run_fixture(tmp_path)
    assert report['matched_samples'] == 4
    assert report['complete_trace_coverage']
    assert max(report['raw_reference_maximum_absolute_error'].values()) < 1e-12
    assert report['applied_reference_maximum_absolute_error']['position_m'] > .0007


@pytest.mark.parametrize('change', ['force', 'timestamp', 'duplicate'])
def test_rejects_wrong_force_clock_and_duplicate_measurements(tmp_path, change):
    with pytest.raises(ValueError):
        run_fixture(tmp_path, change)


def test_missing_measurement_is_reported_and_never_interpolated(tmp_path):
    report = run_fixture(tmp_path, 'sparse')
    assert not report['complete_trace_coverage']
    assert report['matched_samples'] == 3
    assert report['unavailable_trace_times_s'] == [.01]
