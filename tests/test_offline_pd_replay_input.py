import json
from pathlib import Path

import pytest

from scripts.prepare_offline_pd_replay import extract


def fixture():
    return json.loads((Path(__file__).parent/'fixtures/offline_pd_replay_actual_capture.json').read_text())


def test_extracts_actual_captured_force_and_full_precision_reference():
    f=fixture()
    r=extract(f['result'],f['records'],f['native_records'])
    assert r['capture_utime_us']==8299999
    assert r['N']==10
    assert r['u'][0]==[0.07269641291250833,0.19097582474464056,-0.05196702394053141]
    assert r['executed_reference_exactly_matches_native_actor_plan']


@pytest.mark.parametrize('change',['native_position','force_changed_during_replay','missing_arm_timestamp','reposition','nan_force'])
def test_refuses_mismatched_or_invalid_physical_replay_input(change):
    f=fixture()
    if change=='native_position': f['native_records'][0]['raw_states'][1][0] += 1e-5
    elif change=='force_changed_during_replay': f['records'][1]['force_knots_n'][0][0] += .01
    elif change=='missing_arm_timestamp': f['result']['trace'][0]['measurement_utime_us'] += 10_000
    elif change=='reposition': f['records'][0]['c3_mode']=False
    else: f['records'][0]['force_knots_n'][0][0]=float('nan')
    with pytest.raises(ValueError): extract(f['result'],f['records'],f['native_records'])
