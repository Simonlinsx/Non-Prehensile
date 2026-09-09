import copy
import json
from pathlib import Path

import pytest
from scripts.analyze_diagnostic_reference_replay import analyze_replay


def fixture():
    return json.loads((Path(__file__).parent/'fixtures/reference_replay_actual_window.json').read_text())


def test_real_window_has_matching_plan_but_reports_downstream_guard_changes():
    f=fixture();a=analyze_replay(f['records'],f['result'])
    assert a['acceptance_eligible'] is False
    assert a['matched_reference_count']==14
    assert a['duration_s']==.675
    assert a['endpoint_xy_error_m']==pytest.approx(.0031931501065092347)
    assert a['semantic_guard_active_samples']==3
    assert a['servo_raw_reference_vs_ideal_foh_m']['maximum']<1e-12
    assert a['servo_governor_correction_m']['maximum']>.03
    assert a['positive_legal_hand_force_samples']==6


@pytest.mark.parametrize('mutation',['changed_plan','missing_reference','missing_end','not_diagnostic'])
def test_rejects_incomplete_or_mixed_execution(mutation):
    f=fixture()
    if mutation=='changed_plan':
        next(v for v in f['records'] if v['event']=='task_reference')['plan_utime_us']+=50_000
    elif mutation=='missing_reference':
        f['records']=[v for v in f['records'] if not (v['event']=='task_reference' and v['relay_state_utime_us']==8_100_000)]
    elif mutation=='missing_end':
        f['records']=[v for v in f['records'] if v['event']!='diagnostic_reference_replay_end']
    else:
        f['result']['diagnostic_reference_replay']=False
    with pytest.raises(ValueError):analyze_replay(f['records'],f['result'])


def test_readonly_dual_predictions_match_actual_selected_output():
    from scripts.analyze_diagnostic_reference_replay import compare_dual_predictions
    f=json.loads((Path(__file__).parent/'fixtures/reference_replay_dual_pd_actual_window.json').read_text())
    a=compare_dual_predictions(f['native_records'],f['records'],f['result'])
    assert a['recorded_selected_pd_max_state_difference']==0
    assert a['modes']['zoh']['target_endpoint_xy_error_m']==pytest.approx(.016281432821102053)
    assert a['modes']['foh']['target_endpoint_xy_error_m']==pytest.approx(.015749791054174837)
    f['native_records'][0]['zoh_states'][0][7]+=.01
    with pytest.raises(ValueError):compare_dual_predictions(f['native_records'],f['records'],f['result'])


def test_calibrated_predictions_use_same_measured_window_and_reject_mixed_initial_state():
    from scripts.analyze_diagnostic_reference_replay import compare_calibrated_predictions
    f=json.loads((Path(__file__).parent/'fixtures/reference_replay_calibrated_pd_actual_window.json').read_text())
    report=compare_calibrated_predictions(f['calibrations'],f['native_records'],f['records'],f['result'])
    assert report['acceptance_eligible'] is False
    assert report['variants'][0]['target_endpoint_xy_error_m']==pytest.approx(.025335852347499237)
    assert report['variants'][1]['target_endpoint_xy_error_m']==pytest.approx(.0235850724429318)
    f['calibrations'][0]['states'][0][7]+=.01
    with pytest.raises(ValueError):
        compare_calibrated_predictions(f['calibrations'],f['native_records'],f['records'],f['result'])


def test_declared_geometry_failure_is_reported_and_missing_variants_are_rejected():
    from scripts.analyze_diagnostic_reference_replay import compare_calibrated_predictions
    f=json.loads((Path(__file__).parent/'fixtures/reference_replay_calibrated_pd_actual_window.json').read_text())
    f['result']['diagnostic_pd_contact_relinearization']=True
    with pytest.raises(ValueError):
        compare_calibrated_predictions(f['calibrations'],f['native_records'],f['records'],f['result'])
    for mass in [.057,1.]:
        f['calibrations'].append(dict(utime_us=8350000,dt_s=.075,actor_mass_kg=mass,
                                     contact_geometry='relinearized',failed_fine_step=17,error='invalid contact normal'))
    report=compare_calibrated_predictions(f['calibrations'],f['native_records'],f['records'],f['result'])
    assert [v['prediction_valid'] for v in report['variants']]==[True,True,False,False]
    assert all('target_endpoint_xy_error_m' not in v for v in report['variants'] if not v['prediction_valid'])
