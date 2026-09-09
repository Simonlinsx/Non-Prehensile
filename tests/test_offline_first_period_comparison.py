import copy

import pytest

from scripts.analyze_offline_pd_first_period import analyze


def fixture():
    initial=[0.]*19
    initial[3]=1.
    metadata=dict(capture_utime_us=100000,x=[initial])
    states=[dict(utime_us=100000,target_position_m=[0.,0.,0.],target_quaternion_wxyz=[1.,0.,0.,0.]),
            dict(utime_us=150000,target_position_m=[.01,0.,0.],target_quaternion_wxyz=[1.,0.,0.,0.])]
    fine=[]
    for i in range(1,21):
        state=initial.copy()
        state[7]=i*.00025
        fine.append(dict(time_s=i*.0025,state=state))
    trace=[dict(measurement_utime_us=100000+i*10000,raw_task_target_c3_m=[0.,0.,0.],
                governed_task_target_c3_m=[0.,0.,0.],osc_reference_tip_velocity_m_s=[0.,0.,0.],
                applied_feedforward_force_n=[0.,0.,0.],semantic_c1_guard_active=False,
                forbidden_robot_contact=False,legal_safe_robot_contact=True,
                robot_target_contact_force_n_by_sensor={'target_hand_contacts_left':1.}) for i in range(1,6)]
    references=[dict(time_s=i*.01,position_m=[0.,0.,0.],velocity_m_s=[0.,0.,0.],external_force_n=[0.,0.,0.]) for i in range(5)]
    return metadata,dict(control_period_s=.01,trace=trace),states,fine,references,dict(dt_s=.075,resolution=30,sampled_reference_clocks=True)


def test_exact_native_endpoint_and_contact_coverage():
    args=fixture()
    report=analyze(*args)
    assert report['eligible_unshielded_contact']
    assert report['final_prediction_errors']['xy_m']==pytest.approx(.005)
    assert report['final_prediction_errors']['so3_rad']==pytest.approx(0)
    args[1]['trace'].pop()
    report=analyze(*args)
    assert not report['eligible_unshielded_contact']
    assert report['missing_servo_timestamps']==[150000]


def test_missing_fine_step_or_measured_endpoint_is_rejected():
    args=list(fixture())
    args[3]=copy.deepcopy(args[3])
    args[3].pop(10)
    with pytest.raises(ValueError,match='without missing steps'):
        analyze(*args)
    args=fixture()
    args[2][-1]['utime_us']+=2
    with pytest.raises(ValueError,match='exact timestamp'):
        analyze(*args)
