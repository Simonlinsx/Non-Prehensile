import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fr3_semantic_cost_map import query, separation


def box(x,y,w,h):return np.array([[x-w,y-h],[x+w,y-h],[x+w,y+h],[x-w,y+h]])


def test_sweep_detects_blocked_interior_even_when_endpoints_are_clear():
    shape=box(0,0,.01,.01);obstacle=dict(id='cup',mode='forbidden',weight=1,polygon=box(.1,0,.02,.02))
    result=query(shape,[0,0],[.2,0],0,[obstacle])
    assert not result['admissible'] and result['hard_rejections']==['cup']


def test_same_geometry_different_semantics_keeps_contact_observable_and_finite():
    shape=box(0,0,.01,.01);o=dict(id='doll',mode='acceptable',weight=.15,polygon=box(.1,0,.02,.02))
    r=query(shape,[0,0],[.2,0],0,[o])
    assert r['admissible'] and r['soft_cost']==.15 and r['clearance_m']['doll']<0


def test_rotation_changes_feasibility_and_checks_swept_interior():
    shape=box(0,0,.10,.008);o=dict(id='balloon',mode='forbidden',weight=1,polygon=box(.065,.065,.006,.006))
    assert query(shape,[0,0],[0,0],0,[o])['admissible']
    assert not query(shape,[0,0],[0,0],np.pi/2,[o])['admissible']


def test_translation_rotation_equivariance_and_no_nan_acceptance():
    a=box(0,0,.1,.02);b=box(.25,.2,.03,.03)
    theta=.6;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    assert abs(separation(a,b)-separation(a@R.T+3,b@R.T+3))<1e-10
    import pytest
    with pytest.raises(ValueError):query(a,[0,0],[float('nan'),0],0,[])


def test_rotation_priority_still_uses_a_unilateral_inward_push():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/IsaacLab_nonPrehensile'))
    from dapl.contact_planner.contact_servo import goal_wrench_contact_axis
    contact=np.array([-.05,0.]);com=np.zeros(2)
    axis,requested,achieved=goal_wrench_contact_axis(goal_delta_xy_m=np.array([-.008,0.]),
        contact_point_xy_m=contact,object_com_xy_m=com,signed_yaw_error_rad=.3,
        object_yaw_rate_rad_s=0.,yaw_moment_gain_m_per_rad=.04,
        yaw_rate_moment_gain_m_s_per_rad=.02,max_axis_deviation_rad=np.pi)
    assert np.dot(axis,com-contact)>0
    assert abs(achieved-requested)<1e-10 and requested<0
    # This legal rotational contact would be rejected by a forward-translation
    # preference, despite achieving the requested signed moment.
    assert np.dot(axis,[-1.,0.])<.2
