"""Exercise composition of target-map and whole-arm guards at a microstep."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fr3_semantic_cost_map as runtime
import fr3_semantic_sweep3d as volume


def test_volume_microstep_calls_both_distinct_interfaces(monkeypatch):
    base=object();ik=object();bridge=object();calls=[]
    monkeypatch.setenv('FR3_SEMANTIC_BACKEND','volume3d')
    monkeypatch.setattr(runtime,'live_obstacles',lambda b: [])
    monkeypatch.setattr(runtime,'policy',lambda: {'objects':[]})
    monkeypatch.setattr(runtime,'record',lambda *a,**k: None)
    def map_factory(b,rules):
        assert b is base and rules==[];calls.append('target')
        return SimpleNamespace(query=lambda *args:dict(admissible=True)),{}
    def arm_factory(model,b):
        assert model is ik and b is base;calls.append('arm')
        return SimpleNamespace(segment=lambda *args:dict(clearance_m=.1))
    monkeypatch.setattr(volume,'from_live_scene',map_factory)
    monkeypatch.setitem(sys.modules,'fr3_execution_runtime.clutter_route',SimpleNamespace(from_live_scene=arm_factory))
    result=runtime.micro_check(base,ik,np.zeros(7),np.zeros(7),bridge,np.zeros((2,3)),np.zeros(3),np.array([1,0]),.002,0.)
    assert result and calls==['target','arm']


def test_candidate_prediction_uses_actual_nominal_servo_direction():
    # Deliberately omit raw directions: servo-mode scoring must not consult them.
    c=SimpleNamespace(push_distance=np.array([[.015]]))
    checks={0:dict(axis=[.6,.8],achieved_moment=-.007)}
    delta,yaw=runtime.candidate_motion(c,0,.5,.002,checks)
    np.testing.assert_allclose(delta,[.009,.012,0])
    np.testing.assert_allclose(yaw,-.02625)
