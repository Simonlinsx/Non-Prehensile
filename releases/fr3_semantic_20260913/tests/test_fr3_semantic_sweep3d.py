import numpy as np
import trimesh
from scripts.fr3_semantic_sweep3d import convex,SceneSweep3D


def scene(obstacle_position,mode='forbidden',obstacle_size=.02):
    target=convex(trimesh.creation.box(extents=[.02]*3).vertices)
    obstacle=convex(trimesh.creation.box(extents=[obstacle_size]*3).vertices)
    rule=dict(id='object',mode=mode,weight=.15)
    return SceneSweep3D(target,[0,0,0,1,0,0,0],[(rule,(obstacle,),[*obstacle_position,1,0,0,0])])


def test_interior_swept_collision_even_when_endpoints_clear():
    s=scene([.1,0,0]);assert s.query([0,0,0],[0,0,0],0)['admissible']
    r=s.query([0,0,0],[.2,0,0],0)
    assert not r['admissible'] and r['clearance_m']['object']==0


def test_3d_height_and_full_target_radius():
    # A point TCP at origin is >12 mm away, but the full box is only 6 mm away.
    s=scene([.026,0,0]);assert not s.query([0,0,0],[0,0,0],0)['admissible']
    above=scene([0,0,.05]);assert above.query([0,0,0],[0,0,0],0)['admissible']


def test_soft_cost_numeric_and_containment():
    r=scene([.026,0,0],'acceptable').query([0,0,0],[0,0,0],0)
    assert r['admissible'];np.testing.assert_allclose(r['soft_cost'],.075,atol=1e-8)
    r=scene([0,0,0],obstacle_size=.2).query([0,0,0],[0,0,0],0)
    assert not r['admissible'] # Fully contained body must not pass surface-only distance.


def test_rotating_long_body_hits_obstacle():
    target=convex(trimesh.creation.box(extents=[.2,.01,.01]).vertices)
    obstacle=convex(trimesh.creation.box(extents=[.01]*3).vertices)
    s=SceneSweep3D(target,[0,0,0,1,0,0,0],[(dict(id='o',mode='forbidden',weight=1),(obstacle,),[0,.08,0,1,0,0,0])])
    assert s.query([0,0,0],[0,0,0],0)['admissible']
    assert not s.query([0,0,0],[0,0,0],np.pi/2)['admissible']
