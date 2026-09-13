import numpy as np
import trimesh
from scripts.fr3_semantic_volume import distance_to_volume, point_contact_cost


def test_exact_box_volume_distance():
    mesh = trimesh.creation.box(extents=[.1,.1,.1])
    p = np.array([[0,0,0],[.05,0,0],[.056,0,0],[.056,.058,0]])
    np.testing.assert_allclose(distance_to_volume(p, mesh), [0,0,.006,.010], atol=1e-10)


def test_numeric_soft_cost_and_nontradeable_forbidden():
    mesh = trimesh.creation.box(extents=[.1,.1,.1])
    p = np.array([[0,0,0],[.056,0,0],[.063,0,0]])
    soft = {'id':'doll','mode':'acceptable','weight':.15}
    result = point_contact_cost(p, [(soft, mesh)])
    np.testing.assert_allclose(result['total_cost'], [.15,.075,0], atol=1e-10)
    hard = {'id':'cup','mode':'forbidden','weight':1.}
    result = point_contact_cost(p, [(soft,mesh),(hard,mesh)])
    assert np.isinf(result['total_cost'][:2]).all()
    assert result['total_cost'][2] == 0


def test_height_and_hollow_interior_preserved():
    left = trimesh.creation.box(extents=[.01,.1,.1]);left.apply_translation([-.045,0,.05])
    right = left.copy();right.apply_translation([.09,0,0])
    mesh = trimesh.util.concatenate([left,right])
    # Space between components and above them stays empty: no global convex hull.
    p=np.array([[0,0,.05],[.045,0,.05],[.045,0,.12]])
    np.testing.assert_allclose(distance_to_volume(p,mesh), [.04,0,.02], atol=1e-10)
    # Entire protected object envelope is excluded, including cavities.
    hard={'id':'cup','mode':'forbidden','weight':1.}
    assert point_contact_cost(p,[(hard,mesh)])['forbidden'].tolist()==[True,True,False]
