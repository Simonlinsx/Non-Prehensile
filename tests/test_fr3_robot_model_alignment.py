"""Check generated FR3 physics and FK against the untouched official URDF."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_fr3', ROOT / 'scripts/prepare_fr3_robot_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MODEL = ROOT / 'data/robot_models/fr3_stock_20260909'


def poses(root, q):
    joints = root.findall('joint')
    children = {j.find('child').get('link') for j in joints}
    result = {l.get('name'): np.eye(4) for l in root.findall('link') if l.get('name') not in children}
    while joints:
        remaining = []
        for joint in joints:
            parent = joint.find('parent').get('link')
            if parent not in result:
                remaining.append(joint)
                continue
            value = q.get(joint.get('name'), 0.0)
            axis_node = joint.find('axis')
            axis = np.array([float(x) for x in axis_node.get('xyz').split()]) if axis_node is not None else np.array([1, 0, 0])
            motion = np.eye(4)
            if joint.get('type') == 'revolute':
                motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            elif joint.get('type') == 'prismatic':
                motion[:3, 3] = axis * value
            result[joint.find('child').get('link')] = result[parent] @ module.transform(joint) @ motion
        assert len(remaining) < len(joints), 'Disconnected or cyclic robot'
        joints = remaining
    return result


def test_official_fr3_physics_and_random_configuration_fk(tmp_path):
    manifest = module.prepare(MODEL/'fr3_official.urdf', MODEL/'vendor/franka_description', tmp_path)
    official = ET.parse(MODEL/'fr3_official.urdf').getroot()
    sim = ET.parse(manifest['simulation_urdf']).getroot()
    native = ET.parse(manifest['native_urdf']).getroot()
    assert len(sim.findall('link')) == 11
    for link in sim.findall('link'):
        original = official.find(f"link[@name='{link.get('name').replace('panda_', 'fr3_', 1)}']")
        for tag in ('inertial', 'collision/origin', 'visual/origin'):
            node = link.find(tag)
            if node is not None:
                assert node.attrib == original.find(tag).attrib
                assert [(n.tag, n.attrib) for n in node.iter()] == [(n.tag, n.attrib) for n in original.find(tag).iter()]
    random = np.random.default_rng(20260909)
    for _ in range(50):
        q = {name: random.uniform(limits['lower'], limits['upper']) for name, limits in manifest['joint_limits'].items()}
        q['panda_finger_joint2'] = q['panda_finger_joint1']
        off = poses(official, {k.replace('panda_', 'fr3_', 1): v for k, v in q.items()})
        current = poses(sim, q)
        for name, value in current.items():
            np.testing.assert_allclose(value, off[name.replace('panda_', 'fr3_', 1)], atol=2e-14, rtol=0)
        closed = dict(q, panda_finger_joint1=0, panda_finger_joint2=0)
        sim_closed = poses(sim, closed)
        native_closed = poses(native, closed)
        for name, value in sim_closed.items():
            np.testing.assert_allclose(value, native_closed[name], atol=2e-14, rtol=0)
    for i, armature in enumerate(manifest['armature_kg_m2'], 1):
        dynamics = official.find(f"joint[@name='fr3_joint{i}']/dynamics")
        assert armature == float(dynamics.get('gear_ratio')) ** 2 * float(dynamics.get('motor_inertia'))


def test_empty_frame_collapse_preserves_nontrivial_transform():
    root = ET.fromstring('''<robot><link name="base"><inertial/></link>
      <link name="empty"/><link name="body"><inertial/></link>
      <joint name="a" type="fixed"><parent link="base"/><child link="empty"/>
        <origin xyz="1 2 3" rpy="0.4 -0.2 0.8"/></joint>
      <joint name="b" type="revolute"><parent link="empty"/><child link="body"/>
        <origin xyz="-0.1 0.3 0.2" rpy="0.7 0.2 -0.4"/><axis xyz="0 0 1"/></joint></robot>''')
    before = poses(root, {'b': 0.51})['body']
    assert module.collapse_empty_frames(root) == ['empty']
    np.testing.assert_allclose(before, poses(root, {'b': 0.51})['body'], atol=2e-14, rtol=0)
