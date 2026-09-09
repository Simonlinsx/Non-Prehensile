"""Derive matched Isaac/Drake FR3 models from a pinned official expanded URDF."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def transform(joint):
    node = joint.find('origin')
    xyz = [float(x) for x in node.get('xyz', '0 0 0').split()] if node is not None else [0]*3
    rpy = [float(x) for x in node.get('rpy', '0 0 0').split()] if node is not None else [0]*3
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def write_origin(joint, matrix):
    origin = joint.find('origin')
    if origin is None:
        origin = ET.SubElement(joint, 'origin')
    origin.set('xyz', ' '.join(format(x, '.17g') for x in matrix[:3, 3]))
    origin.set('rpy', ' '.join(format(x, '.17g') for x in Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz')))


def collapse_empty_frames(root):
    removed = []
    for link in list(root.findall('link')):
        if link.find('inertial') is not None:
            continue
        if link.find('collision') is not None or link.find('visual') is not None:
            raise ValueError('Do not discard geometry without inertia')
        name = link.get('name')
        incoming = [j for j in root.findall('joint') if j.find('child').get('link') == name]
        children = [j for j in root.findall('joint') if j.find('parent').get('link') == name]
        if not incoming:
            if len(children) != 1 or children[0].get('type') != 'fixed' or not np.allclose(transform(children[0]), np.eye(4), atol=1e-14):
                raise ValueError('Unsupported nonidentity empty root')
            root.remove(children[0])
        else:
            if len(incoming) != 1 or incoming[0].get('type') != 'fixed':
                raise ValueError('Cannot collapse movable massless link')
            parent = incoming[0]
            for child in children:
                write_origin(child, transform(parent) @ transform(child))
                child.find('parent').set('link', parent.find('parent').get('link'))
            root.remove(parent)
        root.remove(link)
        removed.append(name)
    return removed


def prepare(official, package_root, output):
    root = ET.parse(official).getroot()
    if root.get('name') != 'fr3':
        raise ValueError('Expected official fr3 model')
    removed = collapse_empty_frames(root)
    aliases = {}
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if value.startswith('fr3_'):
                aliases[value] = 'panda_' + value[4:]
                node.set(key, aliases[value])
    mesh_hashes = {}
    for mesh in root.iter('mesh'):
        uri = mesh.get('filename')
        if not uri.startswith('package://franka_description/'):
            raise ValueError(f'Unexpected mesh URI: {uri}')
        path = (package_root / uri.removeprefix('package://franka_description/')).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path))
        mesh_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    masses = {}
    joint_limits = {j.get('name'): {k: float(v) for k, v in j.find('limit').attrib.items()}
                    for j in root.findall('joint') if j.find('limit') is not None}
    joint_dynamics = {j.get('name'): {k: float(v) for k, v in j.find('dynamics').attrib.items()}
                      for j in root.findall('joint') if j.find('dynamics') is not None}
    for link in root.findall('link'):
        mass = float(link.find('inertial/mass').get('value'))
        inertia = link.find('inertial/inertia')
        a = {k: float(v) for k, v in inertia.attrib.items()}
        matrix = np.array([[a['ixx'],a['ixy'],a['ixz']], [a['ixy'],a['iyy'],a['iyz']], [a['ixz'],a['iyz'],a['izz']]])
        eigenvalues = np.linalg.eigvalsh(matrix)
        if not math.isfinite(mass) or mass <= 0 or eigenvalues[0] <= 0 or eigenvalues[2] > eigenvalues[:2].sum()+1e-12:
            raise ValueError(f'Invalid rigid-body inertia: {link.get("name")}')
        masses[link.get('name')] = mass
    root.set('name', 'fr3_stock_gripper')
    output.mkdir(parents=True, exist_ok=True)
    sim = output / 'fr3_isaac.urdf'
    ET.indent(root)
    ET.ElementTree(root).write(sim, encoding='utf-8', xml_declaration=True)
    native_root = copy.deepcopy(root)
    native_root.set('xmlns:drake', 'http://drake.mit.edu')
    for link in native_root.findall('link'):
        for node in list(link):
            if node.tag in ('visual','collision'):
                link.remove(node)
    for joint in native_root.findall('joint'):
        if joint.get('name').startswith('panda_finger_joint'):
            joint.set('type', 'fixed')
            for node in list(joint):
                if node.tag in ('axis','limit','mimic','safety_controller','dynamics'):
                    joint.remove(node)
    # Massless task frame is valid in Drake. Isaac uses the same offset on
    # its existing massive hand body rather than importing another rigid body.
    ET.SubElement(native_root, 'link', name='end_effector_tip')
    joint = ET.SubElement(native_root, 'joint', name='fr3_task_frame', type='fixed')
    ET.SubElement(joint, 'parent', link='panda_hand')
    ET.SubElement(joint, 'child', link='end_effector_tip')
    ET.SubElement(joint, 'origin', xyz='0 0 0.1034', rpy=f'{math.pi:.17g} 0 {math.pi/4:.17g}')
    for i in range(1,8):
        transmission = ET.SubElement(native_root,'transmission',name=f'panda_tran{i}')
        ET.SubElement(transmission,'type').text='transmission_interface/SimpleTransmission'
        ET.SubElement(transmission,'joint',name=f'panda_joint{i}')
        actuator=ET.SubElement(transmission,'actuator',name=f'panda_motor{i}')
        ET.SubElement(actuator,'mechanicalReduction').text='1'
        dynamics = joint_dynamics[f'panda_joint{i}']
        ET.SubElement(actuator, 'drake:gear_ratio', value=str(dynamics['gear_ratio']))
        ET.SubElement(actuator, 'drake:rotor_inertia', value=str(dynamics['motor_inertia']))
    native = output / 'fr3_native.urdf'
    ET.indent(native_root)
    ET.ElementTree(native_root).write(native, encoding='utf-8', xml_declaration=True)
    manifest = dict(schema='nonprehensile.fr3_robot_model.v1', robot='FR3', end_effector='stock_closed_gripper',
        official_urdf=str(official.resolve()), official_urdf_sha256=hashlib.sha256(official.read_bytes()).hexdigest(),
        simulation_urdf=str(sim.resolve()), native_urdf=str(native.resolve()),
        simulation_urdf_sha256=hashlib.sha256(sim.read_bytes()).hexdigest(),native_urdf_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        removed_empty_fixed_frames=removed, internal_name_aliases=aliases, body_masses_kg=masses,
        mesh_sha256=mesh_hashes, task_tip_from_hand_m=.1034,
        armature_kg_m2=[joint_dynamics[f'panda_joint{i}']['gear_ratio']**2 * joint_dynamics[f'panda_joint{i}']['motor_inertia'] for i in range(1,8)],
        joint_limits=joint_limits, joint_dynamics=joint_dynamics,
        armature_note='Official expanded FR3 dynamics: gear_ratio squared times motor_inertia; nominal manufacturer model, not measured hardware identification.',
        joint_friction_note='Drake ignores URDF Coulomb friction. Isaac arm friction explicitly zero for the nominal matched baseline; viscous damping retained on both sides.',
        native_fingers='fixed at commanded closed position', native_collision_geometry=False,
        physical_mesh_geometry_changed=False)
    (output/'robot_model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-urdf',type=Path,required=True)
    parser.add_argument('--package-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.official_urdf,args.package_root,args.output)
