"""Read stock FR3 collision samples in their actual rigid-body frames."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def collision_candidates(urdf_path, body_name):
    root = ET.parse(urdf_path).getroot()
    if root.get('name') != 'fr3_stock_gripper':
        raise ValueError('Expected the prepared FR3 stock gripper model')
    body = root.find(f"link[@name='{body_name}']")
    if body is None:
        raise ValueError(f'No collision body {body_name}')
    clouds = []
    for collision in body.findall('collision'):
        mesh_node = collision.find('geometry/mesh')
        box_node = collision.find('geometry/box')
        if box_node is not None:
            mesh = trimesh.creation.box(extents=[float(x) for x in box_node.get('size').split()])
        elif mesh_node is not None:
            filename = Path(mesh_node.get('filename'))
            if not filename.is_absolute():
                filename = Path(urdf_path).parent / filename
            mesh = trimesh.load(filename, force='mesh', process=False)
            mesh.vertices *= np.array([float(x) for x in mesh_node.get('scale', '1 1 1').split()])
        else:
            raise ValueError('Unsupported FR3 collision primitive')
        points = np.concatenate((np.asarray(mesh.vertices), np.asarray(mesh.triangles).mean(axis=1)))
        origin = collision.find('origin')
        if origin is not None:
            rotation = Rotation.from_euler('xyz', [float(x) for x in origin.get('rpy', '0 0 0').split()])
            points = rotation.apply(points) + np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
        clouds.append(points)
    if not clouds:
        raise ValueError(f'No collision meshes on {body_name}')
    return np.unique(np.concatenate(clouds), axis=0)
