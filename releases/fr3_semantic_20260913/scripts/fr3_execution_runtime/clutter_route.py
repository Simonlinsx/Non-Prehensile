"""Sampled whole-FR3 clearance against conservative full-mesh obstacle boxes.

This is a static-obstacle preflight, not a continuous collision certificate.
"""
import os
import numpy as np
import pinocchio as pin
import hppfcl as fcl
import trimesh
from scipy.spatial.transform import Rotation

class ClutterClearance:
    def __init__(self, ik, boxes):
        self.model = ik.model
        self.geom = pin.buildGeomFromUrdf(self.model, os.environ['DAPL_LOCAL_FRANKA_URDF'], pin.GeometryType.COLLISION)
        self.data = self.model.createData()
        self.gdata = self.geom.createData()
        self.boxes = boxes
        self.queries = 0
        self.cache = {}
        self.names = [g.name for g in self.geom.geometryObjects]
        assert len(self.names) == 17, self.names

    def clearance(self, q, bridge):
        key = np.asarray(q, dtype=np.float64).tobytes()
        if key in self.cache:
            return self.cache[key]
        ref, rot, tcp = bridge
        translation = tcp - rot @ ref.translation
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, np.r_[q, 0., 0.])
        minimum, pair = float('inf'), None
        for geo, pose in zip(self.geom.geometryObjects, self.gdata.oMg):
            transform = fcl.Transform3f(rot @ pose.rotation, rot @ pose.translation + translation)
            for name, box, box_transform in self.boxes:
                self.queries += 1
                value = float(fcl.distance(geo.geometry, transform, box, box_transform, fcl.DistanceRequest(), fcl.DistanceResult()))
                if not np.isfinite(value):
                    raise ValueError('Nonfinite FCL distance')
                if value < minimum:
                    minimum, pair = value, (geo.name, name)
        self.cache[key] = (minimum, pair)
        return minimum, pair

    def segment(self, start, end, bridge):
        # Bound joint sample spacing rather than silently using a fixed sparse grid.
        count = max(32, int(np.ceil(np.max(np.abs(end-start)) / .02)) + 1)
        best = dict(clearance_m=float('inf'), pair=None, samples=count)
        for alpha in np.linspace(0, 1, count):
            value, pair = self.clearance((1-alpha)*start+alpha*end, bridge)
            if value < best['clearance_m']:
                best.update(clearance_m=value, pair=pair, alpha=float(alpha))
        return best


def from_live_scene(ik, base):
    count = int(getattr(base, '_clutter_active_obstacle_count', base.cfg.active_obstacle_count) or 0)
    if not count:
        return None
    assert base.num_envs == 1
    obstacles = base.scene['obstacles']
    from fr3_semantic_cost_map import policy
    rules = policy()["objects"]
    boxes = []
    for index, cfg in enumerate(list(obstacles.cfg.rigid_objects.values())[:count]):
        asset = cfg.spawn.assets_cfg[0]
        assert rules[index]["asset_id"] == asset.dapl_asset_id
        if rules[index]["mode"] == "acceptable":
            continue
        mesh = trimesh.load(asset.obj_path, force='scene', process=False)
        mesh = mesh.to_geometry() if isinstance(mesh, trimesh.Scene) else mesh
        vertices = np.asarray(mesh.vertices) * np.asarray(asset.scale or (1., 1., 1.))
        assert vertices.size and np.isfinite(vertices).all()
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        quat = obstacles.data.object_quat_w[0,index].detach().cpu().numpy()
        rot = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        position = (obstacles.data.object_pos_w[0,index]-base.scene.env_origins[0]).detach().cpu().numpy()
        boxes.append((f'obstacle{index}', fcl.Box(*(hi-lo)), fcl.Transform3f(rot, position+rot@((lo+hi)/2))))
    return ClutterClearance(ik, boxes)
