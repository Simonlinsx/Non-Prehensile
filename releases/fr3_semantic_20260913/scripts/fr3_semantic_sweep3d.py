"""Lift per-object 3-D point costs to conservative full-target swept volumes.

FCL convex distances avoid evaluating a dense voxel grid in the control loop.
Target and forbidden objects use conservative convex envelopes; acceptable
objects retain their convex collision components. Hard feasibility is separate
from the finite peak-over-horizon contact cost. Whole-arm guards remain separate.
"""
from functools import lru_cache
import time
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import hppfcl as fcl


def convex(vertices):
    hull=trimesh.convex.convex_hull(np.asarray(vertices))
    v=fcl.StdVec_Vec3f();f=fcl.StdVec_Triangle()
    for p in hull.vertices:v.append(p)
    for face in hull.faces:f.append(fcl.Triangle(*map(int,face)))
    return fcl.Convex(v,f)


@lru_cache(maxsize=32)
def geometry(path,scale,envelope):
    mesh=trimesh.load(path,force='mesh',process=False)
    mesh.vertices=np.asarray(mesh.vertices)*scale
    if envelope:return (convex(mesh.vertices),)
    parts=mesh.split(only_watertight=False)
    if not parts or any(not p.is_watertight or not p.is_convex for p in parts):
        raise ValueError('Expected closed convex collision components')
    return tuple(convex(p.vertices) for p in parts)


def transform(pose):
    pose=np.asarray(pose,dtype=float)
    return Rotation.from_quat(pose[[4,5,6,3]]).as_matrix(),pose[:3]


class SceneSweep3D:
    def __init__(self,target,target_pose,obstacles):
        self.target=target
        self.rotation,self.position=transform(target_pose)
        self.obstacles=[]
        for rule,parts,pose in obstacles:
            r,t=transform(pose)
            self.obstacles.append((rule,parts,fcl.Transform3f(r,t)))

    def query(self,center,delta,yaw,margin=.012):
        started=time.perf_counter()
        center=np.asarray(center,dtype=float);delta=np.asarray(delta,dtype=float)
        if center.shape!=(3,) or delta.shape!=(3,) or not np.isfinite(np.r_[center,delta,yaw,margin]).all() or margin<=0:
            raise ValueError('Finite XYZ center/displacement, yaw and positive margin required')
        count=max(2,int(np.ceil(np.linalg.norm(delta)/.002))+1,int(np.ceil(abs(yaw)/np.deg2rad(1)))+1)
        gaps={rule['id']:float('inf') for rule,_,_ in self.obstacles}
        queries=0
        for alpha in np.linspace(0,1,count):
            r=Rotation.from_rotvec([0,0,float(alpha*yaw)]).as_matrix()
            pose=fcl.Transform3f(r@self.rotation,center+alpha*delta+r@(self.position-center))
            for rule,parts,obstacle_pose in self.obstacles:
                for part in parts:
                    distance=float(fcl.distance(self.target,pose,part,obstacle_pose,fcl.DistanceRequest(),fcl.DistanceResult()))
                    if not np.isfinite(distance):raise ValueError('Nonfinite FCL distance')
                    # Collision penetration is mapped to zero volume distance.
                    gaps[rule['id']]=min(gaps[rule['id']],max(0.,distance));queries+=1
        rejected=[rule['id'] for rule,_,_ in self.obstacles if rule['mode']=='forbidden' and gaps[rule['id']]<=margin]
        soft=sum(rule['weight']*float(np.clip(1-gaps[rule['id']]/margin,0,1))
                 for rule,_,_ in self.obstacles if rule['mode']=='acceptable')
        return dict(admissible=not rejected,hard_rejections=rejected,soft_cost=soft,
            clearance_m=gaps,samples=count,margin_m=margin,backend='fcl_swept_target_3d',
            fcl_queries=queries,query_wall_s=time.perf_counter()-started)


def from_record(record):
    target=record['target']
    shape=geometry(target['path'],tuple(target['scale']),True)[0]
    obstacles=[]
    for item in record['obstacles']:
        parts=geometry(item['path'],tuple(item['scale']),item['rule']['mode']=='forbidden')
        obstacles.append((item['rule'],parts,item['pose']))
    return SceneSweep3D(shape,target['pose'],obstacles)


def from_live_scene(base,rules):
    target=base.scene['target'];asset=target.cfg.spawn.assets_cfg[0]
    origin=base.scene.env_origins[0].cpu().numpy()
    pose=target.data.root_state_w[0,:7].cpu().numpy().copy();pose[:3]-=origin
    record=dict(target=dict(path=str(asset.obj_path),scale=list(asset.scale or (1,1,1)),pose=pose.tolist()),obstacles=[])
    clutter=base.scene['obstacles']
    for index,(rule,cfg) in enumerate(zip(rules,list(clutter.cfg.rigid_objects.values()))):
        asset=cfg.spawn.assets_cfg[0]
        assert asset.dapl_asset_id==rule['asset_id']
        pos=clutter.data.object_pos_w[0,index].cpu().numpy()-origin
        q=clutter.data.object_quat_w[0,index].cpu().numpy()
        record['obstacles'].append(dict(rule=rule,path=str(asset.obj_path),scale=list(asset.scale or (1,1,1)),pose=np.r_[pos,q].tolist()))
    return from_record(record),record
