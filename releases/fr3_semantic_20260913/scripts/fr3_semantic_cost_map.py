"""Action-conditioned planar semantic map plus per-object physical observations.

Geometry is privileged simulation state. Semantics are an explicitly supplied
assistant-authored policy. This is not an autonomous RGB-D reconstruction model.
Planar SAT uses conservative projected convex hulls; it is not a 3D certificate.
"""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def polygon(points):
    p=np.asarray(points,dtype=float)[:,:2]
    return p[ConvexHull(p).vertices]


def separation(a,b):
    """Largest separating-axis gap; <=0 means convex projections intersect."""
    a,b=np.asarray(a),np.asarray(b)
    edges=np.concatenate((np.roll(a,-1,axis=0)-a,np.roll(b,-1,axis=0)-b))
    axes=np.stack((-edges[:,1],edges[:,0]),axis=1)
    axes=axes/np.linalg.norm(axes,axis=1)[:,None]
    pa,pb=a@axes.T,b@axes.T
    return float(np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0)).max())


def query(footprint,center,delta,yaw,obstacles,margin=.012):
    """Sample an object sweep at <=2 mm / <=1 degree increments.

    Hard feasibility remains separate from finite soft costs; no goal gain can
    compensate a hard rejection. Caller supplies measured obstacle transforms.
    """
    footprint=np.asarray(footprint);center=np.asarray(center)[:2];delta=np.asarray(delta)[:2]
    if not all(np.isfinite(x).all() for x in (footprint,center,delta,np.asarray(yaw))):
        raise ValueError('Nonfinite map input')
    n=max(2,int(np.ceil(np.linalg.norm(delta)/.002))+1,int(np.ceil(abs(yaw)/np.deg2rad(1)))+1)
    gaps={o['id']:float('inf') for o in obstacles}
    for t in np.linspace(0,1,n):
        c,s=np.cos(t*yaw),np.sin(t*yaw)
        shape=(footprint-center)@np.array([[c,s],[-s,c]])+center+t*delta
        for o in obstacles:gaps[o['id']]=min(gaps[o['id']],separation(shape,o['polygon']))
    hard=[o['id'] for o in obstacles if o['mode']=='forbidden' and gaps[o['id']]<=margin]
    soft=sum(o['weight']*float(np.clip((margin-gaps[o['id']])/margin,0,1)) for o in obstacles if o['mode']=='acceptable')
    return dict(admissible=not hard,soft_cost=soft,hard_rejections=hard,clearance_m=gaps,samples=n,margin_m=margin)


@lru_cache(maxsize=1)
def policy():
    p=Path(os.environ['FR3_SEMANTIC_POLICY'])
    d=json.loads(p.read_text())
    assert d['schema']=='fr3.semantic_policy.v1'
    assert d['provenance']['semantic_provider']=='assistant_visual_review'
    assert all(o['mode'] in ('forbidden','acceptable') and 0<=o['weight']<=1 for o in d['objects'])
    assert len({o['id'] for o in d['objects']})==len(d['objects'])
    assert all(o['robot_contact']==o['target_contact']==o['mode'] for o in d['objects'])
    return d


def live_obstacles(base):
    import trimesh
    specs=policy()['objects'];objects=base.scene['obstacles']
    assert base.num_envs==1 and int(base._clutter_active_obstacle_count)==len(specs)
    cfgs=list(objects.cfg.rigid_objects.values())[:len(specs)]
    result=[]
    if not hasattr(base,'_semantic_mesh_cache'):base._semantic_mesh_cache={}
    for i,(spec,cfg) in enumerate(zip(specs,cfgs)):
        asset=cfg.spawn.assets_cfg[0]
        assert spec['asset_id']==asset.dapl_asset_id
        key=asset.obj_path
        if key not in base._semantic_mesh_cache:
            m=trimesh.load(key,force='scene',process=False);m=m.to_geometry() if isinstance(m,trimesh.Scene) else m
            base._semantic_mesh_cache[key]=np.asarray(m.vertices)*np.asarray(asset.scale or (1,1,1))
        v=base._semantic_mesh_cache[key];q=objects.data.object_quat_w[0,i].cpu().numpy()
        pos=(objects.data.object_pos_w[0,i]-base.scene.env_origins[0]).cpu().numpy()
        world=Rotation.from_quat(q[[1,2,3,0]]).apply(v)+pos
        result.append(dict(spec,polygon=polygon(world).tolist(),position=pos.tolist(),quaternion_wxyz=q.tolist()))
    return result


def record(base,event,**data):
    out=Path(os.environ['M1_COMPARISON_OUTPUT'])
    with (out/'semantic_queries.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=event,physics_step=int(getattr(base,'_semantic_step',0)),**data))+'\n')


def live_target_vertices(base):
    import trimesh
    target=base.scene['target']
    if not hasattr(base,'_semantic_target_vertices'):
        asset=target.cfg.spawn.assets_cfg[0]
        m=trimesh.load(asset.obj_path,force='scene',process=False);m=m.to_geometry() if isinstance(m,trimesh.Scene) else m
        base._semantic_target_vertices=np.asarray(m.vertices)*np.asarray(asset.scale or (1,1,1))
    q=target.data.root_quat_w[0].cpu().numpy()
    pos=(target.data.root_pos_w[0]-base.scene.env_origins[0]).cpu().numpy()
    return Rotation.from_quat(q[[1,2,3,0]]).apply(base._semantic_target_vertices)+pos


def candidate_motion(candidates,rank,rotation_efficiency,gyration_sq,servo_checks=None):
    distance=float(candidates.push_distance[0,rank])
    if servo_checks is not None:
        check=servo_checks[rank]
        delta=np.r_[np.asarray(check['axis'],dtype=float)*distance,0.]
        moment=float(check['achieved_moment'])
    else:
        delta=(candidates.push_tcp[0,rank]-candidates.contact_tcp[0,rank]).detach().cpu().numpy()
        moment=float(candidates.contact_moment_arm[0,rank])
    return delta,float(rotation_efficiency*distance*moment/gyration_sq)


def candidate_costs(base,candidates,points,center,rotation_efficiency,gyration_sq,servo_checks=None):
    footprint=polygon(live_target_vertices(base));center=center[0].detach().cpu().numpy()
    obstacles=live_obstacles(base);costs={}
    backend=os.environ.get('FR3_SEMANTIC_BACKEND','planar')
    if backend not in ('planar','volume3d'):raise ValueError(backend)
    weight=float(os.environ.get('FR3_SEMANTIC_WEIGHT_M','1'))
    if not np.isfinite(weight) or weight<0:raise ValueError('Invalid semantic cost weight')
    scene3d=geometry_record=None
    if backend=='volume3d':
        from fr3_semantic_sweep3d import from_live_scene
        scene3d,geometry_record=from_live_scene(base,policy()['objects'])
    for rank in range(candidates.valid.shape[1]):
        if not candidates.valid[0,rank]:continue
        delta,yaw=candidate_motion(candidates,rank,rotation_efficiency,gyration_sq,servo_checks)
        result=scene3d.query(center,delta,yaw) if scene3d else query(footprint,center,delta,yaw,obstacles)
        costs[rank]=weight*result['soft_cost']
        if not result['admissible']:candidates.valid[0,rank]=False
        record(base,'candidate',rank=rank,delta=delta.tolist(),delta_dtype=str(delta.dtype),yaw=yaw,center=center.tolist(),center_dtype=str(center.dtype),footprint=footprint.tolist(),obstacles=obstacles,result=result,backend=backend,geometry_record=geometry_record,semantic_weight_m=weight,weighted_cost_m=costs[rank],motion_basis='nominal_contact_servo' if servo_checks is not None else 'raw_candidate',servo_check=servo_checks[rank] if servo_checks is not None else None)
    return costs


def micro_check(base,ik,q_start,q_end,bridge,points,center,axis,distance,yaw):
    from fr3_execution_runtime.clutter_route import from_live_scene as robot_scene_guard
    obstacles=live_obstacles(base)
    delta=np.r_[np.asarray(axis)*distance,0.]
    geometry_record=None
    if os.environ.get('FR3_SEMANTIC_BACKEND','planar')=='volume3d':
        from fr3_semantic_sweep3d import from_live_scene
        scene3d,geometry_record=from_live_scene(base,policy()['objects'])
        result=scene3d.query(center,delta,yaw)
    else:
        result=query(polygon(live_target_vertices(base)),center,delta,yaw,obstacles)
    guard=robot_scene_guard(ik,base)
    robot=guard.segment(q_start,q_end,bridge) if guard is not None else dict(clearance_m=float('inf'))
    valid=result['admissible'] and robot['clearance_m']>.012
    record(base,'micro',result=result,robot=robot,admissible=valid,geometry_record=geometry_record,center=np.asarray(center).tolist(),delta=delta.tolist(),yaw=float(yaw))
    return valid


def phase_check(base,ik,start,end):
    """Whole-arm preflight for all commanded phases, including vertical lift."""
    from fr3_execution_runtime.clutter_route import from_live_scene
    from isaaclab.utils.math import matrix_from_quat
    guard=from_live_scene(ik,base)
    if guard is None:return True
    robot=base.scene['robot'];ids=[robot.joint_names.index(f'panda_joint{i}') for i in range(1,8)]
    measured=robot.data.joint_pos[0,ids].cpu().numpy()
    tcp=(base.scene['ee_frame'].data.target_pos_w[0,0]-base.scene.env_origins[0]).cpu().numpy()
    rot=matrix_from_quat(base.scene['ee_frame'].data.target_quat_w[0,0]).cpu().numpy()
    bridge=ik.bridge(measured,tcp,rot)
    results=[guard.segment(measured,start,bridge),guard.segment(start,end,bridge)]
    valid=all(x['clearance_m']>.012 for x in results)
    record(base,'whole_arm_phase',results=results,admissible=valid)
    return valid


def read_contacts(base):
    import torch
    from fr3_execution_runtime.typed_constraints import ROBOT_SENSORS
    specs=policy()['objects'];n=len(specs)
    assert base.num_envs==1 and int(base._clutter_active_obstacle_count)==n
    def read(name):
        sensor=base.scene.sensors[name]
        # IsaacLab expands ENV_REGEX_NS during scene construction.
        paths=list(sensor.cfg.filter_prim_paths_expr)
        expected=[f'Obstacle_{i:02d}' for i in range(n)]
        assert [p.rsplit('/',1)[-1] for p in paths[:n]]==expected,(name,paths)
        assert len({p.rsplit('/',1)[0] for p in paths[:n]})==1,(name,paths)
        m=sensor.data.force_matrix_w
        assert m.ndim==4 and m.shape[2]>=n and m.shape[0]==1,m.shape
        f=torch.linalg.vector_norm(m[:,:,:n],dim=-1).amax(dim=1)[0]
        assert torch.isfinite(f).all()
        return f
    target=read('target_obstacle_contacts')
    robot=torch.stack([read(name) for name in ROBOT_SENSORS]).amax(dim=0)
    hard=torch.tensor([o['mode']=='forbidden' for o in specs],device=base.device)
    zero=target.new_zeros(1)
    t=target[hard].max().reshape(1) if hard.any() else zero
    r=robot[hard].max().reshape(1) if hard.any() else zero
    base._semantic_step=getattr(base,'_semantic_step',0)+1
    record(base,'physical_contact',objects=[dict(id=o['id'],mode=o['mode'],target_force_n=float(target[i]),robot_force_n=float(robot[i])) for i,o in enumerate(specs)])
    # For forbidden cup/balloon, any target part is forbidden, stronger than
    # the original protected-part-only C2. Acceptable objects are still sensed.
    # This policy forbids whole-object contacts, not only a protected region.
    # Do not invent a protected-region distance: legacy serializer emits null.
    return dict(c2=t>.02,c3=r>.02,target_obstacle_force_n=t,robot_obstacle_force_n=r,protected_clearance_m=zero+float('nan'))


def snapshot(base,markers=None):
    if os.environ.get('FR3_SEMANTIC_SNAPSHOT','1')=='0':
        return  # Unrecorded evaluation; no render call or simulation state change.
    from PIL import Image
    from pxr import UsdGeom
    from omni.kit.viewport.utility import get_active_viewport
    out=Path(os.environ['M1_COMPARISON_OUTPUT'])
    stage=base.sim.stage
    if markers:
        for name,marker in markers.items():
            if name=='goal_ghost':marker.set_visibilities([False])
            else:marker.set_visibility(False)
    for _ in range(3):base.sim.render()
    Image.fromarray(base.render()).save(out/'scene_original.png')
    viewport=get_active_viewport();cam=UsdGeom.Camera(stage.GetPrimAtPath(str(viewport.camera_path))).GetCamera()
    camera=dict(view=np.asarray(cam.frustum.ComputeViewMatrix()).tolist(),projection=np.asarray(cam.frustum.ComputeProjectionMatrix()).tolist())
    if markers:
        for name,marker in markers.items():
            if name=='goal_ghost':marker.set_visibilities([True])
            else:marker.set_visibility(True)
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import get_object_pointcloud_in_env_frame
    from isaaclab.managers import SceneEntityCfg
    points=get_object_pointcloud_in_env_frame(base,SceneEntityCfg('target')).reshape(-1,3).cpu().numpy()
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp
    semantics=mdp.domino_target_affordance(base,SceneEntityCfg('target')).reshape(-1,2).cpu().tolist()
    d=dict(camera=camera,obstacles=live_obstacles(base),target_points=points.tolist(),target_semantics=semantics,
           target_footprint=polygon(live_target_vertices(base)).tolist(),
           target_pose=base.scene['target'].data.root_state_w[0,:7].cpu().tolist(),
           target_center=base.scene['target'].data.root_com_pos_w[0].cpu().tolist(),
           goal=base.command_manager.get_command('target_object_pose')[0].cpu().tolist(),policy=policy())
    (out/'map_snapshot.json').write_text(json.dumps(d,indent=2))


def initialize_stages(base):
    """Explicit task-level subgoal; independent observer retains final goal."""
    if not policy().get('rotate_then_translate',False):return
    command=base.command_manager.get_command('target_object_pose')
    base._semantic_final_goal=command.clone()
    base._semantic_initial_pos=base.scene['target'].data.root_pos_w[0].clone()
    base._semantic_phase='rotate'
    delta=command[0,:2]-(base._semantic_initial_pos-base.scene.env_origins[0])[:2]
    command[0,:2]=(base._semantic_initial_pos-base.scene.env_origins[0])[:2]+.008*delta/delta.norm()
    record(base,'stage_begin',phase='rotate',subgoal=command[0].cpu().tolist(),final_goal=base._semantic_final_goal[0].cpu().tolist())


def update_stages(base):
    if getattr(base,'_semantic_phase',None)!='rotate':return
    q=base.scene['target'].data.root_quat_w[0].cpu().numpy();g=base._semantic_final_goal[0,3:7].cpu().numpy()
    error=float((Rotation.from_quat(g[[1,2,3,0]]).inv()*Rotation.from_quat(q[[1,2,3,0]])).magnitude())
    # Match the existing servo's 0.10 rad orientation deadband. Requiring a
    # smaller subgoal threshold would stall a controller already inside it.
    if error<.10:
        base.command_manager.get_command('target_object_pose').copy_(base._semantic_final_goal)
        base._semantic_phase='translate'
        record(base,'stage_begin',phase='translate',rotation_error_rad=error,translation_before_stage_m=float((base.scene['target'].data.root_pos_w[0,:2]-base._semantic_initial_pos[:2]).norm()))
