"""Apply and verify the FR3 nominal friction contract in the physics backend."""
import torch


def spawn_fr3_with_rigid_finger_coupling(prim_path, cfg, translation=None, orientation=None):
    """Keep the URDF's q2=q1 coupling without importer-added compliance.

    The importer authors naturalFrequency=25, dampingRatio=.005, allowing
    millimetres of unequal finger travel. PhysX defines zero frequency/damping
    as the hard mimic relation. Apply before physics initialization, in the
    live stage only; never modify the shared cached USD asset.
    """
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf
    prim = spawn_from_urdf(prim_path, cfg, translation, orientation)
    joint = prim.GetStage().GetPrimAtPath(str(prim.GetPath()) + "/joints/panda_finger_joint2")
    if not joint.IsValid():
        raise RuntimeError("Missing FR3 second finger joint in spawned USD")
    prefix = "physxMimicJoint:rotY:"
    if joint.GetAttribute(prefix + "gearing").Get() != -1.0 or joint.GetAttribute(prefix + "offset").Get() != 0.0:
        raise RuntimeError("Unexpected FR3 finger mimic relationship")
    reference = joint.GetRelationship(prefix + "referenceJoint").GetTargets()
    if len(reference) != 1 or reference[0].name != "panda_finger_joint1":
        raise RuntimeError("Unexpected FR3 reference finger joint")
    for name in ("naturalFrequency", "dampingRatio"):
        attribute = joint.GetAttribute(prefix + name)
        if not attribute.IsValid() or not attribute.Set(0.0) or attribute.Get() != 0.0:
            raise RuntimeError("Could not apply rigid FR3 finger coupling")
    return prim


def synchronize_finger_limits(robot, contract):
    """Restore official limits after mimic import, then verify PhysX readback.

    Use Articulation writers so IsaacLab's hard/soft limit caches agree with
    PhysX. This preserves the imported finger coupling and closed-hand drives.
    """
    names = ["panda_finger_joint1", "panda_finger_joint2"]
    ids = [robot.joint_names.index(name) for name in names]
    view = robot.root_physx_view
    getters = {"position": view.get_dof_limits,
               "velocity": view.get_dof_max_velocities,
               "effort": view.get_dof_max_forces}
    before = {key: getter().clone() for key, getter in getters.items()}
    limits = [contract["joint_limits"][name] for name in names]
    expected = {
        "position": [[x["lower"], x["upper"]] for x in limits],
        "velocity": [x["velocity"] for x in limits],
        "effort": [x["effort"] for x in limits],
    }
    writers = {"position": robot.write_joint_position_limit_to_sim,
               "velocity": robot.write_joint_velocity_limit_to_sim,
               "effort": robot.write_joint_effort_limit_to_sim}
    for key, writer in writers.items():
        writer(torch.tensor(expected[key], device=robot.device).unsqueeze(0), joint_ids=ids)
    after = {key: getter().clone() for key, getter in getters.items()}
    for key, values in after.items():
        target = torch.tensor(expected[key], dtype=values.dtype, device=values.device)
        if not torch.allclose(values[:, ids], target.expand_as(values[:, ids]), atol=1e-7, rtol=0):
            raise RuntimeError(f"PhysX did not apply official FR3 finger {key} limits")
        others = [i for i in range(len(robot.joint_names)) if i not in ids]
        if not torch.equal(values[:, others], before[key][:, others]):
            raise RuntimeError(f"Applying finger {key} limits modified arm limits")
    return {"joint_names": names, "expected": expected,
            "before": {k: v.detach().cpu().tolist() for k, v in before.items()},
            "after": {k: v.detach().cpu().tolist() for k, v in after.items()}}


def synchronize_arm_friction(view, joint_ids):
    """Commit both legacy and current API values; preserve finger properties.

    IsaacLab 2.2's Sim 5 friction setters mutate a get-buffer without calling
    set_dof_friction_properties. A later getter overwrites that temporary edit.
    Bypass that incomplete setter locally and read the backend back explicitly.
    """
    legacy_before = view.get_dof_friction_coefficients().clone()
    properties_before = view.get_dof_friction_properties().clone()
    indices = torch.arange(legacy_before.shape[0], dtype=torch.int32, device='cpu')
    legacy = legacy_before.clone()
    legacy[:, joint_ids] = 0.0
    view.set_dof_friction_coefficients(legacy, indices)
    properties = properties_before.clone()
    properties[:, joint_ids, :] = 0.0
    view.set_dof_friction_properties(properties, indices)
    legacy_after = view.get_dof_friction_coefficients().clone()
    properties_after = view.get_dof_friction_properties().clone()
    if torch.any(legacy_after[:, joint_ids] != 0) or torch.any(properties_after[:, joint_ids, :] != 0):
        raise RuntimeError('PhysX did not apply the nominal zero arm friction contract')
    return {name: value.detach().cpu().tolist() for name, value in {
        'legacy_before': legacy_before, 'properties_before': properties_before,
        'legacy_after': legacy_after, 'properties_after': properties_after,
    }.items()}
