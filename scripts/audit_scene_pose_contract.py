"""Check executed initial/goal poses against the frozen scene specification."""
import math

try:
    from .audit_strict_pose_dwell import pose_errors
except ImportError:
    from audit_strict_pose_dwell import pose_errors


def yaw_quaternion(degrees):
    angle = math.radians(degrees) / 2
    return [math.cos(angle), 0., 0., math.sin(angle)]


def world_yaw_times_support(degrees, support):
    c, _, _, s = yaw_quaternion(degrees)
    w, x, y, z = support
    return [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]


def audit(scene, result, native_goal, semantic, base_height_m=.029):
    if (scene['asset_id'] != semantic['asset_id']
            or scene['support_pose_index'] != semantic['support_pose_index']):
        raise ValueError('Frozen scene asset/support differs from pinned semantics')
    if result['franka_base_height_m'] != base_height_m:
        raise ValueError('Execution and native frame offsets differ')
    support, height = semantic['support_quaternion_wxyz'], semantic['support_height_m']
    initial_yaw = scene.get('initial_yaw_deg', 0.)
    initial = [*scene['initial_xy_m'], height, *world_yaw_times_support(initial_yaw, support)]
    goal = [*scene['goal_xy_m'], height, *world_yaw_times_support(scene['goal_yaw_deg'], support)]
    native = [*scene['goal_xy_m'], height-base_height_m, *yaw_quaternion(scene['goal_yaw_deg'])]
    if native_goal['goal_mode'] != 2:
        raise ValueError('Native goal is not fixed')
    measured = [result['initial_target_pose_wxyz'], result['goal_pose_wxyz'],
                native_goal['fixed_target_positions'][0]+native_goal['fixed_target_orientations'][0]]
    rows = {}
    for name, actual, expected in zip(('initial', 'isaac_goal', 'native_goal'), measured, (initial, goal, native)):
        errors = pose_errors(actual[:3], actual[3:], expected)
        # Manifest XY is rounded to six decimals; Isaac states are float32.
        if any(error > 2e-6 for error in errors):
            raise ValueError(f'{name} does not match frozen scene pose: {errors}')
        rows[name] = dict(planar_m=errors[0], height_m=errors[1], rotation_rad=errors[2])
    return dict(scene_pose_contract_pass=True, errors=rows,
                initial_yaw_deg=initial_yaw, goal_yaw_deg=scene['goal_yaw_deg'])
