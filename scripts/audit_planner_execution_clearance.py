"""Compare published C3 predictions with staged planner/executor semantic distances."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


def audit(directory, include_plan_details=False):
    runtimes = list(directory.glob('shared_runtime_*'))
    if len(runtimes) != 1:
        raise ValueError('Requires one isolated runtime')
    runtime = runtimes[0]
    params_path = runtime / 'examples/sampling_c3/anything/parameters/sampling_c3_controller_params.yaml'
    params = yaml.safe_load(params_path.read_text())
    if len(params['unsafe_meshes']) != 1:
        raise ValueError('This audit covers target-only C1 scenes')
    mesh_path = runtime / params['unsafe_meshes'][0]
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    planning = params.get('semantic_guard_clearance', .025)
    execution = params.get('semantic_guard_stop_distance', planning+.010)
    requested = planning
    native_log = directory / 'controller_online.log'
    logged = re.findall(r'^C3_SEMANTIC_PLANNING_CLEARANCE requested_m=(\S+) execution_stop_m=(\S+) effective_m=(\S+) closed_gripper=(\S+)',
                        native_log.read_text(), flags=re.MULTILINE)
    if logged:
        if len(logged) != 1:
            raise ValueError('Ambiguous effective planning clearance')
        raw, stop, effective, closed = logged[0]
        if (float(raw) != requested or float(stop) != execution or closed != '1'
                or float(effective) != max(requested, execution)):
            raise ValueError('Effective clearance differs from staged closed-gripper contract')
        planning = float(effective)
    c3_path = runtime / params['sampling_c3_options_file']
    options = yaml.safe_load(c3_path.read_text())
    if any(abs(options[key]-.075) > 1e-12 for key in ('planning_dt_position', 'planning_dt_pose')):
        raise ValueError('First50ms calculation requires staged 75ms C3 knots in both phases')
    path = directory / 'effect_audit.jsonl'
    plans, modes = {}, {}
    with path.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r.get('event') == 'solver_diagnostic' and r['channel'] == 'C3_DEBUG_CURR':
                if r['utime_us'] in plans:
                    raise ValueError('Duplicate solver plan')
                plans[r['utime_us']] = np.asarray(r['x']).T
            elif r.get('event') == 'task_reference':
                modes[r['relay_state_utime_us']] = r['c3_mode']
    if not plans:
        raise ValueError('No recorded current-location predictions')

    def distances(states):
        q = states[:, 3:7]
        if not np.isfinite(states).all() or np.min(np.linalg.norm(q, axis=1)) < 1e-12:
            raise ValueError('Invalid published state or quaternion')
        local = Rotation.from_quat(q[:, [1, 2, 3, 0]]).inv().apply(states[:, :3] - states[:, 7:10])
        _, distance, _ = trimesh.proximity.closest_point(mesh, local)
        return distance, local

    counts = dict(total_published_current_plans=len(plans), selected_c3_plans=0,
                  selected_plans_in_clearance_gap=0, first50ms_enters_stop_from_clear_current=0,
                  selected_current_already_in_stop=0, selected_below_planner_clearance=0,
                  missing_command_modes=0)
    examples = []
    details = []
    minima = []
    mesh_query_check = None
    margin = 1e-4  # avoid classification at float32 publication boundaries
    for stamp, x in plans.items():
        matched = [modes[t] for t in (stamp-1, stamp, stamp+1) if t in modes]
        if len(matched) > 1:
            raise ValueError('Ambiguous command timestamp')
        if not matched:
            counts['missing_command_modes'] += 1
            continue
        if not matched[0]:
            continue
        if x.ndim != 2 or x.shape[1] != 19 or len(x) < 2:
            raise ValueError('Unexpected target-only published plan shape')
        counts['selected_c3_plans'] += 1
        alpha = np.linspace(0, 1, 5)[None, :, None]
        interpolated = ((1-alpha)*x[:-1, None, :] + alpha*x[1:, None, :]).reshape(-1, 19)
        # Native knots are 75 ms in these fixed evaluations. Check this below.
        first50 = np.array([(1-a)*x[0]+a*x[1] for a in (0, 1/3, 2/3)])
        values, local = distances(np.vstack([interpolated, first50]))
        if mesh_query_check is None:
            _, naive, _ = trimesh.proximity.closest_point_naive(mesh, local[:5])
            mesh_query_check = float(np.max(np.abs(naive-values[:5])))
            if mesh_query_check > 1e-10:
                raise ValueError('Indexed and direct mesh distances differ')
        minimum = float(values[:-3].min())
        initial = float(values[-3])
        near = float(values[-3:].min())
        minima.append(minimum)
        gap = planning+margin < minimum < execution-margin
        counts['selected_plans_in_clearance_gap'] += gap
        counts['first50ms_enters_stop_from_clear_current'] += initial > execution+margin and near < execution-margin
        counts['selected_current_already_in_stop'] += initial < execution-margin
        counts['selected_below_planner_clearance'] += minimum < planning-margin
        if include_plan_details:
            details.append(dict(utime_us=stamp, initial_distance_m=initial,
                                minimum_horizon_distance_m=minimum,
                                minimum_first50ms_distance_m=near,
                                enters_stop_from_clear_current=bool(
                                    initial > execution+margin and near < execution-margin)))
        if gap and len(examples) < 12:
            examples.append(dict(utime_us=stamp, initial_distance_m=initial,
                                 minimum_horizon_distance_m=minimum, minimum_first50ms_distance_m=near))
    return dict(schema='nonprehensile.planner_execution_clearance_audit.v1',
                planner_clearance_m=planning, requested_planner_clearance_m=requested,
                effective_clearance_logged=bool(logged), execution_stop_distance_m=execution,
                classification_margin_m=margin, counts=counts, examples=examples,
                minimum_selected_horizon_distance_m=min(minima) if minima else None,
                selected_plan_details=details if include_plan_details else None,
                mesh_distance_query_maximum_difference_m=mesh_query_check,
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (path, params_path, mesh_path, c3_path, native_log)},
                limitations=['Published float32 current-location predictions, not all candidate plans.',
                             'Uses normalized predicted object rotations and native four-substep interpolation.',
                             'Predicted proximity does not itself prove actual contact or explain every execution hold.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-directory', type=Path, required=True)
    parser.add_argument('--include-plan-details', action='store_true')
    args = parser.parse_args()
    report = audit(args.scene_directory, args.include_plan_details)
    (args.scene_directory / 'planner_execution_clearance_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
