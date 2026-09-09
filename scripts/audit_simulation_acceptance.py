#!/usr/bin/env python3
"""Check the frozen simulation goal without dropping failed/missing scenes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

try:
    from .audit_strict_pose_dwell import audit as audit_dwell
    from .audit_scene_pose_contract import audit as audit_scene_pose
except ImportError:
    from audit_strict_pose_dwell import audit as audit_dwell
    from audit_scene_pose_contract import audit as audit_scene_pose


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def physical_safe_contact_observed(result):
    return any(row.get('legal_safe_robot_contact') is True
               and row.get('forbidden_robot_contact') is not True
               and any(name.startswith('target_hand_contacts') and finite_number(force) and force > 0
                       for name, force in row.get('robot_target_contact_force_n_by_sensor', {}).items())
               for row in result.get('trace', []))


def native_controller_parameters(result):
    value = result.get('controller_parameters')
    return value if isinstance(value, dict) else {}


def verify_native_goal_artifact(result, scene_directory):
    artifact = native_controller_parameters(result).get('native_goal_artifact') or {}
    path = Path(artifact.get('path', ''))
    if (not path.is_file() or not path.resolve().is_relative_to(scene_directory.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get('sha256')
            or yaml.safe_load(path.read_text()).get('goal_mode') != 2):
        raise ValueError('Missing, changed, or non-fixed native goal artifact')


def summarize(protocol, scene_ids, results):
    thresholds = protocol['strict_pose_thresholds']
    successes, unsafe, missing, invalid = [], [], [], []
    for scene_id in scene_ids:
        r = results.get(scene_id)
        if r is None:
            missing.append(scene_id)
            continue
        period, duration = r.get('control_period_s'), r.get('executed_sim_time_s')
        steps, dwell_steps = r.get('executed_steps'), r.get('maximum_strict_dwell_steps')
        contact_step = r.get('first_safe_contact_step')
        timing_ok = (finite_number(period) and period > 0 and finite_number(duration)
                     and 0 < duration <= protocol['max_sim_time_s'] + 1e-9
                     and type(steps) is int and steps > 0
                     and math.isclose(duration, steps * period, abs_tol=1e-8)
                     and type(dwell_steps) is int and 0 <= dwell_steps <= steps
                     and (contact_step is None or (type(contact_step) is int and 0 <= contact_step < steps)))
        if (r.get('schema') != 'nonprehensile.c3_online_isaaclab_task.v1'
                or not timing_ok
                or r.get('error') is not None or r.get('strict_pose_thresholds') != thresholds
                or r.get('diagnostic_reference_replay', False) is not False
                or r.get('physical_end_effector') != 'stock_franka_gripper_closed'
                or native_controller_parameters(r).get('native_goal_mode') != 2
                or not isinstance(r.get('forbidden_robot_contact_ever'), bool)):
            invalid.append(scene_id)
            continue
        trace_unsafe = any(row.get('forbidden_robot_contact') is True for row in r.get('trace', []))
        if r['forbidden_robot_contact_ever'] or trace_unsafe:
            unsafe.append(scene_id)
        pose_ok = all(finite_number(r.get(key))
                      and 0 <= r[key] < thresholds[threshold]
                      for key, threshold in [('final_planar_error_m', 'planar_m'),
                                             ('final_height_error_m', 'height_m'),
                                             ('final_rotation_error_rad', 'rotation_rad')])
        dwell = dwell_steps * period
        independent_dwell_ok = True
        if protocol.get('require_independent_dwell_trace', False) and r.get('online_closed_loop_success') is True:
            try:
                independent_dwell_ok = audit_dwell(r)['terminal_dwell_certified']
            except (KeyError, ValueError, TypeError):
                independent_dwell_ok = False
            if not independent_dwell_ok:
                invalid.append(scene_id)
        if (r.get('online_closed_loop_success') is True
                and r.get('stopped_reason') == 'strict_pose_dwell_success'
                and not r['forbidden_robot_contact_ever']
                and not trace_unsafe and physical_safe_contact_observed(r)
                and r.get('first_safe_contact_step') is not None
                and pose_ok and dwell >= thresholds['dwell_time_s'] and independent_dwell_ok):
            successes.append(scene_id)
    count = len(scene_ids)
    passed = (count == protocol['count'] and not missing and not invalid and not unsafe
              and len(successes) >= protocol['minimum_successes']
              and len(successes) / count > protocol['success_rate_strictly_greater_than'])
    return {'goal_passed': passed, 'total_fixed_scenes': count, 'successes': len(successes),
            'success_rate': len(successes) / count if count else 0,
            'successful_scene_ids': successes, 'c1_violations': unsafe,
            'missing_scene_ids': missing, 'invalid_scene_ids': invalid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('data/manifests/contact_planner_m3/hammer_c1_acceptance50_protocol.json'))
    parser.add_argument('--run-root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    manifest = Path(protocol['manifest'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != protocol['manifest_sha256']:
        raise ValueError('Frozen manifest hash changed')
    scenes = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ids = [scene['scene_id'] for scene in scenes]
    scene_by_id = dict(zip(ids, scenes))
    semantic = None
    if 'semantic_manifest' in protocol:
        semantic_path = Path(protocol['semantic_manifest'])
        if hashlib.sha256(semantic_path.read_bytes()).hexdigest() != protocol['semantic_manifest_sha256']:
            raise ValueError('Pinned semantic support pose changed')
        semantic = json.loads(semantic_path.read_text())
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate frozen scene IDs')
    results, configurations, files, scene_pose_audits = {}, [], {}, {}
    for root in args.run_root:
        cfg = json.loads((root/'evaluation_config.json').read_text())
        if (cfg.get('manifest_sha256') != protocol['manifest_sha256']
                or cfg.get('max_sim_time_s') != protocol['max_sim_time_s']
                or not cfg.get('native_binary_sha256') or not cfg.get('execution_source_sha256')
                or cfg.get('force_c3_on_contact') is not False
                or cfg.get('native_goal_mode') != 2
                or cfg.get('diagnostic_reference_replay_start_s') is not None
                or cfg.get('contact_acquisition_speed_m_s') != 0):
            raise ValueError('Run does not have the frozen manifest, versioned code, or required execution contract')
        snapshot = root / 'source_snapshot'
        archived = [(snapshot / name, digest) for name, digest in cfg['execution_source_sha256'].items()]
        archived += [(snapshot / 'native/franka_sampling_c3_controller', cfg['native_binary_sha256']),
                     (snapshot / 'input_manifest.jsonl', protocol['manifest_sha256'])]
        for path, digest in archived:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Missing or altered source snapshot: {path}')
        configurations.append({k: v for k, v in cfg.items() if k != 'gpu'})
        for scene_id in ids:
            path = root/scene_id/'result.json'
            if path.exists():
                if scene_id in results:
                    raise ValueError(f'Duplicate result for {scene_id}; do not select best-of reruns')
                results[scene_id] = json.loads(path.read_text())
                verify_native_goal_artifact(results[scene_id], path.parent)
                if semantic is not None:
                    goal_path = Path(native_controller_parameters(results[scene_id])['native_goal_artifact']['path'])
                    scene_pose_audits[scene_id] = audit_scene_pose(
                        scene_by_id[scene_id], results[scene_id], yaml.safe_load(goal_path.read_text()),
                        semantic, protocol['franka_base_height_m'])
                if native_controller_parameters(results[scene_id]).get('native_osc_matched_coarse_model', False) != cfg.get('c3_osc_matched_coarse_model', False):
                    raise ValueError('Recorded native coarse model differs from frozen configuration')
                planner_clearance = cfg.get('task_height_clearance_m') if cfg.get('enforce_planner_finger_floor', False) else None
                if native_controller_parameters(results[scene_id]).get('native_planner_finger_table_clearance') != planner_clearance:
                    raise ValueError('Recorded native finger floor differs from frozen configuration')
                model_enabled = native_controller_parameters(results[scene_id]).get('native_relinearized_pd_cost', False)
                if type(model_enabled) is not bool or model_enabled != cfg.get('c3_relinearized_pd_cost', False):
                    raise ValueError('Recorded native PD model differs from frozen evaluation configuration')
                if 'planning_horizon' in cfg and native_controller_parameters(results[scene_id]).get('native_planning_horizon') != cfg['planning_horizon']:
                    raise ValueError('Recorded native planning horizon differs from frozen configuration')
                files[scene_id] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if any(c != configurations[0] for c in configurations):
        raise ValueError('Controller configurations differ across the acceptance batch')
    report = summarize(protocol, ids, results)
    report.update(schema='nonprehensile.simulation_acceptance_audit.v1', protocol=protocol,
                  result_files=files, scene_pose_audits=scene_pose_audits,
                  audit_source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [Path(__file__), Path(__file__).with_name('audit_strict_pose_dwell.py'),
                                   Path(__file__).with_name('audit_scene_pose_contract.py')]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['goal_passed', 'total_fixed_scenes', 'successes', 'success_rate']}))
    return 0 if report['goal_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
