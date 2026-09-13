"""Prepare or execute the frozen simulation release. Never opens hardware."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--example', choices=['optional_contact', 'doll_side', 'doll_front', 'bowl_sponge', 'sitting_doll'], default='sitting_doll')
    p.add_argument('--holdout-case', type=int, choices=range(20), help='Run one of the exact frozen held-out manifests (0–19).')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--weight', type=float, choices=[0., .5], default=.5)
    p.add_argument('--video', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--domino-root', type=Path, default=Path(os.environ.get('DOMINO_ROOT', '/data1/linsixu/DOMINO')))
    p.add_argument('--usd-root', type=Path, default=Path(os.environ.get('DOMINO_USD_ROOT', str(REPO / 'data/domino_usd'))))
    p.add_argument('--model-manifest', type=Path, default=REPO / 'data/robot_models/fr3_stock_20260909/robot_model_manifest.json')
    p.add_argument('--robot-usd-cache', type=Path, default=Path('/tmp/IsaacLab/fr3_stock_b500db5814f1b9f4'))
    p.add_argument('--sim-seconds', type=float, default=60.)
    p.add_argument('--execute', action='store_true', help='Without this flag, only validate inputs and write the launch file.')
    a = p.parse_args()
    if not 0 < a.sim_seconds <= 60: p.error('sim-seconds must be in (0, 60]')
    frozen = json.loads((HERE / 'source_manifest.json').read_text())
    for name, item in frozen['files'].items():
        if sha(HERE / name) != item['sha256']: raise ValueError('Frozen source changed: ' + name)
    sys.path.insert(0, str(HERE / 'scripts'))
    from fr3_robot_model_contract import load_contract
    model = load_contract(a.model_manifest)
    for folder in [a.domino_root, a.usd_root]:
        if not folder.is_dir(): raise FileNotFoundError(f'Required external dataset/cache: {folder}')
    config_scene = HERE / 'examples' / ('optional_contact' if a.holdout_case is not None else a.example)
    scene = HERE / 'examples/holdout20' / f'{a.holdout_case:03}' if a.holdout_case is not None else config_scene
    config = json.loads((config_scene / 'config.json').read_text())
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'process.log').exists(): raise FileExistsError('Never overwrite an existing attempt: ' + str(out))
    for name in ['manifest.jsonl', 'semantic_policy.json']: shutil.copy2(scene / name, out / name)
    env = dict(config['environment'])
    env.update(PYTHON_BIN=sys.executable, GPU_ID=a.gpu, OUTPUT_ROOT=str(out), M1_COMPARISON_OUTPUT=str(out),
        MANIFEST=str(out / 'manifest.jsonl'), FR3_SEMANTIC_POLICY=str(out / 'semantic_policy.json'),
        FR3_SEMANTIC_ASSETS=str(HERE / 'assets' / config['assets']),
        DAPL_LOCAL_FRANKA_URDF=model['simulation_urdf'], DAPL_LOCAL_FRANKA_USD_DIR=str(a.robot_usd_cache.resolve()),
        PUSH_ANYTHING_ROBOT_MODEL_MANIFEST=str(a.model_manifest.resolve()),
        DOMINO_ROOT=str(a.domino_root.resolve()), DOMINO_USD_ROOT=str(a.usd_root.resolve()),
        FR3_SEMANTIC_WEIGHT_M=str(a.weight), VIDEO=str(int(a.video)), FR3_SEMANTIC_SNAPSHOT=str(int(a.video)),
        M1_COMPARISON_SIM_S=str(a.sim_seconds), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    command = ['bash', str(HERE / 'scripts/run_contact_planner_m1.sh'), *config['command_arguments']]
    inputs = [HERE / name for name in frozen['files']]
    inputs += [q for q in (HERE / 'assets' / config['assets']).rglob('*') if q.is_file()]
    inputs += [out / 'manifest.jsonl', out / 'semantic_policy.json', a.model_manifest.resolve(),
               Path(model['simulation_urdf'])]
    launch = dict(cwd=str(REPO), command=command, environment=env, wall_limit_s=900,
        input_hashes={str(q): sha(q) for q in inputs},
        scope='Release runner; copied frozen controller. Simulation only, no real robot connection.',
        budget_override=a.sim_seconds != 60.)
    (out / 'launch.json').write_text(json.dumps(launch, indent=2))
    print(out / 'launch.json', flush=True)
    if not a.execute: return
    if shutil.disk_usage(out).free < 2 * 1024**3: raise RuntimeError('Need 2 GiB free for bounded rollout')
    from supervisor import bounded_process
    with (out / 'process.log').open('x') as log:
        result = bounded_process(command, cwd=REPO, env={**os.environ, **env}, log=log, timeout_s=900)
    result.update(timeout=result['reason'] == 'wall_timeout', wall_s=result['process_wall_s'],
        frozen_inputs_unchanged=all(sha(Path(q)) == h for q, h in launch['input_hashes'].items()))
    (out / 'process.json').write_text(json.dumps(result, indent=2))
    valid = result['returncode'] == 0 and not result['timeout'] and result['frozen_inputs_unchanged'] and \
        not result['remaining_live_pids'] and (out / 'result.json').exists() and \
        'Traceback (most recent call last)' not in (out / 'process.log').read_text(errors='replace')
    if not valid: raise RuntimeError('Invalid attempt retained; inspect process.json and process.log')
    from audit_fr3_semantic_demo import main as audit
    audit(out)


if __name__ == '__main__': main()
