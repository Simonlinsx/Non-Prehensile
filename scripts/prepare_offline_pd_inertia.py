"""Extract the recorded partial-OSC translational apparent inertia for offline replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def extract(result, capture_us):
    rows = [r for r in result['trace'] if abs(r['measurement_utime_us'] - capture_us) <= 1]
    if len(rows) != 1:
        raise ValueError('Missing unique captured dynamics record')
    record = rows[0]['osc_dynamics_audit']
    if (record['frame'] != 'robot_root_axes_at_task_point'
            or record['inertial_dynamics_decoupling'] is not True
            or record['partial_inertial_dynamics_decoupling'] is not True):
        raise ValueError('Requires recorded partial OSC at the native task point')
    jacobian = np.asarray(record['jacobian_b'])
    mass = np.asarray(record['joint_mass_matrix_kg_m2'])
    cached = np.asarray(record['osc_operational_inertia_b'])
    if (jacobian.shape != (6, 7) or mass.shape != (7, 7) or cached.shape != (6, 6)
            or not all(np.isfinite(a).all() for a in (jacobian, mass, cached))
            or not np.allclose(mass, mass.T, atol=1e-7, rtol=0)
            or np.linalg.eigvalsh(mass).min() <= 0):
        raise ValueError('Invalid measured dynamics matrices')
    jv = jacobian[:3]
    inertia = np.linalg.inv(jv @ np.linalg.solve(mass, jv.T))
    inertia = (inertia + inertia.T) / 2
    difference = float(np.max(np.abs(inertia - cached[:3, :3])))
    if (not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 1e-6
            or difference > 1e-5 or np.max(np.abs(cached[:3, 3:])) > 1e-7):
        raise ValueError('Reconstructed inertia differs from recorded partial OSC')
    return dict(matrix_kg=inertia.tolist(), eigenvalues_kg=np.linalg.eigvalsh(inertia).tolist(),
                maximum_difference_from_cached_inertia_kg=difference,
                state_timing=record['state_timing'], capture_utime_us=capture_us,
                limitation='Frozen local 3D inertia approximation. Cached state precedes the last physics substep; '
                           'arm rotation, nullspace, joint limits and inertia changes are not modeled.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.directory / 'input_provenance.json').read_text())
    source = Path(meta['source_scene_directory']) / 'result.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != meta['source_sha256'][str(source)]:
        raise ValueError('Physical source hash changed')
    report = extract(json.loads(source.read_text()), meta['capture_utime_us'])
    report['source_sha256'] = {str(source): digest}
    text = ' '.join(str(v) for row in report['matrix_kg'] for v in row)
    (args.directory / 'inertia.txt').write_text(text + '\n')
    report['inertia_input_sha256'] = hashlib.sha256((args.directory / 'inertia.txt').read_bytes()).hexdigest()
    (args.directory / 'inertia_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
