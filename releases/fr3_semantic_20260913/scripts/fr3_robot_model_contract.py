"""Validate the same pinned FR3 input bytes before starting either executor."""
import hashlib
import json
from pathlib import Path


def load_contract(path):
    path = Path(path).expanduser().resolve()
    model = json.loads(path.read_text())
    if (model.get('schema'), model.get('robot'), model.get('end_effector')) != (
        'nonprehensile.fr3_robot_model.v1', 'FR3', 'stock_closed_gripper'
    ):
        raise ValueError('Expected FR3 with the stock closed gripper')
    for key in ('official_urdf', 'simulation_urdf', 'native_urdf'):
        if hashlib.sha256(Path(model[key]).read_bytes()).hexdigest() != model[key + '_sha256']:
            raise ValueError(f'Robot model checksum mismatch: {key}')
    for mesh, digest in model['mesh_sha256'].items():
        if hashlib.sha256(Path(mesh).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Robot mesh checksum mismatch: {mesh}')
    model['manifest_path'] = str(path)
    model['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    args = parser.parse_args()
    print(load_contract(args.manifest)['native_urdf'])
