#!/usr/bin/env python3
"""Compile the actual native C1 quaternion initializers and check frame maps."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--eigen-root', type=Path, required=True)
    args = p.parse_args()
    extracted = []
    for path, variable, parameters in [
        ('systems/controllers/sampling_based_c3_controller.cc', 'object_quaternion',
         'const Eigen::VectorXd& x_lcs, int quaternion_index'),
        ('examples/sampling_c3/semantic_c1_trajectory_guard.cc', 'quaternion',
         'const Eigen::VectorXd& positions'),
    ]:
        source = (args.upstream_root / path).read_text()
        matches = re.findall(r'(?:Eigen::)?Quaterniond\s+' + variable + r'\s*\((.*?)\);', source, re.S)
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one native initializer for {variable}')
        extracted.append(f'Eigen::Quaterniond make_{variable}({parameters}) {{ Eigen::Quaterniond {variable}({matches[0]}); return {variable}.normalized(); }}')
    code = '#include <Eigen/Geometry>\n#include <iostream>\n' + '\n'.join(extracted) + r'''
int main() {
  double maximum_error = 0;
  int checks = 0;
  for (int i = 0; i < 37; ++i) {
    const double yaw = -3.141592653589793 + i * 3.141592653589793 / 18;
    Eigen::Quaterniond expected(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                                Eigen::AngleAxisd(.23, Eigen::Vector3d::UnitY()) *
                                Eigen::AngleAxisd(-.17, Eigen::Vector3d::UnitX()));
    for (double scale : {-2., 1.}) {
      Eigen::Vector4d q(expected.w(), expected.x(), expected.y(), expected.z());
      q *= scale;
      Eigen::VectorXd x_lcs = Eigen::VectorXd::Zero(19), positions = Eigen::VectorXd::Zero(7);
      x_lcs.segment<4>(3) = q; positions.head<4>() = q;
      const Eigen::Vector3d body(.09, -.04, .013), translation(.4, .2, -.016);
      const Eigen::Vector3d world = expected * body + translation;
      for (auto actual : {make_object_quaternion(x_lcs, 3), make_quaternion(positions)}) {
        maximum_error = std::max(maximum_error, (actual.inverse() * (world - translation) - body).norm());
        ++checks;
      }
    }
  }
  std::cout << "{\"frame_map_checks\":" << checks << ",\"maximum_error_m\":" << maximum_error << "}\n";
  return maximum_error < 1e-12 ? 0 : 2;
}
'''
    with tempfile.TemporaryDirectory(prefix='semantic_quaternion_', dir='/tmp') as directory:
        source, binary = Path(directory) / 'check.cc', Path(directory) / 'check'
        source.write_text(code)
        subprocess.run(['g++', '-std=c++17', '-O2', '-I' + str(args.eigen_root), str(source), '-o', str(binary)], check=True)
        result = subprocess.run([str(binary)], capture_output=True, text=True)
        print(result.stdout.strip())
        return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
