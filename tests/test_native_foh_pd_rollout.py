"""Compile the actual patch helper against Eigen; exercise rollout boundaries."""
from pathlib import Path
import subprocess


def test_native_foh_rollout_preserves_measured_state_and_matches_reference(tmp_path):
    root = Path(__file__).resolve().parents[1]
    patch = root / 'third_party/push_anything/patches/0013-optional-foh-pd-rollout.patch'
    subprocess.run(['git', 'apply', '--include=systems/controllers/foh_pd_rollout.h', str(patch)],
                   cwd=tmp_path, check=True, capture_output=True)
    cpp = tmp_path / 'check.cc'
    cpp.write_text(r'''
#include "systems/controllers/foh_pd_rollout.h"
#include <cassert>
#include <limits>
using Eigen::VectorXd;
using std::vector;
int main() {
  // EE position [0:3], an untouched object scalar [3], EE velocity [4:7].
  vector<VectorXd> xs(3, VectorXd::Zero(7));
  xs[0][3] = 42; xs[0][4] = .25;
  xs[1] = xs[0]; xs[1][0] = 1; xs[1][4] = -99;
  xs[2] = VectorXd::Constant(7, 999); // unpublished terminal extension
  vector<VectorXd> us(2, VectorXd::Zero(3)); us[0][0] = 2; us[1][0] = 4;
  VectorXd kp = VectorXd::Zero(7), kd = kp; kp[0] = 3; kd[4] = 2;
  vector<double> seen;
  auto identity = [&](const VectorXd& x, const VectorXd& u) {
    assert(x.isApprox(xs[0], 1e-14)); seen.push_back(u[0]); return x;
  };
  auto result = dairlib::SimulateFOHPDRollout(xs, us, kp, kd, 4, .5, 2, true, identity);
  // FOH reference: x=0,.5,1,1; v=2,2,0,0; ff=2,3,4,4.
  const vector<double> expected{5.5,8.,6.5,6.5};
  assert(seen == expected);
  assert(result.first.size() == 3 && result.second.size() == 2);
  assert(result.first[0].isApprox(xs[0], 1e-14));
  assert(result.second[0][0] == 5.5 && result.second[1][0] == 6.5);
  seen.clear();
  dairlib::SimulateFOHPDRollout(xs, us, kp, kd, 4, .5, 2, false, identity);
  assert((seen == vector<double>{3.5,5.,2.5,2.5}));
  // Stateful transitions and downsampling preserve every fine step.
  int calls = 0;
  auto advance = [&](const VectorXd& x, const VectorXd&) {
    auto next = x; next[3] += 1; ++calls; return next;
  };
  result = dairlib::SimulateFOHPDRollout(xs, us, kp, kd, 4, .5, 4, true, advance);
  assert(calls == 8 && result.first[0][3] == 42 && result.first[1][3] == 46 && result.first[2][3] == 50);
  // One published knot is a constant reference, even at its first instant.
  xs.resize(2); us.resize(1); seen.clear();
  dairlib::SimulateFOHPDRollout(xs, us, kp, kd, 4, .5, 1, true, identity);
  assert((seen == vector<double>{1.5}));
  auto rejects = [&](double dt, int rate) {
    try { dairlib::SimulateFOHPDRollout(xs, us, kp, kd, 4, dt, rate, true, identity); }
    catch (const std::invalid_argument&) { return true; }
    return false;
  };
  assert(rejects(.5,0) && rejects(0,1) && rejects(std::numeric_limits<double>::quiet_NaN(),1));
  xs[1][0] = std::numeric_limits<double>::infinity(); assert(rejects(.5,1));
  xs[1][0] = 0; kp[3] = 1; assert(rejects(.5,1));
}
''')
    exe = tmp_path / 'check'
    subprocess.run(['g++', '-std=c++17', '-O0', '-I/usr/include/eigen3', '-I'+str(tmp_path),
                    str(cpp), '-o', str(exe)], check=True, capture_output=True)
    subprocess.run([str(exe)], check=True, capture_output=True)
