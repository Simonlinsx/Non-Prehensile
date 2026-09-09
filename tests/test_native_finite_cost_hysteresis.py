"""Exercise the actual native helper, including rejected model predictions."""
from pathlib import Path
import subprocess


def test_native_hysteresis_preserves_finite_decisions_and_rejects_invalid_costs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    patch = root / 'third_party/push_anything/patches/0020-finite-cost-hysteresis.patch'
    subprocess.run(['git', 'apply', '--include=systems/controllers/finite_cost_hysteresis.h', str(patch)],
                   cwd=tmp_path, check=True, capture_output=True)
    cpp = tmp_path / 'check.cc'
    cpp.write_text(r'''
#include "systems/controllers/finite_cost_hysteresis.h"
#include <cassert>
#include <limits>
using dairlib::CostImprovesWithHysteresis;
using dairlib::RepositionCostWithHysteresis;
int main() {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (bool relative : {false, true}) {
    for (double fraction : {0., .01, .1, .5}) {
      assert(CostImprovesWithHysteresis(inf, 20., 5., fraction, relative));
      assert(CostImprovesWithHysteresis(nan, 20., 5., fraction, relative));
      assert(!CostImprovesWithHysteresis(20., inf, 5., fraction, relative));
      assert(!CostImprovesWithHysteresis(20., nan, 5., fraction, relative));
      assert(!CostImprovesWithHysteresis(inf, inf, 5., fraction, relative));
      // Original finite arithmetic, strict equality and tie behavior retained.
      for (double current : {0., 10., 20., 100., 567.3465154269113, 703.657033888167})
        for (double candidate : {0., 10., 15., 20., 99., 567.3465154269113}) {
          const bool original = relative ? current > candidate + fraction * current
                                         : current > candidate + 5.;
          assert(CostImprovesWithHysteresis(current, candidate, 5., fraction, relative) == original);
          const double original_repos = relative ? candidate + fraction * current : candidate + 5.;
          assert(RepositionCostWithHysteresis(candidate, current, 5., fraction, relative) == original_repos);
        }
    }
  }
  // Invalid previous reposition target cannot poison a valid new target.
  assert(RepositionCostWithHysteresis(12., inf, 5., .1, true) == 12.);
  assert(RepositionCostWithHysteresis(12., nan, 5., .1, true) == 12.);
  assert(RepositionCostWithHysteresis(12., inf, 5., .1, false) == 17.);
  assert(!CostImprovesWithHysteresis(20., 15., 5., .25, false));
  assert(!CostImprovesWithHysteresis(20., 15., 5., .25, true));
}
''')
    exe = tmp_path / 'check'
    subprocess.run(['g++', '-std=c++17', '-O2', '-include', 'initializer_list', '-I'+str(tmp_path),
                    str(cpp), '-o', str(exe)], check=True, capture_output=True)
    subprocess.run([str(exe)], check=True, capture_output=True)
