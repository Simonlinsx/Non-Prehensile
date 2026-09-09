import json
from pathlib import Path

import pytest

from scripts.prepare_offline_pd_inertia import extract


def fixture():
    return json.loads((Path(__file__).parent / 'fixtures/offline_pd_inertia_actual.json').read_text())


def test_reconstructs_actual_partial_osc_from_joint_dynamics():
    f = fixture()
    report = extract(f['result'], f['capture_us'])
    assert report['maximum_difference_from_cached_inertia_kg'] < 2e-6
    assert report['eigenvalues_kg'] == pytest.approx([.76161266, .98159346, 9.24717503], abs=1e-7)
    assert 'before the final physics substep' in report['state_timing']


@pytest.mark.parametrize('change', ['frame', 'coupled_osc', 'cached', 'mass', 'timestamp'])
def test_refuses_incompatible_or_corrupt_measured_dynamics(change):
    f = fixture()
    row = f['result']['trace'][0]
    r = row['osc_dynamics_audit']
    if change == 'frame': r['frame'] = 'world_origin'
    elif change == 'coupled_osc': r['partial_inertial_dynamics_decoupling'] = False
    elif change == 'cached': r['osc_operational_inertia_b'][0][0] += .001
    elif change == 'mass': r['joint_mass_matrix_kg_m2'][0][0] = -1
    else: row['measurement_utime_us'] += 10000
    with pytest.raises(ValueError):
        extract(f['result'], f['capture_us'])
