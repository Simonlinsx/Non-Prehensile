#!/usr/bin/env python3
"""Reconstruct actual cached OSC inertia from its mass matrix and Jacobian."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def audit(result, start_s, end_s):
    params=result['controller_parameters']
    if params.get('osc_dynamics_audit') is not True:
        raise ValueError('Result does not contain enabled OSC dynamics instrumentation')
    uri=Path(result['push_anything_runtime_root'])/'examples/sampling_c3/urdf/end_effector_simple_model.urdf'
    xml=uri.read_text()
    if "xmlns:drake=" not in xml:
        xml=re.sub(r"<robot\b", '<robot xmlns:drake="urn:drake"', xml, count=1)
    mass=float(ET.fromstring(xml).find("link[@name='end_effector_simple']/inertial/mass").attrib['value'])
    kp=params['osc_translation_stiffness'];kd=2*np.sqrt(kp)*params['osc_damping_ratio']
    rows=[]
    for row in result['trace']:
        clock=row['measurement_utime_us']/1e6-result['control_period_s']
        if not start_s<=clock<end_s:continue
        state=row.get('osc_dynamics_audit')
        if not state:raise ValueError('Missing OSC snapshot within requested window')
        M=np.asarray(state['joint_mass_matrix_kg_m2']);J=np.asarray(state['jacobian_b']);cached=np.asarray(state['osc_operational_inertia_b'])
        if M.shape!=(7,7) or J.shape!=(6,7) or cached.shape!=(6,6) or not all(np.isfinite(v).all() for v in (M,J,cached)):
            raise ValueError('Invalid cached dynamics dimensions or values')
        eigM=np.linalg.eigvalsh(M)
        if eigM.min()<=0:raise ValueError('Joint inertia must be positive definite')
        inverse_operational=J@np.linalg.solve(M,J.T)
        linear=np.linalg.inv(inverse_operational[:3,:3])
        full=np.linalg.inv(inverse_operational)
        discrepancy=float(np.max(np.abs(linear-cached[:3,:3])))
        if discrepancy>1e-3:raise ValueError('Cached partial OSC inertia does not match recorded mass/Jacobian')
        eig=np.linalg.eigvalsh(linear)
        rows.append({'planner_clock_s':clock,'joint_inertia_minimum_eigenvalue':float(eigM.min()),
            'partial_translation_inertia_eigenvalues_kg':eig.tolist(),
            'partial_translation_inertia_diagonal_kg':np.diag(linear).tolist(),
            'effective_mass_along_root_axes_kg':(1/np.diag(inverse_operational[:3,:3])).tolist(),
            'orientation_constrained_translation_inertia_eigenvalues_kg':np.linalg.eigvalsh(full[:3,:3]).tolist(),
            'cached_inertia_max_abs_difference':discrepancy,
            'motion_stiffness_eigenvalues_n_m':(kp*eig).tolist(),
            'motion_damping_eigenvalues_n_s_m':(kd*eig).tolist()})
    if not rows:raise ValueError('No instrumented samples in requested interval')
    eigenvalues=np.array([r['partial_translation_inertia_eigenvalues_kg'] for r in rows])
    return {'schema':'nonprehensile.osc_operational_inertia_audit.v1','sample_count':len(rows),
        'window_planner_clock_s':[start_s,end_s], 'planner_point_mass_kg':mass,
        'planner_model':str(uri),'planner_model_sha256':hashlib.sha256(uri.read_bytes()).hexdigest(),
        'partial_translation_inertia_eigenvalue_range_kg':[float(eigenvalues.min()),float(eigenvalues.max())],
        'partial_translation_inertia_eigenvalue_medians_kg':np.median(eigenvalues,axis=0).tolist(),
        'inertia_eigenvalue_to_point_mass_ratio_range':[float(eigenvalues.min()/mass),float(eigenvalues.max()/mass)],
        'maximum_cached_inertia_disagreement':max(r['cached_inertia_max_abs_difference'] for r in rows),
        'rows':rows,'interpretation':[
            'These are the actual OSC cached matrices, independently reconstructed; no controller gains changed.',
            'Task inertia is direction- and configuration-dependent; a scalar point mass is an approximation.',
            'Partial translational inertia is the matrix used in current OSC motion feedback.',
            'Free-direction and orientation-constrained inertia describe different mechanical conditions; neither alone calibrates closed-loop contact.',
            'The snapshot precedes the final physics substep of its trace interval; it includes the joint state used by that OSC call.',
            'This is dynamics instrumentation, not task acceptance or proof that changing mass improves success.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start-s',type=float,required=True);p.add_argument('--end-s',type=float,required=True);a=p.parse_args()
    if not np.isfinite([a.start_s,a.end_s]).all() or not 0<=a.start_s<a.end_s:raise ValueError('Invalid window')
    report=audit(json.loads(a.result.read_text()),a.start_s,a.end_s);report['result']=str(a.result);report['result_sha256']=hashlib.sha256(a.result.read_bytes()).hexdigest();a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
