"""Independently check pose acceptance and per-object semantic contact policy."""
import argparse
import hashlib
import json
from pathlib import Path
import math
import copy
import numpy as np
from audit_m1_c3_comparison import audit_rows
from fr3_semantic_cost_map import query


def main(folder):
    launch=json.loads((folder/'launch.json').read_text());proc=json.loads((folder/'process.json').read_text())
    assert proc['returncode']==0 and not proc['timeout'] and not proc['remaining_live_pids'],proc
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in launch['input_hashes'].items()),'Frozen input changed'
    result=json.loads((folder/'result.json').read_text())
    rows=[json.loads(s) for s in (folder/'physics.jsonl').read_text().splitlines()]
    events=[json.loads(s) for s in (folder/'semantic_queries.jsonl').read_text().splitlines()]
    contacts=[r for r in events if r['event']=='physical_contact']
    assert len(rows)==len(contacts)==result['steps']
    policy=json.loads(Path(launch['environment']['FR3_SEMANTIC_POLICY']).read_text())
    rules={o['id']:o for o in policy['objects']};maxforces={i:dict(target_n=0.,robot_n=0.,target_contact_steps=0,robot_contact_steps=0) for i in rules}
    violations=[]
    for i,(r,c) in enumerate(zip(rows,contacts),1):
        assert r['step']==c['physics_step']==i
        assert set(o['id'] for o in c['objects'])==set(rules)
        hard_target=hard_robot=0.
        for o in c['objects']:
            rule=rules[o['id']];assert o['mode']==rule['mode']
            tf,rf=o['target_force_n'],o['robot_force_n'];assert all(math.isfinite(x) and x>=0 for x in (tf,rf))
            m=maxforces[o['id']];m['target_n']=max(m['target_n'],tf);m['robot_n']=max(m['robot_n'],rf);m['target_contact_steps']+=tf>.02;m['robot_contact_steps']+=rf>.02
            if rule['mode']=='forbidden':hard_target=max(hard_target,tf);hard_robot=max(hard_robot,rf)
        assert abs(hard_target-r['target_obstacle_force_n'])<1e-7 and abs(hard_robot-r['robot_obstacle_force_n'])<1e-7
        assert r['c2']==(hard_target>.02) and r['c3']==(hard_robot>.02)
        if r['c2'] or r['c3']:violations.append(i)
    original=audit_rows(rows,result['goal_pose_wxyz'],result['initial_pose_wxyz'],dt=1/240)
    success=original['strict_success'] and not violations
    assert success==result['constrained_success']
    candidates=[e for e in events if e['event']=='candidate'];counterfactual=0
    for c in candidates:
        # Candidate centers and displacements come from float32 PhysX/torch
        # tensors in the frozen runtime. Preserve dtype across JSON replay,
        # including ceil-based sweep sample counts near exact 2 mm boundaries.
        center=np.asarray(c['center'],dtype=c.get('center_dtype','float32'))
        delta=np.asarray(c['delta'],dtype=c.get('delta_dtype','float32'))
        if c.get('backend')=='volume3d':
            from fr3_semantic_sweep3d import from_record
            computed=from_record(c['geometry_record']).query(center,delta,c['yaw'])
        else:
            computed=query(c['footprint'],center,delta,c['yaw'],c['obstacles'])
        saved=c['result']
        assert all(computed[k]==saved[k] for k in ('admissible','hard_rejections','samples','margin_m'))
        assert abs(computed['soft_cost']-saved['soft_cost'])<1e-7
        assert max(abs(computed['clearance_m'][k]-saved['clearance_m'][k]) for k in saved['clearance_m'])<1e-7
        if c.get('backend')=='volume3d':
            all_hard=copy.deepcopy(c['geometry_record'])
            for o in all_hard['obstacles']:o['rule']['mode']='forbidden'
            counter=from_record(all_hard).query(center,delta,c['yaw'])
        else:
            all_hard=[dict(o,mode='forbidden') for o in c['obstacles']]
            counter=query(c['footprint'],center,delta,c['yaw'],all_hard)
        counterfactual+=computed['admissible'] and not counter['admissible']
        if 'weighted_cost_m' in c:
            assert abs(c['weighted_cost_m']-saved['soft_cost']*c['semantic_weight_m'])<1e-10
    rankings=[e for e in events if e['event']=='ranking']
    changes=0
    for e in rankings:
        assert e['task_scores_m'].keys()==e['total_scores_m'].keys()
        for rank,value in e['task_scores_m'].items():
            assert abs(value+e['semantic_costs_m'].get(rank,0)-e['total_scores_m'][rank])<1e-9
        changes+=sorted(e['task_scores_m'],key=e['task_scores_m'].get)!=sorted(e['total_scores_m'],key=e['total_scores_m'].get)
    for e in events:
        if e['event']=='micro' and e.get('geometry_record'):
            replay=from_record(e['geometry_record']).query(e['center'],e['delta'],e['yaw'])
            assert replay['admissible']==e['result']['admissible']
            assert abs(replay['soft_cost']-e['result']['soft_cost'])<1e-7
    model=json.loads((folder/'model_audit.json').read_text());assert model['fk_position_error_m']<1e-5 and model['fk_rotation_matrix_error']<1e-5
    stages=[e for e in events if e['event']=='stage_begin']
    stage_sequence_pass=None
    if policy.get('rotate_then_translate',False):
        stage_sequence_pass=([s['phase'] for s in stages]==['rotate','translate'] and
                             stages[1]['translation_before_stage_m']<=.02 and stages[1]['rotation_error_rad']<.10)
    final=dict(**original,strict_semantic_success=success,stage_sequence_pass=stage_sequence_pass,hard_collision_steps=len(violations),contacts=maxforces,
        candidate_queries=len(candidates),candidate_rejections=sum(not c['result']['admissible'] for c in candidates),
        candidates_allowed_only_by_acceptable_contact=counterfactual,
        micro_queries=sum(e['event']=='micro' for e in events),whole_arm_phase_queries=sum(e['event']=='whole_arm_phase' for e in events),
        ranking_events=len(rankings),ranking_changes_due_to_semantic_cost=changes,
        volume_query_wall_s=sum(e['result'].get('query_wall_s',0) for e in events if e['event'] in ('candidate','micro')),
        stages=stages,process_wall_s=proc['wall_s'],source_hashes_pass=True,
        scope='Assistant-authored cached semantics + ground-truth simulation geometry. Rigid proxies. Not autonomous VLM inference, fluid/deformation safety validation, or a population success rate.')
    (folder/'semantic_audit.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);main(p.parse_args().folder.resolve())
