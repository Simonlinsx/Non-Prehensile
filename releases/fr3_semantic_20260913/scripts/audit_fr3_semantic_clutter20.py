"""Incremental independent audit; only completed matched pairs enter comparison."""
from contextlib import redirect_stdout
import io
import json
import argparse
from pathlib import Path
import numpy as np
from audit_fr3_semantic_demo import main as audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/fr3_semantic_map_20260913/clutter20_v1'


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());rows=[];invalid=[]
    for job in protocol['jobs']:
        folder=Path(job['folder'])
        if not (folder/'process.json').exists():continue
        process=json.loads((folder/'process.json').read_text())
        if (process['returncode']!=0 or process['timeout'] or not (folder/'result.json').exists()
            or 'Traceback (most recent call last)' in (folder/'process.log').read_text(errors='replace')):
            invalid.append(dict(case=job['case'],group=job['group'],folder=str(folder),reason='Invalid process / missing result'));continue
        if not (folder/'semantic_audit.json').exists():
            with redirect_stdout(io.StringIO()):audit(folder)
        a=json.loads((folder/'semantic_audit.json').read_text());r=json.loads((folder/'result.json').read_text())
        contacts=a['contacts'];allowed=[o['id'] for o in json.loads((folder/'semantic_policy.json').read_text())['objects'] if o['mode']=='acceptable']
        values=[]
        for l in open(folder/'semantic_queries.jsonl'):
            e=json.loads(l)
            if e['event']=='physical_contact':values.append(sum(o['target_force_n'] for o in e['objects'] if o['id'] in allowed))
        force=np.array(values)
        row=dict(case=job['case'],layout=job['layout'],group=job['group'],strict_success=a['strict_semantic_success'],
            terminal_dwell_s=a['terminal_dwell_s'],safe_contact_gt05N=a['safe_contact_gt05N'],
            safe_contact_gt002N=r['safe_002'],
            pose_dwell_safety_with002N_contact=(a['terminal_dwell_s']>=.5 and r['safe_002'] and not r['c1'] and
                a['hard_collision_steps']==0 and a['max_finger_mm']<1),
            final_xy_mm=1000*r['last']['xy_m'],final_so3_rad=r['last']['rotation_rad'],sim_s=r['sim_s'],
            wall_s=a['process_wall_s'],video=job['video'],c1_violation=r['c1'],hard_collision_steps=a['hard_collision_steps'],
            acceptable_target_force_peak_n=float(force.max()),acceptable_force_norm_integral_ns=float(force.sum()/240),
            acceptable_contact_gt2mN_s=float((force>.002).sum()/240),
            ranking_changes=a['ranking_changes_due_to_semantic_cost'],ranking_events=a['ranking_events'],
            core_volume_query_s=a['volume_query_wall_s'],stop_reason=r['stop_reason'])
        (folder/'evaluation_row.json').write_text(json.dumps(row,indent=2));rows.append(row)
    cases={}
    for r in rows:cases.setdefault(r['case'],{})[r['group']]=r
    paired={k:v for k,v in cases.items() if set(v)=={'hard','soft'}}
    counts={g:dict(completed=sum(r['group']==g for r in rows),successes=sum(r['strict_success'] for r in rows if r['group']==g)) for g in ['hard','soft']}
    for case in paired:
        a=OUT/'hard'/case;b=OUT/'soft'/case
        for name in ['manifest.jsonl','semantic_policy.json']:assert json.loads((a/name).read_text())==json.loads((b/name).read_text())
        ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text())
        assert ra['initial_pose_wxyz']==rb['initial_pose_wxyz'] and ra['goal_pose_wxyz']==rb['goal_pose_wxyz']
    result=dict(complete=len(rows)==40 and not invalid,completed_pairs=len(paired),counts=counts,
        paired_successes={g:sum(v[g]['strict_success'] for v in paired.values()) for g in ['hard','soft']},
        soft_only=[k for k,v in paired.items() if v['soft']['strict_success'] and not v['hard']['strict_success']],
        hard_only=[k for k,v in paired.items() if v['hard']['strict_success'] and not v['soft']['strict_success']],
        invalid_attempts=invalid,rows=rows,scope=protocol['scope'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','scope']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,default=OUT)
    OUT=parser.parse_args().root.resolve()
    main()
