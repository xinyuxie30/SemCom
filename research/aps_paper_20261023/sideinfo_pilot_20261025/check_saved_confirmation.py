"""CPU-only independent recomputation from saved CSV, without the model runner."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
OUT=HERE/'confirm_20261025T140525Z'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    frozen=json.loads((OUT/'confirmation_inputs.json').read_text(encoding='utf8'))
    with (OUT/'confirmation_cases.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==4320
    assert len({tuple(r[k] for k in ('view','mode','scene','snr_db','noise_index','condition')) for r in rows})==4320
    for r in rows:
        r['mse']=float(r['mse']);r['psnr_db']=float(r['psnr_db'])
        assert math.isfinite(r['mse']) and r['mse']>0
        assert abs(r['psnr_db']+10*math.log10(r['mse']))<1e-10
    for p,h in result['files'].items():assert sha(OUT/p)==h,p
    for p in frozen['images']:assert sha(Path(p['path']))==p['sha256'],p
    for s,w in frozen['wrong_reference'].items():assert frozen['physical_groups'][s]!=frozen['physical_groups'][w]
    for methods in frozen['selections'].values():
        for c in methods.values():assert sha(Path(c['path']))==c['sha256']
    assert sha(HERE/'audit_and_confirm.py')==frozen['script_sha256']
    assert sha(HERE/'CONFIRMATION_PROTOCOL.txt')==frozen['confirmation_protocol_sha256']
    computed={};scene_stats={}
    for view,methods in result['summary'].items():
        computed[view]={}
        for m,a in methods.items():
            rr=[r for r in rows if r['view']==view and r['mode']==m]
            main=[r for r in rr if r['condition']!='none']
            assert len(rr)==360 and len(main)==270
            for k in ('mse','psnr_db'):assert abs(mean(r[k] for r in main)-a[k])<1e-10
            for cond,b in a['by_condition'].items():
                cr=[r for r in rr if r['condition']==cond];assert len(cr)==90
                for k in ('mse','psnr_db'):assert abs(mean(r[k] for r in cr)-b[k])<1e-10
            groups={g:mean(r['psnr_db'] for r in main if frozen['physical_groups'][r['scene']]==g) for g in set(frozen['physical_groups'].values())}
            assert abs(mean(groups.values())-result['equal_group_main_psnr'][view][m])<1e-10
            computed[view][m]=mean(r['psnr_db'] for r in main)
        controls=[m for m in methods if m not in ('cycle_photo','frozen_swin')]
        delta=computed[view]['cycle_photo']-max(computed[view][m] for m in controls)
        assert abs(delta-result['decisions'][view]['main_delta_db'])<1e-10
        assert delta<.20 and not result['decisions'][view]['numeric_pass']
        control=result['decisions'][view]['strongest_control']
        per_scene={s:mean(r['psnr_db'] for r in rows if r['view']==view and r['scene']==s and r['mode']=='cycle_photo' and r['condition']!='none')-mean(r['psnr_db'] for r in rows if r['view']==view and r['scene']==s and r['mode']==control and r['condition']!='none') for s in frozen['scenes']}
        scene_stats[view]=dict(positive_pairs=sum(d>0 for d in per_scene.values()),negative_pairs=sum(d<0 for d in per_scene.values()),min_db=min(per_scene.values()),max_db=max(per_scene.values()))
    assert result['additional_training_steps']==0 and result['action']=='STOP_NO_EXTENSION'
    report=dict(status='PASSED',metric_rows=4320,views=['best','final'],scene_stats=scene_stats,
                hashes_verified=True,no_same_group_wrong_references=True,independent_recomputed_main_psnr=computed,
                checks='Saved CSV uniqueness/formulas/counts/means/group weighting/gates, file/input/checkpoint hashes; CPU only.',
                script_sha256=sha(Path(__file__)))
    (OUT/'independent_csv_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
