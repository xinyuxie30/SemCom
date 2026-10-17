"""Prepare or supervise sequential background experiments; no LLM polling."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'dcat_csi_prediction_20261004'
PYTHON=Path('D:/PyChram_project/DCAT-SoM/.venv/Scripts/python.exe')


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)


def prepare():
    job=HERE/'results'/datetime.now(timezone.utc).strftime('b01_%Y%m%dT%H%M%SZ')
    (job/'src').mkdir(parents=True)
    b0=Path(json.loads((OLD/'latest_b0.json').read_text())['run'])
    cfg=json.loads((b0/'protocol.json').read_text())
    cfg['id']='B01-20261008-v1'
    cfg['scope']='Post-B0 exploratory calibration and equalization factorial gate; no formal holdout claim'
    cfg['b01']={'calibration':'Validation pilots 600:800 only; no diagnostic_test labels. This is reused validation data, not an independent calibration split.',
                 'factors':['hold / empirical Wiener / ACF Wiener','original / validation-error-calibrated attention','ZF / error-aware MMSE'],
                 'equalizer':'g=conj(hhat)/(abs(hhat)^2 + validation_error_variance[horizon] + 10^(-SNR/10)); applied equally to hold and predictors.',
                 'images':'Same fixed 32 B0 validation scenes for every noise seed.',
                 'noise_seeds':[20261004,20261005,20261006],
                 'checkpoint_families':['without','with'],
                 'decision':'Report all arms and per-seed differences; no automatic selection on these development outcomes or automatic full training.'}
    cfg['e2e']['policies']=[prefix+suffix for prefix in ['hold','wiener','acf'] for suffix in ['','_cal','_mmse','_cal_mmse']]
    for n in ['run_e2e.py','run_b0.py','csi_predictors.py']:
        shutil.copy2(OLD/n,job/'src'/n)
    for n in ['worker.py','launch.py']:
        shutil.copy2(HERE/n,job/n)
    write(job/'protocol.json',cfg)
    hashes={str(p.relative_to(job)):hashlib.sha256(p.read_bytes()).hexdigest() for p in job.rglob('*.py')}
    status={'status':'PREPARED','job':str(job),'b0':str(b0),'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_hashes':hashes,'expected_transmissions':69120,'runs':[]}
    write(job/'status.json',status)
    write(HERE/'latest_job.json',{'job':str(job)})
    print(job)


def supervise(job):
    state=json.loads((job/'status.json').read_text())
    if state['status']!='PREPARED': raise RuntimeError('Refusing duplicate launch')
    state.update(status='RUNNING',supervisor_pid=os.getpid())
    write(job/'status.json',state)
    cfg=json.loads((job/'protocol.json').read_text())
    try:
        for family in cfg['b01']['checkpoint_families']:
            for seed in cfg['b01']['noise_seeds']:
                name=f'{family}_seed{seed}'
                local=json.loads(json.dumps(cfg));local['e2e']['seed']=seed
                protocol=job/(name+'_protocol.json');write(protocol,local)
                command=[str(PYTHON),'-u',str(job/'worker.py'),'--src',str(job/'src'),'--b0',state['b0'],
                         '--protocol',str(protocol),'--output',str(job/name),'--family',family]
                record={'name':name,'status':'RUNNING','command':command}
                state['runs'].append(record);write(job/'status.json',state)
                with (job/(name+'.log')).open('w',encoding='utf-8') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800,
                                          env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},creationflags=subprocess.CREATE_NO_WINDOW)
                record['exit_code']=result.returncode
                if result.returncode: raise RuntimeError(f'{name} failed; see log')
                record['status']='COMPLETED';write(job/'status.json',state)
        summary=[]
        for r in state['runs']:
            data=json.loads((job/r['name']/'summary.json').read_text())
            for row in data['rows']:
                summary.append({'run':r['name'],**row})
        write(job/'combined_summary.json',summary)
        with (job/'combined_summary.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        state.update(status='COMPLETED',finished_utc=datetime.now(timezone.utc).isoformat())
        (job/'REVIEW_READY.md').write_text('# Experiment completed\n\nAll six runs completed. Review combined_summary.csv and per_state.csv before selecting a method. These are development results, not independent final validation.\n',encoding='utf-8')
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc(),finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write(job/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path);a=p.parse_args()
    supervise(a.job) if a.job else prepare()
