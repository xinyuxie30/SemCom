"""Prepare or supervise sequential background experiments; no LLM polling."""
import argparse
import csv
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'dcat_csi_prediction_20261004'
PYTHON=Path('D:/PyChram_project/DCAT-SoM/.venv/Scripts/python.exe')


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)


def process_alive(pid):
    """Conservative Windows liveness check; never signal a process."""
    if not pid:
        return False
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000,False,int(pid))  # SYNCHRONIZE
    if not handle:
        return ctypes.get_last_error()!=87  # Access denied is not evidence of exit.
    try:
        return kernel.WaitForSingleObject(handle,0)!=0
    finally:
        kernel.CloseHandle(handle)


def prepare(recover_from=None):
    previous=None
    if recover_from:
        recover_from=recover_from.resolve()
        previous=json.loads((recover_from/'status.json').read_text())
        if previous['status'] not in ('RUNNING','FAILED','INTERRUPTED'):
            raise RuntimeError('Only interrupted/failed jobs can be recovered')
        # Restricted Windows sessions can hide live host PIDs. Absence alone
        # cannot authorize restarting a job that has recently written activity.
        activity=[recover_from/'status.json',*recover_from.glob('*.log')]
        if time.time()-max(p.stat().st_mtime for p in activity)<300:
            raise RuntimeError('Recent job activity; verify the host process before recovery')
        pids=[previous.get('supervisor_pid')]+[r.get('child_pid') for r in previous['runs']]
        if any(process_alive(pid) for pid in pids):
            raise RuntimeError('An old process may still be alive; refusing duplicate work')
        # This recovery path is deliberately limited to jobs with no inference output.
        if any((recover_from/r['name']).exists() for r in previous['runs']):
            raise RuntimeError('Prior inference outputs exist; preserve them and plan an explicit resume')
        for name,expected in previous['source_hashes'].items():
            if hashlib.sha256((recover_from/name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'Frozen source changed: {name}')
    job=HERE/'results'/datetime.now(timezone.utc).strftime('b01_%Y%m%dT%H%M%SZ')
    (job/'src').mkdir(parents=True)
    b0=Path(previous['b0']) if previous else Path(json.loads((OLD/'latest_b0.json').read_text())['run'])
    cfg=json.loads(((recover_from if previous else b0)/'protocol.json').read_text())
    if not previous:
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
        shutil.copy2((recover_from/'src' if previous else OLD)/n,job/'src'/n)
    shutil.copy2((recover_from if previous else HERE)/'worker.py',job/'worker.py')
    shutil.copy2(HERE/'launch.py',job/'launch.py')
    write(job/'protocol.json',cfg)
    hashes={str(p.relative_to(job)):hashlib.sha256(p.read_bytes()).hexdigest() for p in job.rglob('*.py')}
    status={'status':'PREPARED','job':str(job),'b0':str(b0),'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_hashes':hashes,'expected_transmissions':69120,'runs':[]}
    if previous:
        status['recovery_from']=str(recover_from)
        status['recovery_reason']='Prior supervisor absent, no inference output directory, no traceback. Root cause unconfirmed; interpreter warning reproduced on a successful invocation.'
        status['experiment_sources_unchanged']=True
        write(job/'prior_status.json',previous)
        write(recover_from/'status_before_recovery.json',previous)
        previous.update(status='INTERRUPTED',interruption_observed_utc=status['created_utc'],recovery_job=str(job))
        write(recover_from/'status.json',previous)
    write(job/'status.json',status)
    write(HERE/'latest_job.json',{'job':str(job)})
    print(job)


def supervise(job):
    state=json.loads((job/'status.json').read_text())
    if state['status']!='PREPARED': raise RuntimeError('Refusing duplicate launch')
    state.update(status='RUNNING',supervisor_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
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
                record={'name':name,'status':'RUNNING','command':command,'started_utc':datetime.now(timezone.utc).isoformat()}
                state['runs'].append(record);write(job/'status.json',state)
                with (job/(name+'.log')).open('w',encoding='utf-8') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                           env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},creationflags=subprocess.CREATE_NO_WINDOW)
                    record['child_pid']=child.pid;write(job/'status.json',state)
                    deadline=time.monotonic()+1800
                    while True:
                        remaining=deadline-time.monotonic()
                        if remaining<=0:
                            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT,check=False)
                            raise TimeoutError(f'{name}: 30-minute hard timeout')
                        try:
                            code=child.wait(timeout=min(30,remaining))
                            break
                        except subprocess.TimeoutExpired:
                            state['heartbeat_utc']=datetime.now(timezone.utc).isoformat()
                            write(job/'status.json',state)
                record['exit_code']=code
                if code: raise RuntimeError(f'{name} failed; see log')
                manifest=json.loads((job/name/'manifest.json').read_text())
                expected=cfg['e2e']['images']*len(cfg['e2e']['channels'])*len(cfg['e2e']['snr_db'])*len(cfg['e2e']['start_phases'])*len(cfg['e2e']['policies'])
                if manifest['status']!='COMPLETED' or manifest['completed_rows']!=expected or not (job/name/'calibration_curves.npz').is_file():
                    raise RuntimeError(f'{name}: incomplete artifacts despite exit code 0')
                record.update(status='COMPLETED',completed_rows=expected,finished_utc=datetime.now(timezone.utc).isoformat())
                write(job/'status.json',state)
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
        for record in state['runs']:
            if record['status']=='RUNNING': record['status']='FAILED'
        state.update(status='FAILED',error=traceback.format_exc(),finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write(job/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group()
    group.add_argument('--job',type=Path)
    group.add_argument('--recover-from',type=Path,help='Explicitly restart a dead job with no inference outputs, preserving its frozen protocol and experiment code')
    a=p.parse_args()
    supervise(a.job) if a.job else prepare(a.recover_from)
