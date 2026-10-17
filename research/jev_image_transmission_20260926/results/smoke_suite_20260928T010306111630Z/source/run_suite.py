"""Single entry point. Default is read-only preflight; formal runs require an explicit mode."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from suite_common import ROOT, BASELINE, CODEC_PYTHON, VISUAL_PYTHON, read, write, digest, cases

FILES=['suite_common.py','suite_codec.py','suite_decisions.py','suite_learning.py','suite_report.py','run_suite.py',
       'pilot_counterfactual.py','pilot_decisions.py','probe_typesafe.py','suite_config.json','suite_data.json','smoke_config.json','smoke_data.json']

def preflight(check_weights=False):
    for python in (CODEC_PYTHON,VISUAL_PYTHON):
        if not python.is_file(): raise FileNotFoundError(python)
    config,data=read(ROOT/'suite_config.json'),read(ROOT/'suite_data.json')
    checks=[]
    for item in data['images']:
        if digest(item['path'])!=item['sha256']: raise ValueError('Image hash mismatch: '+item['image_id'])
    groups={}
    for item in data['images']:
        prior=groups.setdefault(item['scene_id'],item['split'])
        if prior!=item['split']: raise ValueError('Scene leakage')
    expected=list(cases(config,data['images']))
    if len(expected)!=config['api_max_requests']: raise ValueError('API cap and experiment matrix disagree')
    hashes={name:digest(ROOT/name) for name in FILES}
    external=read(BASELINE/'reproduction/preflight_report.json')
    source=BASELINE/'SoM-DCAT-main'
    for p,h in external['source_hashes'].items():
        if digest(source/p)!=h: raise ValueError('External DCAT source changed: '+p)
    assets=read(ROOT/'visual_assets_manifest.json')
    for item in assets['files']:
        if digest(ROOT/'vendor/Visual-Jev'/item['path'])!=item['sha256']: raise ValueError('Visual Jev source changed')
    asset_paths=[source/'Checkpoints'/f'DCAT_without_CP_C{c}.model' for c in (64,128,192)]
    asset_paths += [source/'Channel_inputs'/f'{c}.mat' for c in set(config['channels']+config['external_channels'])]
    asset_paths += sorted((ROOT/'models').rglob('*.safetensors'))+sorted((ROOT/'models').rglob('*.json'))
    weights={str(p):digest(p) for p in asset_paths}
    for name in ('common.py','evaluate_batched.py'): weights[str(BASELINE/'reproduction'/name)]=digest(BASELINE/'reproduction'/name)
    versions={}
    probe="import json, torch; from importlib.metadata import version; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0),'numpy':version('numpy'),'pillow':version('pillow')}))"
    for name,python in [('codec',CODEC_PYTHON),('visual',VISUAL_PYTHON)]:
        result=subprocess.run([str(python),'-c',probe],capture_output=True,text=True,check=True,timeout=60)
        versions[name]=json.loads(result.stdout.strip())
        if not versions[name]['cuda']: raise RuntimeError('CUDA unavailable')
    report={'status':'READY','formal_started':False,'matrix_states':len(expected),'image_counts':dict(Counter(x['split'] for x in data['images'])),
        'api_requests_max':config['api_max_requests'],'api_key_present':bool(os.environ.get('TYPESAFE_API_KEY','').strip()),
        'source_hashes':hashes,'asset_hashes':weights,'versions':versions,'free_disk_gib':shutil.disk_usage(ROOT).free/2**30,
        'external_references':{'DCAT':str(BASELINE),'Visual_Jev':assets['repository'],'Visual_Jev_commit':assets['commit']},
        'limits':'Readiness does not imply positive scientific results; external API availability can change.'}
    if not report['api_key_present']: report['status']='MISSING_API_KEY'
    if report['free_disk_gib']<20: raise RuntimeError('Less than 20 GiB free')
    write(ROOT/'preflight.json',report)
    return report

def run_stage(run,manifest,name,argv,timeout):
    if name in manifest['completed_stages']: return
    manifest.update(status='RUNNING',current_stage=name)
    write(run/'manifest.json',manifest)
    log=run/(name+'.log')
    print('STAGE '+name,flush=True)
    started=time.monotonic()
    with log.open('a',encoding='utf-8') as stream:
        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
        try:
            while True:
                try:
                    code=process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic()-started>timeout:
                        process.terminate()
                        process.wait(timeout=20)
                        raise TimeoutError('Hard stage timeout: '+name)
                    print(f'RUNNING {name} seconds={time.monotonic()-started:.0f} log={log}',flush=True)
        except BaseException:
            if process.poll() is None: process.terminate()
            raise
    if code!=0: raise RuntimeError(f'Stage {name} exited {code}; see {log}')
    manifest['completed_stages'].append(name)
    manifest.setdefault('stage_seconds',{})[name]=time.monotonic()-started
    write(run/'manifest.json',manifest)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['preflight','smoke','formal'],default='preflight')
    parser.add_argument('--live-api',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    readiness=preflight()
    print(json.dumps({k:readiness[k] for k in ('status','matrix_states','image_counts','api_requests_max','api_key_present','free_disk_gib')}),flush=True)
    if args.mode=='preflight': return
    if not args.live_api: raise SystemExit('Explicit --live-api required for a live run; preflight sent no requests.')
    if readiness['status']!='READY': raise SystemExit(readiness['status'])
    prefix='smoke' if args.mode=='smoke' else 'suite'
    config_path=ROOT/f'{prefix}_config.json'
    data_path=ROOT/f'{prefix}_data.json'
    if args.resume:
        run=args.resume.resolve()
        if not run.is_relative_to(ROOT/'results'): raise ValueError('Resume outside results directory')
        manifest=read(run/'manifest.json')
        assert manifest['mode']==args.mode
        assert manifest['source_hashes']==readiness['source_hashes'], 'Code changed: use a new run'
        assert manifest['asset_hashes']==readiness['asset_hashes'], 'Assets changed'
        assert digest(run/'config.json')==digest(config_path) and digest(run/'data.json')==digest(data_path)
    else:
        run=ROOT/'results'/(args.mode+'_suite_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        run.mkdir(parents=True)
        shutil.copyfile(config_path,run/'config.json')
        shutil.copyfile(data_path,run/'data.json')
        (run/'source').mkdir()
        for name in FILES: shutil.copyfile(ROOT/name,run/'source'/name)
        manifest={'mode':args.mode,'status':'PREPARED','created_utc':datetime.now(timezone.utc).isoformat(),
            'source_hashes':readiness['source_hashes'],'asset_hashes':readiness['asset_hashes'],'versions':readiness['versions'],
            'completed_stages':[],'preflight':readiness['status']}
        write(run/'manifest.json',manifest)
    print('RUN_OUTPUT '+str(run),flush=True)
    cfg=read(run/'config.json')
    try:
        for name,python,script,extra in [
            ('generate',CODEC_PYTHON,'suite_codec.py',['--stage','generate']),
            ('decide',VISUAL_PYTHON,'suite_decisions.py',['--live-api']),
            ('learn',CODEC_PYTHON,'suite_learning.py',[]),
            ('replay',CODEC_PYTHON,'suite_codec.py',['--stage','replay']),
            ('report',CODEC_PYTHON,'suite_report.py',[])]:
            run_stage(run,manifest,name,[str(python),str(ROOT/script),'--run',str(run),*extra],cfg['stage_timeout_seconds'])
        expected=list(cases(cfg,read(run/'data.json')['images']))
        test_count=sum(c['image']['split'] in ('test','external_test') for c in expected)
        counts={folder:len(list((run/folder).glob('*.json'))) for folder in ('states','labels','decisions','selections','replay')}
        assert all(counts[k]==len(expected) for k in ('states','labels','decisions'))
        assert counts['selections']==counts['replay']==test_count
        manifest.update(status='COMPLETED',counts=counts,completed_utc=datetime.now(timezone.utc).isoformat())
        write(run/'manifest.json',manifest)
        print('COMPLETED '+str(run/'REPORT.md'),flush=True)
    except BaseException as error:
        manifest.update(status='FAILED',error_type=type(error).__name__)
        write(run/'manifest.json',manifest)
        raise

if __name__=='__main__': main()
