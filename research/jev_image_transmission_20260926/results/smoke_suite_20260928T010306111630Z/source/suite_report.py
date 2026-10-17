"""Paired image/scene-cluster analysis; never promote smoke data to formal evidence."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from suite_common import ACTIONS, read, write, resources, feedback_ok, stable_seed

def collapse_seeds(rows):
    """Average training-seed outcomes within each case; seeds are not extra images."""
    groups=defaultdict(list)
    for r in rows:
        method=r['method'].split('_seed')[0]
        groups[(r['case_id'],method)].append(r)
    result=[]
    for (_,method),items in groups.items():
        row=dict(items[0],method=method)
        for key in ['psnr_db','mse','lpips_normalized','total_symbols','normalized_rf_energy','elapsed_ms','decision_ms','outage','deadline_miss','exact','near','regret_db']:
            row[key]=float(np.mean([x[key] for x in items]))
        result.append(row)
    return result

def paired_ci(rows, method, comparator, key, samples):
    left={r['case_id']:r for r in rows if r['method']==method}
    right={r['case_id']:r for r in rows if r['method']==comparator}
    clusters=defaultdict(list)
    for case in left.keys()&right.keys(): clusters[left[case]['scene_id']].append(left[case][key]-right[case][key])
    if not clusters: return None
    values=np.array([np.mean(clusters[s]) for s in sorted(clusters)])
    rng=np.random.default_rng(stable_seed('bootstrap',method,comparator,key))
    means=[]
    for _ in range(samples): means.append(float(rng.choice(values,len(values),replace=True).mean()))
    return {'paired_scene_mean_difference':float(values.mean()),'ci95_percentile':np.quantile(means,[.025,.975]).tolist(),'clusters':len(values)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    cfg=read(args.run/'config.json')
    tracks={'fixed_time':[],'measured_delay_replay':[]}
    gain_errors=defaultdict(list)
    failures=0
    tokens={'input_tokens':0,'output_tokens':0}
    for path in sorted((args.run/'decisions').glob('*.json')):
        official=read(path)['official_jev']
        failures+=int(official.get('fallback',False))
        usage=official.get('response',{}).get('usage',{})
        for k in tokens: tokens[k]+=usage.get(k,0)
    for path in sorted((args.run/'selections').glob('*.json')):
        state=read(args.run/'states'/path.name)
        label=read(args.run/'labels'/path.name)
        selections=read(path)['methods']
        replay=read(args.run/'replay'/path.name)['methods']
        oracle=min(ACTIONS,key=lambda a:label['actions'][a]['mse'])
        common={k:state[k] for k in ['case_id','scene_id','image_id','split','channel','snr_db','age_group','replica']}
        single_cost=6144+cfg['header_symbols']+cfg['pilot_symbols']*label['single_pilots']
        for track,rows in tracks.items():
            rows.append({**common,'method':'single_C192',**label['single'],'total_symbols':single_cost,'normalized_rf_energy':single_cost,
                'payload_symbols':6144,'feedback_symbols':0,'header_symbols':cfg['header_symbols'],'pilot_symbols':cfg['pilot_symbols']*label['single_pilots'],'unused_symbols':cfg['budget_symbols']-single_cost,
                'elapsed_ms':3+(single_cost-6144)/2048+label['single_codec_ms'],'decision_ms':0,'deadline_miss':0,
                'outage':float(label['single']['psnr_db']<cfg['outage_psnr_db']),'exact':0,'near':0,'regret_db':0})
            for method,selection in selections.items():
                selected=selection['action']
                if track=='fixed_time':
                    executed=selected if feedback_ok(state['case_id'],cfg['feedback_erasure_probability']) else 'stop'
                    costs=resources(executed,label['base_pilots'],label['enhancement_pilots'],cfg,selected!='stop')
                    measured={**label['actions'][executed],**costs,'elapsed_ms':2+cfg['fixed_gap_steps']*.03125+(1 if executed!='stop' else 0),'decision_ms':selection['decision_ms'],'deadline_miss':0}
                else: measured=replay[method]
                rows.append({**common,'method':method,**measured,'outage':float(measured['psnr_db']<cfg['outage_psnr_db']),
                    'exact':float(selected==oracle),'near':float(label['actions'][selected]['mse']<=cfg['near_optimal_mse_ratio']*label['actions'][oracle]['mse']),
                    'regret_db':label['actions'][oracle]['psnr_db']-label['actions'][selected]['psnr_db']})
                if track=='fixed_time' and 'predictions' in selection and not method.startswith('uncertainty_only'):
                    target=np.array([(label['actions']['stop']['mse']-label['actions'][a]['mse'])/label['actions']['stop']['mse'] for a in ACTIONS])
                    gain_errors[method].extend(np.abs(np.array(selection['predictions'])-target).tolist())
            if track=='fixed_time':
                costs=resources(oracle,label['base_pilots'],label['enhancement_pilots'],cfg,oracle!='stop')
                rows.append({**common,'method':'hindsight_oracle',**label['actions'][oracle],**costs,'elapsed_ms':0,'decision_ms':0,'deadline_miss':0,
                    'outage':float(label['actions'][oracle]['psnr_db']<cfg['outage_psnr_db']),'exact':1,'near':1,'regret_db':0})
    report={'scope':read(args.run/'manifest.json')['mode'],'api_failures_including_fallbacks':failures,'api_usage':tokens,'tracks':{},
        'gain_prediction_mae':{k:float(np.mean(v)) for k,v in gain_errors.items()},
        'caveats':['Fixed-time accuracy is an optimistic component diagnostic, not an end-to-end latency claim.',
        'Latency replay waits for and charges one fresh periodic pilot; no free intermediate pilot observations.',
        'Complete base codec wall time is conservatively placed after base reception; replay is not a measured hardware link.',
        'RF energy assumes unit energy per symbol; compute energy and API infrastructure energy are unmeasured.',
        'Three replicas share released propagation traces with independent noise; they are not independent propagation realizations.',
        'UDIS aggregate baseline results were previously observed; DIV2K is the new external test.',
        '95% intervals are paired scene-cluster bootstrap intervals, descriptive and not multiplicity-adjusted; primary contrast official vs description.',
        'Exact/near/regret scores compare intended decisions against the fixed-time hindsight oracle, even in replay.']}
    metrics=['psnr_db','mse','lpips_normalized','total_symbols','normalized_rf_energy','elapsed_ms','decision_ms','outage','deadline_miss','exact','near','regret_db']
    for track,raw in tracks.items():
        with (args.run/(track+'.csv')).open('w',newline='',encoding='utf-8') as file:
            keys=['case_id','scene_id','split','channel','snr_db','age_group','replica','method']+metrics+['payload_symbols','feedback_symbols','header_symbols','pilot_symbols','unused_symbols']
            writer=csv.DictWriter(file,keys,extrasaction='ignore')
            writer.writeheader(); writer.writerows(raw)
        rows=collapse_seeds(raw)
        panels={}
        # No pooled cross-dataset headline; channel/age/SNR strata are also exposed.
        panel_keys=sorted({(r['split'],r['channel'],'all','all') for r in rows}|{(r['split'],r['channel'],str(r['snr_db']),r['age_group']) for r in rows})
        for split,channel,snr,age in panel_keys:
            selected=[r for r in rows if r['split']==split and r['channel']==channel and (snr=='all' or str(r['snr_db'])==snr) and (age=='all' or r['age_group']==age)]
            methods={}
            for method in sorted({r['method'] for r in selected}):
                a=[r for r in selected if r['method']==method]
                methods[method]={k:float(np.mean([r[k] for r in a])) for k in metrics}
                methods[method].update(psnr_p05_db=float(np.quantile([r['psnr_db'] for r in a],.05)),states=len(a),scenes=len({r['scene_id'] for r in a}))
                original=[r for r in raw if r['method'].split('_seed')[0]==method and r['split']==split and r['channel']==channel and (snr=='all' or str(r['snr_db'])==snr) and (age=='all' or r['age_group']==age)]
                seed_groups=defaultdict(list)
                for r in original: seed_groups[r['method']].append(r)
                seeds={name:{'mean_psnr_db':float(np.mean([r['psnr_db'] for r in values])),
                    'psnr_p05_db':float(np.quantile([r['psnr_db'] for r in values],.05)),
                    'outage':float(np.mean([r['outage'] for r in values]))} for name,values in seed_groups.items()}
                methods[method]['training_seed_summaries']=seeds
                methods[method]['psnr_p05_db']=float(np.mean([v['psnr_p05_db'] for v in seeds.values()]))
                methods[method]['seed_mean_psnr_std']=float(np.std([v['mean_psnr_db'] for v in seeds.values()],ddof=1)) if len(seeds)>1 else 0.0
            comparisons={}
            if snr=='all':
                for method,reference in [('official','description'),('official','numeric'),('visual','qwen'),('official','official_no_age'),('visual','visual_no_age'),('official','official_no_stop'),('official','single_C192'),('visual','single_C192')]:
                    comparisons[method+'_minus_'+reference]={key:paired_ci(selected,method,reference,key,cfg['bootstrap_samples']) for key in ['psnr_db','mse','outage']}
            panels['/'.join([split,channel,snr,age])]={'methods':methods,'paired_comparisons':comparisons}
        report['tracks'][track]=panels
    write(args.run/'analysis.json',report)
    lines=['# Experiment report','',f"Mode: **{report['scope']}**. Smoke runs are not scientific evidence.",'',
           f"Official API failures retained: {failures}. Input/output tokens: {tokens}.",'',
           'Main table: means over states, with each trained-seed family averaged within state. See analysis.json for scene-cluster paired intervals and stratification.','']
    for track,panels in report['tracks'].items():
        for panel,content in panels.items():
            if not panel.endswith('/all/all'): continue
            lines += [f'## {track}: {panel}','','| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |','|---|---:|---:|---:|---:|---:|---:|']
            for method,v in content['methods'].items():
                lines.append(f"| {method} | {v['psnr_db']:.3f} | {v['psnr_p05_db']:.3f} | {v['lpips_normalized']:.4f} | {v['outage']:.3f} | {v['decision_ms']:.2f} | {v['total_symbols']:.1f} |")
            lines.append('')
    lines += ['## Interpretation limits','']+['- '+x for x in report['caveats']]
    (args.run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'REPORT {args.run / "REPORT.md"}',flush=True)

if __name__=='__main__': main()
