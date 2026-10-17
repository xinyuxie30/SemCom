"""Train small gain predictors on train only; choose epochs on validation only."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from suite_common import ACTIONS, REGIONS, FEATURE_KEYS, read, write, stable_seed

VARIANTS=['numeric','numeric_no_age','description','official','official_no_age','visual','visual_no_age','qwen','uncertainty_only']
WIDTH=61

def features(state, decision, variant):
    # Padding keeps all gain predictors at the same input/hidden parameter count.
    v=np.zeros(WIDTH,dtype=np.float32)
    v[:24]=[state['region_features'][r][k] for r in REGIONS for k in FEATURE_KEYS]
    if variant!='uncertainty_only':
        v[24:26]=[state['snr_db'],state['last_known_csi_magnitude']]
        if not variant.endswith('_no_age'): v[26]=state['csi_age_at_last_base_symbol_ms']
    if variant=='description' or variant.startswith('official'):
        v[27:43]=[decision['descriptions'][r]['distribution'][g] for r in REGIONS for g in ['low','noticeable','severe','uncertain']]
    if variant.startswith('official'):
        official=decision['official_jev']
        v[43:49]=[official.get('probabilities',{}).get(a,1/6) for a in ACTIONS]
        answer=official.get('response',{}).get('answers',{})
        v[49:54]=[answer.get('benefit_'+a,{}).get('noul',.5) for a in ACTIONS[:-1]]
        v[54]=float(official.get('fallback',False))
    if variant.startswith('visual') or variant=='qwen':
        key='visual_jev' if variant.startswith('visual') else 'qwen_visual'
        v[55:61]=[decision[key]['probabilities'][a] for a in ACTIONS]
    assert np.isfinite(v).all()
    return v

def frontend_ms(state,decision,variant):
    ms=state['feature_ms']
    if variant in ('description',) or variant.startswith('official'):
        ms+=decision['input_preparation_ms']+decision['description_ms']
    if variant.startswith('official'): ms+=decision['official_jev']['elapsed_ms']
    if variant.startswith('visual'): ms+=decision['input_preparation_ms']+decision['visual_jev']['elapsed_ms']
    if variant=='qwen': ms+=decision['input_preparation_ms']+decision['qwen_visual']['elapsed_ms']
    return ms

def network(width,outputs):
    return nn.Sequential(nn.Linear(WIDTH,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,outputs))

def fit(run,config,records,variant,seed):
    target=run/'learned'/f'{variant}_{seed}.pt'
    history_path=target.with_suffix('.json')
    output_dim=4 if variant=='uncertainty_only' else 6
    if target.exists() and history_path.exists():
        bundle=torch.load(target,map_location='cpu',weights_only=True)
        model=network(config['hidden_width'],output_dim)
        model.load_state_dict(bundle['state_dict'])
        return model.eval(),bundle['mean'],bundle['std']
    torch.manual_seed(seed)
    train=[r for r in records if r[0]['split']=='train']
    val=[r for r in records if r[0]['split']=='validation']
    def xy(rows):
        x=torch.from_numpy(np.stack([features(s,d,variant) for s,d,l in rows]))
        if variant=='uncertainty_only': y=torch.tensor([[np.log(max(l['base_region_mse'][a],1e-9)) for a in REGIONS] for s,d,l in rows],dtype=torch.float32)
        else:
            y=torch.tensor([[(l['actions']['stop']['mse']-l['actions'][a]['mse'])/max(l['actions']['stop']['mse'],1e-9) for a in ACTIONS] for s,d,l in rows],dtype=torch.float32).clamp(-4,1)
        return x,y
    tx,ty=xy(train)
    vx,vy=xy(val)
    mean,std=tx.mean(0),tx.std(0,unbiased=False).clamp_min(1e-5)
    tx,vx=(tx-mean)/std,(vx-mean)/std
    model=network(config['hidden_width'],output_dim)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    best,best_state,patience=float('inf'),None,0
    history=[]
    for epoch in range(config['train_epochs']):
        model.train()
        optimizer.zero_grad()
        loss=nn.functional.mse_loss(model(tx),ty)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode(): validation=float(nn.functional.mse_loss(model(vx),vy))
        history.append({'epoch':epoch+1,'train_mse':float(loss.detach()),'validation_mse':validation})
        if validation<best:
            best,best_state,patience=validation,{k:v.detach().clone() for k,v in model.state_dict().items()},0
        else: patience+=1
        if patience>=config['early_stopping_patience']: break
    model.load_state_dict(best_state)
    target.parent.mkdir(exist_ok=True)
    tmp=target.with_suffix('.tmp')
    torch.save({'state_dict':best_state,'mean':mean,'std':std},tmp)
    tmp.replace(target)
    write(history_path,{'variant':variant,'seed':seed,'train_rows':len(train),'validation_rows':len(val),'history':history,'best_validation_mse':best,
        'objective':'log regional MSE' if variant=='uncertainty_only' else 'Relative MSE reduction clipped [-4,1]; stop fixed to zero at action selection',
        'test_labels_used':False})
    print(f'trained {variant} seed={seed} epochs={len(history)}',flush=True)
    return model.eval(),mean,std

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    config=read(args.run/'config.json')
    torch.set_num_threads(4)
    records=[]
    # Test label files are intentionally not opened during fitting/selection.
    for path in sorted((args.run/'states').glob('*.json')):
        s=read(path)
        d=read(args.run/'decisions'/path.name)
        l=read(args.run/'labels'/path.name) if s['split'] in ('train','validation') else None
        records.append((s,d,l))
    models={(v,seed):fit(args.run,config,records,v,seed) for v in VARIANTS for seed in config['decision_seeds']}
    with torch.inference_mode():
        for s,d,_ in records:
            if s['split'] in ('train','validation'): continue
            path=args.run/'selections'/f"{s['case_id']}.json"
            if path.exists(): continue
            methods={}
            for a in ('global','stop','upper_right'):
                methods['fixed_'+a]={'action':a,'decision_ms':0.0}
            methods['random_roi']={'action':REGIONS[stable_seed(s['case_id'],'random_roi')%4],'decision_ms':0.0}
            methods['gradient_importance']={'action':max(REGIONS,key=lambda r:s['region_features'][r]['gradient']),'decision_ms':s['feature_ms']}
            severity={r:sum(d['descriptions'][r]['distribution'][g]*w for g,w in zip(['low','noticeable','severe','uncertain'],[0,.5,1,.5])) for r in REGIONS}
            methods['severity_rule']={'action':max(severity,key=severity.get),'decision_ms':d['input_preparation_ms']+d['description_ms']}
            for v in ('official_jev','visual_jev','qwen_visual','qwen_description_only'):
                extra=d['description_ms'] if v in ('official_jev','qwen_description_only') else 0
                methods[v+'_raw']={'action':d[v]['action'],'decision_ms':d['input_preparation_ms']+extra+d[v]['elapsed_ms']}
            for (v,seed),(model,mean,std) in models.items():
                begun=time.perf_counter()
                x=torch.from_numpy(features(s,d,v))
                values=model((x-mean)/std).numpy()
                if v!='uncertainty_only': values[-1]=0
                action=ACTIONS[int(np.argmax(values))]
                elapsed=(time.perf_counter()-begun)*1000+frontend_ms(s,d,v)
                name=f'{v}_seed{seed}'
                methods[name]={'action':action,'decision_ms':elapsed,'predictions':values.tolist()}
                if v in ('official','visual'):
                    methods[f'{v}_no_stop_seed{seed}']={'action':ACTIONS[int(np.argmax(values[:-1]))],'decision_ms':elapsed}
            write(path,{'case_id':s['case_id'],'methods':methods})

if __name__=='__main__': main()
