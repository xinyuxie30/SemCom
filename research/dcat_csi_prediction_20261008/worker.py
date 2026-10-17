"""B0.1 controlled calibration / MMSE ablation using frozen DCAT weights."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch


def denominator(h, variance, noise):
    """Equivalent divisor for h*Y/(|h|^2+variance+noise), including h=0."""
    power = h.abs().square()
    total = power + variance + noise
    safe = torch.where(power > 0, h.conj(), torch.ones_like(h))
    return torch.where(power > 0, total/safe, torch.full_like(h, 1e30))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--src',type=Path,required=True)
    p.add_argument('--b0',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=['without','with'],default='without')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    sys.path.insert(0,str(a.src))
    import run_e2e as runner
    from csi_predictors import samples,channel_info,simple_predictor
    cfg=json.loads(a.protocol.read_text(encoding='utf-8'))
    sys.path.insert(0,str(Path(cfg['baseline_root'])/'reproduction'))
    import common
    ctx={'policy':None,'rows':[]}
    curves={}
    original_build=common.build_model
    original_select=runner.select_images
    original_trace=runner.load_trace
    def tagged_trace(path):
        trace=original_trace(path)
        trace['_case_name']=Path(path).stem
        return trace
    runner.load_trace=tagged_trace

    def fixed_images(e):
        e=dict(e); e['seed']=20261004
        return original_select(e)
    runner.select_images=fixed_images

    def info(trace,start,policy,empirical,acf):
        predictor=policy.split('_')[0]
        model={'hold':simple_predictor('hold'),'wiener':empirical,'acf':acf}[predictor]
        key=(trace['_case_name'],predictor)
        if key not in curves:
            # All calibration labels stay in the old validation time block.
            _,x,y=samples(trace,cfg['pilot_blocks']['validation'])
            error=np.mean(abs(model.predict(x)-y)**2,axis=0)
            power=np.mean(abs(y)**2,axis=0)
            curves[key]=(error,error/np.maximum(power,1e-12))
        error,norm=curves[key]
        times=start+np.arange(64)
        txh=times-(start//128)*128
        rxh=times%128
        if ctx['policy']!=policy:
            ctx['policy']=policy;ctx['rows']=[]
        ctx['rows'].append({'tx':float(norm[txh].mean()),'rx':float(norm[rxh].mean()),'var':error[rxh]})
        if predictor=='hold':
            return channel_info(trace,start)
        return channel_info(trace,start,tx_model=model,rx_model=model)
    runner.policy_info=info

    def build(*args,**kwargs):
        model=original_build(*args,**kwargs)
        def calibrated(which):
            def hook(module,args,output):
                if '_cal' not in ctx['policy']:
                    return output
                assert len(ctx['rows'])==len(output)
                return torch.tensor([r[which] for r in ctx['rows']],device=output.device,dtype=output.dtype)
            return hook
        model.pred_nmse_tx.register_forward_hook(calibrated('tx'))
        model.pred_nmse_rx.register_forward_hook(calibrated('rx'))
        def equalize(module,args,kwargs):
            if '_mmse' not in ctx['policy']:
                return args,kwargs
            kwargs=dict(kwargs)
            data=dict(kwargs['CHANNEL_INFO'])
            h=data['CSI_ZF']
            variance=torch.tensor(np.stack([r['var'] for r in ctx['rows']]),device=h.device,dtype=h.real.dtype)
            # Physical CSI_GT and all decoder-side CSI remain untouched.
            noise=10**(-float(kwargs['chan_param'])/10)
            data['CSI_ZF']=denominator(h,variance,noise)
            kwargs['CHANNEL_INFO']=data
            return args,kwargs
        model.ours_channel.register_forward_pre_hook(equalize,with_kwargs=True)
        return model
    common.build_model=build
    sys.argv=['run_e2e.py','--b0',str(a.b0),'--protocol',str(a.protocol),
              '--output',str(a.output),'--checkpoint-family',a.family]
    if a.smoke: sys.argv.append('--smoke')
    runner.main()
    # Record the exact validation-only variance curves used by the worker.
    np.savez_compressed(a.output/'calibration_curves.npz',
                        **{f'curve_{i}_{j}':v for i,pair in enumerate(curves.values()) for j,v in enumerate(pair)})


if __name__=='__main__':
    main()
