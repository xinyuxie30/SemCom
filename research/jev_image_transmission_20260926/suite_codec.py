"""Frozen, explicitly referenced DCAT codec and causal channel replay."""
import argparse
import contextlib
import math
from pathlib import Path
import sys
import time
import types
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from suite_common import ROOT, BASELINE, ACTIONS, STEP_MS, cases, read, write, stable_seed, feedback_ok, resources
from pilot_counterfactual import REGIONS, blind_region_features, metrics
sys.path.insert(0, str(BASELINE/'reproduction'))
from common import build_model, load_channel, channel_slice
from evaluate_batched import adapt

class Codec:
    def __init__(self, log):
        self.log, self.channels, self.noise_context = log, {}, {}
        self.active_case, self.blocks = None, {}
        self.models = {}
        torch.set_num_threads(4)
        from utils import seed_torch
        seed_torch(20260927)
        with contextlib.redirect_stdout(log):
            for c in (64,128,192):
                model = adapt(build_model(c))
                context = self.noise_context
                def noise(_self, inputs, std, name=None):
                    return inputs + context['noise'].to(inputs.device).unsqueeze(0)*std
                model.ours_channel.gaussian_noise_layer = types.MethodType(noise, model.ours_channel)
                self.models[c] = model

    def channel(self, name):
        if name not in self.channels: self.channels[name] = load_channel(name)
        return self.channels[name]

    def noise(self, case_id, start, count):
        if self.active_case != case_id:
            self.active_case, self.blocks = case_id, {}
        first, last = start//256, (start+count-1)//256
        for block in range(first, last+1):
            if block not in self.blocks:
                rng = np.random.default_rng(stable_seed('noise_v1',case_id,block))
                self.blocks[block] = (rng.standard_normal((256,64), dtype=np.float32)+1j*rng.standard_normal((256,64), dtype=np.float32)).astype(np.complex64)
        bank = np.concatenate([self.blocks[b] for b in range(first,last+1)])
        return torch.from_numpy(bank[start%256:start%256+count])

    def infer(self, c, x, start, name, case_id, snr):
        if not 0 <= start <= 128000-c//2: raise ValueError('Trace bounds')
        self.noise_context['noise'] = self.noise(case_id, start, c//2)
        one = channel_slice(self.channel(name), start, c//2, False)
        info = {k:v*x.shape[0] for k,v in one.items()}
        torch.cuda.synchronize()
        begun = time.perf_counter()
        with contextlib.redirect_stdout(self.log):
            result, _, _ = self.models[c](x, info, snr)
        torch.cuda.synchronize()
        ms = (time.perf_counter()-begun)*1000
        if not bool(torch.isfinite(result).all()): raise ValueError('Nonfinite reconstruction')
        return result.clamp(0,1), one, ms

    def pilots(self, name, start, count):
        age = self.channel(name)['AgingTime'][0,start:start+count]
        return 1+int((np.diff(age)<0).sum())

    def refresh_after(self, name, earliest):
        age = self.channel(name)['AgingTime'][0]
        end = min(earliest+130, len(age)-32)
        # Wait for a newly observed periodic pilot; do not consume free CSI during API wait.
        indices = np.flatnonzero(np.diff(age[max(0,earliest-1):end+1])<0)+max(0,earliest-1)+1
        valid = indices[indices>=earliest]
        if not len(valid): raise ValueError('No refresh pilot in trace window')
        return int(valid[0])

    def score(self, x, y):
        return metrics(x,y,self.models[64].lpips_loss)

def load_image(path):
    with Image.open(path) as image:
        return torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255).permute(2,0,1).unsqueeze(0).cuda()

def input_for(x, action):
    if action == 'global': return x
    y0,y1,x0,x1 = REGIONS[action]
    return F.interpolate(x[:,:,y0:y1,x0:x1], size=(128,128), mode='bilinear', align_corners=False)

def fuse(base, enhancement, action, alpha):
    if action == 'global': return (1-alpha)*base+alpha*enhancement
    y0,y1,x0,x1 = REGIONS[action]
    out = base.clone()
    patch = F.interpolate(enhancement, size=(64,64), mode='bilinear', align_corners=False)
    out[:,:,y0:y1,x0:x1] = (1-alpha)*base[:,:,y0:y1,x0:x1]+alpha*patch
    return out

def generate(run, config, images):
    completed, started = 0, time.monotonic()
    (run/'receiver_arrays').mkdir(exist_ok=True)
    with (run/'codec.log').open('a',encoding='utf-8') as log, torch.inference_mode():
        engine = Codec(log)
        for case in cases(config,images):
            case_id, item = case['case_id'], case['image']
            label_path = run/'labels'/f'{case_id}.json'
            if label_path.exists():
                assert (run/'states'/f'{case_id}.json').exists() and (run/'receiver_arrays'/f'{case_id}.npz').exists()
                continue
            name, snr = case['channel'], case['snr_db']
            low, high = config['age_at_last_base_symbol_ms'][case['age_group']]
            age = engine.channel(name)['AgingTime'][0]
            left = 256+case['replica']*10000
            candidates = np.flatnonzero((age[left:left+10000]>=low)&(age[left:left+10000]<=high))+left
            rng = np.random.default_rng(stable_seed(config['seed'], item['image_id'], name, case['age_group'],case['replica']))
            last = int(rng.choice(candidates))
            base_start, enhancement_start = last-63, last+1+config['fixed_gap_steps']
            x = load_image(item['path'])
            base, info, base_ms = engine.infer(128,x,base_start,name,case_id,snr)
            single, _, single_ms = engine.infer(192,x,base_start,name,case_id,snr)
            enhancements, _, _ = engine.infer(64,torch.cat([input_for(x,a) for a in ACTIONS[:-1]]),enhancement_start,name,case_id,snr)
            finals = torch.cat([fuse(base,enhancements[i:i+1],a,config['alpha']) for i,a in enumerate(ACTIONS[:-1])]+[base])
            action_metrics = dict(zip(ACTIONS,engine.score(x,finals)))
            single_metrics = engine.score(x,single)[0]
            # Supervision for the uncertainty-only control remains in the label file.
            regional_errors = {a:float((x[:,:,y0:y1,x0:x1]-base[:,:,y0:y1,x0:x1]).square().mean()) for a,(y0,y1,x0,x1) in REGIONS.items()}
            np.savez_compressed(run/'receiver_arrays'/f'{case_id}.npz',received=base.cpu().numpy())
            feature_began = time.perf_counter()
            region_features = blind_region_features(base)
            feature_ms = (time.perf_counter()-feature_began)*1000
            state = {'case_id':case_id, 'image_id':item['image_id'], 'scene_id':item['scene_id'], 'split':item['split'],
                'channel':name, 'snr_db':snr, 'age_group':case['age_group'], 'replica':case['replica'],
                'csi_age_at_last_base_symbol_ms':info['aging_time'][0][-1], 'last_known_csi_magnitude':abs(info['csi_txrx'][0][-1]),
                'region_features':region_features, 'feature_ms':feature_ms, 'array':f'receiver_arrays/{case_id}.npz'}
            write(run/'states'/f'{case_id}.json',state)
            write(label_path, {'case_id':case_id,'actions':action_metrics, 'single':single_metrics, 'base_region_mse':regional_errors,
                'base_start':base_start, 'enhancement_start':enhancement_start, 'base_pilots':engine.pilots(name,base_start,64),
                'enhancement_pilots':engine.pilots(name,enhancement_start,32), 'single_pilots':engine.pilots(name,base_start,96),
                'base_codec_ms':base_ms, 'single_codec_ms':single_ms})
            completed += 1
            if completed%20 == 0: print(f'codec completed_new={completed} elapsed_s={time.monotonic()-started:.1f}',flush=True)

def replay(run, config, images):
    image_map = {x['image_id']:x for x in images}
    completed, started = 0, time.monotonic()
    with (run/'replay.log').open('a',encoding='utf-8') as log, torch.inference_mode():
        engine = Codec(log)
        for selection_path in sorted((run/'selections').glob('*.json')):
            case_id = selection_path.stem
            output = run/'replay'/f'{case_id}.json'
            if output.exists(): continue
            state = read(run/'states'/selection_path.name)
            label = read(run/'labels'/selection_path.name)
            selected = read(selection_path)['methods']
            with np.load(run/state['array']) as archive:
                base = torch.from_numpy(archive['received']).cuda()
            x = load_image(image_map[state['image_id']]['path'])
            base_end = label['base_start']+64
            replay_rows, cached = {}, {}
            for method, decision in selected.items():
                action, decision_ms = decision['action'], decision['decision_ms']
                # Conservative declared model: measured complete base codec call is post-reception compute.
                wait_ms = decision_ms + label['base_codec_ms']
                elapsed = 2+label['base_pilots']*config['pilot_symbols']/2048 + wait_ms
                reason = None
                feedback_sent = action != 'stop'
                if feedback_sent and not feedback_ok(case_id,config['feedback_erasure_probability']):
                    action, reason = 'stop','feedback_erasure'
                    elapsed += config['feedback_symbols']/2048
                if elapsed > config['deadline_ms']:
                    action, reason = 'stop','decision_deadline'
                start, extra_pilots, enhancement_ms = None, 0, 0.0
                metric = label['actions']['stop']
                if action != 'stop':
                    intended = base_end+math.ceil((wait_ms+(config['feedback_symbols']+config['header_symbols'])/2048)/STEP_MS)
                    start = engine.refresh_after(state['channel'],intended)
                    extra_pilots = engine.pilots(state['channel'],start,32)
                    if resources(action,label['base_pilots'],extra_pilots,config,True)['total_symbols'] > config['budget_symbols']:
                        action, reason = 'stop','budget'
                    else:
                        key = (action,start)
                        if key not in cached:
                            enhancement, _, enhancement_ms = engine.infer(64,input_for(x,action),start,state['channel'],case_id,state['snr_db'])
                            metric = engine.score(x,fuse(base,enhancement,action,config['alpha']))[0]
                            cached[key] = (metric,enhancement_ms)
                        metric,enhancement_ms = cached[key]
                        elapsed = (start-base_end)*STEP_MS+3+(label['base_pilots']+extra_pilots)*config['pilot_symbols']/2048+enhancement_ms
                        if elapsed > config['deadline_ms']:
                            metric, reason = label['actions']['stop'],'reconstruction_deadline'
                            # Actual late transmission still consumes its full resource budget.
                cost = resources(action,label['base_pilots'],extra_pilots,config,feedback_sent)
                assert cost['total_symbols'] <= config['budget_symbols']
                replay_rows[method] = {'requested_action':decision['action'],'executed_action':action,**metric,**cost,
                    'elapsed_ms':elapsed,'decision_ms':decision_ms,'enhancement_codec_ms':enhancement_ms,'enhancement_start':start,
                    'fallback_reason':reason,'deadline_miss':bool(elapsed>config['deadline_ms'])}
            write(output, {'case_id':case_id,'methods':replay_rows})
            completed += 1
            if completed%10 == 0: print(f'replay completed_new={completed} elapsed_s={time.monotonic()-started:.1f}',flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['generate','replay'],required=True)
    args=parser.parse_args()
    config, images = read(args.run/'config.json'),read(args.run/'data.json')['images']
    (generate if args.stage=='generate' else replay)(args.run,config,images)
