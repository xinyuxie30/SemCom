"""Label-blind frozen foundation-model feature extraction.

Only states/ and receiver_arrays/ are read. No labels or original images are read.
Channel metadata is deliberately excluded from every foundation-model prompt;
it enters the downstream gain estimator, making the age ablation interpretable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request
import numpy as np
from PIL import Image
import torch
from peft import PeftModel
from suite_common import ROOT, ACTIONS, REGIONS, read, write
from pilot_decisions import VDM, ARTIFACTS, CRITERIA, QUESTION, PROTOCOL, question_record, visual_predict, prediction, official_request
from probe_typesafe import ENDPOINT, MODEL, NoRedirect

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--live-api',action='store_true',required=True)
    args=parser.parse_args()
    config=read(args.run/'config.json')
    key=os.environ.get('TYPESAFE_API_KEY','').strip()
    if not key: raise RuntimeError('TYPESAFE_API_KEY absent')
    torch.manual_seed(config['seed'])
    model=VDM(str(ROOT/'models/qwen_base'),dtype=torch.bfloat16,device='cuda',with_heads=False)
    model.processor.image_processor.max_pixels=200704
    model.backbone=PeftModel.from_pretrained(model.backbone,str(ROOT/'models/visual_adapter')).eval()
    opener=urllib.request.build_opener(NoRedirect())
    gray=Image.new('RGB',(448,448),(127,127,127))
    descriptions_questions=[question_record('Consider only the '+r.replace('_',' ')+' quadrant of the RECEIVED image. Which description best matches visible blur/noise/blocking affecting edge and texture fidelity? Do not assume smooth scene content is necessarily damaged. The original is unavailable.',ARTIFACTS) for r in REGIONS]
    # New formal prompt version is frozen before formal outcomes; no ages, SNR or CSI.
    context={'protocol':PROTOCOL,'available_evidence':'Received pixels only. Link measurements are handled by a separate numerical controller.'}
    existing=list((args.run/'decisions').glob('*.json'))
    total_calls=len(existing)
    failures, completed, started=0,0,time.monotonic()
    with torch.inference_mode():
        for path in sorted((args.run/'states').glob('*.json')):
            output=args.run/'decisions'/path.name
            if output.exists(): continue
            if total_calls >= config['api_max_requests']: raise RuntimeError('Explicit API request cap reached')
            state=read(path)
            stage=time.perf_counter()
            with np.load(args.run/state['array']) as archive: array=archive['received'][0].transpose(1,2,0)
            received=Image.fromarray(np.clip(np.rint(array*255),0,255).astype(np.uint8)).resize((448,448),Image.Resampling.BICUBIC)
            preparation_ms=(time.perf_counter()-stage)*1000
            order=ACTIONS.copy()
            random.Random(int(hashlib.sha256(state['case_id'].encode()).hexdigest()[:8],16)).shuffle(order)
            criteria={a:CRITERIA[a] for a in order}
            question=question_record(QUESTION,criteria)
            with model.backbone.disable_adapter():
                raw,raw_ms=visual_predict(model,received,[question],context)
                desc,desc_ms=visual_predict(model,received,descriptions_questions,context)
            descriptions={r:{'grade':list(ARTIFACTS)[int(np.argmax(p))], 'distribution':dict(zip(ARTIFACTS,[round(v,6) for v in p]))} for r,p in zip(REGIONS,desc)}
            text_context={**context,'receiver_region_descriptions':descriptions,'description_origin':'Frozen visual front end on received pixels; uncertain observations, not ground truth'}
            with model.backbone.disable_adapter(): text_probs,text_ms=visual_predict(model,gray,[question],text_context)
            adapted,adapted_ms=visual_predict(model,received,[question],context)
            result={'case_id':state['case_id'],'input_preparation_ms':preparation_ms,'description_ms':desc_ms,'descriptions':descriptions,'option_order':order,
                'qwen_visual':prediction(order,raw[0],raw_ms),'qwen_description_only':prediction(order,text_probs[0],text_ms),'visual_jev':prediction(order,adapted[0],adapted_ms)}
            request=official_request(text_context,criteria)
            result['official_request']=request
            begin=time.perf_counter()
            fatal=False
            try:
                payload=json.dumps(request).encode()
                http=urllib.request.Request(ENDPOINT,data=payload,method='POST',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                total_calls+=1
                with opener.open(http,timeout=config['api_timeout_seconds']) as response: body=json.loads(response.read())
                elapsed=(time.perf_counter()-begin)*1000
                assert body['model']==MODEL==config['api_model']
                answer=body['answers']['action']
                assert answer['type']=='choice' and set(answer['probabilities'])==set(ACTIONS)
                official=prediction(order,[answer['probabilities'][a] for a in order],elapsed)
                assert answer['choice'] in ACTIONS and official['action']==answer['choice']
                for a in ACTIONS[:-1]:
                    probability=body['answers']['benefit_'+a]['noul']
                    assert isinstance(probability,(int,float)) and 0<=probability<=1
                official.update(response=body,confidence=answer['confidence'],request_sha256=hashlib.sha256(payload).hexdigest())
                result['official_jev']=official
                failures=0
            except Exception as error:
                failures+=1
                status=error.code if isinstance(error,urllib.error.HTTPError) else None
                fatal=status in (401,403)
                result['official_jev']={'action':'stop','fallback':True,'error_type':type(error).__name__,'http_status':status,'elapsed_ms':(time.perf_counter()-begin)*1000}
            result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            write(output,result)
            completed+=1
            if completed%5==0: print(f'decisions completed_new={completed} total_calls={total_calls} elapsed_s={time.monotonic()-started:.1f}',flush=True)
            if fatal or failures>=config['api_consecutive_failures_limit']:
                raise RuntimeError('API authorization failure or consecutive failures; partial records preserved. No automatic retries.')

if __name__=='__main__': main()
