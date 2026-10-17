"""Freeze image-group splits without reading model outcomes."""
from collections import Counter
from pathlib import Path
from suite_common import ROOT, BASELINE, read, write, digest, stable_seed

def main():
    source = BASELINE/'datasets/UDIS-D/test128/prepared_manifest.json'
    items = read(source)['images']
    # Different numbered pairs reuse identical views. Merge those pairs transitively
    # before splitting, not just the two images with the same numbered basename.
    parent={Path(x['source'].replace('\\','/')).stem:Path(x['source'].replace('\\','/')).stem for x in items}
    def find(s):
        if parent[s]!=s: parent[s]=find(parent[s])
        return parent[s]
    seen={}
    for x in items:
        s=Path(x['source'].replace('\\','/')).stem
        if x['sha256'] in seen:
            a,b=find(s),find(seen[x['sha256']])
            parent[max(a,b)]=min(a,b)
        seen[x['sha256']]=s
    components={}
    for x in items:
        s=find(Path(x['source'].replace('\\','/')).stem)
        components.setdefault(s,{})[x['sha256']]=x
    scenes=sorted([s for s,unique in components.items() if len(unique)>=2],key=lambda x:stable_seed('split_v2',x))
    assert len(scenes)>=576
    assignments = {s:split for split, group in [('train',scenes[:256]), ('validation',scenes[256:320]), ('test',scenes[320:576])] for s in group}
    representatives=[]
    for s in assignments:
        representatives.extend((s,x) for x in sorted(components[s].values(),key=lambda x:stable_seed('representative',x['sha256']))[:2])
    images = []
    for scene,x in representatives:
        images.append({'image_id':'udis_'+x['file'], 'scene_id':'udis_'+scene,
            'path':str(source.parent/x['file']), 'sha256':x['sha256'], 'split':assignments[scene], 'source':x['source']})
    images += read(ROOT/'div2k_manifest.json')['images']
    assert all(digest(x['path']) == x['sha256'] for x in images)
    assert len({x['sha256'] for x in images}) == len(images), 'Exact duplicate across selected images'
    manifest = {'seed':20260927, 'images':images, 'counts':dict(Counter(x['split'] for x in images)),
        'udis_manifest_sha256':digest(source), 'div2k_manifest_sha256':digest(ROOT/'div2k_manifest.json'),
        'split_rule':'Union-find of original pair basenames and shared exact prepared-image hashes, then split_v2 hash order; 256/64/256 components, two unique representatives each; all other UDIS images unused',
        'udis_pair_count':len(parent),'udis_connected_components':len(components),'eligible_components':len(scenes),
        'grouping_limit':'Components are scene proxies from known pairs/shared pixels; visually similar but nonidentical scenes may remain. New DIV2K external test is reported separately.',
        'exposure':'UDIS images previously used for aggregate published-weight inference, never for this selector training; DIV2K outcomes unobserved before formal run. Kodak is development only.',
        'validation':'No statistical power guarantee; cluster at UDIS scene or DIV2K image, average channel replicas first.'}
    write(ROOT/'suite_data.json', manifest)
    kodak = read(ROOT/'kodak_manifest.json')['images']
    smoke = []
    for i, split in enumerate(['train','train','validation','test','external_test']):
        x = kodak[i]
        smoke.append({'image_id':x['image_id'], 'scene_id':Path(x['image_id']).stem,
            'path':str(ROOT/'datasets/kodak/128'/x['image_id']), 'sha256':x['prepared_sha256'], 'split':split})
    write(ROOT/'smoke_data.json', {'scope':'Previously observed Kodak development fixtures only', 'images':smoke})
    cfg = read(ROOT/'suite_config.json')
    cfg.update(replicas=1, snrs=[0], external_channels=['A9','A1'], train_epochs=3,
               early_stopping_patience=2, bootstrap_samples=50, api_max_requests=12, stage_timeout_seconds=1200)
    write(ROOT/'smoke_config.json', cfg)
    print(manifest['counts'])

if __name__ == '__main__': main()
