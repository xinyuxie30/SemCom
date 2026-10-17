"""Download the public DIV2K validation HR archive for an external, frozen test.
Official academic-research dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
No extraction of arbitrary archive paths; only the 100 expected PNGs are read.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def main():
    target = ROOT/'datasets/div2k'
    target.mkdir(parents=True, exist_ok=True)
    archive = target/'DIV2K_valid_HR.zip'
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as output:
            size = int(response.headers.get('Content-Length', 0))
            received, last = 0, time.monotonic()
            while True:
                b = response.read(2**20)
                if not b: break
                output.write(b)
                received += len(b)
                if time.monotonic()-last > 20:
                    print(json.dumps({'downloaded_bytes':received, 'total_bytes':size}), flush=True)
                    last = time.monotonic()
        if size and partial.stat().st_size != size: raise ValueError('Incomplete archive')
        partial.replace(archive)
    prepared = target/'128'
    prepared.mkdir(exist_ok=True)
    images = []
    with zipfile.ZipFile(archive) as bundle:
        for index in range(801, 901):
            name = f'{index:04d}.png'
            member = f'DIV2K_valid_HR/{name}'
            raw = bundle.read(member)
            import io
            with Image.open(io.BytesIO(raw)) as image:
                dimensions = image.size
                image = ImageOps.exif_transpose(image).convert('RGB').resize((128,128), Image.Resampling.BICUBIC)
                image.save(prepared/name)
            images.append({'image_id':'div2k_'+name, 'scene_id':'div2k_'+name[:-4], 'path':str(prepared/name),
                'source_member':member, 'source_size':dimensions, 'source_sha256':hashlib.sha256(raw).hexdigest(), 'sha256':sha(prepared/name), 'split':'external_test'})
    result = {'source':URL, 'archive_sha256':sha(archive), 'source_split':'DIV2K validation HR; used here only as external test',
        'license':'Academic research only; cite Agustsson and Timofte, CVPRW 2017', 'preprocessing':'EXIF transpose / RGB / BICUBIC 128x128', 'images':images}
    (ROOT/'div2k_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'images':len(images), 'archive_sha256':result['archive_sha256']}), flush=True)

if __name__ == '__main__': main()
