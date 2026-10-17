"""Shared, label-free experiment contracts. Only standard-library imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = Path('D:/PyChram_project/DCAT-SoM')
CODEC_PYTHON = BASELINE/'.venv/Scripts/python.exe'
VISUAL_PYTHON = ROOT/'visual_env/Scripts/python.exe'
REGIONS = ['upper_left', 'upper_right', 'lower_left', 'lower_right']
ACTIONS = REGIONS + ['global', 'stop']
STEP_MS = .03125
FEATURE_KEYS = ['mean', 'std', 'gradient', 'laplacian_variance', 'entropy_bits', 'clipped_fraction']

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def cases(config, images):
    for item in images:
        channels = config['external_channels'] if item['split'] == 'external_test' else config['channels']
        for channel in channels:
            for snr in config['snrs']:
                for age in config['ages']:
                    for replica in range(config['replicas']):
                        case_id = f"{Path(item['image_id']).stem}_{channel}_{snr}_{age}_{replica}"
                        yield {'case_id':case_id, 'image':item, 'channel':channel, 'snr_db':snr, 'age_group':age, 'replica':replica}

def feedback_ok(case_id, probability):
    # A common uniform draw for all policies: matched erasure-event counterfactuals.
    return stable_seed(case_id, 'feedback_erasure')/2**32 >= probability

def resources(action, base_pilots, enhancement_pilots, config, feedback_sent=True):
    transmit = action != 'stop'
    payload = 4096 + (2048 if transmit else 0)
    feedback = config['feedback_symbols'] if feedback_sent else 0
    header = config['header_symbols'] * (2 if transmit else 1)
    pilots = config['pilot_symbols'] * (base_pilots + (enhancement_pilots if transmit else 0))
    total = payload + feedback + header + pilots
    return {'payload_symbols':payload, 'feedback_symbols':feedback, 'header_symbols':header,
            'pilot_symbols':pilots, 'total_symbols':total, 'unused_symbols':config['budget_symbols']-total,
            'normalized_rf_energy':total, 'energy_scope':'Unit transmit energy per complex symbol in either direction; excludes compute and service infrastructure'}
