"""CPU-only audit of frozen B0.1 artifacts; never changes inference outputs."""
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'dcat_csi_prediction_20261004'
ROOT = HERE.parents[1]
METRICS = ['mse', 'psnr_db', 'lpips_normalized', 'tx_nmse', 'rx_nmse',
           'tx_error_estimate', 'rx_error_estimate', 'prediction_ms', 'codec_ms']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(delta, block_size=4):
    # [noise seed, scene/time position, SNR, phase]. Keep all 18 repeats together.
    scene_means = delta.mean(axis=(0, 2, 3))
    blocks = scene_means.reshape(-1, block_size).mean(axis=1)
    rng = np.random.default_rng(20261008)
    boot = blocks[rng.integers(0, len(blocks), size=(10000, len(blocks)))].mean(axis=1)
    return np.quantile(boot, [.025, .975]).tolist()


def main():
    job = Path(read(HERE / 'latest_job.json')['job'])
    state, cfg = read(job / 'status.json'), read(job / 'protocol.json')
    assert state['status'] == 'COMPLETED'
    out = job / 'review'
    out.mkdir(exist_ok=True)
    e = cfg['e2e']
    families = cfg['b01']['checkpoint_families']
    seeds = cfg['b01']['noise_seeds']
    channels, policies, snrs, phases = e['channels'], e['policies'], e['snr_db'], e['start_phases']
    images = read(job / state['runs'][0]['name'] / 'images.json')
    ids = [r['image_id'] for r in images]
    scenes = {r['image_id']: r['scene_id'] for r in images}
    assert len(set(scenes.values())) == len(ids) == 32
    indexes = [{v: i for i, v in enumerate(values)} for values in
               (families, seeds, channels, policies, ids, snrs, phases)]
    expected = len(channels)*len(policies)*len(ids)*len(snrs)*len(phases)
    cube = np.full(tuple(map(len, (families,seeds,channels,policies,ids,snrs,phases))) + (len(METRICS),), np.nan)
    starts = {}
    manifests, checks, input_hashes = {}, [], {}
    source = Path(cfg['baseline_root']) / 'SoM-DCAT-main'
    hash_cache = {}

    def check_hash(path, value):
        path = Path(path)
        if path not in hash_cache:
            hash_cache[path] = digest(path)
        assert hash_cache[path] == value, f'Hash mismatch: {path}'

    for name, value in state['source_hashes'].items():
        check_hash(job/name, value)
    b0 = Path(state['b0'])
    for family, seed in itertools.product(families, seeds):
        name = f'{family}_seed{seed}'
        folder = job/name
        m = read(folder/'manifest.json')
        manifests[name] = m
        assert m['status'] == 'COMPLETED' and not m['smoke']
        assert m['expected_rows'] == m['completed_rows'] == expected == 11520
        assert m['repeated_call_max_abs_difference'] == 0
        assert read(folder/'images.json') == images
        for item in images:
            check_hash(item['path'], item['sha256'])
        local_cfg = read(job/(name+'_protocol.json'))
        expected_cfg = json.loads(json.dumps(cfg))
        expected_cfg['e2e']['seed'] = seed
        assert local_cfg == expected_cfg
        check_hash(job/(name+'_protocol.json'), m['protocol_sha256'])
        for file, sha in m['code_hashes'].items():
            check_hash(folder/file, sha)
            check_hash(job/'src'/file, sha)
        for file, sha in m['baseline_source_hashes'].items():
            check_hash(source/file, sha)
        for file, sha in m['baseline_helper_hashes'].items():
            check_hash(Path(cfg['baseline_root'])/'reproduction'/file, sha)
        check_hash(source/'Checkpoints'/m['checkpoint'], m['checkpoint_sha256'])
        for channel, predictors in m['predictor_hashes'].items():
            for predictor, sha in predictors.items():
                check_hash(b0/channel/(predictor+'.npz'), sha)
        raw_path = folder/'per_state.csv'
        input_hashes[name] = digest(raw_path)
        with raw_path.open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        seen = set()
        for r in rows:
            key = (r['channel'],r['policy'],r['image_id'],int(float(r['snr_db'])),int(r['phase']))
            assert key not in seen
            seen.add(key)
            assert r['scene_id'] == scenes[r['image_id']]
            assert r['reconstruction_finite'] == 'True'
            values = np.array([float(r[k]) for k in METRICS])
            assert np.isfinite(values).all() and values[0] > 0
            assert abs(values[1]+10*np.log10(values[0])) < 1e-4
            full_key = (family,seed,*key)
            ix = tuple(indexes[i][v] for i,v in enumerate(full_key))
            cube[ix] = values
            start_key = (r['channel'],r['image_id'],int(r['phase']))
            if start_key in starts:
                assert starts[start_key] == int(r['start'])
            starts[start_key] = int(r['start'])
        assert len(rows) == len(seen) == expected
        for r in read(folder/'summary.json')['rows']:
            a = cube[indexes[0][family],indexes[1][seed],indexes[2][r['channel']],indexes[3][r['policy']]]
            hold = cube[indexes[0][family],indexes[1][seed],indexes[2][r['channel']],0]
            for j, metric in enumerate(METRICS):
                assert abs(float(a[...,j].mean())-r[metric]) < 1e-10
            assert abs(float((a[...,1]-hold[...,1]).mean())-r['delta_psnr_vs_hold_db']) < 1e-10
        checks.append({'run':name,'unique_rows':len(seen),'all_finite':True,'summary_matches_raw':True,
                       'seconds':m['elapsed_seconds'],'peak_allocated_GiB':m['peak_allocated_bytes']/2**30,
                       'peak_reserved_GiB':m['peak_reserved_bytes']/2**30})
    assert np.isfinite(cube).all()
    assert len(checks)*expected == state['expected_transmissions'] == 69120
    combined = read(job/'combined_summary.json')
    expected_combined = [{'run':run['name'],**r} for run in state['runs'] for r in read(job/run['name']/'summary.json')['rows']]
    assert combined == expected_combined

    # Historical controls were already rerun by B0.1. Compare files; no GPU rerun here.
    historical = []
    for family, pointer in [('without','latest_e2e.json'),('with','latest_e2e_with.json')]:
        previous = Path(read(OLD/pointer)['run'])
        if not previous.is_absolute(): previous = ROOT/previous
        with (previous/'per_state.csv').open(encoding='utf-8') as stream:
            old_rows = list(csv.DictReader(stream))
        mapping = {'hold':'hold','wiener_both':'wiener','acf_both':'acf'}
        diffs = {k:[] for k in ('mse','psnr_db','lpips_normalized','tx_nmse','rx_nmse')}
        count = 0
        for r in old_rows:
            if r['policy'] not in mapping: continue
            values = (family,20261004,r['channel'],mapping[r['policy']],r['image_id'],int(float(r['snr_db'])),int(r['phase']))
            a = cube[tuple(indexes[i][v] for i,v in enumerate(values))]
            for metric in diffs:
                diffs[metric].append(abs(float(r[metric])-a[METRICS.index(metric)]))
            count += 1
        historical.append({'family':family,'paired_rows':count,'max_abs_differences':{k:float(max(v)) for k,v in diffs.items()}})

    # Independently rebuild calibration with validation labels only. Verify hook
    # outputs against recorded per-state estimates and prove held-out label isolation.
    sys.path.insert(0,str(job/'src'))
    from csi_predictors import load_trace, samples, Predictor, simple_predictor, channel_info
    calibration, pooled_power = [], {}
    channel_metric_max_diff = 0.
    for ci, channel in enumerate(channels):
        check_hash(source/'Channel_inputs'/(channel+'.mat'),read(b0/channel/'summary.json')['source_sha256'])
        trace = load_trace(source/'Channel_inputs'/(channel+'.mat'))
        anchors, x, y = samples(trace,cfg['pilot_blocks']['validation'])
        test_anchors = samples(trace,cfg['pilot_blocks']['diagnostic_test'])[0]
        chosen = test_anchors[np.linspace(0,len(test_anchors)-1,e['images']).astype(int)]
        for i,image in enumerate(ids):
            for phase in phases:
                start = starts[(channel,image,phase)]
                assert start == int(chosen[i])*128+phase
                pooled_power[(channel,image,phase)] = float(np.sum(abs(trace['CSI'][start:start+64])**2))
        altered = dict(trace)
        altered['CSI'] = trace['CSI'].copy()
        altered['CSI'][800*128:] = 999999+999999j
        _, x2, y2 = samples(altered,cfg['pilot_blocks']['validation'])
        np.testing.assert_array_equal(x,x2); np.testing.assert_array_equal(y,y2)
        for pi,predictor in enumerate(['hold','wiener','acf']):
            model = simple_predictor('hold') if predictor=='hold' else Predictor.load(b0/channel/('wiener_empirical.npz' if predictor=='wiener' else 'wiener_acf.npz'))
            for ii,image in enumerate(ids):
                for hi,phase in enumerate(phases):
                    info = channel_info(trace,starts[(channel,image,phase)],tx_model=model,rx_model=model)
                    truth = np.asarray(info['csi_gt'][0])
                    for metric,field in [('tx_nmse','csi_encoder'),('rx_nmse','csi_txrx')]:
                        measured = float(np.sum(abs(np.asarray(info[field][0])-truth)**2)/np.sum(abs(truth)**2))
                        for suffix in ('','_cal','_mmse','_cal_mmse'):
                            actual = cube[:,:,ci,indexes[3][predictor+suffix],ii,:,hi,METRICS.index(metric)]
                            channel_metric_max_diff = max(channel_metric_max_diff,float(abs(actual-measured).max()))
            assert channel_metric_max_diff < 1e-12
            error = np.mean(abs(model.predict(x)-y)**2,axis=0)
            norm = error / np.maximum(np.mean(abs(y)**2,axis=0),1e-12)
            curve_i = ci*3+pi  # Order follows frozen channel and policy loops.
            max_diff = 0.
            for family, seed in itertools.product(families,seeds):
                with np.load(job/f'{family}_seed{seed}'/'calibration_curves.npz',allow_pickle=False) as curves:
                    np.testing.assert_allclose(curves[f'curve_{curve_i}_0'],error,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(curves[f'curve_{curve_i}_1'],norm,rtol=0,atol=1e-12)
                fi,si = indexes[0][family],indexes[1][seed]
                for suffix in ('_cal','_cal_mmse'):
                    p = indexes[3][predictor+suffix]
                    for hi,phase in enumerate(phases):
                        txh = phase+np.arange(64)
                        rxh = txh%128
                        for metric, horizons in [('tx_error_estimate',txh),('rx_error_estimate',rxh)]:
                            actual = cube[fi,si,ci,p,:,:,hi,METRICS.index(metric)]
                            max_diff=max(max_diff,float(abs(actual-norm[horizons].mean()).max()))
            assert max_diff < 5e-7
            calibration.append({'channel':channel,'predictor':predictor,'max_hook_abs_difference':max_diff,
                                'curves_match':True,'validation_independent_of_diagnostic_labels':True,
                                'mean_validation_rx_nmse':float(norm[:128].mean())})

    comparisons, stratified, factorial = [], [], []
    for fi,family in enumerate(families):
        for ci,channel in enumerate(channels):
            hold = cube[fi,:,ci,0]
            reference = cube[0,:,ci,0]
            power = np.array([[pooled_power[(channel,image,phase)] for phase in phases] for image in ids])[None,:,None,:]
            for pi,policy in enumerate(policies):
                a = cube[fi,:,ci,pi]
                suffix = policy[len(policy.split('_')[0]):]
                matched = cube[fi,:,ci,indexes[3]['hold'+suffix]]
                delta = a[...,1]-hold[...,1]
                seed_delta = delta.mean(axis=(1,2,3))
                paired_delta = a[...,1]-matched[...,1]
                ci4,ci8 = interval(delta,4),interval(delta,8)
                paired_ci = interval(paired_delta,4)
                row = {'family':family,'channel':channel,'policy':policy,'states':int(a[...,1].size),
                       'mean_psnr_db':float(a[...,1].mean()),'delta_vs_hold_db':float(delta.mean()),
                       'seed_delta_min':float(seed_delta.min()),'seed_delta_max':float(seed_delta.max()),
                       'delta_vs_matched_hold_db':float(paired_delta.mean()),
                       'matched_ci4_low':paired_ci[0],'matched_ci4_high':paired_ci[1],
                       'delta_vs_without_hold_db':float((a[...,1]-reference[...,1]).mean()),
                       'ci4_low':ci4[0],'ci4_high':ci4[1],'ci8_low':ci8[0],'ci8_high':ci8[1],
                       'mean_mse':float(a[...,0].mean()),
                       'mse_relative_change_pct':float(100*(a[...,0].mean()/hold[...,0].mean()-1)),
                       'mean_lpips':float(a[...,2].mean()),'delta_lpips':float((a[...,2]-hold[...,2]).mean()),
                       'psnr_p05_db':float(np.quantile(a[...,1],.05)),
                       'below20_fraction_exploratory':float((a[...,1]<20).mean()),
                       'below20_change_pp':float(100*((a[...,1]<20).mean()-(hold[...,1]<20).mean())),
                       'paired_win_fraction':float((delta>0).mean()),
                       'pooled_rx_nmse':float((a[...,4]*power).sum()/np.broadcast_to(power,a[...,4].shape).sum()),
                       'mean_tx_error_estimate':float(a[...,5].mean()),'mean_rx_error_estimate':float(a[...,6].mean()),
                       'prediction_median_ms':float(np.median(a[...,7])),'codec_mean_ms':float(a[...,8].mean())}
                comparisons.append(row)
                for si,snr in enumerate(snrs):
                    for hi,phase in enumerate(phases):
                        stratified.append({'family':family,'channel':channel,'policy':policy,'snr_db':snr,'phase':phase,
                                           'delta_vs_hold_db':float(delta[:,:,si,hi].mean())})
            for predictor in ('hold','wiener','acf'):
                for left,right,label in [(predictor,predictor+'_cal','calibration_with_ZF'),
                                          (predictor+'_mmse',predictor+'_cal_mmse','calibration_with_MMSE'),
                                          (predictor,predictor+'_mmse','MMSE_without_calibration'),
                                          (predictor+'_cal',predictor+'_cal_mmse','MMSE_with_calibration')]:
                    delta = cube[fi,:,ci,indexes[3][right],...,1]-cube[fi,:,ci,indexes[3][left],...,1]
                    factorial.append({'family':family,'channel':channel,'predictor':predictor,'contrast':label,
                                      'difference_psnr_db':float(delta.mean()),'seed_min':float(delta.mean(axis=(1,2,3)).min()),
                                      'seed_max':float(delta.mean(axis=(1,2,3)).max())})
    table(out/'comparisons.csv',comparisons)
    table(out/'stratified.csv',stratified)
    table(out/'factorial_contrasts.csv',factorial)
    lookup = {(r['family'],r['channel'],r['policy']):r for r in comparisons}
    gates = []
    for family in families:
        for policy in policies:
            a1,a7 = lookup[family,'A1',policy],lookup[family,'A7',policy]
            gates.append({'family':family,'policy':policy,'A7_delta':a7['delta_vs_hold_db'],
                          'A1_delta':a1['delta_vs_hold_db'],
                          'descriptive_A7_0p3_A1_minus0p05_gate':bool(a7['delta_vs_hold_db']>=.3 and a1['delta_vs_hold_db']>=-.05)})
    total_seconds = (datetime.fromisoformat(state['finished_utc'])-datetime.fromisoformat(state['started_utc'])).total_seconds()
    report = {'verification_status':'ANALYZED','created_utc':datetime.now(timezone.utc).isoformat(),
              'job':str(job),'total_rows':int(cube[...,0].size),'unique_scenes':32,'wall_seconds':total_seconds,
              'source_and_artifact_checks':checks,'calibration_checks':calibration,'historical_control_agreement':historical,
              'independent_channel_nmse_max_abs_difference':channel_metric_max_diff,
              'interval_scope':'Post-hoc descriptive 10000 block resamples of the SAME 32 scene/time pairs, all 3 noise seeds and 6 SNR/phase repeats kept together. Block sizes 4 and 8 are sensitivity checks, not independent-trace population CIs. No p-values or corrected significance claims.',
              'gates':gates,'comparisons':comparisons,'input_csv_sha256':input_hashes,
              'analysis_script_sha256':digest(__file__),
              'review_did_new_gpu_inference':False}
    write(out/'audit.json',report)
    make_plot(out,lookup,families,channels,policies)
    print(json.dumps({'review_directory':str(out),'rows':report['total_rows'],'seconds':total_seconds,
                      'all_gates_failed':not any(g['descriptive_A7_0p3_A1_minus0p05_gate'] for g in gates),
                      'historical_control_agreement':historical},ensure_ascii=False))
    for family in families:
        for channel in ['A1','A5','A7','A9']:
            for policy in ['hold','wiener','acf','acf_cal','hold_mmse']:
                r=lookup[family,channel,policy]
                print(json.dumps({k:r[k] for k in ['family','channel','policy','mean_psnr_db','delta_vs_hold_db',
                                                  'delta_vs_matched_hold_db','ci4_low','ci4_high','mean_lpips',
                                                  'psnr_p05_db','below20_fraction_exploratory','mse_relative_change_pct']}))


def make_plot(out,lookup,families,channels,policies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,family in zip(axes,families):
        values=np.array([[lookup[family,c,p]['delta_vs_hold_db'] for c in channels] for p in policies])
        img=ax.imshow(values,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-12,vcenter=0,vmax=1),aspect='auto')
        ax.set_xticks(range(len(channels)),channels)
        ax.set_yticks(range(len(policies)),policies)
        ax.set_title(f'Frozen {family}_CP_C128 checkpoint')
        ax.set_xlabel('Released channel trace (one per speed)')
        for (i,j),v in np.ndenumerate(values):
            ax.text(j,i,f'{v:+.3f}',ha='center',va='center',fontsize=8,color='white' if v<-8 else 'black')
    fig.colorbar(img,ax=axes,shrink=.7,label='Mean PSNR difference vs own hold (dB)')
    fig.suptitle('B0.1 development diagnostic: 32 scenes, 3 noise seeds, 3 SNRs, 2 phases\nGains are not relative to an independent SOTA benchmark',fontsize=11)
    fig.savefig(out/'b01_effects.png',dpi=170)
    fig.savefig(out/'b01_effects.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
