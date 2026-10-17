"""Join frozen development decisions to outcomes only after inference is complete."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def main():
    run = ROOT / 'results/decision_component_20260926T162054038821Z'
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED'
    predictions = [json.loads(s) for s in (run / 'predictions.jsonl').read_text().splitlines()]
    outcomes = [json.loads(s) for s in (Path(manifest['states_path']).parent / 'action_outcomes.jsonl').read_text().splitlines()]
    by_case = {}
    for row in outcomes:
        if row['base_c'] == 128 and row['alpha'] == .5:
            by_case.setdefault(row['case_id'], {})[row['action']] = row
    methods = ['official_jev', 'visual_jev', 'qwen_visual', 'qwen_description_only', 'severity_rule', 'fixed_upper_right', 'global', 'stop', 'oracle', 'single_C192']
    report = {'scope': 'Development only: 8 images, 32 correlated states; not formal test accuracy', 'methods': {}}
    for method in methods:
        rows, correct, near, regret, times, counts = [], [], [], [], [], {}
        for p in predictions:
            actions = by_case[p['case_id']]
            oracle = min(actions, key=lambda a: actions[a]['mse'])
            if method == 'single_C192':
                row = {'mse': actions['stop']['single_mse'], 'psnr_db': -10*np.log10(actions['stop']['single_mse'])}
                action = None
            else:
                action = oracle if method == 'oracle' else 'upper_right' if method == 'fixed_upper_right' else method if method in ('global', 'stop') else p[method]['action']
                row = actions[action]
                correct.append(action == oracle)
                near.append(row['mse'] <= 1.02*actions[oracle]['mse'])
                counts[action] = counts.get(action, 0)+1
            rows.append(row)
            regret.append(actions[oracle]['psnr_db'] - row['psnr_db'])
            if method in p and 'elapsed_ms' in p[method]:
                extra = p['description_ms'] if method in ('official_jev', 'qwen_description_only') else 0
                times.append(p[method]['elapsed_ms'] + extra + p['input_preparation_ms'])
        report['methods'][method] = {'mean_psnr_db': float(np.mean([r['psnr_db'] for r in rows])), 'mean_mse': float(np.mean([r['mse'] for r in rows])),
            'exact_accuracy': float(np.mean(correct)) if correct else None, 'within_2pct_mse_accuracy': float(np.mean(near)) if near else None,
            'mean_psnr_regret_db': float(np.mean(regret)), 'median_total_decision_ms': float(np.median(times)) if times else None, 'action_counts': counts}
    report['api_failures'] = [{k:v for k,v in p['official_jev'].items() if k != 'response'} for p in predictions if p['official_jev'].get('fallback')]
    calibration = []
    usage = {'input_tokens':0, 'output_tokens':0}
    for p in predictions:
        reply = p['official_jev'].get('response')
        if reply:
            for k in usage: usage[k] += reply.get('usage', {}).get(k, 0)
            rows = by_case[p['case_id']]
            for a in list(rows):
                key = 'benefit_' + a
                if key in reply['answers']:
                    probability = reply['answers'][key]['noul']
                    calibration.append((probability, float(rows[a]['mse'] < rows['stop']['mse'])))
    report['api_usage'] = usage
    report['noul_diagnostic'] = {'n':len(calibration), 'brier':float(np.mean([(p-y)**2 for p,y in calibration])), 'positive_fraction':float(np.mean([y for _,y in calibration])), 'note':'Five correlated action labels per successful request; descriptive only'}
    (run/'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
