"""Original diagnostic plotting code; only audited, completed experiment data.

Style/API reference: Matplotlib official Linestyles example:
https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
No example implementation is vendored. No smoothing, fitted curve, or edited
score is used. This is an experiment diagnostic, not a manuscript figure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    audit = json.loads((run / 'review_analysis.json').read_text(encoding='utf8'))
    assert audit['status'] == 'AUDITED'
    history_path = run / 'history.json'
    history = json.loads(history_path.read_text(encoding='utf8'))
    assert history[-1]['step'] == 12000
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    styles = [
        ('ssm', 'SS2D', '#111111', '-', 'o'),
        ('attention', '摘要注意力 48', '#555555', '--', 's'),
        ('attention_wide', '摘要注意力 96', '#222222', '-.', '^'),
        ('pool', '全局池化', '#8a8a8a', ':', 'd'),
    ]
    steps = np.array([row['step'] for row in history])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    for mode, label, color, line, marker in styles:
        dev = [row['development'][mode]['psnr'] for row in history]
        mse = np.array([row['training_probe'][mode]['mse'] for row in history])
        kwargs = dict(label=label, color=color, linestyle=line, linewidth=1.6,
                      marker=marker, markersize=3.1, markevery=2, markerfacecolor='white')
        axes[0].plot(steps, dev, **kwargs)
        axes[1].plot(steps, mse / mse[0] * 100, **kwargs)
    axes[0].set(title='开发集平均 PSNR（24 张图）', ylabel='PSNR / dB')
    axes[1].set(title='固定训练样本 MSE（8 张图）', ylabel='MSE / 初始 MSE（%）')
    for axis in axes:
        axis.set(xlabel='完整图像训练步数', xlim=(-150, 12150), xticks=np.arange(0, 12001, 2000))
        axis.grid(axis='y', color='#dddddd', linewidth=.6)
        axis.spines[['top', 'right']].set_visible(False)
        for step in (3000, 5000, 8000, 10000):
            axis.axvline(step, color='#dddddd', linewidth=.65, zorder=0)
        axis.axvline(6000, color='#888888', linestyle='--', linewidth=.9, zorder=0)
    axes[0].text(6150, .045, '6000 → 12000 续训', transform=axes[0].get_xaxis_transform(),
                 fontsize=8, color='#555555', bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, .89))
    fig.suptitle('严格 SSM 复核：原始训练轨迹', y=.99, fontsize=14)
    fig.subplots_adjust(left=.075, right=.985, top=.75, bottom=.22, wspace=.23)
    fig.text(.5, .035, 'seed=17；AWGN 7/13 dB；每 500 步评估；原始检查点连线，无平滑。开发集结果，不是独立测试集。',
             ha='center', fontsize=8, color='#444444')
    outputs = []
    for suffix in ('png', 'pdf'):
        path = run / ('training_diagnostics.' + suffix)
        fig.savefig(path, dpi=180, facecolor='white', bbox_inches='tight')
        outputs.append(dict(path=str(path), sha256=digest(path)))
    plt.close(fig)
    manifest = dict(scope='Post-experiment diagnostic only; no manuscript drafted',
                    data=str(history_path), data_sha256=digest(history_path),
                    source_sha256=digest(Path(__file__)), matplotlib=matplotlib.__version__,
                    style_api_reference='https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html',
                    implementation='Original local code using the documented line-style API; no smoothing or edited values',
                    outputs=outputs)
    (run / 'training_diagnostics_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(outputs), flush=True)


if __name__ == '__main__':
    main()
