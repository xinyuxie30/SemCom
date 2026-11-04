"""Plot existing audited measurements only; no model imports or GPU operations.

Original plotting code written for this project, using installed Matplotlib.
Input CSVs and study artifacts are read-only. All outputs stay beside this file.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "attnrx_20261104"
CONFIRM = STUDY / "confirmation"
MODELS = ("public_original", "plain", "global", "local", "affinity")
CONDITIONS = ("rayleigh", "rician", "aged", "awgn")
MAIN = CONDITIONS[:3]
SNRS = (1, 7, 13)
LABELS = {
    "public_original": "公开权重直接迁移",
    "plain": "累计适配原版（同预算对照）",
    "global": "全局修正",
    "local": "局部修正",
    "affinity": "注意力修正",
}
COLORS = {
    "public_original": "#7B8491",
    "plain": "#236A9D",
    "global": "#C78125",
    "local": "#34866F",
    "affinity": "#9463A8",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_files = [
        CONFIRM / "public_fresh.csv",
        CONFIRM / "fresh.csv",
        CONFIRM / "public_fresh.json",
        CONFIRM / "fresh.json",
        CONFIRM / "INTERPRETATION_AUDIT.json",
        STUDY / "PROTOCOL.txt",
    ]
    source_hashes = {str(p): digest(p) for p in source_files}
    rows = []
    for path in source_files[:2]:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    raw = {}
    for row in rows:
        model, condition = row["model"], row["condition"]
        image, snr, draw = (int(row[k]) for k in ("image", "snr_db", "draw"))
        key = (model, image, condition, snr, draw)
        assert key not in raw, ("Duplicate cell", key)
        value, mse = float(row["psnr_db"]), float(row["mse"])
        assert math.isfinite(value) and mse > 0
        assert abs(value + 10 * math.log10(mse)) < 1e-10
        assert int(row["forward_symbols"]) == 12383
        assert int(row["reverse_symbols"]) == 0
        assert abs(float(row["payload_power"]) - 1) < 1e-5
        raw[key] = value
    expected = set(itertools.product(MODELS, range(733, 749), CONDITIONS, SNRS, range(4)))
    assert set(raw) == expected
    assert len(raw) == 3840

    cells = {
        (m, c, s): mean(raw[m, i, c, s, d] for i in range(733, 749) for d in range(4))
        for m, c, s in itertools.product(MODELS, CONDITIONS, SNRS)
    }
    means = {
        m: {c: mean(cells[m, c, s] for s in SNRS) for c in CONDITIONS}
        for m in MODELS
    }
    for m in MODELS:
        means[m]["main"] = mean(means[m][c] for c in MAIN)
    references = {}
    for path in source_files[2:4]:
        references.update(json.loads(path.read_text(encoding="utf-8")))
    for m in MODELS:
        assert abs(means[m]["main"] - references[m]["main_psnr"]) < 1e-10
        assert abs(means[m]["awgn"] - references[m]["awgn_psnr"]) < 1e-10
        for c, s in itertools.product(CONDITIONS, SNRS):
            assert abs(cells[m, c, s] - references[m]["by_condition"][f"{c}_{s}"]) < 1e-10

    with (HERE / "aggregate_values.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "condition", "snr_db", "n_images", "n_draws", "mean_psnr_db", "delta_vs_plain_db"])
        writer.writeheader()
        for m, c, s in itertools.product(MODELS, CONDITIONS, SNRS):
            writer.writerow(dict(model=m, condition=c, snr_db=s, n_images=16, n_draws=4,
                                 mean_psnr_db=cells[m, c, s], delta_vs_plain_db=cells[m, c, s] - cells["plain", c, s]))

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "font.family": font_name,
        "font.size": 10,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#A8B0BA",
        "axes.labelcolor": "#303945",
        "text.color": "#27323E",
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(14, 9.6), facecolor="white")
    fig.text(.065, .956, "现有 PSNR 结果：适配收益与结构净增益", fontsize=21, fontweight="bold")
    fig.text(.065, .919, "相同 Wiener＋固定衰减补偿前端  |  同图、同信道噪声、相同传输预算", fontsize=11.5, color="#5B6673")
    fig.text(.065, .875, "A  公开权重直接迁移与累计目标信道适配", fontsize=13, fontweight="bold")
    fig.text(.965, .875, f"三信道平均差：+{means['plain']['main'] - means['public_original']['main']:.4f} dB", ha="right", fontsize=12, color=COLORS["plain"])
    top_keys = ("public_original", "plain", "global")
    handles = [Line2D([0], [0], color=COLORS[m], linewidth=2.0,
                       linestyle="--" if m != "plain" else "-", marker="o" if m != "global" else "s",
                       markerfacecolor="white" if m != "plain" else COLORS[m], markersize=5,
                       label=LABELS[m] + ("（与原版近乎重合）" if m == "global" else "")) for m in top_keys]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.06, .836), frameon=False, ncol=3, columnspacing=2.4)
    titles = {"rayleigh": "Rayleigh 多径", "rician": "Rician 多径（K＝4）", "aged": "aged：Rayleigh＋额外 CSI 误差"}
    width, lefts = .265, (.065, .385, .705)
    for col, c in enumerate(MAIN):
        ax = fig.add_axes([lefts[col], .562, width, .229])
        for m in top_keys:
            ax.plot(SNRS, [cells[m, c, s] for s in SNRS], color=COLORS[m], linewidth=2,
                    linestyle="--" if m != "plain" else "-", marker="o" if m != "global" else "s",
                    markerfacecolor="white" if m != "plain" else COLORS[m], markersize=5)
        ax.set_title(titles[c], fontsize=11.5, pad=10)
        ax.set(xlim=(.3, 13.7), ylim=(15, 36), xticks=SNRS, xlabel="SNR / dB")
        ax.yaxis.set_major_locator(MultipleLocator(5))
        if col == 0:
            ax.set_ylabel("平均 PSNR / dB")
        ax.grid(axis="both", color="#E4E8EC", linewidth=.7)
        ax.set_axisbelow(True)
        delta = means["plain"][c] - means["public_original"][c]
        ax.text(.04, .94, f"适配原版 − 公开权重：+{delta:.3f} dB", transform=ax.transAxes, va="top", fontsize=9,
                color=COLORS["plain"], bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=3))

    fig.text(.065, .472, "B  本轮结构改动相对同预算原版的净变化", fontsize=13, fontweight="bold")
    keys = ("global", "local", "affinity")
    handles = [Line2D([0], [0], color=COLORS[m], linewidth=1.8, marker=mk, markersize=5,
                       label=f"{LABELS[m]}：主均值 {means[m]['main'] - means['plain']['main']:+.5f} dB")
               for m, mk in zip(keys, ("s", "^", "D"))]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.06, .433), frameon=False, ncol=3, columnspacing=2.4)
    biggest = max(abs(cells[m, c, s] - cells["plain", c, s]) for m, c, s in itertools.product(keys, MAIN, SNRS))
    limit = max(.1, math.ceil(biggest * 1.2 / .05) * .05)
    for col, c in enumerate(MAIN):
        ax = fig.add_axes([lefts[col], .179, width, .223])
        ax.axhline(0, color="#596573", linewidth=1.2, linestyle="--")
        for m, mk in zip(keys, ("s", "^", "D")):
            ax.plot(SNRS, [cells[m, c, s] - cells["plain", c, s] for s in SNRS], color=COLORS[m], marker=mk,
                    linewidth=1.8, markersize=5)
        ax.set(xlim=(.3, 13.7), ylim=(-limit, limit), xticks=SNRS, xlabel="SNR / dB")
        if col == 0:
            ax.set_ylabel("ΔPSNR / dB（相对原版）")
        ax.grid(axis="both", color="#E4E8EC", linewidth=.7)
        ax.set_axisbelow(True)

    fig.text(.065, .106, "统计范围：16 幅已使用开发图（733–748），每点 4 次新噪声；单训练种子 17。各点为逐图 PSNR 的均值。", fontsize=9.5)
    fig.text(.065, .079, "解释：A 中训练历史不同，展示累计适配收益；B 中本轮训练与选择预算相同。当前结果不是新图像测试或多种子结论。", fontsize=9.5)
    awgn = "，".join(f"{LABELS[m].split('（')[0]} {means[m]['awgn']:.4f}" for m in MODELS)
    fig.text(.065, .052, "AWGN 均值 / dB：" + awgn + "。", fontsize=9)
    fig.text(.065, .026, "共用已知 PDP、理想尺度元数据；aged 使用当前接收端 CSI。每幅 12,383 个前向复符号、0 反馈；曲线连接仅用于阅读。", fontsize=9, color="#5B6673")
    fig.savefig(HERE / "psnr_evidence.png", dpi=200)
    fig.savefig(HERE / "psnr_evidence.pdf")
    plt.close(fig)

    audit = json.loads((CONFIRM / "INTERPRETATION_AUDIT.json").read_text(encoding="utf-8"))
    manifest = dict(
        material_passport="Local audited experiment CSVs; no external data or new inference.",
        verification_status="VERIFIED: CPU aggregate recomputation against stored study summaries; not a model rerun",
        rows=3840, grid="5 models x 16 images x 4 conditions x 3 SNRs x 4 channel draws",
        sources=source_hashes, means=means,
        main_adaptation_gap=means["plain"]["main"] - means["public_original"]["main"],
        structural_deltas={m: means[m]["main"] - means["plain"]["main"] for m in keys},
        aggregation="Arithmetic mean of image-wise PSNR, equal image/channel/SNR/draw weights; no outlier removal",
        displayed_main_conditions=list(MAIN), auxiliary_condition="AWGN: all model means retained in figure footer and table",
        inclusion="All 4 current experimental arms retained; public original reference shown; no favorable seed/cell selection",
        selection="Existing protocol selection on separate noise draws of the same development images; no new reselection",
        limitations=["Single training seed", "Repeatedly observed development images; new noise is not unseen images",
                     "Public original vs adapted models have different cumulative training histories",
                     "No SOTA, structural gain, confidence interval, or significance claim from the adaptation gap"],
        statistical_fallacy_scan=audit,
        plotting_source="Original project-specific Python script; no external plotting script copied",
        matplotlib_version=importlib.metadata.version("matplotlib"),
        matplotlib_license=importlib.metadata.metadata("matplotlib").get("License", ""),
        local_modifications="No dependency changes; original plotting and arithmetic validation code only",
        plotting_script_sha256=digest(Path(__file__)), font=str(font_path),
        gpu_work_started=False, training_started=False, source_files_modified=False,
    )
    (HERE / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    caption = (
        "开发集结果展示，非完整论文实验图。\n"
        "图：共同 Wiener 与固定衰减补偿前端下的 SwinJSCC 适配收益与结构净增益。\n"
        "Figure: SwinJSCC adaptation gains and incremental architecture effects with a common Wiener and fixed attenuation-compensation frontend.\n\n"
        "上排将公开权重直接迁移与累计目标信道适配比较，训练历史不同；该差距不能归因于新增结构。\n"
        "下排为同父权重、同本轮训练和选择预算下，全局、局部、注意力修正相对原版的净差。\n"
        "所有模型共用同一前端、图像、信道噪声和传输预算。主图保留三种规定信道及全部 SNR。\n"
        "16 幅已多次用于开发的图像，4 次新噪声，单训练种子；无新图像 holdout、多训练种子或统计显著性结论。\n"
        "所有三种结构候选、负差值与 AWGN 均值保留。未拼接其他论文数字，未选择最佳测试种子。\n"
        "Aged 是 Rayleigh 加额外 CSI 误差且使用当前接收 CSI，不代表实现了反馈延迟补偿。\n"
        "绘图为本项目原创 Matplotlib 脚本，无外部绘图代码复制或依赖修改；依赖版本、许可、输入哈希见 provenance.json。\n"
        "本次仅 CPU 读取既有结果与绘图，无训练、GPU 推理、权重更改、删除或运行状态更改。\n"
        "复现：使用现有 Python 环境运行本目录 plot_results.py。\n"
    )
    (HERE / "CAPTION.txt").write_text(caption, encoding="utf-8")
    assert source_hashes == {str(p): digest(p) for p in source_files}
    print(json.dumps({"rows_verified": len(raw), "means": means,
                      "main_adaptation_gap": manifest["main_adaptation_gap"],
                      "structural_deltas": manifest["structural_deltas"],
                      "png": str(HERE / "psnr_evidence.png"), "pdf": str(HERE / "psnr_evidence.pdf")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
