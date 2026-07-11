"""Render E3 fair-protocol comparison figures from result CSVs.

Reads experiments/results/end_to_end_summary_*.csv and writes to
experiments/figures/:
  e3_fair_main.{pdf,png}        - path_success + semantic_consistency per method
  e3_fair_robustness.{pdf,png}  - path_success vs task reasoning depth (1/2/3 hop)
  e3_fair_efficiency.{pdf,png}  - payload vs path_success (task coverage at cost)

Uses only the stdlib csv reader + matplotlib (no pandas dependency).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

METHODS = [
    "DynamicPruned_Ours",
    "HAKE_StaticLayer_Sun2026",
    "BudgetedNoPruning_iRML_JSAC2023",
]
LABEL = {
    "DynamicPruned_Ours": "Ours",
    "HAKE_StaticLayer_Sun2026": "HAKE (static)",
    "BudgetedNoPruning_iRML_JSAC2023": "NoPruning",
}
COLOR = {
    "DynamicPruned_Ours": "#2ca02c",
    "HAKE_StaticLayer_Sun2026": "#ff7f0e",
    "BudgetedNoPruning_iRML_JSAC2023": "#9467bd",
}

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120})


def _load(tag):
    p = RESULTS / f"end_to_end_summary_{tag}.csv"
    if not p.exists():
        return None
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _row(rows, method):
    if rows is None:
        return None
    for r in rows:
        if r.get("model") == method:
            return r
    return None


def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def fig_main():
    datasets = [("WN18RR", "fair_formal"), ("NELL-995", "fair_formal_nell")]
    avail = [(name, _load(tag)) for name, tag in datasets]
    avail = [(n, d) for n, d in avail if d]
    if not avail:
        print("  [main] no formal summary found, skip")
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5.2 * len(avail), 4.2), squeeze=False)
    for ax, (name, rows) in zip(axes[0], avail):
        present = [m for m in METHODS if _row(rows, m) is not None]
        x = np.arange(len(present))
        w = 0.38
        ps = [_f(_row(rows, m), "path_success_mean") for m in present]
        ps_e = [_f(_row(rows, m), "path_success_std") for m in present]
        sc = [_f(_row(rows, m), "semantic_consistency_mean") for m in present]
        sc_e = [_f(_row(rows, m), "semantic_consistency_std") for m in present]
        b1 = ax.bar(x - w / 2, ps, w, yerr=ps_e, capsize=3, label="path_success", color="#1f77b4")
        b2 = ax.bar(x + w / 2, sc, w, yerr=sc_e, capsize=3, label="semantic_consistency", color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[m] for m in present], rotation=12)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{name} (fair, 5 seeds)")
        ax.set_ylabel("score")
        for b in list(b1) + list(b2):
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("E3 fair comparison: Ours preserves the intent task, baselines do not", y=1.03)
    _save(fig, "e3_fair_main")


def fig_robustness():
    rows_spec = [(1, "fair_h1"), (2, "fair_seeds3"), (3, "fair_h3")]
    hops, ours, ours_e, hake, hake_e = [], [], [], [], []
    for h, tag in rows_spec:
        rows = _load(tag)
        ro, rh = _row(rows, "DynamicPruned_Ours"), _row(rows, "HAKE_StaticLayer_Sun2026")
        if ro is None or rh is None:
            continue
        hops.append(h)
        ours.append(_f(ro, "path_success_mean"))
        ours_e.append(_f(ro, "path_success_std"))
        hake.append(_f(rh, "path_success_mean"))
        hake_e.append(_f(rh, "path_success_std"))
    if not hops:
        print("  [robustness] no data, skip")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(hops, ours, yerr=ours_e, marker="o", capsize=4, lw=2,
                color=COLOR["DynamicPruned_Ours"], label="Ours (connected pruning)")
    ax.errorbar(hops, hake, yerr=hake_e, marker="s", capsize=4, lw=2,
                color=COLOR["HAKE_StaticLayer_Sun2026"], label="HAKE (static layer)")
    ax.set_xticks(hops)
    ax.set_xlabel("task reasoning depth (reference_hops)")
    ax.set_ylabel("path_success")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Task-hop robustness (WN18RR): fair win holds at every depth")
    ax.legend()
    _save(fig, "e3_fair_robustness")


def fig_efficiency():
    rows = _load("fair_formal")
    if not rows:
        print("  [efficiency] no formal summary, skip")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in METHODS:
        r = _row(rows, m)
        if r is None:
            continue
        px, py = _f(r, "payload_kb_mean"), _f(r, "path_success_mean")
        ax.scatter(px, py, s=140, color=COLOR[m], label=LABEL[m], edgecolor="black", zorder=3)
        ax.annotate(LABEL[m], (px, py), textcoords="offset points", xytext=(7, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("payload (KB, log scale)")
    ax.set_ylabel("path_success")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Task coverage vs payload (WN18RR, fair)")
    ax.legend(loc="center right")
    _save(fig, "e3_fair_efficiency")


def main():
    print(f"Rendering E3 fair figures -> {FIGS}")
    fig_main()
    fig_robustness()
    fig_efficiency()
    print("Done.")


if __name__ == "__main__":
    main()
