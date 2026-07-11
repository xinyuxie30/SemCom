"""
plot_results.py — Phase 5: IEEE publication-ready figures.

Reads convergence_data.csv, latency_breakdown.csv, ood_robustness_snr.csv
and produces three PDF figures for TWC/JSAC submission.

Dependencies: matplotlib, numpy, pandas (all in requirements.txt)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOUR_OURS = "#2166AC"       # deep blue
COLOUR_BASE = "#B2182B"       # deep red
COLOUR_INJ_ON = "#2166AC"
COLOUR_INJ_OFF = "#B2182B"


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: sliding-window moving average
# ═══════════════════════════════════════════════════════════════════════════════

def moving_average(x: np.ndarray, window: int = 4) -> np.ndarray:
    """Simple centred moving average; pads edges with nearest valid value."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    y = np.convolve(x, kernel, mode="same")
    # Fix edges: replace with shorter-window means
    half = window // 2
    for i in range(half):
        y[i] = np.mean(x[:i + half + 1])
        y[-i - 1] = np.mean(x[-i - half - 1:])
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Convergence: G_loss vs Epoch
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence(csv_path: Path, window: int = 4):
    df = pd.read_csv(csv_path)
    # Newer convergence runs store seed-aggregated columns (g_loss_mean/std);
    # older runs store raw per-seed g_loss. Support both so figure generation
    # remains reproducible across experiment versions.
    if "g_loss_mean" in df.columns:
        value_col = "g_loss_mean"
        std_col = "g_loss_std" if "g_loss_std" in df.columns else None
    else:
        value_col = "g_loss"
        std_col = None

    # Keep only rows with valid GAIL updates.
    df = df[df[value_col].notna() & (df[value_col] != "")]
    df[value_col] = df[value_col].astype(float)
    df["epoch"] = df["epoch"].astype(int)

    ours = df[df["model"] == "dynamic_pruned"].sort_values("epoch")
    base = df[df["model"].isin(["budgeted_no_pruning", "static_random_baseline"])].sort_values("epoch")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if len(ours) > window:
        ours_y = moving_average(ours[value_col].values, window)
        ax.plot(ours["epoch"], ours_y,
                color=COLOUR_OURS, linewidth=1.5, label="Dynamic Pruned (Ours)")
        ax.plot(ours["epoch"], ours[value_col].values,
                color=COLOUR_OURS, alpha=0.15, linewidth=0.5)
    else:
        ax.plot(ours["epoch"], ours[value_col].values,
                color=COLOUR_OURS, linewidth=1.5, label="Dynamic Pruned (Ours)")
        ours_y = ours[value_col].values
    if std_col and len(ours) > 0:
        ours_std = ours[std_col].fillna(0).astype(float).values
        ax.fill_between(
            ours["epoch"],
            np.asarray(ours_y) - ours_std,
            np.asarray(ours_y) + ours_std,
            color=COLOUR_OURS,
            alpha=0.08,
            linewidth=0,
        )

    if len(base) > window:
        base_y = moving_average(base[value_col].values, window)
        ax.plot(base["epoch"], base_y,
                color=COLOUR_BASE, linewidth=1.5, linestyle="--",
                label="Budgeted No-Pruning (Baseline)")
        ax.plot(base["epoch"], base[value_col].values,
                color=COLOUR_BASE, alpha=0.15, linewidth=0.5)
    else:
        ax.plot(base["epoch"], base[value_col].values,
                color=COLOUR_BASE, linewidth=1.5, linestyle="--",
                label="Budgeted No-Pruning (Baseline)")
        base_y = base[value_col].values
    if std_col and len(base) > 0:
        base_std = base[std_col].fillna(0).astype(float).values
        ax.fill_between(
            base["epoch"],
            np.asarray(base_y) - base_std,
            np.asarray(base_y) + base_std,
            color=COLOUR_BASE,
            alpha=0.08,
            linewidth=0,
        )

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Generator Loss  $\\mathcal{L}_G$")
    ax.set_title("Convergence Speed: Pruned Sub-graph vs. Budgeted Global KG")
    ax.legend(framealpha=0.9, edgecolor="gray", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=0)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig1_convergence.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Figure 1] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Latency Breakdown (log-scale bar chart)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_latency(csv_path: Path):
    """Real measured latency breakdown + architectural reference comparison.

    Left: measured per-component latency (intent lookup / GNN / policy / channel).
    Right: measured system amortized/decision vs a learned NL extractor whose
    latency is a LITERATURE reference (not measured, not a system component).
    """
    summary = pd.read_csv(csv_path.parent / "latency_summary.csv")

    def comp(name: str) -> float:
        row = summary[summary["component"] == name]
        return float(row["mean_ms"].iloc[0]) if len(row) else float("nan")

    intent = comp("intent_lookup")
    gnn = comp("gnn_encode")
    policy = comp("policy_inference")
    channel = comp("channel")
    system = comp("system_amortized_per_decision")
    learned = comp("learned_extractor_amortized_per_decision")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: measured per-component latency
    comps = ["intent\nlookup", "GNN\nencode", "policy\ninfer", "channel"]
    vals = [intent, gnn, policy, channel]
    bars1 = ax1.bar(comps, vals, color=COLOUR_OURS, edgecolor="black", linewidth=0.6, width=0.6)
    ax1.set_yscale("log")
    ax1.set_ylabel("Latency [ms] (log scale)")
    ax1.set_title("Measured per-component latency")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    for b, v in zip(bars1, vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Right: measured system vs learned-extractor literature reference
    labels = ["CognitiveSemCom\n(measured)", "Learned extractor\n(literature ref)"]
    vals2 = [system, learned]
    bars2 = ax2.bar(labels, vals2, color=[COLOUR_OURS, COLOUR_BASE], edgecolor="black",
                    linewidth=0.6, width=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Amortized per-decision [ms] (log scale)")
    speedup = learned / max(system, 1e-9)
    ax2.set_title(f"Per-decision latency: {speedup:.0f}x lower")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15,
                 f"{v:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.text(0.5, 0.02, "learned extractor = literature reference, NOT measured",
             transform=ax2.transAxes, ha="center", va="bottom", fontsize=8,
             style="italic", color="gray")

    fig.suptitle("E4 Latency: real fast-slow measurements", y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "fig2_latency.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 2] {out}  (measured system/decision = {system:.3f} ms, ref speedup = {speedup:.0f}x)")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — OOD Robustness vs SNR
# ═══════════════════════════════════════════════════════════════════════════════

def plot_robustness(csv_path: Path):
    df = pd.read_csv(csv_path)
    df["snr_db"] = df["snr_db"].astype(int)
    df["cumulative_reward"] = df["cumulative_reward"].astype(float)
    df["fidelity"] = df["fidelity"].astype(float)

    snr_levels = sorted(df["snr_db"].unique())

    inj_on = df[df["injection_enabled"] == True]
    inj_off = df[df["injection_enabled"] == False]

    def agg_by_snr(sub_df):
        grp = sub_df.groupby("snr_db")
        means = grp["cumulative_reward"].mean()
        stds = grp["cumulative_reward"].std()
        fid_means = grp["fidelity"].mean()
        return (np.array([means.get(s, np.nan) for s in snr_levels]),
                np.array([stds.get(s, np.nan) for s in snr_levels]),
                np.array([fid_means.get(s, np.nan) for s in snr_levels]))

    m_on, s_on, _ = agg_by_snr(inj_on)
    m_off, s_off, _ = agg_by_snr(inj_off)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.errorbar(snr_levels, m_on, yerr=s_on, color=COLOUR_INJ_ON,
                marker="o", markersize=8, linewidth=1.8, capsize=6,
                label="Injection ON (Ours)")
    ax.errorbar(snr_levels, m_off, yerr=s_off, color=COLOUR_INJ_OFF,
                marker="s", markersize=8, linewidth=1.8, capsize=6,
                linestyle="--", label="Injection OFF (Baseline)")

    # Annotate key values
    for snr, v_on, v_off in zip(snr_levels, m_on, m_off):
        if not np.isnan(v_on):
            ax.annotate(f"{v_on:.2f}", (snr, v_on),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9, color=COLOUR_INJ_ON)
    for snr, v_on, v_off in zip(snr_levels, m_on, m_off):
        if not np.isnan(v_off):
            ax.annotate(f"{v_off:.2f}", (snr, v_off),
                        textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=9, color=COLOUR_INJ_OFF)

    ax.axhline(y=0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Mean Cumulative Reward")
    ax.set_title("OOD Robustness: Dynamic Injection vs. Static KG\n"
                 "under Channel Degradation")
    # Legend placed below the axes so it never overlaps the inset/data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              framealpha=0.9, edgecolor="gray")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(snr_levels)
    ax.set_xlim(min(snr_levels) - 1.5, max(snr_levels) + 1.5)

    # Inset: fidelity comparison at lowest SNR. Use axes-relative coordinates
    # (ax.inset_axes) and the empty top-left region so it stays clear of both
    # the legend and the data markers.
    ax_inset = ax.inset_axes([0.10, 0.63, 0.33, 0.33])
    fid_on_low = inj_on[inj_on["snr_db"] == snr_levels[0]]["fidelity"].mean()
    fid_off_low = inj_off[inj_off["snr_db"] == snr_levels[0]]["fidelity"].mean()
    xpos = [0, 1]
    ax_inset.bar(xpos, [fid_on_low, fid_off_low],
                 color=[COLOUR_INJ_ON, COLOUR_INJ_OFF], edgecolor="black",
                 linewidth=0.5, width=0.4)
    ax_inset.set_xticks(xpos)
    ax_inset.set_xticklabels(["ON", "OFF"])
    # Headroom so the bar value labels never collide with the inset title.
    fid_vals = [v for v in (fid_on_low, fid_off_low) if not np.isnan(v)]
    fid_top = (max(fid_vals) if fid_vals else 0.01) * 1.45 + 1e-3
    ax_inset.set_ylim(0, fid_top)
    ax_inset.set_title(f"Fidelity @ {snr_levels[0]} dB", fontsize=9, pad=4)
    ax_inset.set_ylabel("Fidelity", fontsize=8)
    ax_inset.tick_params(labelsize=8)
    ax_inset.grid(True, linestyle="--", alpha=0.3, axis="y")
    for b, v in zip(ax_inset.patches, [fid_on_low, fid_off_low]):
        ax_inset.text(b.get_x() + b.get_width() / 2,
                      b.get_height() + fid_top * 0.03,
                      f"{v:.3f}", ha="center", fontsize=8)

    out = OUTPUT_DIR / "fig3_ood_robustness.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Figure 3] {out}")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Phase 5 — IEEE Publication Figures")
    print(f"  Reading CSV from: {RESULTS_DIR}")
    print(f"  Writing PDF to:   {OUTPUT_DIR}")
    print()

    plot_convergence(RESULTS_DIR / "convergence_data.csv")
    plot_latency(RESULTS_DIR / "latency_breakdown.csv")
    plot_robustness(RESULTS_DIR / "ood_robustness_snr.csv")

    print(f"\nDone. Figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
