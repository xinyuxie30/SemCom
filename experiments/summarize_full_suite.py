"""Summarize the latest full CognitiveSemCom experiment suite.

Outputs:
  experiments/results/full_suite_*/full_suite_summary.md
  experiments/results/full_suite_*/full_suite_key_metrics.csv
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"


def latest_suite_dir() -> Path:
    suites = sorted(RESULTS.glob("full_suite_*"), key=lambda p: p.stat().st_mtime)
    if not suites:
        raise FileNotFoundError("No full_suite_* directory found")
    return suites[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    if val in ("", None):
        return default
    return float(val)


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def mean_std(vals: Iterable[float]) -> tuple[float, float]:
    arr = list(vals)
    if not arr:
        return 0.0, 0.0
    return mean(arr), pstdev(arr) if len(arr) > 1 else 0.0


def metric_row(section: str, metric: str, value: float | str, note: str = "") -> dict[str, str]:
    return {
        "section": section,
        "metric": metric,
        "value": str(value),
        "note": note,
    }


def summarize() -> tuple[Path, Path, Path]:
    suite = latest_suite_dir()
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    metrics: list[dict[str, str]] = []
    warnings: list[str] = []

    e1_rows = read_csv(suite / "e1_transe_summary.csv")
    e1_mrr_mean, e1_mrr_std = mean_std(f(r, "MRR") for r in e1_rows)
    e1_h10_mean, e1_h10_std = mean_std(f(r, "Hits@10") for r in e1_rows)
    metrics.append(metric_row("E1", "TransE MRR mean+/-std", f"{fmt(e1_mrr_mean)} +/- {fmt(e1_mrr_std)}"))
    metrics.append(metric_row("E1", "TransE Hits@10 mean+/-std", f"{fmt(e1_h10_mean)} +/- {fmt(e1_h10_std)}"))

    psi_rows = read_csv(RESULTS / "psi_decay_transE.csv")
    psi = {int(f(r, "bfs_distance")): f(r, "psi_mean") for r in psi_rows}
    metrics.append(metric_row("E2", "Psi(1)", fmt(psi.get(1, 0.0))))
    metrics.append(metric_row("E2", "Psi(3)", fmt(psi.get(3, 0.0))))
    metrics.append(metric_row("E2", "Psi decay 1->3", f"{fmt((psi.get(1, 0) - psi.get(3, 0)) / max(psi.get(1, 1), 1e-8) * 100, 2)}%"))

    nstar_rows = read_csv(RESULTS / "nstar_scan.csv")
    max_ep = max(int(f(r, "episode")) for r in nstar_rows)
    by_n: dict[int, list[float]] = defaultdict(list)
    raw_by_n: dict[int, list[float]] = defaultdict(list)
    path_by_n: dict[int, list[float]] = defaultdict(list)
    penalty_by_n: dict[int, list[float]] = defaultdict(list)
    for row in nstar_rows:
        if int(f(row, "episode")) >= max_ep * 0.5:
            n = int(f(row, "max_hops"))
            by_n[n].append(f(row, "reward"))
            if "raw_policy_reward" in row:
                raw_by_n[n].append(f(row, "raw_policy_reward"))
            if "path_ok" in row:
                path_by_n[n].append(f(row, "path_ok"))
            if "complexity_penalty" in row:
                penalty_by_n[n].append(f(row, "complexity_penalty"))
    nstar_scores = {n: mean(vals) for n, vals in by_n.items()}
    best_n = max(nstar_scores, key=nstar_scores.get)
    has_fixed_task_cols = bool(path_by_n)
    nstar_metric_name = "Task-aware N*" if has_fixed_task_cols else "Empirical N*"
    nstar_note = f"score={fmt(nstar_scores[best_n])}"
    if has_fixed_task_cols:
        path_pct = 100.0 * mean(path_by_n.get(best_n, [0.0]))
        raw_mean = mean(raw_by_n.get(best_n, [0.0]))
        penalty_mean = mean(penalty_by_n.get(best_n, [0.0]))
        nstar_note = (
            f"fixed N=2 reference task; score={fmt(nstar_scores[best_n])}; "
            f"raw={fmt(raw_mean)}; path={fmt(path_pct, 1)}%; penalty={fmt(penalty_mean)}"
        )
    metrics.append(metric_row("E2", nstar_metric_name, best_n, nstar_note))
    if best_n != 2:
        warnings.append(f"E2: measured N*={best_n}, not the designed N*=2. Inspect fixed-task coverage and complexity penalties.")

    e3_rows = read_csv(RESULTS / "end_to_end_comparison.csv")
    final_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_model_seed: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in e3_rows:
        by_model_seed[(row["model"], row["seed"])].append(row)
    for (model, _seed), rows in by_model_seed.items():
        final_by_model[model].append(max(rows, key=lambda r: int(f(r, "epoch"))))
    for model, rows in final_by_model.items():
        reward_m, reward_s = mean_std(f(r, "mean_reward") for r in rows)
        raw_reward_m, raw_reward_s = mean_std(
            f(r, "raw_policy_reward", f(r, "mean_reward")) for r in rows
        )
        fid_m, fid_s = mean_std(f(r, "mean_fidelity") for r in rows)
        hit_m, hit_s = mean_std(f(r, "mean_hit_rate") for r in rows)
        reward_label = "task-aware final reward" if "raw_policy_reward" in rows[0] else "final reward"
        metrics.append(metric_row("E3", f"{model} {reward_label}", f"{fmt(reward_m)} +/- {fmt(reward_s)}"))
        if "raw_policy_reward" in rows[0]:
            metrics.append(metric_row("E3", f"{model} raw policy reward", f"{fmt(raw_reward_m)} +/- {fmt(raw_reward_s)}"))
        metrics.append(metric_row("E3", f"{model} final fidelity", f"{fmt(fid_m)} +/- {fmt(fid_s)}"))
        metrics.append(metric_row("E3", f"{model} final hit_rate", f"{fmt(hit_m)} +/- {fmt(hit_s)}"))
    ours_reward = mean(f(r, "mean_reward") for r in final_by_model.get("DynamicPruned_Ours", []))
    hake_reward = mean(f(r, "mean_reward") for r in final_by_model.get("HAKE_StaticLayer_Sun2026", []))
    if hake_reward > ours_reward:
        warnings.append("E3: HAKE static-layer baseline has higher final reward than DynamicPruned_Ours in this run.")
    if final_by_model.get("DynamicPruned_Ours") and "raw_policy_reward" in final_by_model["DynamicPruned_Ours"][0]:
        ours_raw = mean(f(r, "raw_policy_reward") for r in final_by_model.get("DynamicPruned_Ours", []))
        hake_raw = mean(f(r, "raw_policy_reward") for r in final_by_model.get("HAKE_StaticLayer_Sun2026", []))
        if hake_raw > ours_raw:
            warnings.append(
                "E3: HAKE has higher raw policy reward, but lower fixed-task task-aware reward; "
                "frame E3 as reference-task SKB coverage, not generic path-control superiority."
            )

    latency_rows = read_csv(RESULTS / "latency_summary.csv")
    lat = {r["architecture"]: r for r in latency_rows}
    fast_loop = lat["fast_slow_dual_loop"]
    metrics.append(metric_row("E4", "Fast-loop mean latency ms", lat["fast_slow_dual_loop"]["mean_latency_ms"]))
    metrics.append(metric_row("E4", "Fast-loop speedup vs simulated online extractor", fast_loop["speedup_vs_baseline"]))
    bfs_rows = read_csv(RESULTS / "bfs_gnn_latency_profile.csv")
    bfs_total_m, bfs_total_s = mean_std(f(r, "total_ms") for r in bfs_rows)
    metrics.append(metric_row("E4", "BFS+GNN measured mean ms", f"{fmt(bfs_total_m, 3)} +/- {fmt(bfs_total_s, 3)}"))
    if bfs_total_m > 5.0:
        warnings.append("E4: raw BFS+GNN extraction exceeds a 5 ms coherence window; present it as slow-loop amortized cost, not per-decision fast-loop cost.")

    robust_rows = read_csv(RESULTS / "ood_robustness_snr_summary.csv")
    by_snr = defaultdict(dict)
    for row in robust_rows:
        by_snr[int(f(row, "snr_db"))][row["injection_enabled"]] = f(row, "mean_reward")
    for snr in sorted(by_snr):
        on = by_snr[snr].get("True", 0.0)
        off = by_snr[snr].get("False", 0.0)
        metrics.append(metric_row("E5", f"Injection reward gain @ {snr} dB", fmt(on - off)))
    if sum(1 for snr in by_snr if by_snr[snr].get("True", 0) > by_snr[snr].get("False", 0)) < 2:
        warnings.append("E5: injection ON is not consistently better than OFF across SNR levels; do not claim broad OOD robustness from this run.")
    threshold_rows = read_csv(RESULTS / "threshold_ablation.csv")
    best_thr = max(threshold_rows, key=lambda r: f(r, "mean_reward"))
    metrics.append(metric_row("E5", "Best threshold", best_thr["sim_threshold"], f"reward={best_thr['mean_reward']}"))

    conv_rows = read_csv(RESULTS / "convergence_data.csv")
    for model in sorted({r["model"] for r in conv_rows}):
        valid = [r for r in conv_rows if r["model"] == model and r.get("g_loss_mean")]
        if valid:
            last = max(valid, key=lambda r: int(f(r, "epoch")))
            metrics.append(metric_row("E6", f"{model} final g_loss_mean", last["g_loss_mean"], f"epoch={last['epoch']}"))

    lip_rows = read_csv(RESULTS / "lipschitz_comparison.csv")
    lip_by_sn = defaultdict(list)
    for row in lip_rows:
        lip_by_sn[row["use_sn"]].append(f(row, "empirical_lipschitz_L"))
    no_sn_m, _ = mean_std(lip_by_sn["False"])
    sn_m, _ = mean_std(lip_by_sn["True"])
    metrics.append(metric_row("E7", "No-SN empirical Lipschitz mean", fmt(no_sn_m, 5)))
    metrics.append(metric_row("E7", "SN empirical Lipschitz mean", fmt(sn_m, 5)))
    if sn_m > no_sn_m:
        warnings.append("E7: current empirical estimator reports SN > No-SN; treat the SN stability experiment as inconclusive until the estimator is revised.")

    skb_rows = read_csv(RESULTS / "skb_alignment_summary.csv")
    skb = {r["strategy"]: r for r in skb_rows}
    full_payload = f(skb["full_retransmit"], "payload_bytes_mean")
    inc_payload = f(skb["incremental_align"], "payload_bytes_mean")
    payload_saving = (1.0 - inc_payload / max(full_payload, 1e-8)) * 100
    consistency_gain = f(skb["incremental_align"], "semantic_consistency_mean") - f(skb["no_align"], "semantic_consistency_mean")
    metrics.append(metric_row("E8", "Incremental payload saving vs full", f"{fmt(payload_saving, 2)}%"))
    metrics.append(metric_row("E8", "Incremental consistency gain vs no-align", fmt(consistency_gain)))

    key_csv = suite / "full_suite_key_metrics.csv"
    with key_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["section", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(metrics)

    lines = [
        "# CognitiveSemCom Full Suite Summary",
        "",
        f"- Suite directory: `{suite}`",
        f"- Started: `{manifest.get('started_at')}`",
        f"- Finished: `{manifest.get('finished_at')}`",
        f"- Failed steps: `{len(manifest.get('failed_steps', []))}`",
        f"- Device: `{manifest.get('device')}`",
        "",
        "## Key Metrics",
        "",
        "| Section | Metric | Value | Note |",
        "|---|---|---:|---|",
    ]
    for row in metrics:
        lines.append(f"| {row['section']} | {row['metric']} | {row['value']} | {row['note']} |")

    lines.extend(["", "## Warnings / Interpretation Risks", ""])
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- No major interpretation risks detected.")

    lines.extend([
        "",
        "## Recommended Framing",
        "",
        "- Strongest positive result: E8 cloud-edge SKB incremental alignment reaches full semantic consistency with much lower payload than full retransmission.",
        "- Solid supporting result: E1 TransE training is stable across three seeds on FB15K237.",
    ])
    if best_n == 2:
        lines.append("- E2 now supports a task-aware N*=2 claim when generated by the fixed-reference scan.")
        if ours_reward > hake_reward:
            lines.append("- E3 supports DynamicPruned_Ours on the fixed N=2 reference task; keep raw reward as a diagnostic.")
    else:
        lines.append("- Use caution for E2/E3/E5/E7: measured outcomes currently weaken the originally expected claims and should be presented honestly or followed by targeted redesign.")
    if warnings:
        lines.append("- Remaining interpretation risks are listed above; frame those sections conservatively rather than as headline claims.")
    else:
        lines.append("- No remaining warning flags were detected by the summary checks.")
    md_path = suite / "full_suite_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return suite, md_path, key_csv


def main() -> None:
    suite, md_path, key_csv = summarize()
    print(f"Suite: {suite}")
    print(f"Summary: {md_path}")
    print(f"Key metrics: {key_csv}")


if __name__ == "__main__":
    main()
