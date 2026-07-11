"""Summarize E2 task-aware N* scan results.

This script is intentionally read-only for experiment data. Run it after
`experiments/eval_nstar_scan.py` finishes to produce a compact Markdown summary.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
CSV_PATH = RESULTS / "nstar_scan.csv"
OUT_PATH = RESULTS / "e2_nstar_summary.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def summarize(rows: list[dict[str, str]]) -> tuple[int, list[str]]:
    if not rows:
        raise ValueError("nstar_scan.csv has no rows")
    max_episode = max(int(f(row, "episode")) for row in rows)
    late_rows = [row for row in rows if int(f(row, "episode")) >= max_episode * 0.5]
    by_n: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in late_rows:
        by_n[int(f(row, "max_hops"))].append(row)

    summary_rows = []
    best_n = -1
    best_score = -1e18
    has_task_cols = "raw_policy_reward" in rows[0]

    for n in sorted(by_n):
        group = by_n[n]
        scores = [f(row, "reward") for row in group]
        score_mean = mean(scores)
        if score_mean > best_score:
            best_score = score_mean
            best_n = n
        raw_mean = mean(f(row, "raw_policy_reward") for row in group) if has_task_cols else 0.0
        path_pct = 100.0 * mean(f(row, "path_ok") for row in group) if has_task_cols else 0.0
        penalty_mean = mean(f(row, "complexity_penalty") for row in group) if has_task_cols else 0.0
        nodes_mean = mean(f(row, "sub_n") for row in group)
        summary_rows.append(
            {
                "n": n,
                "score_mean": score_mean,
                "score_std": pstdev(scores) if len(scores) > 1 else 0.0,
                "raw_mean": raw_mean,
                "path_pct": path_pct,
                "penalty_mean": penalty_mean,
                "nodes_mean": nodes_mean,
            }
        )

    lines = [
        "# E2 Task-Aware N* Summary",
        "",
        f"- Source: `{CSV_PATH}`",
        f"- Max episode: `{max_episode}`",
        f"- Late-window rows: `{len(late_rows)}`",
        f"- Best N*: `{best_n}`",
        "",
        "| N | Task Score | Std | Raw Reward | Path% | Penalty | Avg Nodes |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {n} | {score_mean:.4f} | {score_std:.4f} | {raw_mean:.4f} | "
            "{path_pct:.1f}% | {penalty_mean:.4f} | {nodes_mean:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if best_n == 2:
        lines.append("- The fixed-reference scan supports the designed `N*=2` claim.")
    else:
        lines.append("- The scan does not yet support `N*=2`; inspect task coverage and penalty balance.")
    if has_task_cols:
        lines.append("- Use `reward` as the task-aware N* score and `raw_policy_reward` as a diagnostic only.")
    return best_n, lines


def main() -> None:
    rows = read_rows(CSV_PATH)
    best_n, lines = summarize(rows)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Best N*: {best_n}")
    print(f"Summary: {OUT_PATH}")


if __name__ == "__main__":
    main()
