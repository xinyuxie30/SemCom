"""Summarize E1 TransE FB15K237 run directories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def latest_run_dir() -> Path:
    runs = sorted((ROOT / "experiments" / "results").glob("e1_transe_fb15k237_*"))
    if not runs:
        raise FileNotFoundError("No e1_transe_fb15k237_* result directory found")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    meta_path = run_dir / "run_meta.json"
    launch_path = run_dir / "launch_meta.json"

    if not meta_path.exists():
        print(f"Run directory exists but training has not written run_meta.json yet: {run_dir}")
        if launch_path.exists():
            print(launch_path.read_text(encoding="utf-8"))
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    save_dir = Path(meta["save_dir"])
    rows = []
    for eval_path in sorted(save_dir.glob("transe_fb15k237_transe_seed*_eval.json")):
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        history = data.get("history", [{}])
        rows.append({
            "file": eval_path.name,
            "seed": data.get("seed"),
            "MRR": data.get("MRR"),
            "Hits@1": data.get("Hits@1"),
            "Hits@3": data.get("Hits@3"),
            "Hits@10": data.get("Hits@10"),
            "epochs_recorded": len(history),
            "final_train_mrr": history[-1].get("train_mrr"),
            "final_test_mrr_fast": history[-1].get("test_mrr"),
            "elapsed_sec": data.get("elapsed_sec"),
        })

    print(f"run_dir={run_dir}")
    print(f"save_dir={save_dir}")
    print(f"started_at={meta.get('started_at')} finished_at={meta.get('finished_at')}")
    print(f"completed_runs={len(meta.get('runs', []))}/{len(meta.get('seeds', []))}")
    for run in meta.get("runs", []):
        print(f"seed={run.get('seed')} returncode={run.get('returncode')} elapsed={run.get('elapsed_sec')}s")

    if not rows:
        print("No completed eval JSON files yet.")
        return

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary={summary_path}")
    for row in rows:
        print(
            f"seed={row['seed']} MRR={row['MRR']} H@10={row['Hits@10']} "
            f"final_fast_test={row['final_test_mrr_fast']} elapsed={row['elapsed_sec']}s"
        )


if __name__ == "__main__":
    main()
