r"""Launch E1 TransE FB15K237 runs from the active Python environment.

This avoids PowerShell/cmd quoting issues on Windows. Run this file with the
conda pytorch interpreter:

    D:\anconda\envs\pytorch\python.exe scripts\launch_transe_e1_fb15k237.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 123, 2025]


def make_run_dirs() -> tuple[str, Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = ROOT / "checkpoints" / f"e1_transe_fb15k237_{stamp}"
    log_dir = ROOT / "experiments" / "results" / f"e1_transe_fb15k237_{stamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return stamp, save_dir, log_dir


def train_command(seed: int, save_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "train_transe_unified.py"),
        "--dataset", "FB15K237",
        "--model", "transE",
        "--epochs", "3000",
        "--dim", "128",
        "--seed", str(seed),
        "--device", "cuda",
        "--nbatches", "100",
        "--lr", "1.0",
        "--eval-every", "50",
        "--fast-eval-size", "2000",
        "--eval-max", "0",
        "--save-dir", str(save_dir),
        "--tag-suffix", f"seed{seed}",
    ]


def run_all(save_dir: Path, log_dir: Path) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "python": sys.executable,
        "save_dir": str(save_dir),
        "log_dir": str(log_dir),
        "seeds": SEEDS,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": [],
    }
    (log_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for seed in SEEDS:
        out_path = log_dir / f"transe_fb15k237_seed{seed}.out.log"
        err_path = log_dir / f"transe_fb15k237_seed{seed}.err.log"
        start = time.time()
        with out_path.open("w", encoding="utf-8") as out_fh, err_path.open("w", encoding="utf-8") as err_fh:
            proc = subprocess.run(
                train_command(seed, save_dir),
                cwd=ROOT,
                stdout=out_fh,
                stderr=err_fh,
                text=True,
            )
        meta["runs"].append({
            "seed": seed,
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - start, 2),
            "stdout": str(out_path),
            "stderr": str(err_path),
        })
        (log_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            meta["failed_at_seed"] = seed
            meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            (log_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return proc.returncode

    summary_rows = []
    for eval_path in sorted(save_dir.glob("transe_fb15k237_transe_seed*_eval.json")):
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        history = data.get("history", [{}])
        summary_rows.append({
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

    if summary_rows:
        import csv
        with (log_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (log_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


def launch_background() -> None:
    _, save_dir, log_dir = make_run_dirs()
    child_out = log_dir / "launcher.out.log"
    child_err = log_dir / "launcher.err.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run",
        "--save-dir", str(save_dir),
        "--log-dir", str(log_dir),
    ]
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    with child_out.open("w", encoding="utf-8") as out_fh, child_err.open("w", encoding="utf-8") as err_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=out_fh,
            stderr=err_fh,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    launch_meta = {
        "launcher_pid": proc.pid,
        "python": sys.executable,
        "save_dir": str(save_dir),
        "log_dir": str(log_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (log_dir / "launch_meta.json").write_text(json.dumps(launch_meta, indent=2), encoding="utf-8")
    print(json.dumps(launch_meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()

    if args.run:
        if args.save_dir is None or args.log_dir is None:
            raise SystemExit("--run requires --save-dir and --log-dir")
        raise SystemExit(run_all(args.save_dir, args.log_dir))

    launch_background()


if __name__ == "__main__":
    main()
