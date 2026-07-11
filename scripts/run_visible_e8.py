"""Visible E8 runner for PyCharm, cmd, or PowerShell windows.

This script tees subprocess output to both the visible console and a log file,
then waits for Enter before closing. It avoids PowerShell pipeline edge cases
that can turn otherwise normal experiment output into red error text.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(r"D:\anconda\envs\pytorch\python.exe")
LOG_PATH = ROOT / "experiments" / "results" / "e8_visible_run.log"


def _python_exe() -> str:
    return str(PYTHON) if PYTHON.exists() else sys.executable


def _write(line: str, log_file) -> None:
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def run_step(name: str, args: Iterable[str], log_file) -> None:
    cmd = [_python_exe(), *args]
    _write("", log_file)
    _write(f"---- {name} ----", log_file)
    _write(" ".join(cmd), log_file)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _write(line.rstrip("\n"), log_file)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--node-keep-prob", type=float, default=0.65)
    parser.add_argument("--edge-keep-prob", type=float, default=0.55)
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        _write("=" * 72, log_file)
        _write("E8 KG-SKB visible experiment run", log_file)
        _write(f"Project: {ROOT}", log_file)
        _write(f"Python:  {_python_exe()}", log_file)
        _write(f"Trials:  {args.trials}", log_file)
        _write(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}", log_file)
        _write("=" * 72, log_file)

        if not args.skip_checks:
            run_step("Unit tests", ["scripts/run_unittest_discover.py"], log_file)
            run_step("Compile checks", ["scripts/run_compile_checks.py"], log_file)

        run_step(
            "E8 SKB alignment experiment",
            [
                "experiments/eval_skb_alignment.py",
                "--trials",
                str(args.trials),
                "--node-keep-prob",
                str(args.node_keep_prob),
                "--edge-keep-prob",
                str(args.edge_keep_prob),
            ],
            log_file,
        )

        _write("", log_file)
        _write(f"Completed: {datetime.now():%Y-%m-%d %H:%M:%S}", log_file)
        _write(r"Raw CSV: experiments\results\skb_alignment.csv", log_file)
        _write(r"Summary CSV: experiments\results\skb_alignment_summary.csv", log_file)
        _write(f"Visible log: {LOG_PATH}", log_file)
        _write("=" * 72, log_file)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"\nERROR: {exc}", flush=True)
        exit_code = 1
    finally:
        try:
            input("\nPress Enter to close this window...")
        except EOFError:
            pass
    sys.exit(exit_code)
