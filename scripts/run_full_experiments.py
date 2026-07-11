"""Visible full experiment-suite runner for CognitiveSemCom.

Runs E1-E8 with paper-scale defaults, tees every subprocess to the console and
to timestamped logs, and records a machine-readable manifest for later audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = Path(r"D:\anconda\envs\pytorch\python.exe")
E1_SEEDS = [42, 123, 2025]


def python_exe() -> str:
    return str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else sys.executable


def detect_device() -> str:
    code = "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    try:
        out = subprocess.check_output([python_exe(), "-c", code], cwd=ROOT, text=True)
        return out.strip() or "cpu"
    except Exception:
        return "cpu"


def write_line(text: str, log_file) -> None:
    print(text, flush=True)
    log_file.write(text + "\n")
    log_file.flush()


def run_step(
    name: str,
    args: Iterable[str],
    suite_dir: Path,
    manifest: dict,
    continue_on_error: bool,
) -> bool:
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
    log_path = suite_dir / f"{safe_name}.log"
    cmd = [python_exe(), *args]
    start = time.time()

    with log_path.open("w", encoding="utf-8") as log_file:
        write_line("=" * 80, log_file)
        write_line(f"{name}", log_file)
        write_line(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}", log_file)
        write_line(f"Command: {' '.join(cmd)}", log_file)
        write_line("=" * 80, log_file)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            write_line(line.rstrip("\n"), log_file)
        code = proc.wait()

        elapsed = round(time.time() - start, 2)
        write_line("=" * 80, log_file)
        write_line(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}", log_file)
        write_line(f"Exit code: {code}", log_file)
        write_line(f"Elapsed seconds: {elapsed}", log_file)

    manifest["steps"].append({
        "name": name,
        "command": cmd,
        "log": str(log_path),
        "returncode": code,
        "elapsed_sec": elapsed,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_manifest(suite_dir, manifest)

    if code != 0 and not continue_on_error:
        raise RuntimeError(f"{name} failed with exit code {code}")
    return code == 0


def save_manifest(suite_dir: Path, manifest: dict) -> None:
    (suite_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def e1_command(seed: int, save_dir: Path, device: str, epochs: int) -> list[str]:
    return [
        "scripts/train_transe_unified.py",
        "--dataset", "FB15K237",
        "--model", "transE",
        "--epochs", str(epochs),
        "--dim", "128",
        "--seed", str(seed),
        "--device", device,
        "--nbatches", "100",
        "--lr", "1.0",
        "--eval-every", "50",
        "--fast-eval-size", "2000",
        "--eval-max", "0",
        "--save-dir", str(save_dir),
        "--tag-suffix", f"seed{seed}",
    ]


def publish_seed42_embeddings(save_dir: Path, suite_dir: Path) -> None:
    """Expose E1 seed42 outputs at legacy checkpoint paths used by E2."""
    pairs = [
        ("transe_fb15k237_transe_seed42_entity_emb.npy", "transe_fb15k237_entity_emb.npy"),
        ("transe_fb15k237_transe_seed42_relation_emb.npy", "transe_fb15k237_relation_emb.npy"),
        ("transe_fb15k237_transe_seed42_entity2id.json", "transe_fb15k237_entity2id.json"),
        ("transe_fb15k237_transe_seed42_relation2id.json", "transe_fb15k237_relation2id.json"),
        ("transe_fb15k237_transe_seed42_eval.json", "transe_fb15k237_eval.json"),
    ]
    published = []
    for src_name, dst_name in pairs:
        src = save_dir / src_name
        dst = ROOT / "checkpoints" / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            published.append({"src": str(src), "dst": str(dst)})
    (suite_dir / "published_seed42_embeddings.json").write_text(
        json.dumps(published, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize_e1(save_dir: Path, suite_dir: Path) -> None:
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
            "final_train_mrr": history[-1].get("train_mrr") if history else "",
            "final_test_mrr_fast": history[-1].get("test_mrr") if history else "",
            "elapsed_sec": data.get("elapsed_sec"),
        })
    if not rows:
        return
    out = suite_dir / "e1_transe_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_steps(device: str, e1_save_dir: Path, e1_epochs: int) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [
        ("Preflight unit tests", ["scripts/run_unittest_discover.py"]),
        ("Preflight compile checks", ["scripts/run_compile_checks.py"]),
    ]
    for seed in E1_SEEDS:
        steps.append((f"E1 TransE FB15K237 seed{seed}", e1_command(seed, e1_save_dir, device, e1_epochs)))
    steps.extend([
        ("E2 Psi(d) TransE", ["experiments/measure_psi_transE.py", "--num-seeds", "300"]),
        ("E2 N-star scan", ["experiments/eval_nstar_scan.py", "--dataset", "WN18RR", "--seeds", "3", "--device", device, "--output-tag", "fair"]),
        ("E3 End-to-end comparison", ["experiments/eval_end_to_end.py", "--epochs", "500", "--seeds", "5", "--dataset", "WN18RR", "--device", device]),
        ("E4 Latency breakdown", ["experiments/eval_latency.py", "--trials", "500", "--device", device]),
        ("E4 BFS-GNN profile", ["experiments/profile_bfs_gnn.py", "--samples", "100", "--device", device]),
        ("E5 OOD robustness", ["experiments/eval_robustness.py", "--episodes-per-config", "30", "--device", device]),
        ("E6 Multi-seed convergence", ["experiments/eval_convergence.py", "--epochs", "500", "--seeds", "3", "--device", device]),
        ("E7 Lipschitz profile", ["experiments/profile_lipschitz.py"]),
        ("E8 SKB alignment", ["experiments/eval_skb_alignment.py", "--trials", "40"]),
    ])
    return steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, default=None)
    parser.add_argument("--e1-epochs", type=int, default=3000)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = args.suite_dir or (ROOT / "experiments" / "results" / f"full_suite_{stamp}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    e1_save_dir = ROOT / "checkpoints" / f"full_suite_transe_fb15k237_{stamp}"
    e1_save_dir.mkdir(parents=True, exist_ok=True)
    device = detect_device()

    manifest = {
        "suite": "CognitiveSemCom full E1-E8",
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "suite_dir": str(suite_dir),
        "python": python_exe(),
        "device": device,
        "e1_save_dir": str(e1_save_dir),
        "e1_epochs": args.e1_epochs,
        "continue_on_error": args.continue_on_error,
        "steps": [],
    }
    save_manifest(suite_dir, manifest)

    steps = build_steps(device, e1_save_dir, args.e1_epochs)
    print("=" * 80, flush=True)
    print("CognitiveSemCom full experiment suite E1-E8", flush=True)
    print(f"Suite dir: {suite_dir}", flush=True)
    print(f"Device:    {device}", flush=True)
    print(f"E1 save:   {e1_save_dir}", flush=True)
    print("=" * 80, flush=True)

    if args.dry_run:
        for idx, (name, cmd) in enumerate(steps, start=1):
            print(f"[{idx:02d}/{len(steps):02d}] {name}: {python_exe()} {' '.join(cmd)}")
        return 0

    failures = []
    for idx, (name, cmd) in enumerate(steps, start=1):
        print(f"\n[{idx:02d}/{len(steps):02d}] START {name}", flush=True)
        ok = run_step(name, cmd, suite_dir, manifest, args.continue_on_error)
        if name.startswith("E1 TransE FB15K237 seed42"):
            publish_seed42_embeddings(e1_save_dir, suite_dir)
        if name.startswith("E1 TransE"):
            summarize_e1(e1_save_dir, suite_dir)
        if ok:
            print(f"[{idx:02d}/{len(steps):02d}] DONE  {name}", flush=True)
        else:
            print(f"[{idx:02d}/{len(steps):02d}] FAIL  {name}", flush=True)
            failures.append(name)

    summarize_e1(e1_save_dir, suite_dir)
    manifest["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest["failed_steps"] = failures
    save_manifest(suite_dir, manifest)

    print("\n" + "=" * 80, flush=True)
    print("Full experiment suite finished.", flush=True)
    print(f"Failed steps: {len(failures)}", flush=True)
    for name in failures:
        print(f"  - {name}", flush=True)
    print(f"Manifest: {suite_dir / 'manifest.json'}", flush=True)
    print(f"E1 summary: {suite_dir / 'e1_transe_summary.csv'}", flush=True)
    print("=" * 80, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception as exc:
        print(f"\nERROR: {exc}", flush=True)
    finally:
        if "--no-pause" not in sys.argv:
            try:
                input("\nPress Enter to close this window...")
            except EOFError:
                pass
    sys.exit(code)
