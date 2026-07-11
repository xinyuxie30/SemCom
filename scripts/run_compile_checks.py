"""Compile key project modules for a fast PyCharm-visible sanity check."""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    ROOT / "agents" / "slow_loop" / "dynamic_kb_pruner.py",
    ROOT / "agents" / "slow_loop" / "intent_extractor.py",
    ROOT / "agents" / "slow_loop" / "intent_parser.py",
    ROOT / "agents" / "slow_loop" / "mock_expert.py",
    ROOT / "agents" / "slow_loop" / "skb_server.py",
    ROOT / "data" / "dataloader.py",
    ROOT / "data" / "kg_interface.py",
    ROOT / "env" / "mdp_env.py",
    ROOT / "experiments" / "eval_end_to_end.py",
    ROOT / "experiments" / "eval_latency.py",
    ROOT / "experiments" / "eval_nstar_scan.py",
    ROOT / "experiments" / "eval_robustness.py",
    ROOT / "utils" / "semantic_consistency.py",
    ROOT / "experiments" / "eval_skb_alignment.py",
    ROOT / "experiments" / "plot_results.py",
    ROOT / "experiments" / "profile_bfs_gnn.py",
    ROOT / "experiments" / "profile_lipschitz.py",
    ROOT / "experiments" / "summarize_full_suite.py",
    ROOT / "models" / "transe_embedder.py",
    ROOT / "tests" / "test_transe_pipeline.py",
    ROOT / "scripts" / "run_full_experiments.py",
    ROOT / "scripts" / "run_visible_e8.py",
    ROOT / "scripts" / "diagnose_intent_match.py",
    ROOT / "experiments" / "plot_e3_fair.py",
    ROOT / "test_slow_loop.py",
    ROOT / "train.py",
]


def main() -> int:
    print("Compile checks")
    print("=" * 64)
    for path in TARGETS:
        py_compile.compile(str(path), doraise=True)
        print(f"OK  {path.relative_to(ROOT)}")
    print("All compile checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except py_compile.PyCompileError as exc:
        print(exc.msg, file=sys.stderr)
        sys.exit(1)
