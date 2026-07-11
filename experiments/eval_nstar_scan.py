"""E2: task-aware optimal pruning radius N* scan (FAIR protocol).

The OLD scan defined the reference task from an N=2 DynamicKBPruner (the method's
own output) and scored with a node_overlap-vs-that-reference term, so "N*=2" was a
self-referential artifact (tautological: use N=2 to define the task, then conclude
N=2 is best).

This FAIR version (matches E3_FAIR_DESIGN_20260624.md):
  - defines a NEUTRAL start->target task on the FULL KG (multi-seed reasoning hop),
    independent of any pruning radius;
  - prunes candidate radii N in {1,2,3} and scores each on the SAME task with
    symmetric metrics (path_success, semantic_consistency, payload) — no term
    references any radius's own output;
  - selects N* = the SMALLEST radius that covers the task (path_success saturates)
    at the lowest complexity. N* therefore emerges from real task coverage.

Default dataset is WN18RR: its synset entities support meaningful intent-keyword
matching (FB15K237 Freebase mids degenerate to noise; see E3_FAIR_DESIGN).

Output: experiments/results/nstar_scan_fair.csv

Usage:
  python experiments/eval_nstar_scan.py --seeds 3 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.eval_end_to_end import (
    DATASET_DIRS, QUERY_POOL, REFERENCE_HOPS,
    load_base_kg, clone_kg, set_seed,
    match_seeds_full_kg, build_ground_truth_task, compute_fair_metrics,
    apply_reference_task,
)
from agents.slow_loop.dynamic_kb_pruner import EntityEmbedder, DynamicKBPruner
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from agents.slow_loop.skb_server import CloudSKBServer

SAT_THRESHOLD = 0.9        # path_success considered "covering" the task
SIM_THRESHOLD = 0.30       # matches E3 Ours config


def scan(dataset: str, radii: List[int], seeds: int) -> List[Dict]:
    base_kg = load_base_kg(dataset)
    shared_embedder = EntityEmbedder(dim=128)
    cloud = CloudSKBServer()
    extractor = IntentExtractor()

    records: List[Dict] = []
    for N in radii:
        for si in range(seeds):
            seed = 42 + si
            set_seed(seed)
            for qi, query in enumerate(QUERY_POOL):
                resp = extractor.query(query)
                intent, entities = parse_intent(resp)
                # Neutral task from the FULL KG (independent of N).
                seeds_kg = match_seeds_full_kg(entities, base_kg, shared_embedder)
                start, target, reference, dist, mode = build_ground_truth_task(
                    base_kg, seeds_kg, intent, shared_embedder, max_hops=REFERENCE_HOPS)
                # Prune candidate radius N (its own clone; injection cannot leak).
                pruner = DynamicKBPruner(clone_kg(base_kg), max_hops=N,
                                         similarity_threshold=SIM_THRESHOLD,
                                         enable_injection=True)
                candidate = pruner.update_sub_graph(entities, intent)
                apply_reference_task(candidate, reference, start, target, dist)
                fair = compute_fair_metrics(candidate, reference, start, target, 0.0, cloud)
                records.append({
                    "radius_N": N,
                    "seed": seed,
                    "query_id": qi,
                    "path_success": fair["path_success"],
                    "semantic_consistency": round(float(fair["semantic_consistency"]), 4),
                    "payload_kb": round(float(fair["payload_kb"]), 3),
                    "num_nodes": len(candidate.nodes),
                    "num_edges": len(candidate.edges),
                    "task_distance": dist,
                    "task_mode": mode,
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WN18RR", choices=list(DATASET_DIRS))
    parser.add_argument("--radii", type=str, default="1,2,3")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-tag", type=str, default="fair")
    args = parser.parse_args()

    radii = [int(x) for x in args.radii.split(",")]
    print("E2 N* scan — FAIR protocol (neutral full-KG task, no self-reference)")
    print(f"  Dataset: {args.dataset} | Radii: {radii} | Seeds: {args.seeds}")
    print(f"  Task: multi-seed {REFERENCE_HOPS}-hop reasoning hop on the full KG")
    print("=" * 72)

    records = scan(args.dataset, radii, args.seeds)

    # Aggregate per radius.
    by_n: Dict[int, List[Dict]] = defaultdict(list)
    for r in records:
        by_n[r["radius_N"]].append(r)
    agg: Dict[int, Dict] = {}
    for N in sorted(by_n):
        rows = by_n[N]
        agg[N] = {
            "path_success": float(np.mean([r["path_success"] for r in rows])),
            "consistency": float(np.mean([r["semantic_consistency"] for r in rows])),
            "payload_kb": float(np.mean([r["payload_kb"] for r in rows])),
            "nodes": float(np.mean([r["num_nodes"] for r in rows])),
        }

    # N* = smallest radius that covers the task (path_success >= SAT), lowest nodes
    # on ties; if none saturate, the best-coverage radius.
    covering = [N for N in sorted(agg) if agg[N]["path_success"] >= SAT_THRESHOLD]
    if covering:
        nstar = min(covering, key=lambda N: agg[N]["nodes"])
        reason = "smallest covering radius (lowest complexity)"
    else:
        nstar = max(sorted(agg), key=lambda N: agg[N]["path_success"])
        reason = "best coverage (none fully saturate)"

    print(f"\n  {'N':>3s}  {'path_success':>12s}  {'consistency':>11s}  "
          f"{'payload_kb':>10s}  {'avg_nodes':>9s}  {'verdict':>10s}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*10}")
    for N in sorted(agg):
        v = "*** N* ***" if N == nstar else ("insufficient" if agg[N]["path_success"] < SAT_THRESHOLD else "redundant")
        print(f"  {N:3d}  {agg[N]['path_success']:12.2f}  {agg[N]['consistency']:11.3f}  "
              f"{agg[N]['payload_kb']:10.1f}  {agg[N]['nodes']:9.1f}  {v:>10s}")
    print(f"\n  N* = {nstar}  ({reason})")
    if nstar == REFERENCE_HOPS:
        print(f"  Confirmed (fair): N*={REFERENCE_HOPS} emerges from real task coverage, "
              f"not a self-referential score.")
    else:
        print(f"  Note: fair scan selects N*={nstar} (task depth ~{REFERENCE_HOPS}); inspect coverage.")
    print("=" * 72)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    csv_path = out_dir / f"nstar_scan{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved -> {csv_path}")


if __name__ == "__main__":
    main()
