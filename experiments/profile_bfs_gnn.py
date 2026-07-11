"""
profile_bfs_gnn.py — Profile BFS extraction + GNN encoding latency.

Measures the wall-clock time of the slow-loop's sub-graph extraction pipeline
at different sub-graph sizes, and compares against the coherence-time budget.

Output: experiments/results/bfs_gnn_latency_profile.csv

Usage:
  python experiments/profile_bfs_gnn.py
  python experiments/profile_bfs_gnn.py --samples 200 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.slow_loop.dynamic_kb_pruner import GlobalKnowledgeGraph, DynamicKBPruner
from models.gnn_encoder import SubGraphEncoder, SUB_EMB_DIM
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent

FB15K_DIR = "C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/FB15K237"
QUERIES = [
    "emergency vehicle collision avoidance on highway",
    "drone surveillance with video streaming and camera",
    "autonomous robot navigation with lidar and depth sensor",
    "telemedicine remote surgery with haptic feedback",
    "smart grid power surge detection across substations",
    "pedestrian crossing street radar depth priority",
]


def profile_extraction(kg_dir: str, device: str, samples: int,
                       max_hops: int = 2, max_nodes: int = 64) -> List[Dict]:
    """Profile BFS extraction + GNN encoding latency across multiple queries."""
    kg = GlobalKnowledgeGraph(kg_dir).load()
    extractor = IntentExtractor()
    pruner = DynamicKBPruner(kg, max_subgraph_nodes=max_nodes, max_hops=max_hops,
                             similarity_threshold=0.7, enable_injection=True)
    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=SUB_EMB_DIM).to(device)

    # Warmup
    for _ in range(5):
        resp = extractor.query(QUERIES[0])
        _, entities = parse_intent(resp)
        _ = pruner.update_sub_graph(entities, np.zeros(64, dtype=np.float32))

    results: List[Dict] = []
    for _ in range(samples):
        for qi, query in enumerate(QUERIES):
            resp = extractor.query(query)
            _, entities = parse_intent(resp)

            t0 = time.perf_counter()
            sub_g = pruner.update_sub_graph(entities, np.zeros(64, dtype=np.float32))
            t_bfs = time.perf_counter() - t0

            sg_data = sub_g.to_tensors()
            sg_data["edge_index"] = sg_data["edge_index"].to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                sub_emb = gnn(sg_data)
            if device == "cuda":
                torch.cuda.synchronize()
            t_gnn = time.perf_counter() - t0

            results.append({
                "query_id": qi,
                "num_nodes": len(sub_g.nodes),
                "num_edges": len(sub_g.edges),
                "bfs_extract_ms": round(t_bfs * 1000, 4),
                "gnn_encode_ms": round(t_gnn * 1000, 4),
                "total_ms": round((t_bfs + t_gnn) * 1000, 4),
            })

    return results


def profile_scaling(kg_dir: str, device: str, samples: int) -> List[Dict]:
    """Profile latency vs sub-graph node count N."""
    kg = GlobalKnowledgeGraph(kg_dir).load()
    extractor = IntentExtractor()
    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=SUB_EMB_DIM).to(device)

    results: List[Dict] = []
    for N in [16, 32, 48, 64, 80, 128]:
        pruner = DynamicKBPruner(kg, max_subgraph_nodes=N, max_hops=3,
                                 similarity_threshold=0.7, enable_injection=True)
        # Warm up TransE/entity cache construction and CUDA kernels for each
        # pruner. Without this, the first measured sample per N includes one-off
        # cache initialization and inflates p99 by seconds.
        for warm_idx in range(3):
            resp = extractor.query(QUERIES[warm_idx % len(QUERIES)])
            _, entities = parse_intent(resp)
            sub_g = pruner.update_sub_graph(entities, np.zeros(64, dtype=np.float32))
            sg_data = sub_g.to_tensors()
            sg_data["edge_index"] = sg_data["edge_index"].to(device)
            with torch.no_grad():
                _ = gnn(sg_data)
            if device == "cuda":
                torch.cuda.synchronize()

        for _ in range(samples):
            resp = extractor.query(QUERIES[_ % len(QUERIES)])
            _, entities = parse_intent(resp)

            t0 = time.perf_counter()
            sub_g = pruner.update_sub_graph(entities, np.zeros(64, dtype=np.float32))
            t_bfs = time.perf_counter() - t0

            sg_data = sub_g.to_tensors()
            sg_data["edge_index"] = sg_data["edge_index"].to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = gnn(sg_data)
            if device == "cuda":
                torch.cuda.synchronize()
            t_gnn = time.perf_counter() - t0

            results.append({
                "max_nodes_setting": N,
                "actual_nodes": len(sub_g.nodes),
                "bfs_ms": round(t_bfs * 1000, 4),
                "gnn_ms": round(t_gnn * 1000, 4),
                "total_ms": round((t_bfs + t_gnn) * 1000, 4),
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kg-dir", type=str, default=FB15K_DIR)
    args = parser.parse_args()

    device = args.device
    print(f"Profile BFS+GNN latency | {args.samples} samples | device={device}")
    print("=" * 60)

    # ── Experiment 1: Detailed profile per query ──────────────────────
    print("\n[Exp 1] Per-query latency profile")
    results_detail = profile_extraction(args.kg_dir, device, args.samples)
    bfs_vals = [r["bfs_extract_ms"] for r in results_detail]
    gnn_vals = [r["gnn_encode_ms"] for r in results_detail]
    total_vals = [r["total_ms"] for r in results_detail]
    n_vals = [r["num_nodes"] for r in results_detail]

    print(f"  BFS extract:  mean={np.mean(bfs_vals):.3f} ms, p99={np.percentile(bfs_vals, 99):.3f} ms")
    print(f"  GNN encode:   mean={np.mean(gnn_vals):.3f} ms, p99={np.percentile(gnn_vals, 99):.3f} ms")
    print(f"  TOTAL:        mean={np.mean(total_vals):.3f} ms, p99={np.percentile(total_vals, 99):.3f} ms")
    print(f"  Avg sub-graph: {np.mean(n_vals):.1f} nodes")

    # ── Experiment 2: Scaling vs N ────────────────────────────────────
    print(f"\n[Exp 2] Latency vs sub-graph node count")
    results_scale = profile_scaling(args.kg_dir, device, args.samples // 5)

    from collections import defaultdict
    by_n = defaultdict(list)
    for r in results_scale:
        by_n[r["max_nodes_setting"]].append(r["total_ms"])

    print("  Slow-loop (BFS match + GNN) cost per INTENT update; runs once per")
    print("  intent change, NOT per coherence window -> see Exp 3 for amortized/decision.")
    print(f"  {'N':>5s}  {'Mean (ms)':>10s}  {'P99 (ms)':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}")
    for N in sorted(by_n.keys()):
        vals = by_n[N]
        mean_v = np.mean(vals)
        p99_v = np.percentile(vals, 99)
        print(f"  {N:5d}  {mean_v:10.3f}  {p99_v:10.3f}")

    # ── Energy-efficiency analysis (ILLUSTRATIVE assumptions, not measured) ──
    print(f"\n[Exp 3] Energy-Reward Trade-off  [illustrative assumptions, not measured]")
    # NOTE: reward_gain and the power figures below are ILLUSTRATIVE assumptions for
    # an order-of-magnitude energy estimate, NOT measured here. The BFS+GNN
    # latencies above ARE real measurements; treat Exp 3 as a back-of-envelope only.
    reward_gain = 0.85  # illustrative: injection ON vs OFF at low SNR (cf. E5)
    total_cost_ms = np.mean(total_vals)  # per query

    # Coherence time budget
    Tc_ms = 5.0  # conservative estimate
    budget_used_pct = total_cost_ms / Tc_ms * 100

    # Amortized cost over fast-loop episodes
    episodes_per_query = 200
    amortized_ms = total_cost_ms / episodes_per_query

    print(f"  Reward gain from injection (low SNR):  {reward_gain:.2f}")
    print(f"  BFS+GNN cost per query:               {total_cost_ms:.3f} ms")
    print(f"  Coherence time budget Tc:              {Tc_ms:.1f} ms")
    print(f"  Budget used:                           {budget_used_pct:.1f}%")
    print(f"  Amortized over {episodes_per_query} fast-loop episodes: {amortized_ms:.4f} ms/decision")
    print(f"  Amortized cost vs Tc:                 {amortized_ms/Tc_ms*100:.2f}% of coherence window")

    # EE metric: Δreward / Δenergy
    # Assume GPU power ~200W, CPU extraction power ~50W
    gpu_power_w = 200.0
    cpu_power_w = 50.0
    gnn_energy_j = np.mean(gnn_vals) / 1000 * gpu_power_w  # Joules per query
    bfs_energy_j = np.mean(bfs_vals) / 1000 * cpu_power_w
    total_energy_j = gnn_energy_j + bfs_energy_j
    ee = reward_gain / total_energy_j  # reward per Joule

    print(f"\n  Energy consumption per query:")
    print(f"    BFS (CPU):  {bfs_energy_j*1000:.2f} mJ")
    print(f"    GNN (GPU):  {gnn_energy_j*1000:.2f} mJ")
    print(f"    Total:      {total_energy_j*1000:.2f} mJ")
    print(f"  Energy-reward efficiency: {ee:.2f} reward/Joule")

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "bfs_gnn_latency_profile.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results_detail[0].keys()))
        w.writeheader()
        w.writerows(results_detail)
    print(f"\nSaved detail -> {csv_path}")

    csv_scale = out_dir / "bfs_gnn_scaling.csv"
    with open(csv_scale, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results_scale[0].keys()))
        w.writeheader()
        w.writerows(results_scale)
    print(f"Saved scaling -> {csv_scale}")


if __name__ == "__main__":
    main()
