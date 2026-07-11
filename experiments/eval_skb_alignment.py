"""Evaluate cloud-edge SKB alignment for KG semantic communication.

This experiment is the KG analogue of Fang et al.'s SKB knowledge alignment:
the receiver has a stale edge-side SKB cache, while the cloud stores the full
semantic payload. We compare no alignment, full retransmission, and our
incremental cloud-edge alignment.

Output:
  experiments/results/skb_alignment.csv
  experiments/results/skb_alignment_summary.csv

Usage:
  python experiments/eval_skb_alignment.py --trials 40
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.slow_loop.dynamic_kb_pruner import DynamicKBPruner, GlobalKnowledgeGraph
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from agents.slow_loop.skb_server import CloudSKBServer, EdgeSKBServer
from data.kg_interface import SubGraph
from utils.semantic_consistency import kg_semantic_consistency


DEFAULT_KG_DIR = "C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/WN18RR"

QUERIES = [
    "High priority on radar and depth for pedestrian crossing street.",
    "Drone surveillance with video streaming and camera monitoring.",
    "Vehicle platoon on highway requires low-latency V2V coordination.",
    "Telemedicine remote surgery with haptic feedback and 8K video.",
    "Smart factory AGV fleet requires lidar collision avoidance.",
    "Smart grid sensor network detects power surge across substation transformers.",
]

STRATEGIES = ["no_align", "full_retransmit", "incremental_align"]


def _estimate_node_id_bytes(kg: GlobalKnowledgeGraph) -> int:
    sample = list(kg.entity2id.keys())[: min(256, len(kg.entity2id))]
    if not sample:
        return 16
    return max(4, int(np.mean([len(x.encode("utf-8")) for x in sample])))


def _make_reference_subgraph(
    pruner: DynamicKBPruner,
    extractor: IntentExtractor,
    query: str,
) -> SubGraph:
    response = extractor.query(query)
    intent_vec, entities = parse_intent(response)
    return pruner.update_sub_graph(entities, intent_vec)


def run_eval(args: argparse.Namespace) -> List[Dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    kg = GlobalKnowledgeGraph(args.kg_dir).load()
    extractor = IntentExtractor()
    pruner = DynamicKBPruner(
        kg,
        max_subgraph_nodes=args.max_subgraph_nodes,
        max_hops=args.max_hops,
        similarity_threshold=args.sim_threshold,
        enable_injection=True,
    )
    cloud = CloudSKBServer(
        node_id_bytes=_estimate_node_id_bytes(kg),
        relation_id_bytes=8,
        embedding_dim=128,
        cloud_base_latency_ms=args.cloud_latency_ms,
        transfer_ms_per_kb=args.transfer_ms_per_kb,
    )

    rows: List[Dict[str, object]] = []
    for trial in range(args.trials):
        query = QUERIES[trial % len(QUERIES)]
        reference = _make_reference_subgraph(pruner, extractor, query)
        if args.progress_every > 0 and (
            trial == 0 or (trial + 1) % args.progress_every == 0 or (trial + 1) == args.trials
        ):
            print(
                f"[{trial + 1:>4}/{args.trials}] "
                f"query={trial % len(QUERIES)} "
                f"ref_nodes={len(reference.nodes)} ref_edges={len(reference.edges)}",
                flush=True,
            )

        stale_edge = EdgeSKBServer(
            capacity_nodes=args.edge_capacity,
            edge_base_latency_ms=args.edge_latency_ms,
        )
        stale_edge.prefill_from_subgraph(
            reference,
            node_keep_prob=args.node_keep_prob,
            edge_keep_prob=args.edge_keep_prob,
            rng=rng,
        )

        for strategy in STRATEGIES:
            edge = stale_edge.clone()
            report = edge.align(reference, cloud, strategy)
            received = edge.materialize(reference)
            metrics = kg_semantic_consistency(reference, received)
            rows.append({
                "trial": trial,
                "query_id": trial % len(QUERIES),
                "strategy": strategy,
                "ref_nodes": len(reference.nodes),
                "ref_edges": len(reference.edges),
                "rx_nodes": len(received.nodes),
                "rx_edges": len(received.edges),
                "kappa": report.kappa,
                "missing_nodes": report.missing_nodes,
                "missing_edges": report.missing_edges,
                "missing_embeddings": report.missing_embeddings,
                "payload_bytes": report.payload_bytes,
                "latency_ms": report.latency_ms,
                **metrics,
            })
    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    numeric = [
        "semantic_consistency",
        "node_coverage",
        "edge_coverage",
        "path_connected",
        "embedding_coverage",
        "seed_coverage",
        "kappa",
        "payload_bytes",
        "latency_ms",
    ]
    by_strategy: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_strategy[str(row["strategy"])].append(row)

    summary: List[Dict[str, object]] = []
    for strategy, vals in by_strategy.items():
        out: Dict[str, object] = {"strategy": strategy, "trials": len(vals)}
        for key in numeric:
            arr = np.asarray([float(v[key]) for v in vals], dtype=np.float64)
            out[f"{key}_mean"] = round(float(arr.mean()), 6)
            out[f"{key}_std"] = round(float(arr.std()), 6)
        summary.append(out)
    return sorted(summary, key=lambda row: str(row["strategy"]))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-dir", type=str, default=DEFAULT_KG_DIR)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-subgraph-nodes", type=int, default=64)
    parser.add_argument("--sim-threshold", type=float, default=0.30,
                        help="Pruner matching threshold; 0.30 follows the E5 operating point. "
                             "Default KG is WN18RR (FB15K237 mids cannot be intent-matched, cf. E3).")
    parser.add_argument("--node-keep-prob", type=float, default=0.65)
    parser.add_argument("--edge-keep-prob", type=float, default=0.55)
    parser.add_argument("--edge-capacity", type=int, default=512)
    parser.add_argument("--edge-latency-ms", type=float, default=1.5)
    parser.add_argument("--cloud-latency-ms", type=float, default=18.0)
    parser.add_argument("--transfer-ms-per-kb", type=float, default=0.08)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    print("KG-SKB cloud-edge alignment simulation")
    print("=" * 64)
    print(f"KG: {args.kg_dir}")
    print(f"Trials: {args.trials} | cache keep nodes={args.node_keep_prob} edges={args.edge_keep_prob}")

    rows = run_eval(args)
    summary = summarize(rows)

    out_dir = Path(__file__).resolve().parent / "results"
    write_csv(out_dir / "skb_alignment.csv", rows)
    write_csv(out_dir / "skb_alignment_summary.csv", summary)

    print(f"{'Strategy':<22} {'Consistency':>12} {'Payload(B)':>12} {'Latency(ms)':>12} {'kappa':>8}")
    for row in summary:
        print(
            f"{row['strategy']:<22} "
            f"{row['semantic_consistency_mean']:>12.4f} "
            f"{row['payload_bytes_mean']:>12.1f} "
            f"{row['latency_ms_mean']:>12.3f} "
            f"{row['kappa_mean']:>8.4f}"
        )
    print(f"Saved -> {out_dir / 'skb_alignment.csv'}")
    print(f"Saved -> {out_dir / 'skb_alignment_summary.csv'}")


if __name__ == "__main__":
    main()
