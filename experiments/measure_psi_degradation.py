"""
measure_psi_degradation.py — Track how OOD injection degrades the semantic
decay function Ψ(d) over time, and verify the self-limiting mechanism.

Key metric: Ψ_eff(d, t) = effective semantic similarity at BFS distance d
after t OOD injections into the KG.

Output: experiments/results/psi_degradation.csv

Usage:
  python experiments/measure_psi_degradation.py --device cpu --max-injections 500
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from agents.slow_loop.dynamic_kb_pruner import (
    GlobalKnowledgeGraph, DynamicKBPruner, get_entity_display_name,
    EntityEmbedder,
)

FB15K_DIR = "C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/FB15K237"


def measure_psi(G: GlobalKnowledgeGraph, seeds: List[str],
                max_hops: int = 6) -> Dict[int, Tuple[float, float]]:
    """
    Measure semantic decay Ψ(d) from `seeds` on the current KG.

    Uses the char n-gram EntityEmbedder as a deterministic semantic proxy.
    Returns {d: (mean_cos_sim, std_cos_sim)} for each BFS distance d.
    """
    embedder = EntityEmbedder(dim=128)

    from collections import deque
    seed_set = set(seeds)
    distances: Dict[str, int] = {}
    for s in seeds:
        if s in G.graph:
            distances[s] = 0

    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        nd = distances[node]
        if nd >= max_hops:
            continue
        for _, nb in G.graph.out_edges(node):
            if nb not in distances:
                distances[nb] = nd + 1
                queue.append(nb)

    # Compute cosine similarity of each reached node to its nearest seed
    by_dist = defaultdict(list)
    seed_embs = {s: embedder.embed(get_entity_display_name(s)) for s in seeds}

    for node, d in distances.items():
        if d == 0:  # seed itself
            by_dist[0].append(1.0)
        else:
            node_emb = embedder.embed(get_entity_display_name(node))
            sims = [float(np.dot(node_emb, seed_embs[s])) for s in seeds]
            by_dist[d].append(max(sims))

    result = {}
    for d in sorted(by_dist.keys()):
        vals = by_dist[d]
        result[d] = (float(np.mean(vals)), float(np.std(vals)))
    return result


def measure_psi_using_entity_names(G: GlobalKnowledgeGraph, seeds: List[str],
                                   max_hops: int = 6
                                   ) -> Dict[int, Tuple[float, float]]:
    """
    Alternative: measure Ψ(d) using character-level string similarity
    between entity display names. This is fully deterministic and
    independent of embedding quality.
    """
    from collections import deque
    seed_set = set(seeds)
    distances: Dict[str, int] = {}
    for s in seeds:
        if s in G.graph:
            distances[s] = 0

    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        if distances[node] >= max_hops:
            continue
        for _, nb in G.graph.out_edges(node):
            if nb not in distances:
                distances[nb] = distances[node] + 1
                queue.append(nb)

    seed_names = {s: set(get_entity_display_name(s).lower().replace('_', ' ').split())
                  for s in seeds}

    def token_overlap(a_tokens: set, b_tokens: set) -> float:
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        return inter / union if union > 0 else 0.0

    by_dist = defaultdict(list)
    for node, d in distances.items():
        if d == 0:
            by_dist[0].append(1.0)
        else:
            node_tokens = set(get_entity_display_name(node).lower().replace('_', ' ').split())
            sims = [token_overlap(node_tokens, seed_names[s]) for s in seeds]
            by_dist[d].append(max(sims))

    result = {}
    for d in sorted(by_dist.keys()):
        vals = by_dist[d]
        result[d] = (float(np.mean(vals)), float(np.std(vals)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-injections", type=int, default=500)
    parser.add_argument("--measure-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("Ψ(d) Degradation Under OOD Injection")
    print("=" * 65)

    # Load clean KG
    G = GlobalKnowledgeGraph(FB15K_DIR).load()
    extractor = IntentExtractor()
    embedder = EntityEmbedder(dim=128)

    # Baseline: measure Ψ(d) on clean KG
    print("\n[Baseline] Measuring Ψ(d) on clean KG...")
    seeds_base = list(G.entity2id.keys())[:20]
    psi_baseline = measure_psi(G, seeds_base, max_hops=6)

    print(f"  {'d':>3s}  {'Ψ(d) mean':>10s}  {'σ':>10s}  {'ΔΨ from d=0':>12s}")
    psi_0 = psi_baseline[0][0]
    for d in sorted(psi_baseline.keys()):
        mean_v, std_v = psi_baseline[d]
        delta = psi_0 - mean_v
        print(f"  {d:3d}  {mean_v:10.4f}  {std_v:10.4f}  {delta:12.4f}")

    # Simulate progressive OOD injection and track Ψ(d) degradation
    print(f"\n[Injection Phase] Injecting OOD entities, measuring every "
          f"{args.measure_every} injections...")

    pruner = DynamicKBPruner(G, similarity_threshold=0.7,
                             enable_injection=True)
    all_results: List[Dict] = []

    for t in range(0, args.max_injections + 1, args.measure_every):
        if t > 0:
            # Inject `measure_every` OOD entities
            for _ in range(args.measure_every):
                # Generate synthetic OOD entities with random names
                ood_name = f"ood_entity_{np.random.randint(0, 1000000)}"
                G.inject_node(ood_name)
                hubs = G.get_top_hub_nodes(k=2)
                for hub in hubs:
                    if hub != ood_name:
                        G.inject_edge(hub, ood_name, relation="dynamic_related_to")
                        G.inject_edge(ood_name, hub, relation="dynamic_related_to")

        # Re-measure Ψ(d) at current injection count
        psi_t = measure_psi(G, seeds_base[:10], max_hops=6)

        psi_base = psi_baseline
        for d in sorted(psi_t.keys()):
            if d in psi_base:
                all_results.append({
                    "injection_count": t,
                    "bfs_distance": d,
                    "psi_mean": round(psi_t[d][0], 6),
                    "psi_std": round(psi_t[d][1], 6),
                    "psi_baseline_mean": round(psi_base[d][0], 6),
                    "degradation_pct": round(
                        max(0, (psi_base[d][0] - psi_t[d][0]) / max(psi_base[d][0], 1e-8)) * 100, 2
                    ),
                    "total_kg_nodes": G.graph.number_of_nodes(),
                })

        if t % 200 == 0:
            # Print snapshot
            print(f"\n  [{t:4d} injections | KG={G.graph.number_of_nodes():,} nodes]")
            for d in [1, 2, 3]:
                if d in psi_t and d in psi_base:
                    deg = (psi_base[d][0] - psi_t[d][0]) / max(psi_base[d][0], 1e-8) * 100
                    print(f"    d={d}: Ψ={psi_t[d][0]:.4f} (degradation={deg:.1f}%)")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Summary: Ψ(d) degradation after {args.max_injections} injections")
    print(f"  KG: {G.graph.number_of_nodes():,} nodes "
          f"(native={len(G.entity2id):,}, injected≈{args.max_injections})")

    # Find the injection count where Ψ(d) drops below threshold
    by_d_last = defaultdict(list)
    for r in all_results:
        if r["injection_count"] == args.max_injections:
            by_d_last[r["bfs_distance"]].append(r["degradation_pct"])

    print(f"\n  Final degradation at each BFS distance:")
    for d in sorted(by_d_last.keys()):
        deg = np.mean(by_d_last[d])
        status = "STABLE" if deg < 20 else ("DEGRADED" if deg < 50 else "COLLAPSED")
        print(f"    d={d}: {deg:.1f}% [{status}]")

    # Critical injection count: where Ψ(1) degrades by 20%
    critical_t = args.max_injections
    for r in all_results:
        if r["bfs_distance"] == 1 and r["degradation_pct"] >= 20:
            critical_t = min(critical_t, r["injection_count"])

    print(f"\n  Critical injection count (Ψ(1) drops 20%): t_crit = {critical_t}")
    print(f"  Recommended recalibration interval: every ~{critical_t} injections")

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "psi_degradation.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
