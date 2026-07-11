"""KG semantic consistency metrics for SKB-aligned communication."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np

from data.kg_interface import SubGraph


def kg_semantic_consistency(reference: SubGraph, received: SubGraph) -> Dict[str, float]:
    """Score whether the receiver-side KG sub-graph preserves the intended meaning.

    This is the KG analogue of generation-similarity metrics in generative
    SemCom: it does not compare pixels, but checks whether the semantic support
    needed for reasoning survives transmission/alignment.
    """
    ref_nodes = set(reference.nodes)
    rx_nodes = set(received.nodes)
    ref_edges = {tuple(e) for e in reference.edges}
    rx_edges = {tuple(e) for e in received.edges}

    node_coverage = _safe_ratio(len(ref_nodes & rx_nodes), len(ref_nodes))
    edge_coverage = _safe_ratio(len(ref_edges & rx_edges), len(ref_edges))
    embedding_coverage = _embedding_coverage(reference, received)
    path_connected = 1.0 if _has_start_target_path(received) else 0.0
    seed_coverage = _seed_coverage(reference, received)

    # Weighting rationale: node/edge coverage are the primary measures of whether
    # the reasoning sub-structure survived transmission (highest weight);
    # start->target connectivity is the next-most task-critical; embedding and seed
    # coverage are supporting signals. The ORDERING (node >= edge > path > emb >
    # seed) is the design choice; the absolute values are not tuned to any method.
    # A weight-sensitivity ablation is recommended for the paper — the Ours >
    # baseline ranking is robust to any reasonable reweighting because Ours
    # dominates every individual coverage term (see E3_FAIR_DESIGN).
    score = (
        0.30 * node_coverage
        + 0.25 * edge_coverage
        + 0.20 * path_connected
        + 0.15 * embedding_coverage
        + 0.10 * seed_coverage
    )

    return {
        "semantic_consistency": round(float(score), 6),
        "node_coverage": round(float(node_coverage), 6),
        "edge_coverage": round(float(edge_coverage), 6),
        "path_connected": round(float(path_connected), 6),
        "embedding_coverage": round(float(embedding_coverage), 6),
        "seed_coverage": round(float(seed_coverage), 6),
    }


def should_request_alignment(
    reference: SubGraph,
    received: SubGraph,
    threshold: float = 0.85,
) -> bool:
    """Return True when the receiver-side graph is semantically insufficient."""
    return kg_semantic_consistency(reference, received)["semantic_consistency"] < threshold


def _safe_ratio(num: int, denom: int) -> float:
    if denom <= 0:
        return 1.0
    return float(num) / float(denom)


def _embedding_coverage(reference: SubGraph, received: SubGraph) -> float:
    ref_emb = reference.node_embeddings or {}
    if not ref_emb:
        return 1.0
    rx_emb = received.node_embeddings or {}
    required = set(reference.nodes) & set(ref_emb)
    present = required & set(rx_emb)
    return _safe_ratio(len(present), len(required))


def _seed_coverage(reference: SubGraph, received: SubGraph) -> float:
    seeds = _seed_nodes_from_metadata(reference)
    if not seeds:
        return 1.0
    return _safe_ratio(len(set(received.nodes) & seeds), len(seeds))


def _seed_nodes_from_metadata(sub_graph: SubGraph) -> Set[str]:
    seeds: Set[str] = set()
    for row in sub_graph.metadata.get("match_trace", []):
        if not isinstance(row, dict):
            continue
        if row.get("action") == "match" and row.get("best_match"):
            seeds.add(str(row["best_match"]))
        elif row.get("action") == "inject" and row.get("keyword"):
            seeds.add(str(row["keyword"]))
    if not seeds:
        raw = sub_graph.metadata.get("seed_entities", [])
        seeds.update(str(x) for x in raw if str(x) in sub_graph.nodes)
    return seeds


def _has_start_target_path(sub_graph: SubGraph) -> bool:
    if len(sub_graph.nodes) < 2:
        return bool(sub_graph.nodes)
    start, target = sub_graph.nodes[0], sub_graph.nodes[-1]
    if start == target:
        return True
    adj: Dict[str, List[str]] = {node: [] for node in sub_graph.nodes}
    for h, _, t in sub_graph.edges:
        if h in adj:
            adj[h].append(t)
    queue = deque([start])
    seen = {start}
    while queue:
        node = queue.popleft()
        for nb in adj.get(node, []):
            if nb == target:
                return True
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return False
