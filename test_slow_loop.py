#!/usr/bin/env python
"""Current slow-loop integration check.

Verifies the supported no-LLM pipeline:
  query -> keyword IntentExtractor -> parse_intent -> DynamicKBPruner
        -> SubGraph with semantic node embeddings -> GNN sub_emb[128]
        -> STATE_DIM contract remains 452.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.slow_loop.dynamic_kb_pruner import DynamicKBPruner, GlobalKnowledgeGraph
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from data.dataloader import STATE_DIM
from data.kg_interface import INTENT_DIM
from env.mdp_env import CSI_DIM, NODE_EMB_DIM
from models.gnn_encoder import SUB_EMB_DIM, SubGraphEncoder


TEST_QUERY = (
    "The user is driving and suddenly a pedestrian crosses the street. "
    "High priority on radar and depth."
)
DEFAULT_KG_DIR = "C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/WN18RR"


def has_path(nodes: list[str], edges: list[tuple[str, str, str]], start: str, target: str) -> bool:
    adj = {node: [] for node in nodes}
    for h, _, t in edges:
        if h in adj and t in adj:
            adj[h].append(t)
    queue = deque([start])
    seen = {start}
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for nb in adj.get(node, []):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Slow-loop integration check")
    parser.add_argument("--kg-dir", type=str, default=DEFAULT_KG_DIR)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--sim-threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 72)
    print("Slow-loop integration check: keyword lookup, no LLM/LAM API")
    print("=" * 72)

    extractor = IntentExtractor()
    response = extractor.query(TEST_QUERY)
    intent_vec, entities = parse_intent(response)
    assert intent_vec.shape == (INTENT_DIM,)
    assert intent_vec.dtype == np.float32
    assert len(entities) > 0
    print(f"Intent: {response['primary_intent']} | entities={len(entities)}")

    kg = GlobalKnowledgeGraph(args.kg_dir).load()
    print(f"KG: entities={len(kg.entity2id):,}, edges={kg.graph.number_of_edges():,}")

    pruner = DynamicKBPruner(
        kg,
        max_subgraph_nodes=args.max_nodes,
        max_hops=args.max_hops,
        similarity_threshold=args.sim_threshold,
        enable_injection=True,
    )
    sub_graph = pruner.update_sub_graph(entities, intent_vec)
    assert 0 < len(sub_graph.nodes) <= args.max_nodes
    assert sub_graph.intent_vec.shape == (INTENT_DIM,)
    assert sub_graph.node_embeddings is not None

    start = sub_graph.metadata.get("preferred_start")
    target = sub_graph.metadata.get("preferred_target")
    assert start in sub_graph.nodes
    assert target in sub_graph.nodes
    assert has_path(sub_graph.nodes, sub_graph.edges, start, target) or start == target
    print(
        f"SubGraph: nodes={len(sub_graph.nodes)}, edges={len(sub_graph.edges)}, "
        f"injected={sub_graph.metadata.get('num_injected')}, "
        f"target=({start} -> {target})"
    )

    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=SUB_EMB_DIM).to(args.device)
    sg_data = sub_graph.to_tensors(embedding_dim=128)
    sg_data["edge_index"] = sg_data["edge_index"].to(args.device)
    with torch.no_grad():
        sub_emb = gnn(sg_data)
    assert tuple(sub_emb.shape) == (SUB_EMB_DIM,)
    print(f"GNN sub_emb shape: {tuple(sub_emb.shape)}")

    assert STATE_DIM == NODE_EMB_DIM + SUB_EMB_DIM + INTENT_DIM + CSI_DIM == 452
    print(f"STATE_DIM contract OK: {STATE_DIM}")
    print("=" * 72)
    print("Slow-loop integration check PASSED")


if __name__ == "__main__":
    main()
