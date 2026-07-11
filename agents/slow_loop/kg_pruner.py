"""
Knowledge graph pruner — reduces the global KG to a compact sub-graph
guided by the extracted intent.

Phase 1: random mock pruning. Phase 2: semantic relevance scoring + pruning.

[Phase 1 Fix]: intent_vec dim = INTENT_DIM (64).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from data.kg_interface import INTENT_DIM, SubGraph


def prune_kg(response: dict, global_kg: Dict | None = None,
             max_nodes: int = 64, embedding_dim: int = INTENT_DIM) -> SubGraph:
    """
    Prune the global KG to a sub-graph relevant to the intent.
    """
    sg_data = response.get("sub_graph", {})
    nodes: List[str] = sg_data.get("nodes", [f"mock_{i}" for i in range(min(16, max_nodes))])
    edges_raw = sg_data.get("edges", [])
    edges: List[tuple[str, str, str]] = [
        (str(h), str(r), str(t)) for h, r, t in edges_raw
    ]
    intent_vec = np.asarray(response.get("intent_vec",
                             np.zeros(embedding_dim, dtype=np.float32)))

    return SubGraph(nodes=nodes[:max_nodes], edges=edges, intent_vec=intent_vec)
