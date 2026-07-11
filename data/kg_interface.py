"""
KG Interface — decouples G-RML from any static local file dependency.

In Phase 1 this module provides:
  - A typed contract (SubGraph) that the slow loop must satisfy.
  - A mock provider so the fast loop can be developed & tested standalone.
  - A registry for deterministic local sub-graph providers.

Design constraint: the fast loop must NEVER read from txt/csv/json files directly;
all KG data flows through this interface.

[Phase 1 Fix]: intent_dim defaults to 64 (matches config model_dims.intent_dim).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np
import torch

# [Phase 1 Fix]: intent dimension = 64, matching config/default.yaml model_dims.intent_dim
INTENT_DIM = 64


# ── Typed contract ──────────────────────────────────────────────────────────

@dataclass
class SubGraph:
    """
    Macro-pruned sub-graph produced by the slow loop.

    Attributes
    ----------
    nodes : List[str]
        Entity / concept IDs in this sub-graph (≤ max_subgraph_nodes).
    edges : List[tuple[str, str, str]]
        (head, relation, tail) triples.
    intent_vec : np.ndarray
        Dense intent embedding, shape [INTENT_DIM] (64).
    metadata : dict
        Opaque payload (e.g. NL query, confidence, timestamps).
    node_embeddings : dict
        Optional semantic node features keyed by node ID. When present, these
        are consumed by the GNN instead of random initialization.
    """
    nodes: List[str]
    edges: List[tuple[str, str, str]]          # (h, r, t)
    intent_vec: np.ndarray                     # [INTENT_DIM] = [64]
    metadata: Dict = field(default_factory=dict)
    node_embeddings: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.intent_vec, list):
            self.intent_vec = np.asarray(self.intent_vec, dtype=np.float32)
        if self.intent_vec.ndim != 1:
            raise ValueError(f"intent_vec must be 1-D, got {self.intent_vec.shape}")
        if self.intent_vec.shape[0] != INTENT_DIM:
            raise ValueError(f"intent_vec must have length {INTENT_DIM}, got {self.intent_vec.shape[0]}")

    def to_tensors(self, embedding_dim: int = 128) -> Dict[str, object]:
        """
        Convert to a dict of tensors ready for the GNN encoder.

        Returns:
            {
                "node_ids":     List[str],
                "edge_index":   [2, E] torch.long,
                "num_nodes":    int,
                "intent_vec":   [INTENT_DIM] torch.float32,
                "node_embeddings": Optional[[N, embedding_dim] torch.float32],
            }
        """
        n2i = {n: i for i, n in enumerate(self.nodes)}
        head_idx = [n2i[h] for h, _, _ in self.edges]
        tail_idx = [n2i[t] for _, _, t in self.edges]
        edge_index = torch.tensor([head_idx, tail_idx], dtype=torch.long)  # [2, E]
        intent = torch.from_numpy(self.intent_vec).float()                  # [INTENT_DIM]
        node_emb = None
        if self.node_embeddings:
            rows = []
            for node in self.nodes:
                vec = self.node_embeddings.get(node)
                if vec is None:
                    rows.append(np.zeros(embedding_dim, dtype=np.float32))
                    continue
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                if arr.shape[0] < embedding_dim:
                    arr = np.pad(arr, (0, embedding_dim - arr.shape[0]))
                elif arr.shape[0] > embedding_dim:
                    arr = arr[:embedding_dim]
                rows.append(arr.astype(np.float32))
            node_emb = torch.from_numpy(np.stack(rows, axis=0)).float()
        return {
            "node_ids": self.nodes,
            "edge_index": edge_index,
            "num_nodes": len(self.nodes),
            "intent_vec": intent,
            "node_embeddings": node_emb,
        }


# ── Provider protocol ───────────────────────────────────────────────────────

class SubGraphProvider(Protocol):
    """Interface that every local KG or cached sub-graph source must satisfy."""

    def fetch(self, query: str) -> SubGraph:
        ...


# ── Mock provider (Phase 1 development) ─────────────────────────────────────

class MockSubGraphProvider:
    """
    Returns a small fixed sub-graph for Phase 1 standalone testing.
    Main experiments inject pruned SubGraph objects directly through the slow loop.

    [Phase 1 Fix]: intent_vec dim = INTENT_DIM (64).
    """

    def __init__(self, embedding_dim: int = 128, num_nodes: int = 16):
        self.dim = embedding_dim
        self.num_nodes = num_nodes

    def fetch(self, query: str = "") -> SubGraph:
        digest = hashlib.sha256(query.encode("utf-8", errors="surrogatepass")).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**31)
        rng = np.random.default_rng(seed)
        nodes = [f"entity_{i}" for i in range(self.num_nodes)]
        edges = [
            (f"entity_{rng.integers(0, self.num_nodes)}",
             f"rel_{rng.integers(0, 4)}",
             f"entity_{rng.integers(0, self.num_nodes)}")
            for _ in range(self.num_nodes * 2)
        ]
        # [Phase 1 Fix]: intent_vec dim = INTENT_DIM (64)
        intent_vec = rng.normal(0, 1, (INTENT_DIM,)).astype(np.float32)
        return SubGraph(nodes=nodes, edges=edges, intent_vec=intent_vec)


# ── Global registry ─────────────────────────────────────────────────────────

_PROVIDER: Optional[SubGraphProvider] = None


def register_provider(p: SubGraphProvider) -> None:
    global _PROVIDER
    _PROVIDER = p


def get_provider() -> SubGraphProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = MockSubGraphProvider()
    return _PROVIDER
