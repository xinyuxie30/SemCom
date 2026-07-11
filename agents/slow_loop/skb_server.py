"""Cloud/edge semantic knowledge-base simulation.

This module deliberately does not open sockets or deploy real services.  It
models the parts that matter for the paper experiment: edge cache misses,
cloud/edge alignment payload, latency, and knowledge misalignment ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from data.kg_interface import SubGraph

Edge = Tuple[str, str, str]


@dataclass
class AlignmentReport:
    """Metrics produced by one simulated SKB alignment step."""

    strategy: str
    required_nodes: int
    required_edges: int
    missing_nodes: int
    missing_edges: int
    missing_embeddings: int
    kappa: float
    payload_bytes: int
    latency_ms: float


class CloudSKBServer:
    """Authoritative SKB that stores the full KG-side semantic payload."""

    def __init__(
        self,
        node_id_bytes: int = 16,
        relation_id_bytes: int = 8,
        embedding_dim: int = 128,
        cloud_base_latency_ms: float = 18.0,
        transfer_ms_per_kb: float = 0.08,
    ) -> None:
        self.node_id_bytes = int(node_id_bytes)
        self.relation_id_bytes = int(relation_id_bytes)
        self.embedding_dim = int(embedding_dim)
        self.cloud_base_latency_ms = float(cloud_base_latency_ms)
        self.transfer_ms_per_kb = float(transfer_ms_per_kb)

    def estimate_payload_bytes(
        self,
        nodes: Iterable[str],
        edges: Iterable[Edge],
        embeddings: Iterable[str],
    ) -> int:
        node_count = len(list(nodes))
        edge_count = len(list(edges))
        emb_count = len(list(embeddings))
        node_bytes = node_count * self.node_id_bytes
        edge_bytes = edge_count * (2 * self.node_id_bytes + self.relation_id_bytes)
        embedding_bytes = emb_count * self.embedding_dim * 4
        return int(node_bytes + edge_bytes + embedding_bytes)

    def latency_ms(self, payload_bytes: int, contacted: bool = True) -> float:
        if not contacted:
            return 0.0
        return self.cloud_base_latency_ms + (payload_bytes / 1024.0) * self.transfer_ms_per_kb


class EdgeSKBServer:
    """Capacity-limited edge SKB cache used by the receiver."""

    def __init__(
        self,
        capacity_nodes: int = 512,
        edge_base_latency_ms: float = 1.5,
    ) -> None:
        self.capacity_nodes = int(capacity_nodes)
        self.edge_base_latency_ms = float(edge_base_latency_ms)
        self.cached_nodes: Set[str] = set()
        self.cached_edges: Set[Edge] = set()
        self.cached_embeddings: Dict[str, np.ndarray] = {}

    def clone(self) -> "EdgeSKBServer":
        other = EdgeSKBServer(
            capacity_nodes=self.capacity_nodes,
            edge_base_latency_ms=self.edge_base_latency_ms,
        )
        other.cached_nodes = set(self.cached_nodes)
        other.cached_edges = set(self.cached_edges)
        other.cached_embeddings = {k: v.copy() for k, v in self.cached_embeddings.items()}
        return other

    def prefill_from_subgraph(
        self,
        sub_graph: SubGraph,
        node_keep_prob: float,
        edge_keep_prob: float,
        rng: np.random.Generator,
    ) -> None:
        """Seed the edge cache with a controlled stale partial copy."""
        for node in sub_graph.nodes:
            if rng.random() <= node_keep_prob:
                self.cached_nodes.add(node)
                if sub_graph.node_embeddings and node in sub_graph.node_embeddings:
                    self.cached_embeddings[node] = np.asarray(
                        sub_graph.node_embeddings[node], dtype=np.float32
                    ).copy()
        for edge in sub_graph.edges:
            if edge[0] in self.cached_nodes and edge[2] in self.cached_nodes:
                if rng.random() <= edge_keep_prob:
                    self.cached_edges.add(tuple(edge))
        self._enforce_capacity()

    def align(
        self,
        reference: SubGraph,
        cloud: CloudSKBServer,
        strategy: str,
    ) -> AlignmentReport:
        """Apply a no-align/full/incremental alignment policy."""
        if strategy not in {"no_align", "full_retransmit", "incremental_align"}:
            raise ValueError(f"unknown SKB alignment strategy: {strategy}")

        required_nodes = list(reference.nodes)
        required_edges = [tuple(e) for e in reference.edges]
        required_embeddings = set((reference.node_embeddings or {}).keys()) & set(required_nodes)

        missing_nodes = [n for n in required_nodes if n not in self.cached_nodes]
        missing_edges = [e for e in required_edges if e not in self.cached_edges]
        missing_embeddings = [n for n in required_embeddings if n not in self.cached_embeddings]

        denom = max(1, len(required_nodes) + len(required_edges) + len(required_embeddings))
        kappa = (len(missing_nodes) + len(missing_edges) + len(missing_embeddings)) / denom

        if strategy == "no_align":
            payload_nodes: List[str] = []
            payload_edges: List[Edge] = []
            payload_embeddings: List[str] = []
        elif strategy == "full_retransmit":
            payload_nodes = required_nodes
            payload_edges = required_edges
            payload_embeddings = list(required_embeddings)
        else:
            payload_nodes = missing_nodes
            payload_edges = missing_edges
            payload_embeddings = missing_embeddings

        payload_bytes = cloud.estimate_payload_bytes(
            payload_nodes, payload_edges, payload_embeddings
        )

        if strategy != "no_align":
            self._store_payload(reference, payload_nodes, payload_edges, payload_embeddings)

        latency = self.edge_base_latency_ms + cloud.latency_ms(
            payload_bytes, contacted=(strategy != "no_align" and payload_bytes > 0)
        )

        return AlignmentReport(
            strategy=strategy,
            required_nodes=len(required_nodes),
            required_edges=len(required_edges),
            missing_nodes=len(missing_nodes),
            missing_edges=len(missing_edges),
            missing_embeddings=len(missing_embeddings),
            kappa=round(float(kappa), 6),
            payload_bytes=int(payload_bytes),
            latency_ms=round(float(latency), 4),
        )

    def materialize(self, reference: SubGraph) -> SubGraph:
        """Return the receiver-side sub-graph currently available at the edge."""
        nodes = [n for n in reference.nodes if n in self.cached_nodes]
        node_set = set(nodes)
        edges = [
            tuple(e) for e in reference.edges
            if tuple(e) in self.cached_edges and e[0] in node_set and e[2] in node_set
        ]
        node_embeddings = {
            n: self.cached_embeddings[n].copy()
            for n in nodes
            if n in self.cached_embeddings
        }
        return SubGraph(
            nodes=nodes,
            edges=edges,
            intent_vec=reference.intent_vec.copy(),
            metadata={**reference.metadata, "receiver_cache": "edge_skb"},
            node_embeddings=node_embeddings,
        )

    def _store_payload(
        self,
        reference: SubGraph,
        nodes: Iterable[str],
        edges: Iterable[Edge],
        embeddings: Iterable[str],
    ) -> None:
        for node in nodes:
            self.cached_nodes.add(node)
        for edge in edges:
            if edge[0] in self.cached_nodes and edge[2] in self.cached_nodes:
                self.cached_edges.add(tuple(edge))
        source_embeddings = reference.node_embeddings or {}
        for node in embeddings:
            if node in source_embeddings:
                self.cached_embeddings[node] = np.asarray(
                    source_embeddings[node], dtype=np.float32
                ).copy()
                self.cached_nodes.add(node)
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        if len(self.cached_nodes) <= self.capacity_nodes:
            return
        # Deterministic FIFO-like approximation using sorted IDs keeps the
        # simulation reproducible without maintaining extra mutable clocks.
        keep = set(sorted(self.cached_nodes)[-self.capacity_nodes:])
        self.cached_nodes = keep
        self.cached_edges = {
            e for e in self.cached_edges if e[0] in keep and e[2] in keep
        }
        self.cached_embeddings = {
            k: v for k, v in self.cached_embeddings.items() if k in keep
        }
