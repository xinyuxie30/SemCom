"""
eval_end_to_end.py — End-to-end semantic communication evaluation (FAIR protocol).

Compares three methods on the SAME neutral task with identical, symmetric metrics:
  1. DynamicPruned_Ours      : intent-driven dynamic connected BFS pruning + OOD injection
  2. HAKE_StaticLayer_Sun2026: static hierarchy pruning, seed-aware layer (non-strawman)
  3. BudgetedNoPruning_iRML  : intent-blind budgeted KG slice

Fair protocol (see E3_FAIR_DESIGN_20260624.md):
  - The reference start->target task is defined on the FULL global KG (a multi-seed
    2-hop reasoning hop), NOT by any pruner. No circular task definition.
  - Every method is scored identically with multi-metric, no composite that
    references any method's own output:
      * path_success         : transmitted sub-graph contains a connected start->target path
      * semantic_consistency : kg_semantic_consistency(full-KG reference, transmitted)
      * payload_bytes        : CloudSKBServer payload cost model
      * raw_policy_reward    : natural Semi-MDP reward of the trained policy (no -5.0 floor)
      * efficiency           : success_per_kb / consistency_per_kb
  - No Ours-only step. task_focused_refinement is only available as an
    all-methods ablation behind --task-focus-ablation.

Output: experiments/results/end_to_end_comparison.csv

Usage:
  python experiments/eval_end_to_end.py --epochs 300 --seeds 3 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.kg_interface import MockSubGraphProvider, register_provider, SubGraph, INTENT_DIM
from data.dataloader import ReplayBuffer, EpisodeCollector, STATE_DIM
from env.mdp_env import SemiMDPEnv, EnvConfig, NODE_EMB_DIM, CSI_DIM
from models.gnn_encoder import SubGraphEncoder, SUB_EMB_DIM
from models.generator import Generator, DEFAULT_ACTION_DIM
from models.discriminator import Discriminator
from agents.gail_agent import GAILAgent
from agents.slow_loop.mock_expert import MockExpert
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from agents.slow_loop.dynamic_kb_pruner import (
    EntityEmbedder, GlobalKnowledgeGraph, DynamicKBPruner, get_entity_display_name,
)
from utils.metrics import semantic_fidelity, path_hit_rate
from utils.semantic_consistency import kg_semantic_consistency
from agents.slow_loop.skb_server import CloudSKBServer
from utils.logger import setup_logger

DATA_ROOT = Path("C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks")
DATASET_DIRS = {
    "FB15K237": DATA_ROOT / "FB15K237",
    "WN18RR": DATA_ROOT / "WN18RR",
    "NELL-995": DATA_ROOT / "NELL-995",
}
KG_CACHE: Dict[str, GlobalKnowledgeGraph] = {}
QUERY_POOL = [
    "High priority on radar and depth for pedestrian crossing street.",
    "Drone surveillance with video streaming and camera monitoring.",
    "Vehicle platoon on highway requires low-latency V2V coordination.",
    "Telemedicine remote surgery with haptic feedback and 8K video.",
]
BATCH_SIZE = 48
NUM_EXPERT_EPISODES = 40
EVAL_EPISODES = 20
REFERENCE_HOPS = 2

# Shared display-name embedding matrix per dataset, so seed matching is identical
# across every method (Ours / HAKE / NoPruning).
_SEED_MATCH_CACHE: Dict[str, Tuple[List[str], np.ndarray]] = {}


def load_base_kg(dataset_name: str) -> GlobalKnowledgeGraph:
    """Load each dataset once; pruners receive isolated clones for fairness."""
    if dataset_name not in KG_CACHE:
        KG_CACHE[dataset_name] = GlobalKnowledgeGraph(str(DATASET_DIRS[dataset_name])).load()
    return KG_CACHE[dataset_name]


def clone_kg(base: GlobalKnowledgeGraph) -> GlobalKnowledgeGraph:
    kg = GlobalKnowledgeGraph(str(base.data_dir))
    kg.entity2id = dict(base.entity2id)
    kg.id2entity = dict(base.id2entity)
    kg.relation2id = dict(base.relation2id)
    kg.id2relation = dict(base.id2relation)
    kg.triples = list(base.triples)
    kg.graph = base.graph.copy()
    kg._next_dynamic_id = base._next_dynamic_id
    kg._loaded = base._loaded
    return kg


def stable_int(text: str, modulo: int = 2**31) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
    return int(digest[:12], 16) % modulo


def build_node_embeddings(nodes: List[str], embedder: EntityEmbedder) -> Dict[str, np.ndarray]:
    return {
        node: embedder.embed(get_entity_display_name(node)).astype(np.float32)
        for node in nodes
    }


def choose_reachable_pair(nodes: List[str], edges: List[Tuple[str, str, str]]) -> Tuple[str, str, int]:
    """Pick a reachable start-target pair and return its graph distance."""
    if not nodes:
        return "", "", -1
    adj: Dict[str, List[str]] = {node: [] for node in nodes}
    for h, _, t in edges:
        if h in adj and t in adj and t not in adj[h]:
            adj[h].append(t)
    best_start, best_target, best_dist = nodes[0], nodes[0], 0
    for start in nodes[: min(len(nodes), 16)]:
        queue = [(start, 0)]
        seen = {start}
        while queue:
            node, dist = queue.pop(0)
            if dist > best_dist:
                best_start, best_target, best_dist = start, node, dist
            for nb in adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, dist + 1))
    if best_dist > 0:
        return best_start, best_target, best_dist
    if len(nodes) >= 2:
        return nodes[0], nodes[-1], -1
    return nodes[0], nodes[0], 0


def shortest_distance(
    nodes: List[str],
    edges: List[Tuple[str, str, str]],
    start: str,
    target: str,
) -> Optional[int]:
    if start not in nodes or target not in nodes:
        return None
    adj: Dict[str, List[str]] = {node: [] for node in nodes}
    for h, _, t in edges:
        if h in adj and t in adj and t not in adj[h]:
            adj[h].append(t)
    queue = [(start, 0)]
    seen = {start}
    while queue:
        node, dist = queue.pop(0)
        if node == target:
            return dist
        for nb in adj.get(node, []):
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, dist + 1))
    return None


def shortest_path_nodes(
    nodes: List[str],
    edges: List[Tuple[str, str, str]],
    start: str,
    target: str,
) -> Optional[List[str]]:
    if start not in nodes or target not in nodes:
        return None
    adj: Dict[str, List[str]] = {node: [] for node in nodes}
    for h, _, t in edges:
        if h in adj and t in adj and t not in adj[h]:
            adj[h].append(t)
    queue = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == target:
            return path
        for nb in adj.get(node, []):
            if nb not in seen:
                seen.add(nb)
                queue.append(path + [nb])
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Neutral task on the FULL KG + symmetric multi-metric scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _entity_embedding_matrix(kg: GlobalKnowledgeGraph,
                             embedder: EntityEmbedder) -> Tuple[List[str], np.ndarray]:
    """Cache an L2-normalized display-name embedding matrix for the full KG.

    Shared by every method so seed matching is identical across methods.
    """
    key = str(kg.data_dir)
    cached = _SEED_MATCH_CACHE.get(key)
    if cached is not None:
        return cached
    eid_list = sorted(e for e, i in kg.entity2id.items() if isinstance(i, int) and i >= 0)
    if eid_list:
        mat = np.stack([embedder.embed(get_entity_display_name(e)) for e in eid_list]).astype(np.float32)
    else:
        mat = np.zeros((0, int(getattr(embedder, "dim", 128))), dtype=np.float32)
    _SEED_MATCH_CACHE[key] = (eid_list, mat)
    return eid_list, mat


def match_seeds_full_kg(core_entities: List[str], kg: GlobalKnowledgeGraph,
                        embedder: EntityEmbedder) -> List[str]:
    """Map intent keywords to their best-match FULL-KG entity (shared by all methods)."""
    eid_list, mat = _entity_embedding_matrix(kg, embedder)
    if not eid_list:
        return []
    seeds: List[str] = []
    seen: set = set()
    for ent in core_entities:
        ent_clean = str(ent).lower().strip().replace(" ", "_")
        if not ent_clean:
            continue
        q = embedder.embed(ent_clean)
        sims = mat @ q
        best = eid_list[int(np.argmax(sims))]
        if best not in seen:
            seen.add(best)
            seeds.append(best)
    return seeds


def _bfs_distances(graph, src: str, limit: int) -> Dict[str, int]:
    """Forward BFS distances on the FULL KG up to `limit` hops."""
    dist = {src: 0}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if dist[u] >= limit:
            continue
        for _, v in graph.out_edges(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def build_neighborhood_subgraph(kg: GlobalKnowledgeGraph, seeds: List[str],
                                intent_vec: np.ndarray, embedder: EntityEmbedder,
                                max_nodes: int = 64, max_hops: int = REFERENCE_HOPS) -> SubGraph:
    """Multi-source <=max_hops BFS on the FULL KG -> ground-truth task neighborhood.

    Method-independent: this is the shared semantic-consistency reference and is
    never produced by any pruner.
    """
    graph = kg.graph
    visited: set = set()
    layer: Dict[str, int] = {}
    order: List[str] = []
    edges: List[Tuple[str, str, str]] = []
    dq = deque()
    for s in seeds:
        if s in graph and s not in visited:
            visited.add(s)
            layer[s] = 0
            order.append(s)
            dq.append(s)
    while dq:
        cur = dq.popleft()
        if layer[cur] >= max_hops:
            continue
        for _, nb in graph.out_edges(cur):
            data = graph.get_edge_data(cur, nb)
            rel = data.get("relation", "unknown") if data else "unknown"
            edges.append((cur, rel, nb))
            if nb not in visited:
                visited.add(nb)
                layer[nb] = layer[cur] + 1
                order.append(nb)
                dq.append(nb)
    kept = order[:max_nodes]
    kept_set = set(kept)
    kept_edges = [(h, r, t) for (h, r, t) in edges if h in kept_set and t in kept_set]
    return SubGraph(
        nodes=kept,
        edges=kept_edges,
        intent_vec=intent_vec.astype(np.float32),
        metadata={"seed_entities": list(seeds), "source": "full_kg_ground_truth"},
        node_embeddings=build_node_embeddings(kept, embedder),
    )


def _subgraph_distances(nodes: List[str], edges: List[Tuple[str, str, str]],
                        src: str, limit: int) -> Dict[str, int]:
    """BFS distances from `src` within a sub-graph (edges list), up to `limit` hops."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for h, _, t in edges:
        if h in adj and t in adj:
            adj[h].append(t)
    dist = {src: 0}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if dist[u] >= limit:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def build_ground_truth_task(kg: GlobalKnowledgeGraph, seeds: List[str],
                            intent_vec: np.ndarray, embedder: EntityEmbedder,
                            max_hops: int = REFERENCE_HOPS,
                            max_nodes: int = 64) -> Tuple[str, str, SubGraph, int, str]:
    """Define a NEUTRAL start->target task INSIDE the full-KG reference neighborhood.

    The task target is chosen from the reference sub-graph itself, so the reference
    always contains the task path; path_success then measures whether a method's
    pruning preserves it. (On sparse KGs the reference is the full <=max_hops
    neighborhood; on dense KGs it is the budgeted neighborhood, so the task stays
    well-defined either way.)

    Priority: multi-seed reasoning hop -> single-seed max_hops -> reachable pair.
    Returns (start, target, reference_neighborhood, distance, task_mode).
    """
    graph = kg.graph
    valid_seeds = sorted(s for s in seeds if s in graph)
    reference = build_neighborhood_subgraph(kg, valid_seeds, intent_vec, embedder, max_nodes, max_hops)
    ref_seeds = [s for s in valid_seeds if s in reference.nodes]

    start, target, distance, mode = "", "", -1, "none"
    seed_dists = {s: _subgraph_distances(reference.nodes, reference.edges, s, max_hops)
                  for s in ref_seeds}

    # 1a) seed-to-seed reasoning: two intent seeds connected within the neighborhood.
    #     Both endpoints are seeds, hence present in every method's sub-graph, so
    #     path_success isolates whether the connecting hop survives pruning. This is
    #     robust on dense KGs where a 2-hop target node would be truncated.
    for a in ref_seeds:
        reachable = sorted(b for b in ref_seeds
                           if b != a and 0 < seed_dists[a].get(b, 999) <= max_hops)
        if reachable:
            b = min(reachable, key=lambda n: seed_dists[a][n])
            start, target, distance, mode = a, b, seed_dists[a][b], "seed_to_seed"
            break

    # 1b) multi-seed reasoning hop: target k-hop from one seed and reachable from another
    if not target:
        for a in ref_seeds:
            two_hop = sorted(t for t, d in seed_dists[a].items() if d == max_hops)
            for t in two_hop:
                if any(b != a and t in seed_dists[b] for b in ref_seeds):
                    start, target, distance, mode = a, t, max_hops, "multi_seed"
                    break
            if target:
                break

    # 2) single-seed max_hops, highest global out-degree target
    if not target:
        for a in ref_seeds:
            two_hop = sorted(t for t, d in seed_dists[a].items() if d == max_hops)
            if two_hop:
                t = max(two_hop, key=lambda n: graph.out_degree(n))
                start, target, distance, mode = a, t, max_hops, "single_seed"
                break

    # 3) any reachable pair inside the reference neighborhood
    if not target:
        s, t, d = choose_reachable_pair(reference.nodes, reference.edges)
        start, target, distance, mode = s, t, d, "reachable_pair"

    reference.metadata["preferred_start"] = start
    reference.metadata["preferred_target"] = target
    reference.metadata["task_mode"] = mode
    return start, target, reference, int(distance), mode


def has_path(sub_graph: SubGraph, start: str, target: str) -> bool:
    return shortest_distance(sub_graph.nodes, sub_graph.edges, start, target) is not None


def compute_fair_metrics(received: SubGraph, reference: SubGraph, start: str, target: str,
                         raw_reward: float, cloud: CloudSKBServer) -> Dict[str, float | int]:
    """Symmetric metrics applied identically to every method.

    No term references any method's own pruned output; `reference` is the
    full-KG ground-truth neighborhood shared by all methods.
    """
    cons = kg_semantic_consistency(reference, received)
    path_success = 1.0 if has_path(received, start, target) else 0.0
    emb_keys = list((received.node_embeddings or {}).keys())
    payload = cloud.estimate_payload_bytes(received.nodes, received.edges, emb_keys)
    kb = max(1e-6, payload / 1024.0)
    return {
        "path_success": float(path_success),
        "semantic_consistency": cons["semantic_consistency"],
        "node_coverage": cons["node_coverage"],
        "edge_coverage": cons["edge_coverage"],
        "path_connected": cons["path_connected"],
        "embedding_coverage": cons["embedding_coverage"],
        "seed_coverage": cons["seed_coverage"],
        "payload_bytes": int(payload),
        "payload_kb": round(payload / 1024.0, 3),
        "success_per_kb": round(path_success / kb, 6),
        "consistency_per_kb": round(cons["semantic_consistency"] / kb, 6),
        "raw_policy_reward": float(raw_reward),
    }


def task_focused_refinement(
    sub_graph: SubGraph,
    task_start: str,
    task_target: str,
    max_nodes: int = 24,
    context_nodes: int = 8,
) -> SubGraph:
    """Compact a sub-graph around the reference path.

    Only used as an ALL-METHODS ablation behind --task-focus-ablation, never as
    an Ours-only step in the headline comparison.
    """
    path = shortest_path_nodes(sub_graph.nodes, sub_graph.edges, task_start, task_target)
    if not path or len(path) < 2:
        return sub_graph

    selected: List[str] = []
    for node in path:
        if node not in selected:
            selected.append(node)

    path_edges: List[Tuple[str, str, str]] = []
    for src, dst in zip(path[:-1], path[1:]):
        rel = "task_corridor"
        for h, r, t in sub_graph.edges:
            if h == src and t == dst:
                rel = r
                break
        path_edges.append((src, rel, dst))

    path_set = set(path)
    context_budget = max(0, min(context_nodes, max_nodes - len(selected)))
    for h, _, t in sub_graph.edges:
        if context_budget <= 0 or len(selected) >= max_nodes:
            break
        if h in path_set and t not in selected:
            selected.append(t)
            context_budget -= 1
        if len(selected) >= max_nodes:
            break
        if t in path_set and h not in selected:
            selected.append(h)
            context_budget -= 1

    for node in sub_graph.nodes:
        if context_budget <= 0 or len(selected) >= max_nodes:
            break
        if node not in selected:
            selected.append(node)
            context_budget -= 1

    selected_set = set(selected)
    focused_edges = list(path_edges)
    path_edge_set = {(h, t) for h, _, t in path_edges}
    for h, r, t in sub_graph.edges:
        if h not in selected_set or t not in selected_set:
            continue
        if (h, t) in path_edge_set:
            continue
        if h in path_set:
            continue
        focused_edges.append((h, r, t))
    focused_embeddings = None
    if sub_graph.node_embeddings:
        focused_embeddings = {
            node: emb
            for node, emb in sub_graph.node_embeddings.items()
            if node in selected_set
        }
    metadata = dict(sub_graph.metadata)
    metadata.update({
        "task_focused_refinement": True,
        "pre_refine_nodes": len(sub_graph.nodes),
        "pre_refine_edges": len(sub_graph.edges),
    })
    return SubGraph(
        nodes=selected,
        edges=focused_edges,
        intent_vec=sub_graph.intent_vec,
        metadata=metadata,
        node_embeddings=focused_embeddings,
    )


def apply_reference_task(
    sub_graph: SubGraph,
    reference_graph: SubGraph,
    task_start: str,
    task_target: str,
    reference_distance: int,
) -> None:
    sub_graph.metadata["preferred_start"] = task_start
    sub_graph.metadata["preferred_target"] = task_target
    sub_graph.metadata["reference_distance"] = reference_distance
    sub_graph.metadata["reference_nodes"] = len(reference_graph.nodes)
    sub_graph.metadata["reference_edges"] = len(reference_graph.edges)


def make_subgraph(
    nodes: List[str],
    edges: List[Tuple[str, str, str]],
    intent_vec: np.ndarray,
    embedder: EntityEmbedder,
    metadata: Optional[Dict] = None,
) -> SubGraph:
    start, target, distance = choose_reachable_pair(nodes, edges)
    meta = dict(metadata or {})
    meta.update({
        "preferred_start": start,
        "preferred_target": target,
        "task_distance": distance,
    })
    return SubGraph(
        nodes=nodes,
        edges=edges,
        intent_vec=intent_vec.astype(np.float32),
        metadata=meta,
        node_embeddings=build_node_embeddings(nodes, embedder),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HAKE Baseline: static hierarchy pruning (Sun 2026), seed-aware layer selection
# ═══════════════════════════════════════════════════════════════════════════════

class HAKEBaselinePruner:
    """
    Approximates Sun 2026's static hierarchy pruning using TransE norm.

    Entities are assigned to fixed (query-independent) layers. The high-level
    policy is allowed to SELECT which layer to read (Sun 2026's high-level action),
    so we pick the layer holding the most intent seeds — but the layer ASSIGNMENT
    is static and traversal stays within the selected layer. This makes HAKE a
    fair, non-strawman opponent: it is not blind-rotated, yet its query-independent
    partition still fragments cross-layer reasoning paths.
    """

    def __init__(self, kg: GlobalKnowledgeGraph,
                 transe_embeddings: Optional[np.ndarray] = None,
                 num_layers: int = 3,
                 seed: int = 42):
        self.kg = kg
        self.num_layers = num_layers
        self.layer_assignments: Dict[str, int] = {}
        self.embedder = EntityEmbedder(dim=128)
        self.rng = random.Random(seed)

        if transe_embeddings is not None and transe_embeddings.shape[0] == len(kg.entity2id):
            norms = np.linalg.norm(transe_embeddings, axis=1)
            eid_list = list(kg.entity2id.keys())
            try:
                import pandas as pd
                labels = pd.qcut(norms, num_layers, labels=False, duplicates='drop')
            except ImportError:
                labels = np.digitize(norms, np.percentile(norms, np.linspace(0, 100, num_layers+1)[1:-1]))
            for i, eid in enumerate(eid_list):
                self.layer_assignments[eid] = int(labels[i])
        else:
            # Fallback: deterministic static layer assignment (query-independent).
            for eid in kg.entity2id:
                self.layer_assignments[eid] = self.rng.randint(0, num_layers - 1)

    def _select_active_layer(self, seeds: List[str]) -> int:
        """High-level layer selection: the layer holding the most intent seeds."""
        counts = Counter(
            self.layer_assignments.get(s, 0)
            for s in seeds if s in self.kg.entity2id
        )
        if counts:
            # deterministic tie-break by (count desc, layer asc)
            return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return 0

    def update_sub_graph(self, entities: List[str], intent_vec: np.ndarray,
                         seeds: Optional[List[str]] = None,
                         active_layer: Optional[int] = None,
                         max_hops: int = REFERENCE_HOPS,
                         node_budget: int = 64) -> SubGraph:
        """Static-hierarchy pruning within a seed-selected layer (within-layer BFS)."""
        seeds = list(seeds or [])
        if active_layer is None:
            active_layer = self._select_active_layer(seeds)

        candidates = [eid for eid in self.kg.entity2id
                      if self.layer_assignments.get(eid, 0) == active_layer]
        candidate_set = set(candidates)

        # Prefer the shared seeds that fall inside the selected layer.
        seed_nodes = [s for s in seeds if s in candidate_set]
        n_seeds_in_layer = len(seed_nodes)

        # If no shared seed is in this layer, match intent entities within the layer.
        if not seed_nodes:
            for ent in entities:
                ent_clean = str(ent).lower().strip().replace(' ', '_')
                ent_emb = self.embedder.embed(ent_clean)
                best_sim, best_eid = -1.0, None
                for eid in candidates:
                    sim = float(np.dot(ent_emb, self.embedder.embed(get_entity_display_name(eid))))
                    if sim > best_sim:
                        best_sim, best_eid = sim, eid
                if best_eid and best_eid not in seed_nodes:
                    seed_nodes.append(best_eid)

        # In-layer hub fallback for connectivity.
        if len(seed_nodes) < 3:
            for h in self.kg.get_top_hub_nodes(k=12):
                if self.layer_assignments.get(h, 0) == active_layer and h not in seed_nodes:
                    seed_nodes.append(h)
                if len(seed_nodes) >= 3:
                    break

        # Within-layer BFS (the legitimate static-hierarchy constraint).
        visited = set(seed_nodes)
        visited_order = list(seed_nodes)
        layer_map = {s: 0 for s in seed_nodes}
        edge_order = []
        queue = deque(seed_nodes)
        while queue:
            curr = queue.popleft()
            if layer_map[curr] >= max_hops:
                continue
            for _, nb in self.kg.graph.out_edges(curr):
                if nb in visited:
                    continue
                if self.layer_assignments.get(nb, 0) != active_layer:
                    continue  # stay within the selected static layer
                edge_data = self.kg.graph.get_edge_data(curr, nb)
                r_name = edge_data.get("relation", "unknown") if edge_data else "unknown"
                edge_order.append((curr, r_name, nb))
                visited.add(nb)
                visited_order.append(nb)
                layer_map[nb] = layer_map[curr] + 1
                queue.append(nb)

        kept = visited_order[:node_budget]
        kept_set = set(kept)
        kept_edges = [(h, r, t) for (h, r, t) in edge_order
                      if h in kept_set and t in kept_set]
        return make_subgraph(
            kept,
            kept_edges,
            intent_vec,
            self.embedder,
            metadata={
                "baseline": "HAKE_StaticLayer_Sun2026",
                "active_layer": int(active_layer),
                "n_seeds_in_layer": int(n_seeds_in_layer),
                "seed_entities": entities,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# No-pruning Baseline (iRML: budgeted full-KG slice)
# ═══════════════════════════════════════════════════════════════════════════════

class NoPruningProvider:
    """Returns a budgeted intent-blind slice of the full KG (mimics iRML)."""

    def __init__(self, kg: GlobalKnowledgeGraph, max_nodes: int = 512, seed: int = 42):
        self.kg = kg
        self.max_nodes = max_nodes
        self.seed = seed
        self.embedder = EntityEmbedder(dim=128)
        self._fixed_nodes: Optional[List[str]] = None

    def _select_fixed_nodes(self) -> List[str]:
        if self._fixed_nodes is not None:
            return list(self._fixed_nodes)
        all_nodes = list(self.kg.entity2id.keys())
        hubs = self.kg.get_top_hub_nodes(k=min(64, len(all_nodes)))
        remaining = [node for node in all_nodes if node not in set(hubs)]
        rng = random.Random(self.seed)
        rng.shuffle(remaining)
        self._fixed_nodes = (hubs + remaining)[: min(self.max_nodes, len(all_nodes))]
        return list(self._fixed_nodes)

    def update_sub_graph(self, entities: List[str], intent_vec: np.ndarray,
                         seeds: Optional[List[str]] = None) -> SubGraph:
        # Deliberately not intent-pruned, but deterministic per seed.
        sampled = self._select_fixed_nodes()
        sampled_set = set(sampled)
        edges = []
        for h, t, data in self.kg.graph.edges(data=True):
            if h in sampled_set and t in sampled_set:
                r = data.get("relation", "unknown")
                edges.append((h, r, t))
        return make_subgraph(
            sampled,
            edges,
            intent_vec,
            self.embedder,
            metadata={
                "baseline": "BudgetedNoPruning_iRML_JSAC2023",
                "seed_entities": entities,
                "bounded_full_kg_nodes": self.max_nodes,
            },
        )


class OracleTaskIncludedNoPruningProvider(NoPruningProvider):
    """Diagnostic no-pruning baseline that is allowed to include reference-task nodes.

    Stronger than a blind budgeted slice: controls for the concern that no-pruning
    lost only because its bounded GNN input did not sample the reference task.
    """

    def __init__(self, kg: GlobalKnowledgeGraph, max_nodes: int = 512, seed: int = 42):
        super().__init__(kg=kg, max_nodes=max_nodes, seed=seed)
        self.reference_graph: Optional[SubGraph] = None

    def set_reference_graph(self, reference_graph: SubGraph) -> None:
        self.reference_graph = reference_graph
        self._fixed_nodes = None

    def _select_fixed_nodes(self) -> List[str]:
        if self._fixed_nodes is not None:
            return list(self._fixed_nodes)
        all_nodes = list(self.kg.entity2id.keys())
        ref_nodes = []
        if self.reference_graph is not None:
            for node in self.reference_graph.nodes:
                if node in self.kg.entity2id and node not in ref_nodes:
                    ref_nodes.append(node)
        hubs = [node for node in self.kg.get_top_hub_nodes(k=min(64, len(all_nodes))) if node not in ref_nodes]
        remaining = [node for node in all_nodes if node not in set(ref_nodes) and node not in set(hubs)]
        rng = random.Random(self.seed)
        rng.shuffle(remaining)
        self._fixed_nodes = (ref_nodes + hubs + remaining)[: min(self.max_nodes, len(all_nodes))]
        return list(self._fixed_nodes)

    def update_sub_graph(self, entities: List[str], intent_vec: np.ndarray,
                         seeds: Optional[List[str]] = None) -> SubGraph:
        sub_graph = super().update_sub_graph(entities, intent_vec, seeds=seeds)
        sub_graph.metadata["baseline"] = "OracleTaskIncluded_NoPruning"
        sub_graph.metadata["oracle_reference_nodes"] = len(self.reference_graph.nodes) if self.reference_graph else 0
        return sub_graph


# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prune_with_method(pruner, entities: List[str], intent_vec: np.ndarray,
                      seeds: List[str], reference_hops: int = REFERENCE_HOPS,
                      node_budget: int = 64) -> SubGraph:
    """Uniform pruning entry-point. HAKE additionally consumes the shared seeds
    for seed-aware layer selection, matches the task reasoning depth, and shares
    the node budget; other methods ignore them (Ours' budget/depth are set on the
    pruner instance)."""
    if isinstance(pruner, HAKEBaselinePruner):
        return pruner.update_sub_graph(entities, intent_vec, seeds=seeds,
                                       max_hops=reference_hops, node_budget=node_budget)
    return pruner.update_sub_graph(entities, intent_vec)


def evaluate_episode(env, collector, sub_graph, intent_vec, generator, task_target: str,
                     deterministic: bool = False) -> dict:
    """Roll out the trained policy on the transmitted sub-graph.

    No artificial -5.0 floor: a sub-graph that does not contain the neutral task
    simply cannot reach it, so the natural Semi-MDP reward stays low (method-blind).
    `task_goal` is True only when the policy actually walks to the neutral target.
    """
    if not deterministic:
        traj = collector.collect(sub_graph=sub_graph, intent_vec_override=intent_vec)
        if not traj:
            return {"raw_reward": 0.0, "env_goal": False, "task_goal": False, "hops": 0}
        cumulative = sum(e.reward for e in traj)
        final_path = list(getattr(env, "_path_history", []))
        env_goal = any(e.reward > 5.0 for e in traj)
        task_goal = bool(task_target) and (task_target in final_path)
        return {"raw_reward": round(cumulative, 4), "env_goal": env_goal,
                "task_goal": task_goal, "hops": len(traj)}

    sub_graph_data = sub_graph.to_tensors()
    sub_graph_data["edge_index"] = sub_graph_data["edge_index"].to(collector.device)
    with torch.no_grad():
        sub_emb = collector.gnn_encoder(sub_graph_data).detach().cpu()

    intent_vec_np = np.asarray(intent_vec, dtype=np.float32)
    intent_vec_t = torch.from_numpy(intent_vec_np).float()
    env_state = env.reset(sub_graph=sub_graph, intent_vec=intent_vec_np)

    cumulative = 0.0
    env_goal = False
    final_path: List[str] = list(env_state.get("path_history", []))
    hops = 0
    for _ in range(collector.max_hops):
        state = collector._build_state(env_state, sub_emb, intent_vec_t)
        state_batch = state.unsqueeze(0).to(collector.device)
        with torch.no_grad():
            action_t, _ = generator.get_action(state_batch, deterministic=True)
        next_env_state, reward, done, info = env.step(int(action_t.item()))
        cumulative += float(reward)
        hops += 1
        final_path = list(next_env_state.get("path_history", final_path))
        if info.get("reason") == "goal_reached":
            env_goal = True
        env_state = next_env_state
        if done:
            break

    task_goal = bool(task_target) and (task_target in final_path)
    return {"raw_reward": round(cumulative, 4), "env_goal": env_goal,
            "task_goal": task_goal, "hops": hops}


def run_training_and_eval(label: str, pruner, task_kg: GlobalKnowledgeGraph,
                          shared_embedder: EntityEmbedder, device: str, epochs: int,
                          seed: int, node_emb_dim: int, state_dim: int,
                          action_dim: int, sub_emb_dim: int, intent_dim: int,
                          csi_dim: int, eval_mode: str = "stochastic",
                          reference_hops: int = REFERENCE_HOPS,
                          node_budget: int = 64,
                          task_focus_ablation: bool = False,
                          task_focus_nodes: int = 24,
                          task_context_nodes: int = 8) -> List[Dict]:
    """Train GAIL on each method's transmitted sub-graph and score it on the
    shared neutral full-KG task with symmetric multi-metrics."""
    set_seed(seed)
    log = setup_logger()
    cloud = CloudSKBServer()

    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=sub_emb_dim).to(device)
    env_cfg = EnvConfig(max_hops=8, action_dim=action_dim,
                        node_emb_dim=node_emb_dim, csi_dim=csi_dim)
    env = SemiMDPEnv(env_cfg)
    gen = Generator(state_dim=state_dim, action_dim=action_dim, use_sn=True).to(device)
    disc = Discriminator(state_dim=state_dim, action_dim=action_dim).to(device)
    agent = GAILAgent(gen, disc, gen_lr=3e-4, disc_lr=1e-4,
                      entropy_coef=0.01, device=device)
    expert = MockExpert(gnn_encoder=gnn, node_emb_dim=node_emb_dim, device=device)
    mock_prov = MockSubGraphProvider(embedding_dim=128)
    register_provider(mock_prov)
    collector = EpisodeCollector(env=env, provider=mock_prov, gnn_encoder=gnn,
                                 state_dim=state_dim, max_hops=8,
                                 generator=gen, device=device)
    extractor = IntentExtractor()

    def build_task(query: str):
        resp = extractor.query(query)
        intent_local, entities_local = parse_intent(resp)
        seeds = match_seeds_full_kg(entities_local, task_kg, shared_embedder)
        start, target, reference, dist, mode = build_ground_truth_task(
            task_kg, seeds, intent_local, shared_embedder,
            max_hops=reference_hops, max_nodes=node_budget)
        if hasattr(pruner, "set_reference_graph"):
            pruner.set_reference_graph(reference)
        candidate = prune_with_method(pruner, entities_local, intent_local, seeds,
                                      reference_hops, node_budget)
        apply_reference_task(candidate, reference, start, target, dist)
        if task_focus_ablation and task_focus_nodes > 0:
            candidate = task_focused_refinement(
                candidate, start, target,
                max_nodes=task_focus_nodes, context_nodes=task_context_nodes)
            apply_reference_task(candidate, reference, start, target, dist)
        return intent_local, entities_local, seeds, candidate, reference, start, target, dist, mode

    # Initial task
    (intent_vec, entities, seeds, sub_graph, reference_graph,
     task_start, task_target, reference_distance, task_mode) = build_task(QUERY_POOL[0])
    expert_buf = ReplayBuffer(capacity=10000)
    for _ in range(NUM_EXPERT_EPISODES):
        traj = expert.sample_trajectory(sub_graph)
        if traj:
            expert_buf.push_batch(traj)
    gen_buf = ReplayBuffer(capacity=10000)

    eval_interval = max(epochs // 10, 10)
    records: List[Dict] = []

    for ep in range(epochs):
        if ep > 0 and ep % 15 == 0:
            qi = (ep // 15) % len(QUERY_POOL)
            try:
                (intent_vec, entities, seeds, sub_graph, reference_graph,
                 task_start, task_target, reference_distance, task_mode) = build_task(QUERY_POOL[qi])
                expert_buf = ReplayBuffer(capacity=10000)
                for _ in range(NUM_EXPERT_EPISODES):
                    traj = expert.sample_trajectory(sub_graph)
                    if traj:
                        expert_buf.push_batch(traj)
            except Exception as exc:
                log.warning(f"[E3] task refresh failed, reusing previous task: {exc}")

        traj = collector.collect(sub_graph=sub_graph, intent_vec_override=intent_vec)
        gen_buf.push_batch(traj)

        if len(gen_buf) >= BATCH_SIZE and len(expert_buf) >= BATCH_SIZE:
            gb_s = gen_buf.to_tensors(gen_buf.sample(BATCH_SIZE))
            eb_s = expert_buf.to_tensors(expert_buf.sample(BATCH_SIZE))
            agent.update_discriminator(gb_s, eb_s)
            for _ in range(2):
                gb = gen_buf.to_tensors(gen_buf.sample(BATCH_SIZE))
                agent.update_generator(gb)

        if ep % eval_interval == 0 or ep == epochs - 1:
            evals = [
                evaluate_episode(env, collector, sub_graph, intent_vec, gen, task_target,
                                 deterministic=(eval_mode == "deterministic"))
                for _ in range(EVAL_EPISODES)
            ]
            raw_mean = float(np.mean([e["raw_reward"] for e in evals]))
            raw_std = float(np.std([e["raw_reward"] for e in evals]))
            env_goal_rate = float(np.mean([1 if e["env_goal"] else 0 for e in evals]))
            task_goal_rate = float(np.mean([1 if e["task_goal"] else 0 for e in evals]))
            mean_hops = float(np.mean([e["hops"] for e in evals]))

            fair = compute_fair_metrics(sub_graph, reference_graph, task_start, task_target,
                                        raw_mean, cloud)

            records.append({
                "epoch": ep, "model": label, "seed": seed,
                "path_success": round(fair["path_success"], 4),
                "task_goal_rate": round(task_goal_rate, 4),
                "semantic_consistency": round(float(fair["semantic_consistency"]), 4),
                "node_coverage": round(float(fair["node_coverage"]), 4),
                "edge_coverage": round(float(fair["edge_coverage"]), 4),
                "path_connected": round(float(fair["path_connected"]), 4),
                "embedding_coverage": round(float(fair["embedding_coverage"]), 4),
                "seed_coverage": round(float(fair["seed_coverage"]), 4),
                "payload_bytes": int(fair["payload_bytes"]),
                "payload_kb": round(float(fair["payload_kb"]), 3),
                "success_per_kb": round(float(fair["success_per_kb"]), 6),
                "consistency_per_kb": round(float(fair["consistency_per_kb"]), 6),
                "raw_policy_reward": round(raw_mean, 4),
                "raw_policy_std": round(raw_std, 4),
                "env_goal_rate": round(env_goal_rate, 4),
                "mean_hops": round(mean_hops, 4),
                "sub_graph_nodes": len(sub_graph.nodes),
                "sub_graph_edges": len(sub_graph.edges),
                "active_layer": int(sub_graph.metadata.get("active_layer", -1)),
                "n_seeds_in_layer": int(sub_graph.metadata.get("n_seeds_in_layer", -1)),
                "num_shared_seeds": len(seeds),
                "task_start": task_start,
                "task_target": task_target,
                "reference_distance": reference_distance,
                "task_mode": task_mode,
                "reference_hops": reference_hops,
                "node_budget": node_budget,
                "task_focused_refinement": int(bool(task_focus_ablation)),
            })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="WN18RR",
                        help="Single dataset name. Ignored when --datasets is provided. "
                             "WN18RR is the fair-protocol default: its synset entities support "
                             "meaningful intent-keyword matching, unlike FB15K237 Freebase mids.")
    parser.add_argument("--datasets", type=str, default="",
                        help="Comma-separated datasets or 'all'. Options: FB15K237, WN18RR, NELL-995.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ours-max-hops", type=int, default=2,
                        help="DynamicPruned depth. Superseded by --reference-hops (Ours depth follows task depth).")
    parser.add_argument("--reference-hops", type=int, default=2,
                        help="Neutral task reasoning depth (k-hop). Ours and HAKE pruning depth both "
                             "follow this, so the comparison is at equal hop budget. Use 1/2/3 for the "
                             "task-hop robustness sweep; 2 is the E2 N*=2 default.")
    parser.add_argument("--node-budget", type=int, default=64,
                        help="Max nodes in the transmitted sub-graph AND the reference neighborhood, "
                             "shared by Ours/HAKE/reference (symmetric). Increase (128/256) for dense "
                             "KGs like NELL-995 where 64 truncates the task neighborhood.")
    parser.add_argument("--ours-sim-threshold", type=float, default=0.30,
                        help="DynamicPruned matching threshold. Default follows E5 threshold scan.")
    parser.add_argument("--eval-mode", choices=["stochastic", "deterministic"],
                        default="stochastic",
                        help="Report expected stochastic policy performance or argmax-policy performance.")
    parser.add_argument("--output-tag", type=str, default="",
                        help="Optional suffix for smoke/debug outputs, e.g. smoke.")
    parser.add_argument("--noprune-max-nodes", type=int, default=512,
                        help="Bounded GNN input size for no-pruning baselines.")
    parser.add_argument("--task-focus-ablation", action="store_true",
                        help="Apply task-focused refinement to ALL methods (symmetric ablation, off in headline).")
    parser.add_argument("--task-focus-nodes", type=int, default=24,
                        help="Max nodes kept by task-focused refinement (ablation only).")
    parser.add_argument("--task-context-nodes", type=int, default=8,
                        help="One-hop context nodes retained around the task corridor (ablation only).")
    parser.add_argument("--include-oracle-noprune", action="store_true",
                        help="Include a stronger diagnostic no-pruning baseline that is given reference-task nodes.")
    args = parser.parse_args()
    device = args.device

    nd, sd, ad = NODE_EMB_DIM, STATE_DIM, DEFAULT_ACTION_DIM
    subd, intd, csid = SUB_EMB_DIM, INTENT_DIM, CSI_DIM
    if args.datasets:
        dataset_names = list(DATASET_DIRS) if args.datasets.lower() == "all" else [
            item.strip() for item in args.datasets.split(",") if item.strip()
        ]
    else:
        dataset_names = [args.dataset]
    unknown = [name for name in dataset_names if name not in DATASET_DIRS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Valid: {sorted(DATASET_DIRS)}")

    protocol = f"fair_neutral_full_kg_task_h{args.reference_hops}_b{args.node_budget}" + ("_taskfocus_all" if args.task_focus_ablation else "")
    print(f"End-to-End Semantic Communication Evaluation (FAIR protocol)")
    print(f"  Datasets: {', '.join(dataset_names)}")
    print(f"  Epochs: {args.epochs} | Seeds: {args.seeds} | Device: {device}")
    print(f"  Task reasoning depth (reference_hops)={args.reference_hops}; node_budget={args.node_budget}; Ours/HAKE follow both")
    print(f"  Ours config: sim_threshold={args.ours_sim_threshold}")
    print(f"  Eval mode: {args.eval_mode}")
    print(f"  Metric protocol: {protocol}")
    print(f"  Headline metrics: path_success, semantic_consistency, payload_kb, success_per_kb, raw_policy_reward")
    print("=" * 72)

    shared_embedder = EntityEmbedder(dim=128)

    def build_pruner(label: str, seed: int, dataset_name: str):
        kg = clone_kg(load_base_kg(dataset_name))
        if label == "DynamicPruned_Ours":
            return DynamicKBPruner(
                kg,
                max_subgraph_nodes=args.node_budget,
                max_hops=args.reference_hops,
                similarity_threshold=args.ours_sim_threshold,
                enable_injection=True,
            )
        if label == "HAKE_StaticLayer_Sun2026":
            return HAKEBaselinePruner(kg, num_layers=3, seed=seed)
        if label == "BudgetedNoPruning_iRML_JSAC2023":
            return NoPruningProvider(kg, max_nodes=args.noprune_max_nodes, seed=seed)
        if label == "OracleTaskIncluded_NoPruning":
            return OracleTaskIncludedNoPruningProvider(kg, max_nodes=args.noprune_max_nodes, seed=seed)
        raise ValueError(f"Unknown E3 model label: {label}")

    configs = [
        "DynamicPruned_Ours",
        "HAKE_StaticLayer_Sun2026",
        "BudgetedNoPruning_iRML_JSAC2023",
    ]
    if args.include_oracle_noprune:
        configs.append("OracleTaskIncluded_NoPruning")

    all_records: List[Dict] = []

    for dataset_name in dataset_names:
        dataset_dir = DATASET_DIRS[dataset_name]
        base_kg = load_base_kg(dataset_name)   # read-only task KG (no pruner mutates it)
        print(f"\n{'=' * 72}")
        print(f"Dataset: {dataset_name} ({dataset_dir})")
        for label in configs:
            print(f"\n[{dataset_name} | {label}]")
            for si in range(args.seeds):
                seed = 42 + si
                print(f"  Seed {si+1}/{args.seeds} (seed={seed})...")
                try:
                    pruner = build_pruner(label, seed, dataset_name)
                    recs = run_training_and_eval(
                        label=label, pruner=pruner, task_kg=base_kg,
                        shared_embedder=shared_embedder, device=device,
                        epochs=args.epochs, seed=seed,
                        node_emb_dim=nd, state_dim=sd, action_dim=ad,
                        sub_emb_dim=subd, intent_dim=intd, csi_dim=csid,
                        eval_mode=args.eval_mode,
                        reference_hops=args.reference_hops,
                        node_budget=args.node_budget,
                        task_focus_ablation=args.task_focus_ablation,
                        task_focus_nodes=args.task_focus_nodes,
                        task_context_nodes=args.task_context_nodes)
                    for row in recs:
                        row["dataset"] = dataset_name
                        row["metric_protocol"] = protocol
                        row["baseline_budget_nodes"] = args.noprune_max_nodes if "NoPruning" in label else 64
                    all_records.extend(recs)
                    if recs:
                        final = recs[-1]
                        print(f"    Final: path={final['path_success']:.2f} | "
                              f"consistency={final['semantic_consistency']:.3f} | "
                              f"payload={final['payload_kb']:.1f}KB | "
                              f"raw={final['raw_policy_reward']:.2f}")
                except Exception as e:
                    import traceback
                    print(f"    FAILED: {e}")
                    traceback.print_exc()

    # ── Aggregate and save ────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if all_records:
        suffix = f"_{args.output_tag}" if args.output_tag else ""
        csv_path = out_dir / f"end_to_end_comparison{suffix}.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_records[0].keys()))
            w.writeheader()
            w.writerows(all_records)

        # Use each seed's last evaluation point only.
        final_by_model = defaultdict(list)
        by_model_seed = defaultdict(list)
        for r in all_records:
            by_model_seed[(r["dataset"], r["model"], r["seed"])].append(r)
        for (dataset, model, _seed), rows in by_model_seed.items():
            final_by_model[(dataset, model)].append(max(rows, key=lambda x: int(x["epoch"])))

        def agg(rows, key):
            vals = [float(r.get(key, 0.0)) for r in rows]
            return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4))

        summary_rows = []
        print(f"\n{'='*72}")
        print(f"Final Comparison (last eval point, avg across seeds) — FAIR protocol")
        summary_by_key: Dict[Tuple[str, str], Dict] = {}
        for dataset_name in dataset_names:
            print(f"\nDataset: {dataset_name}")
            for label in configs:
                key = (dataset_name, label)
                if key not in final_by_model:
                    continue
                rows = final_by_model[key]
                ps_m, ps_s = agg(rows, "path_success")
                sc_m, sc_s = agg(rows, "semantic_consistency")
                spkb_m, _ = agg(rows, "success_per_kb")
                cpkb_m, _ = agg(rows, "consistency_per_kb")
                raw_m, raw_s = agg(rows, "raw_policy_reward")
                pay_m, _ = agg(rows, "payload_kb")
                tgr_m, _ = agg(rows, "task_goal_rate")
                nodes_m, _ = agg(rows, "sub_graph_nodes")
                out = {
                    "dataset": dataset_name,
                    "model": label,
                    "num_seeds": len(rows),
                    "path_success_mean": ps_m,
                    "path_success_std": ps_s,
                    "semantic_consistency_mean": sc_m,
                    "semantic_consistency_std": sc_s,
                    "task_goal_rate_mean": tgr_m,
                    "payload_kb_mean": pay_m,
                    "success_per_kb_mean": spkb_m,
                    "consistency_per_kb_mean": cpkb_m,
                    "raw_policy_reward_mean": raw_m,
                    "raw_policy_reward_std": raw_s,
                    "sub_graph_nodes_mean": nodes_m,
                    "eval_mode": args.eval_mode,
                    "metric_protocol": protocol,
                    "baseline_budget_nodes": args.noprune_max_nodes if "NoPruning" in label else 64,
                }
                summary_rows.append(out)
                summary_by_key[key] = out
                print(f"  {label:34s}: path={out['path_success_mean']:.2f}+/-{out['path_success_std']:.2f} | "
                      f"consist={out['semantic_consistency_mean']:.3f} | "
                      f"succ/KB={out['success_per_kb_mean']:.3f} | "
                      f"payload={out['payload_kb_mean']:.1f}KB | "
                      f"raw={out['raw_policy_reward_mean']:.2f}")
            ours = summary_by_key.get((dataset_name, "DynamicPruned_Ours"))
            hake = summary_by_key.get((dataset_name, "HAKE_StaticLayer_Sun2026"))
            if ours and hake:
                print(f"  --- Ours vs HAKE (fair) ---")
                print(f"      path_success : {ours['path_success_mean']:.2f} vs {hake['path_success_mean']:.2f} "
                      f"(+{ours['path_success_mean']-hake['path_success_mean']:.2f})")
                print(f"      consistency  : {ours['semantic_consistency_mean']:.3f} vs {hake['semantic_consistency_mean']:.3f} "
                      f"(+{ours['semantic_consistency_mean']-hake['semantic_consistency_mean']:.3f})")
                print(f"      success/KB   : {ours['success_per_kb_mean']:.3f} vs {hake['success_per_kb_mean']:.3f}")
                ours_wins = (ours['path_success_mean'] > hake['path_success_mean'] and
                             ours['semantic_consistency_mean'] >= hake['semantic_consistency_mean'])
                print(f"      VERDICT: Ours fairly {'WINS' if ours_wins else 'does NOT clearly win'} vs HAKE")
        if summary_rows:
            summary_path = out_dir / f"end_to_end_summary{suffix}.csv"
            with open(summary_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)
            print(f"\nSummary -> {summary_path}")
        print(f"Saved -> {csv_path}")


if __name__ == "__main__":
    main()
