"""
Mock Expert — generates pseudo-expert trajectories on dynamic sub-graphs.

[Phase 1 Fix]: Replaces pure-random tensor generation with BFS/random-walk
trajectories that respect actual graph topology. The discriminator can now
learn meaningful decision boundaries from these structured paths.

Algorithm (per sub-graph):
  1. Run BFS from start node to target node to find the shortest path.
  2. If no path exists, fall back to a random walk with backtracking avoidance.
  3. For each hop, construct the full flat state vector using the same
     dimensional contract as EpisodeCollector:
       flat_state = cat([node_emb, sub_emb, intent_vec, csi])
                  = [NODE_EMB_DIM + SUB_EMB_DIM + INTENT_DIM + CSI_DIM]
                  = [452]
  4. The "expert action" at each hop is the index of the correct next node
     in the current node's neighbour list (matching env.step() decoding).
"""
from __future__ import annotations

import hashlib
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data.dataloader import Experience, STATE_DIM
from data.kg_interface import INTENT_DIM, SubGraph
from env.channel import channel_snapshot
from env.mdp_env import (
    CSI_DIM,
    NODE_EMB_DIM,
    encode_semimdp_action,
    project_semantic_node_embedding,
)
from models.gnn_encoder import SUB_EMB_DIM


class MockExpert:
    """
    Produces expert trajectories by solving shortest-path on the sub-graph.

    The expert:
      - Encodes the sub-graph via GNN to get sub_emb.
      - Builds per-node random embeddings (consistent with env Phase 1).
      - Finds BFS shortest path; maps each hop to a discrete action index
        that matches how SemiMDPEnv.step() decodes actions.
      - Constructs full [STATE_DIM] state vectors for each transition.
    """

    def __init__(self, gnn_encoder, node_emb_dim: int = NODE_EMB_DIM,
                 device: str = "cpu"):
        self.gnn_encoder = gnn_encoder
        self.node_emb_dim = node_emb_dim
        self.device = device

    # ── Public API ─────────────────────────────────────────────────────────
    def sample_trajectory(self, sub_graph: SubGraph) -> List[Experience]:
        """
        Generate one expert trajectory on the given sub-graph.

        Returns:
            List[Experience] — a full path from start to target (or max_hops).
        """
        # 1. Build node embeddings from the same semantic features as the env.
        rng = np.random.default_rng(self._stable_seed(sub_graph))
        semantic_features = sub_graph.node_embeddings or {}
        node_embs: Dict[str, np.ndarray] = {
            n: project_semantic_node_embedding(
                semantic_features.get(n),
                out_dim=self.node_emb_dim,
                fallback_key=n,
            )
            for n in sub_graph.nodes
        }

        # 2. Encode sub-graph
        sg_data = sub_graph.to_tensors()
        sg_data["edge_index"] = sg_data["edge_index"].to(self.device)
        with torch.no_grad():
            sub_emb: torch.Tensor = self.gnn_encoder(sg_data).cpu()   # [SUB_EMB_DIM]
        intent_vec_t = torch.from_numpy(sub_graph.intent_vec).float() # [INTENT_DIM]

        # 3. Determine start / target (same convention as SemiMDPEnv)
        nodes = sub_graph.nodes
        if len(nodes) < 2:
            return []
        start_node, target_node = self._resolve_start_target(sub_graph)

        # 4. Find expert path via BFS
        path = self._bfs_shortest_path(sub_graph, start_node, target_node)
        if path is None or len(path) < 2:
            path = self._random_walk(sub_graph, start_node, target_node,
                                     max_hops=8, rng=rng)

        # 5. Build trajectory from path
        trajectory: List[Experience] = []
        for i in range(len(path) - 1):
            current = path[i]
            nxt = path[i + 1]
            done = (nxt == target_node) or (i == len(path) - 2)

            # Get neighbours of current node → determine action index
            neighbours = self._get_neighbours(sub_graph, current)
            if nxt in neighbours:
                action = encode_semimdp_action(
                    node_idx=neighbours.index(nxt),
                    num_neighbours=len(neighbours),
                    bw_idx=12,
                    mcs_idx=3,
                )
            else:
                action = 0  # fallback (shouldn't happen with BFS path)

            # Build state / next_state
            node_emb_t = torch.from_numpy(node_embs[current]).float()
            next_node_emb_t = torch.from_numpy(node_embs[nxt]).float()
            csi = channel_snapshot(dim=CSI_DIM)
            next_csi = channel_snapshot(dim=CSI_DIM)  # re-sample for next step

            state = torch.cat([node_emb_t, sub_emb, intent_vec_t, csi], dim=0)      # [452]
            next_state = torch.cat([next_node_emb_t, sub_emb, intent_vec_t, next_csi], dim=0)

            # Expert reward: positive at goal, small penalty per hop
            reward = 10.0 if done else -0.01

            trajectory.append(Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                sub_graph_id="mock_expert",
                log_prob=0.0,  # expert has no log_prob
            ))

        return trajectory

    # ── BFS shortest path ──────────────────────────────────────────────────
    @staticmethod
    def _bfs_shortest_path(sub_graph: SubGraph, start: str, target: str
                           ) -> Optional[List[str]]:
        """BFS to find shortest path. Returns None if unreachable."""
        if start == target:
            return [start]
        adj = MockExpert._build_adjacency(sub_graph)
        queue = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbour in adj.get(current, []):
                if neighbour == target:
                    return path + [neighbour]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])
        return None

    # ── Random walk fallback ───────────────────────────────────────────────
    @staticmethod
    def _random_walk(sub_graph: SubGraph, start: str, target: str,
                     max_hops: int, rng: np.random.Generator) -> List[str]:
        """Random walk with bias toward target when possible."""
        adj = MockExpert._build_adjacency(sub_graph)
        path = [start]
        current = start
        visited = {start}
        for _ in range(max_hops):
            neighbours = adj.get(current, [])
            if not neighbours:
                break
            # Prefer unvisited nodes; bias toward target
            unvisited = [n for n in neighbours if n not in visited]
            candidates = unvisited if unvisited else neighbours
            if target in candidates:
                path.append(target)
                break
            nxt = candidates[rng.integers(0, len(candidates))]
            path.append(nxt)
            visited.add(nxt)
            current = nxt
            if nxt == target:
                break
        return path

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _build_adjacency(sub_graph: SubGraph) -> Dict[str, List[str]]:
        """Build adjacency list from sub-graph edges."""
        adj: Dict[str, List[str]] = {n: [] for n in sub_graph.nodes}
        for h, _, t in sub_graph.edges:
            if h in adj and t not in adj[h]:
                adj[h].append(t)
        return adj

    @staticmethod
    def _get_neighbours(sub_graph: SubGraph, node: str) -> List[str]:
        """Get neighbour list for a node (same logic as SemiMDPEnv)."""
        neighbours = []
        for h, _, t in sub_graph.edges:
            if h == node and t not in neighbours:
                neighbours.append(t)
        if not neighbours:
            neighbours = [n for n in sub_graph.nodes if n != node]
        return neighbours

    @staticmethod
    def _resolve_start_target(sub_graph: SubGraph) -> Tuple[str, str]:
        preferred_start = sub_graph.metadata.get("preferred_start")
        preferred_target = sub_graph.metadata.get("preferred_target")
        if preferred_start in sub_graph.nodes and preferred_target in sub_graph.nodes:
            return str(preferred_start), str(preferred_target)

        start = sub_graph.nodes[0]
        adj = MockExpert._build_adjacency(sub_graph)
        queue = deque([(start, 0)])
        seen = {start}
        farthest = start
        farthest_dist = 0
        while queue:
            node, dist = queue.popleft()
            if dist > farthest_dist:
                farthest = node
                farthest_dist = dist
            for nb in adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, dist + 1))
        if farthest != start:
            return start, farthest
        return start, sub_graph.nodes[-1]

    @staticmethod
    def _stable_seed(sub_graph: SubGraph) -> int:
        h = hashlib.sha256()
        for node in sub_graph.nodes:
            h.update(node.encode("utf-8", errors="surrogatepass"))
            h.update(b"\0")
        for head, rel, tail in sub_graph.edges:
            h.update(head.encode("utf-8", errors="surrogatepass"))
            h.update(b"\t")
            h.update(rel.encode("utf-8", errors="surrogatepass"))
            h.update(b"\t")
            h.update(tail.encode("utf-8", errors="surrogatepass"))
            h.update(b"\0")
        return int.from_bytes(h.digest()[:8], "big") % (2**31)
