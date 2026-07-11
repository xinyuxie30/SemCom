"""
Semi-MDP environment for graph-traversal-based resource scheduling.

[Phase 1 Fix]: State is now an EXPLICIT dict (not a flat opaque vector).
  state = {
      "node_emb":    np.ndarray [NODE_EMB_DIM],   # current node embedding
      "csi":         np.ndarray [CSI_DIM],         # channel state snapshot
      "hop_count":   int,
      "current_node": str,
      "target_node":  str,
      "done":         bool,
  }

The EpisodeCollector is responsible for flattening this dict + sub_emb + intent_vec
into the full [STATE_DIM] tensor consumed by the policy network.

This environment is fully decoupled from static KG files.
The sub-graph and intent are injected dynamically at reset().
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data.kg_interface import INTENT_DIM, SubGraph
from env.channel import channel_snapshot

# ── [Phase 1 Fix]: Dimension constants — must match config/default.yaml ────
NODE_EMB_DIM = 256
SUB_EMB_DIM = 128
CSI_DIM = 4
_PROJECTION_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def project_semantic_node_embedding(
    semantic_emb: Optional[np.ndarray],
    out_dim: int = NODE_EMB_DIM,
    fallback_key: str = "",
) -> np.ndarray:
    """Project semantic KG features into the fixed environment node state."""
    if semantic_emb is None:
        rng = np.random.default_rng(_stable_seed(fallback_key))
        vec = rng.normal(0, 1, out_dim).astype(np.float32)
    else:
        vec = np.asarray(semantic_emb, dtype=np.float32).reshape(-1)
        if vec.shape[0] != out_dim:
            key = (int(vec.shape[0]), int(out_dim))
            if key not in _PROJECTION_CACHE:
                rng = np.random.default_rng(20260621 + key[0] * 1009 + key[1])
                scale = 1.0 / max(1.0, np.sqrt(float(key[0])))
                _PROJECTION_CACHE[key] = rng.normal(0, scale, key).astype(np.float32)
            vec = vec @ _PROJECTION_CACHE[key]
    vec = vec.astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / (norm + 1e-8) if norm > 1e-8 else vec


def encode_semimdp_action(
    node_idx: int,
    num_neighbours: int,
    bw_idx: int,
    mcs_idx: int,
    action_dim: int = 128,
    num_bw_bins: int = 16,
    num_mcs_levels: int = 8,
) -> int:
    """Encode a topology/resource choice into the fixed discrete action space."""
    n = max(1, int(num_neighbours))
    action_dim = max(1, int(action_dim))
    node_code = int(node_idx) % n
    desired_resource = (int(bw_idx) % num_bw_bins)
    desired_resource += num_bw_bins * (int(mcs_idx) % num_mcs_levels)
    max_resource = max(0, (action_dim - 1 - node_code) // n)
    resource_code = min(desired_resource, max_resource)
    return node_code + n * resource_code


@dataclass
class EnvConfig:
    max_hops: int = 8
    num_bw_bins: int = 16
    num_mcs_levels: int = 8
    action_dim: int = 128            # total discrete actions (masked per state)
    node_emb_dim: int = NODE_EMB_DIM
    sub_emb_dim: int = SUB_EMB_DIM
    intent_dim: int = INTENT_DIM
    csi_dim: int = CSI_DIM
    penalty_hop: float = -0.05
    reward_goal: float = 10.0
    reward_suboptimal: float = 1.0
    snr_db_low: float = 0.0
    snr_db_high: float = 30.0
    doppler_hz: float = 10.0


class SemiMDPEnv:
    """
    A single-episode Semi-MDP on a dynamically-injected sub-graph.

    The agent must choose a path from the start entity to a target entity
    (both encoded in the intent), while jointly selecting BW & MCS per hop
    to maximise semantic fidelity under channel constraints.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._sub_graph: Optional[SubGraph] = None
        self._intent_vec: Optional[np.ndarray] = None    # [Phase 1 Fix]: stored from reset()
        self._node_embeddings: Dict[str, np.ndarray] = {}
        self._current_node: str = ""
        self._target_node: str = ""
        self._hop_count: int = 0
        self._done: bool = False
        self._path_history: List[str] = []                # [Phase 1 Fix]: track visited nodes
        self._rng = np.random.default_rng(42)
        self._last_csi = np.zeros(self.cfg.csi_dim, dtype=np.float32)

    # ── [Phase 1 Fix]: reset() now accepts intent_vec ──────────────────────
    def reset(self, sub_graph: SubGraph, intent_vec: Optional[np.ndarray] = None
              ) -> Dict[str, object]:
        """
        Inject a new sub-graph + intent and reset the episode.

        Returns:
            state dict with keys: node_emb, csi, hop_count, current_node,
                                  target_node, done, path_history
        """
        self._sub_graph = sub_graph
        if intent_vec is not None:
            self._intent_vec = np.asarray(intent_vec, dtype=np.float32)
        else:
            self._intent_vec = np.asarray(sub_graph.intent_vec, dtype=np.float32)
        if self._intent_vec.shape != (self.cfg.intent_dim,):
            raise ValueError(
                f"intent_vec must be [{self.cfg.intent_dim}], got {self._intent_vec.shape}"
            )
        self._hop_count = 0
        self._done = False
        self._path_history = []

        semantic_features = sub_graph.node_embeddings or {}
        self._node_embeddings = {
            n: project_semantic_node_embedding(
                semantic_features.get(n),
                out_dim=self.cfg.node_emb_dim,
                fallback_key=n,
            )
            for n in sub_graph.nodes
        }

        self._current_node, self._target_node = self._resolve_start_target(sub_graph)

        self._path_history.append(self._current_node)
        return self._get_state_dict()

    # ── [Phase 1 Fix]: step() returns explicit dict state ──────────────────
    def step(self, action: int) -> Tuple[Dict[str, object], float, bool, Dict]:
        """
        Execute one hop.

        Args:
            action: discrete index, decoded into (next_node_idx, bw_idx, mcs_idx).

        Returns:
            (state_dict, reward, done, info)
            where state_dict keys = {node_emb, csi, hop_count, current_node,
                                     target_node, done, path_history}
        """
        if self._done:
            return self._get_state_dict(), 0.0, True, {"reason": "already_done"}

        self._hop_count += 1

        # Decode action → next node
        neighbours = self._get_neighbours(self._current_node)
        if not neighbours:
            self._done = True
            return self._get_state_dict(), self.cfg.penalty_hop, True, {"reason": "dead_end"}

        node_idx, bw_idx, mcs_idx = self._decode_action(action, len(neighbours))
        next_node = neighbours[node_idx]
        channel_quality = self._channel_quality()
        bw_norm = (bw_idx + 1) / max(1, self.cfg.num_bw_bins)
        mcs_norm = (mcs_idx + 1) / max(1, self.cfg.num_mcs_levels)
        reliability = float(np.clip(channel_quality + 0.15 * bw_norm - 0.35 * mcs_norm, -1.0, 1.0))
        resource_cost = 0.01 * bw_norm + 0.02 * mcs_norm

        # ── Reward signal ──────────────────────────────────────────────────
        reward = self.cfg.penalty_hop + 0.2 * reliability - resource_cost
        info: Dict = {
            "hop": self._hop_count,
            "node": next_node,
            "target": self._target_node,
            "bw_idx": bw_idx,
            "mcs_idx": mcs_idx,
            "channel_quality": round(channel_quality, 4),
            "link_reliability": round(reliability, 4),
        }

        if next_node == self._target_node:
            reward = self.cfg.reward_goal + max(0.0, reliability)
            self._done = True
            info["reason"] = "goal_reached"
        elif self._hop_count >= self.cfg.max_hops:
            reward += self.cfg.reward_suboptimal * max(0.0, reliability)
            self._done = True
            info["reason"] = "max_hops"

        self._current_node = next_node
        self._path_history.append(next_node)
        return self._get_state_dict(), reward, self._done, info

    def sample_action(self) -> int:
        """Random action (for exploration / debugging only)."""
        neighbours = self._get_neighbours(self._current_node)
        if not neighbours:
            return 0
        return self._rng.integers(0, len(neighbours))

    # ── [Phase 1 Fix]: explicit state dict builder ─────────────────────────
    def _get_state_dict(self) -> Dict[str, object]:
        """
        Build the environment-side state dict.

        Keys:
            node_emb:     np.ndarray [NODE_EMB_DIM]  — current node embedding
            csi:          np.ndarray [CSI_DIM]        — channel snapshot (re-sampled each call)
            hop_count:    int
            current_node: str
            target_node:  str
            done:         bool
            path_history: List[str]
        """
        node_emb = self._node_embeddings.get(
            self._current_node,
            np.zeros(self.cfg.node_emb_dim, dtype=np.float32),
        )

        # [Phase 1 Fix]: generate CSI per step to simulate time-varying channel
        snr_db = float(self._rng.uniform(self.cfg.snr_db_low, self.cfg.snr_db_high))
        csi = channel_snapshot(dim=self.cfg.csi_dim, snr_db=snr_db,
                               doppler_hz=self.cfg.doppler_hz).numpy()
        self._last_csi = csi.astype(np.float32)

        return {
            "node_emb": node_emb,
            "csi": csi,
            "hop_count": self._hop_count,
            "current_node": self._current_node,
            "target_node": self._target_node,
            "done": self._done,
            "path_history": list(self._path_history),
        }

    # ── Neighbour extraction from dynamic sub-graph ────────────────────────
    def _get_neighbours(self, node: str) -> list[str]:
        """Extract neighbour list from the current sub-graph."""
        if self._sub_graph is None:
            return []
        neighbours = []
        for h, _, t in self._sub_graph.edges:
            if h == node and t not in neighbours:
                neighbours.append(t)
        if not neighbours:
            # Fallback: allow teleport to any other node (for Phase 1)
            neighbours = [n for n in self._sub_graph.nodes if n != node]
        return neighbours

    def _resolve_start_target(self, sub_graph: SubGraph) -> tuple[str, str]:
        """Choose a reachable start/target pair for the current task.

        Dynamic multi-source pruning can produce node lists where the last node
        is not reachable from the first node. Using that arbitrary ordering as
        the task target hides the quality of the pruned graph behind an
        evaluation artifact, so we prefer explicit metadata and otherwise pick
        the farthest reachable node from the start.
        """
        if not sub_graph.nodes:
            return "", ""
        preferred_start = sub_graph.metadata.get("preferred_start")
        preferred_target = sub_graph.metadata.get("preferred_target")
        if preferred_start in sub_graph.nodes and preferred_target in sub_graph.nodes:
            return str(preferred_start), str(preferred_target)

        start = sub_graph.nodes[0]
        adj: Dict[str, List[str]] = {node: [] for node in sub_graph.nodes}
        for h, _, t in sub_graph.edges:
            if h in adj and t in adj and t not in adj[h]:
                adj[h].append(t)
        frontier = [(start, 0)]
        seen = {start}
        farthest = start
        farthest_dist = 0
        while frontier:
            node, dist = frontier.pop(0)
            if dist > farthest_dist:
                farthest = node
                farthest_dist = dist
            for nb in adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    frontier.append((nb, dist + 1))
        if farthest != start:
            return start, farthest
        if len(sub_graph.nodes) >= 2:
            return start, sub_graph.nodes[-1]
        return start, start

    def _decode_action(self, action: int, num_neighbours: int) -> Tuple[int, int, int]:
        """Decode action into next-node, bandwidth, and MCS choices."""
        n = max(1, int(num_neighbours))
        action = int(action) % max(1, self.cfg.action_dim)
        node_idx = action % n
        resource_code = action // n
        bw_idx = resource_code % max(1, self.cfg.num_bw_bins)
        mcs_idx = (resource_code // max(1, self.cfg.num_bw_bins)) % max(1, self.cfg.num_mcs_levels)
        return node_idx, bw_idx, mcs_idx

    def _channel_quality(self) -> float:
        """Estimate link quality from the CSI observed before this action."""
        if self._last_csi.shape[0] < 4:
            return 0.0
        snr_norm = float(self._last_csi[0])
        sinr_est = float(self._last_csi[2])
        path_loss = float(self._last_csi[3])
        return float(np.clip(0.45 * snr_norm + 0.45 * sinr_est - 0.25 * path_loss, -1.0, 1.0))

    @property
    def state_dim(self) -> int:
        return self.cfg.node_emb_dim + self.cfg.sub_emb_dim + self.cfg.intent_dim + self.cfg.csi_dim

    @property
    def observation_dim(self) -> int:
        """Environment-local observation size before subgraph and intent context."""
        return self.cfg.node_emb_dim + self.cfg.csi_dim

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim
