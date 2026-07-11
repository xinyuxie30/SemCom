"""
Dataloader — constructs experience tuples for G-RML training.

Key design decisions (Phase 1):
  - NEVER imports from txt/csv/pickle KG files.
  - All KG data arrives via kg_interface.SubGraphProvider.
  - Produces (state, action, reward, next_state, done) tuples
    consumable by the GAIL buffer.

[Phase 1 Fix]: State composition is now EXPLICIT and documented.
  flat_state = cat([node_emb, sub_emb, intent_vec, csi])
             = cat([256,      128,      64,         4  ])
             = [452]  = STATE_DIM

[Phase 1 Fix]: EpisodeCollector.collect() now uses Generator policy
  (action, log_prob = generator.get_action(state)) instead of random sampling.
"""
from __future__ import annotations

import hashlib
from collections import deque
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from data.kg_interface import INTENT_DIM, SubGraph
from env.channel import channel_snapshot
from env.mdp_env import CSI_DIM, NODE_EMB_DIM
from models.gnn_encoder import SUB_EMB_DIM

# ── [Phase 1 Fix]: Dimension constants — single source of truth ─────────────
# State composition:
#   [B, NODE_EMB_DIM] cat [B, SUB_EMB_DIM] cat [B, INTENT_DIM] cat [B, CSI_DIM]
#   = [B, 256] cat [B, 128] cat [B, 64] cat [B, 4] -> [B, 452]
STATE_DIM = NODE_EMB_DIM + SUB_EMB_DIM + INTENT_DIM + CSI_DIM  # 452


def _stable_subgraph_id(query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8", errors="surrogatepass")).hexdigest()
    return digest[:16]


# ── Experience tuple ────────────────────────────────────────────────────────

class Experience(NamedTuple):
    state: torch.Tensor           # [STATE_DIM] = [452]
    action: int                   # discrete action index
    reward: float
    next_state: torch.Tensor      # [STATE_DIM] = [452]
    done: bool
    sub_graph_id: str             # opaque id for the sub-graph used in this step
    log_prob: float = 0.0         # [Phase 1 Fix]: log π(a|s) for policy gradient


# ── Replay buffer ───────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size FIFO buffer for GAIL on-policy / off-policy transitions."""

    def __init__(self, capacity: int = 10000):
        self._buf: deque[Experience] = deque(maxlen=capacity)

    def push(self, exp: Experience) -> None:
        self._buf.append(exp)

    def push_batch(self, exps: List[Experience]) -> None:
        for e in exps:
            self._buf.append(e)

    def sample(self, batch_size: int) -> List[Experience]:
        indices = np.random.choice(len(self._buf), size=min(batch_size, len(self._buf)), replace=False)
        return [self._buf[i] for i in indices]

    def to_tensors(self, batch: List[Experience]) -> Dict[str, torch.Tensor]:
        """Stack a list of Experiences into training-ready tensors."""
        states = torch.stack([e.state for e in batch])              # [B, STATE_DIM]
        actions = torch.tensor([e.action for e in batch], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in batch])    # [B, STATE_DIM]
        dones = torch.tensor([e.done for e in batch], dtype=torch.bool)
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    def __len__(self) -> int:
        return len(self._buf)


# ── Episode collector ───────────────────────────────────────────────────────

class EpisodeCollector:
    """
    Drives the Semi-MDP environment for one sub-graph, collecting
    (s, a, r, s', done) trajectories.

    [Phase 1 Fix]: Uses Generator policy for action selection instead of
    env.sample_action(). The collector is the bridge that:
      1. Fetches sub-graph + intent from provider
      2. Encodes sub-graph via GNN → sub_emb [SUB_EMB_DIM]
      3. Resets env with sub-graph + intent
      4. At each step: builds flat state, queries generator for action,
         steps env, builds next_state, stores Experience
    """

    def __init__(self, env, provider, gnn_encoder,
                 state_dim: int = STATE_DIM,
                 max_hops: int = 8,
                 generator=None,         # [Phase 1 Fix]: generator policy (set after init or at init)
                 device: str = "cpu"):
        self.env = env
        self.provider = provider
        self.gnn_encoder = gnn_encoder
        self.state_dim = state_dim
        self.max_hops = max_hops
        self.generator = generator
        self.device = device

    # [Phase 1 Fix]: attach generator after construction (for train.py flow)
    def set_generator(self, generator) -> None:
        self.generator = generator

    # ── [Phase 3 Fix]: collect() accepts optional sub_graph / intent_vec overrides ──
    def collect(self, query: str = "",
                sub_graph: SubGraph = None,
                intent_vec_override: np.ndarray = None) -> List[Experience]:
        """
        Run one episode using the generator policy and return collected transitions.

        [Phase 3 Fix]: When sub_graph and intent_vec_override are provided by the
        slow loop, they bypass the provider. Otherwise falls back to Phase 1/2
        behaviour (provider.fetch).

        Data flow:
          if sub_graph is None: provider.fetch(query) → SubGraph
          gnn_encoder(sub_graph) → sub_emb [SUB_EMB_DIM]
          env.reset(sub_graph, intent_vec) → env_state_dict
          loop:
            flat_state = cat([node_emb, sub_emb, intent_vec, csi])  [STATE_DIM]
            action, log_prob = generator.get_action(flat_state)
            next_env_dict, reward, done, _ = env.step(action)
            flat_next = cat([next_node_emb, sub_emb, intent_vec, next_csi])
            store Experience(flat_state, action, reward, flat_next, done)
        """
        # [Phase 3 Fix]: accept externally-injected sub-graph from slow loop
        if sub_graph is None:
            sub_graph = self.provider.fetch(query)

        # Encode sub-graph once (static for the episode)
        sub_graph_data = sub_graph.to_tensors()
        sub_graph_data["edge_index"] = sub_graph_data["edge_index"].to(self.device)
        sub_emb: torch.Tensor = self.gnn_encoder(sub_graph_data).detach().cpu()  # [SUB_EMB_DIM]

        # [Phase 3 Fix]: use override intent_vec if provided
        if intent_vec_override is not None:
            intent_vec_np = np.asarray(intent_vec_override, dtype=np.float32)
        else:
            intent_vec_np = sub_graph.intent_vec
        intent_vec_t = torch.from_numpy(intent_vec_np).float()         # [INTENT_DIM]

        # Reset env with sub-graph + intent
        env_state: Dict = self.env.reset(sub_graph=sub_graph, intent_vec=intent_vec_np)

        trajectory: List[Experience] = []

        for _ in range(self.max_hops):
            # [Phase 1 Fix]: build flat state from env dict + global context
            state = self._build_state(env_state, sub_emb, intent_vec_t)  # [STATE_DIM]

            # [Phase 1 Fix]: use generator policy, fallback to random if no generator set
            if self.generator is not None:
                state_batch = state.unsqueeze(0).to(self.device)         # [1, STATE_DIM]
                with torch.no_grad():
                    action_t, log_prob_t = self.generator.get_action(state_batch)
                action = int(action_t.item())
                log_prob = float(log_prob_t.item())
            else:
                action = self.env.sample_action()
                log_prob = 0.0

            next_env_state, reward, done, _ = self.env.step(action)

            # [Phase 1 Fix]: build next_state with updated CSI / node_emb
            next_state = self._build_state(next_env_state, sub_emb, intent_vec_t)

            trajectory.append(Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                sub_graph_id=_stable_subgraph_id(query),
                log_prob=log_prob,
            ))

            env_state = next_env_state
            if done:
                break

        return trajectory

    # ── [Phase 1 Fix]: Explicit dimension-aware state builder ───────────────
    def _build_state(self, env_state: Dict, sub_emb: torch.Tensor,
                     intent_vec: torch.Tensor) -> torch.Tensor:
        """
        Concatenate environment observation with global semantic context.

        Composition (documented for debug):
          node_emb:  np.ndarray [NODE_EMB_DIM]  from env (current node)
          sub_emb:   torch.Tensor [SUB_EMB_DIM] from GNN (static per episode)
          intent_vec: torch.Tensor [INTENT_DIM] from SubGraph (static per episode)
          csi:        np.ndarray [CSI_DIM]       from env (re-sampled per step)

        Result: torch.Tensor [STATE_DIM] = [452]
        """
        node_emb_np: np.ndarray = env_state["node_emb"]        # [256]
        csi_np: np.ndarray = env_state["csi"]                  # [4]

        node_emb_t = torch.from_numpy(node_emb_np).float()     # [256]
        csi_t = torch.from_numpy(csi_np).float()               # [4]

        # [Phase 1 Fix]: cat([256], [128], [64], [4]) → [452]
        #   [B, 256] cat [B, 128] cat [B, 64] cat [B, 4] -> [B, 452]
        full = torch.cat([node_emb_t, sub_emb, intent_vec, csi_t], dim=0)

        assert full.shape[0] == self.state_dim, \
            f"[Phase 1 Fix] State dim mismatch: {full.shape[0]} vs expected {self.state_dim}. " \
            f"Breakdown: node_emb={node_emb_t.shape[0]}, sub_emb={sub_emb.shape[0]}, " \
            f"intent={intent_vec.shape[0]}, csi={csi_t.shape[0]}"
        return full
