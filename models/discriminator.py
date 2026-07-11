"""
GAIL Discriminator — D_φ(s, a) → score ∈ [0, 1].

Distinguishes expert trajectories (from slow-loop / heuristic) from
generator rollouts. The reward for the generator is:
  r(s, a) = -log(1 - D(s, a))

[Phase 1 Fix]: Input = cat([state, action_emb]).
  state:     [B, STATE_DIM] = [B, 452]
  action:    [B] discrete → Embedding → [B, 8]
  concat:    [B, 452 + 8] = [B, 460]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data.dataloader import STATE_DIM

DEFAULT_ACTION_DIM = 128
ACTION_EMB_DIM = 8          # [Phase 1 Fix]: small action embedding for better gradient


class Discriminator(nn.Module):
    """GAIL discriminator: score how likely a (s, a) pair comes from the expert."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = DEFAULT_ACTION_DIM,
                 hidden_dim: int = 512, num_layers: int = 3,
                 action_emb_dim: int = ACTION_EMB_DIM):
        super().__init__()
        self.action_emb = nn.Embedding(action_dim, action_emb_dim)   # [Phase 1 Fix]: embed action

        # Input: state [B, STATE_DIM] + action_emb [B, ACTION_EMB_DIM]
        #      = [B, 452 + 8] = [B, 460]
        in_dim = state_dim + action_emb_dim
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  [B, STATE_DIM].
            action: [B] (discrete indices).
        Returns:
            logits: [B, 1] (raw scores before sigmoid).
        """
        # [Phase 1 Fix]: embed discrete action instead of raw scalar
        if action.dim() == 0:
            action = action.unsqueeze(0)
        action_emb = self.action_emb(action.long())           # [B, ACTION_EMB_DIM]
        sa = torch.cat([state, action_emb], dim=-1)           # [B, STATE_DIM + ACTION_EMB_DIM]
        return self.net(sa)

    def score(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Probability that (s, a) is expert."""
        return torch.sigmoid(self.forward(state, action))

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        GAIL-style reward: r = -log(1 - D(s,a)).
        Higher when the discriminator thinks the sample is expert-like.
        """
        d = self.score(state, action)
        return -torch.log(1.0 - d + 1e-8)
