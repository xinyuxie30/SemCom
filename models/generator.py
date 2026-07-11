"""
G-RML Generator (policy network) — GAIL-style stochastic policy π_θ(a|s).

[Spectral Norm Fix]: Optional Spectral Normalization on all linear layers
constrains the Lipschitz constant L_π ≤ 1, making the theoretical bound
||π_new - π_old||_TV ≤ L_π · η · (1-ρ) meaningful rather than vacuous.

State composition:
  [B, 256] cat [B, 128] cat [B, 64] cat [B, 4] -> [B, 452]

Output: action logits over the discrete action space [action_dim].
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils import spectral_norm

from data.dataloader import STATE_DIM

DEFAULT_ACTION_DIM = 128


class Generator(nn.Module):
    """Stochastic policy for the fast-loop G-RML agent.

    [Spectral Norm Fix]: When use_sn=True, all linear layers are wrapped
    with spectral_norm(), constraining the spectral norm to ≤1.
    This guarantees L_π ≤ 1, making the policy-jump bound tight.
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = DEFAULT_ACTION_DIM,
                 hidden_dim: int = 512, num_layers: int = 3,
                 use_sn: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_sn = use_sn

        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(spectral_norm(linear) if use_sn else linear)
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        linear = nn.Linear(hidden_dim, action_dim)
        layers.append(spectral_norm(linear) if use_sn else linear)
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> Categorical:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(state)
        if deterministic:
            action = dist.logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if action.shape[0] == 1:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

    def estimate_lipschitz(self, n_samples: int = 500) -> float:
        """Empirically estimate L_π = max ||π(s)-π(s')||_TV / ||s-s'||."""
        self.eval()
        device = next(self.parameters()).device
        max_ratio = 0.0
        with torch.no_grad():
            for _ in range(n_samples):
                s = torch.randn(1, self.state_dim, device=device)
                eps = 0.01 * torch.randn(1, self.state_dim, device=device)
                s_prime = s + eps
                d1 = self.forward(s)
                d2 = self.forward(s_prime)
                tv = 0.5 * (d1.probs - d2.probs).abs().sum(dim=-1).item()
                norm_eps = eps.norm().item()
                if norm_eps > 1e-8:
                    max_ratio = max(max_ratio, tv / norm_eps)
        return max_ratio

    def spectral_norm_bound(self) -> float:
        """Theoretical bound: product of spectral norms of all linear layers."""
        prod = 1.0
        for m in self.net:
            if isinstance(m, nn.Linear):
                w = m.weight
                u = torch.randn(1, w.shape[0], device=w.device)
                v = torch.randn(1, w.shape[1], device=w.device)
                for _ in range(5):
                    v = torch.nn.functional.normalize(u @ w, dim=1)
                    u = torch.nn.functional.normalize(v @ w.t(), dim=1)
                prod *= (u @ w @ v.t()).item()
        return prod
