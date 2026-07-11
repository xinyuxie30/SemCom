"""
GAIL training agent — orchestrates generator/discriminator updates.

[Phase 1 Fix]: MockExpert moved to agents/slow_loop/mock_expert.py.
Dimension defaults now sourced from dataloader.STATE_DIM.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from data.dataloader import Experience, ReplayBuffer, STATE_DIM
from models.generator import Generator
from models.discriminator import Discriminator


class GAILAgent:
    """GAIL trainer wrapping generator π_θ and discriminator D_φ."""

    def __init__(self, generator: Generator, discriminator: Discriminator,
                 gen_lr: float = 3e-4, disc_lr: float = 1e-4,
                 entropy_coef: float = 0.01, device: str = "cpu"):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

        self.gen_opt = optim.Adam(self.generator.parameters(), lr=gen_lr)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=disc_lr)
        self.entropy_coef = entropy_coef

        self.bce = nn.BCEWithLogitsLoss()

    def update_discriminator(self, gen_batch: Dict[str, torch.Tensor],
                             expert_batch: Dict[str, torch.Tensor]) -> float:
        """
        One discriminator update step.
        Returns: discriminator loss (float).
        """
        gen_s = gen_batch["states"].to(self.device)
        gen_a = gen_batch["actions"].to(self.device)
        exp_s = expert_batch["states"].to(self.device)
        exp_a = expert_batch["actions"].to(self.device)

        # Expert logits → label 1; Generator logits → label 0
        exp_logits = self.discriminator(exp_s, exp_a)
        gen_logits = self.discriminator(gen_s.detach(), gen_a.detach())

        exp_loss = self.bce(exp_logits, torch.ones_like(exp_logits))
        gen_loss = self.bce(gen_logits, torch.zeros_like(gen_logits))
        loss = exp_loss + gen_loss

        self.disc_opt.zero_grad()
        loss.backward()
        self.disc_opt.step()
        return loss.item()

    def update_generator(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        One generator (policy) update step using GAIL reward.
        Returns: dict with 'policy_loss', 'entropy'.
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)

        log_probs, entropy = self.generator.evaluate(states, actions)

        # GAIL reward from discriminator
        with torch.no_grad():
            rewards = self.discriminator.reward(states, actions).squeeze(-1)

        policy_loss = -(log_probs * rewards).mean()
        entropy_loss = -self.entropy_coef * entropy.mean()
        total_loss = policy_loss + entropy_loss

        self.gen_opt.zero_grad()
        total_loss.backward()
        self.gen_opt.step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": total_loss.item(),
        }
