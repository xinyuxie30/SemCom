"""Profile generator Lipschitz behavior under spectral normalization.

The useful question is not whether a random untrained No-SN policy happens to
be smooth. Small random weights can make that trivially true. We therefore test
two non-trivial conditions:

  1. Synthetic stress: amplify the underlying linear weights x3 and check that
     SN keeps the logits Lipschitz estimate bounded while No-SN grows.
  2. REAL GAIL training: run the actual adversarial training loop (generator +
     discriminator + MockExpert demonstrations on a deterministic mock
     sub-graph) and measure the trained generator — the condition the paper's
     policy-jump bound ||pi_new - pi_old||_TV <= L_pi * eta * (1-rho) actually
     assumes.

Output:
  experiments/results/lipschitz_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.generator import DEFAULT_ACTION_DIM, STATE_DIM, Generator
from models.discriminator import Discriminator
from models.gnn_encoder import SubGraphEncoder, SUB_EMB_DIM
from agents.gail_agent import GAILAgent
from agents.slow_loop.mock_expert import MockExpert
from data.kg_interface import MockSubGraphProvider, register_provider
from data.dataloader import EpisodeCollector, ReplayBuffer
from env.mdp_env import SemiMDPEnv, EnvConfig


def _linear_modules(gen: Generator) -> list[nn.Linear]:
    return [m for m in gen.net if isinstance(m, nn.Linear)]


def amplify_weights(gen: Generator, scale: float) -> None:
    """Scale raw linear parameters; for SN layers this scales weight_orig."""
    with torch.no_grad():
        for layer in _linear_modules(gen):
            if hasattr(layer, "weight_orig"):
                layer.weight_orig.mul_(scale)
            else:
                layer.weight.mul_(scale)
            if layer.bias is not None:
                layer.bias.mul_(scale)


def effective_weight(layer: nn.Linear, exact_sn: bool) -> torch.Tensor:
    """Return the exact effective weight used for analysis."""
    if exact_sn and hasattr(layer, "weight_orig"):
        w = layer.weight_orig.detach()
        sigma = torch.linalg.svdvals(w).max()
        return w / (sigma + 1e-12)
    return layer.weight.detach()


def effective_linear_params(gen: Generator, exact_sn: bool) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Precompute effective linear weights once per model."""
    params = []
    for layer in _linear_modules(gen):
        params.append((effective_weight(layer, exact_sn), layer.bias.detach() if layer.bias is not None else None))
    return params


def exact_logits(
    gen: Generator,
    state: torch.Tensor,
    params: list[tuple[torch.Tensor, torch.Tensor | None]],
) -> torch.Tensor:
    """Forward through the MLP with exact spectral normalization when requested."""
    x = state
    linear_idx = 0
    layers = list(gen.net)
    for layer in layers:
        if isinstance(layer, nn.Linear):
            weight, bias = params[linear_idx]
            x = F.linear(x, weight, bias)
            linear_idx += 1
        elif isinstance(layer, nn.ReLU):
            x = F.relu(x)
        else:
            x = layer(x)
    return x


def spectral_bound(gen: Generator, exact_sn: bool) -> float:
    """Product of exact spectral norms of effective linear weights."""
    prod = 1.0
    with torch.no_grad():
        for weight, _bias in effective_linear_params(gen, exact_sn):
            sigma = torch.linalg.svdvals(weight).max().item()
            prod *= float(sigma)
    return prod


def estimate_logits_lipschitz(
    gen: Generator,
    params: list[tuple[torch.Tensor, torch.Tensor | None]],
    n_samples: int = 300,
) -> float:
    """Estimate max ||logits(s)-logits(s')||_2 / ||s-s'||_2."""
    gen.eval()
    max_ratio = 0.0
    with torch.no_grad():
        for _ in range(n_samples):
            s = torch.randn(1, gen.state_dim)
            eps = 0.01 * torch.randn(1, gen.state_dim)
            z1 = exact_logits(gen, s, params)
            z2 = exact_logits(gen, s + eps, params)
            ratio = (z1 - z2).norm().item() / max(eps.norm().item(), 1e-8)
            max_ratio = max(max_ratio, ratio)
    return float(max_ratio)


def estimate_policy_tv_lipschitz(
    gen: Generator,
    params: list[tuple[torch.Tensor, torch.Tensor | None]],
    n_samples: int = 300,
) -> float:
    """Estimate max TV(pi(s), pi(s')) / ||s-s'||_2 as a secondary metric."""
    gen.eval()
    max_ratio = 0.0
    with torch.no_grad():
        for _ in range(n_samples):
            s = torch.randn(1, gen.state_dim)
            eps = 0.01 * torch.randn(1, gen.state_dim)
            p1 = torch.softmax(exact_logits(gen, s, params), dim=-1)
            p2 = torch.softmax(exact_logits(gen, s + eps, params), dim=-1)
            tv = 0.5 * (p1 - p2).abs().sum().item()
            max_ratio = max(max_ratio, tv / max(eps.norm().item(), 1e-8))
    return float(max_ratio)


def train_generator_gail(use_sn: bool, seed: int, epochs: int,
                         device: str) -> Generator:
    """Train a generator with the REAL GAIL loop on a deterministic mock sub-graph.

    Weights evolve through genuine adversarial updates (discriminator + policy
    gradient + MockExpert BFS demonstrations); no real KG is needed for the
    Lipschitz question, only real training dynamics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    prov = MockSubGraphProvider(embedding_dim=128, num_nodes=32)
    register_provider(prov)
    sub_graph = prov.fetch("lipschitz_train")

    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=SUB_EMB_DIM).to(device)
    env = SemiMDPEnv(EnvConfig(max_hops=8, action_dim=DEFAULT_ACTION_DIM))
    gen = Generator(state_dim=STATE_DIM, action_dim=DEFAULT_ACTION_DIM,
                    use_sn=use_sn).to(device)
    disc = Discriminator(state_dim=STATE_DIM, action_dim=DEFAULT_ACTION_DIM).to(device)
    agent = GAILAgent(gen, disc, gen_lr=3e-4, disc_lr=1e-4,
                      entropy_coef=0.01, device=device)
    expert = MockExpert(gnn_encoder=gnn, node_emb_dim=256, device=device)
    collector = EpisodeCollector(env=env, provider=prov, gnn_encoder=gnn,
                                 state_dim=STATE_DIM, max_hops=8,
                                 generator=gen, device=device)

    expert_buf = ReplayBuffer(capacity=10000)
    for _ in range(40):
        traj = expert.sample_trajectory(sub_graph)
        if traj:
            expert_buf.push_batch(traj)
    gen_buf = ReplayBuffer(capacity=10000)

    batch = 48
    for _ in range(epochs):
        traj = collector.collect(sub_graph=sub_graph,
                                 intent_vec_override=sub_graph.intent_vec)
        gen_buf.push_batch(traj)
        if len(gen_buf) >= batch and len(expert_buf) >= batch:
            gb = gen_buf.to_tensors(gen_buf.sample(batch))
            eb = expert_buf.to_tensors(expert_buf.sample(batch))
            agent.update_discriminator(gb, eb)
            for _ in range(2):
                gb2 = gen_buf.to_tensors(gen_buf.sample(batch))
                agent.update_generator(gb2)
    return gen.cpu()


def run_profile(train_epochs: int, device: str) -> List[dict]:
    configs = [
        ("NoSN_init", False, 1.0, False),
        ("NoSN_stress_x3", False, 3.0, False),
        ("NoSN_trained", False, 1.0, True),
        ("SN_init", True, 1.0, False),
        ("SN_stress_x3", True, 3.0, False),
        ("SN_trained", True, 1.0, True),
    ]
    results: List[dict] = []
    for label, use_sn, scale, trained in configs:
        for seed in range(5):
            if trained:
                gen = train_generator_gail(use_sn, 20260622 + seed,
                                           epochs=train_epochs, device=device)
            else:
                torch.manual_seed(20260622 + seed)
                gen = Generator(
                    state_dim=STATE_DIM,
                    action_dim=DEFAULT_ACTION_DIM,
                    use_sn=use_sn,
                )
                amplify_weights(gen, scale)
            params = effective_linear_params(gen, exact_sn=use_sn)
            results.append({
                "config": label,
                "use_sn": use_sn,
                "weight_scale": scale,
                "gail_trained": trained,
                "train_epochs": train_epochs if trained else 0,
                "seed": seed,
                "spectral_bound": round(spectral_bound(gen, exact_sn=use_sn), 6),
                "logits_lipschitz": round(estimate_logits_lipschitz(gen, params), 6),
                "policy_tv_lipschitz": round(estimate_policy_tv_lipschitz(gen, params), 6),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-epochs", type=int, default=150,
                        help="GAIL epochs for the *_trained configs.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Lipschitz profile: spectral normalization (stress + real GAIL training)")
    print("=" * 65)
    rows = run_profile(args.train_epochs, args.device)

    by_config = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row)

    print(f"{'Config':<18} {'Bound':>10} {'Logit-L':>10} {'TV-L':>10}")
    print("-" * 55)
    for config, vals in by_config.items():
        bound = np.mean([v["spectral_bound"] for v in vals])
        logit_l = np.mean([v["logits_lipschitz"] for v in vals])
        tv_l = np.mean([v["policy_tv_lipschitz"] for v in vals])
        print(f"{config:<18} {bound:10.4f} {logit_l:10.4f} {tv_l:10.6f}")

    no_sn = np.mean([v["logits_lipschitz"] for v in by_config["NoSN_stress_x3"]])
    sn = np.mean([v["logits_lipschitz"] for v in by_config["SN_stress_x3"]])
    print("\nStress verdict:")
    print(f"  NoSN stress logits-L = {no_sn:.4f}")
    print(f"  SN stress logits-L   = {sn:.4f}")
    print(f"  Reduction            = {no_sn / max(sn, 1e-8):.2f}x")

    no_sn_tr = np.mean([v["spectral_bound"] for v in by_config["NoSN_trained"]])
    sn_tr = np.mean([v["spectral_bound"] for v in by_config["SN_trained"]])
    sn_tr_l = np.mean([v["logits_lipschitz"] for v in by_config["SN_trained"]])
    no_sn_tr_l = np.mean([v["logits_lipschitz"] for v in by_config["NoSN_trained"]])
    print("\nTrained verdict (real GAIL training):")
    print(f"  NoSN trained: bound={no_sn_tr:.4f}  logits-L={no_sn_tr_l:.4f}")
    print(f"  SN trained:   bound={sn_tr:.4f}  logits-L={sn_tr_l:.4f}")
    print(f"  SN keeps the theoretical bound <= 1 after training: {'YES' if sn_tr <= 1.0 + 1e-6 else 'NO'}")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lipschitz_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
