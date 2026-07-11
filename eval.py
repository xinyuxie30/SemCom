"""
eval.py — evaluation & visualization entry point.

[Phase 1 Fix]: Updated to use STATE_DIM = 452 and BFS-based MockExpert.

Usage:
  python eval.py --checkpoint ./checkpoints/phase1_checkpoint.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.dataloader import STATE_DIM
from models.generator import Generator, DEFAULT_ACTION_DIM
from models.discriminator import Discriminator
from models.gnn_encoder import SubGraphEncoder
from env.mdp_env import SemiMDPEnv, EnvConfig
from utils.metrics import semantic_fidelity, path_hit_rate, latency_breakdown
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--state_dim", type=int, default=STATE_DIM,
                        help=f"Flat state dimension (default: {STATE_DIM})")
    parser.add_argument("--action_dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    log = setup_logger()
    device = args.device

    # Load models
    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=128).to(device)
    gen = Generator(state_dim=args.state_dim, action_dim=args.action_dim).to(device)
    disc = Discriminator(state_dim=args.state_dim, action_dim=args.action_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    gnn.load_state_dict(ckpt["gnn_encoder"])
    gen.load_state_dict(ckpt["generator"])
    disc.load_state_dict(ckpt["discriminator"])
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # Latency breakdown (mock values — replace with real profiling in Phase 4)
    breakdown = latency_breakdown(
        slow_loop_ms=1200.0,    # simulated online intent extraction
        gnn_enc_ms=0.5,         # GNN encode
        policy_inf_ms=0.3,      # G-RML policy
        channel_ms=1.2,         # Tx + propagation
    )
    log.info("Latency breakdown (mock):")
    for k, v in breakdown.items():
        log.info(f"  {k}: {v:.2f} ms")

    log.info("Evaluation complete. Extend with full rollout in Phase 4.")


if __name__ == "__main__":
    main()
