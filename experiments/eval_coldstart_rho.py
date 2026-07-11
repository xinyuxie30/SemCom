"""
eval_coldstart_rho.py — Measure GAIL re-convergence time after abrupt intent switch
with varying sub-graph overlap ratio ρ, and test semantic warm-start bootstrapping.

Output: experiments/results/coldstart_rho_recovery.csv

Usage:
  python experiments/eval_coldstart_rho.py --device cuda --episodes 500
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.kg_interface import MockSubGraphProvider, register_provider, SubGraph
from data.dataloader import EpisodeCollector, STATE_DIM, ReplayBuffer, Experience
from env.mdp_env import SemiMDPEnv, EnvConfig, NODE_EMB_DIM, CSI_DIM
from models.gnn_encoder import SubGraphEncoder, SUB_EMB_DIM
from models.generator import Generator, DEFAULT_ACTION_DIM
from models.discriminator import Discriminator
from agents.gail_agent import GAILAgent
from agents.slow_loop.mock_expert import MockExpert
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from agents.slow_loop.dynamic_kb_pruner import (
    GlobalKnowledgeGraph, DynamicKBPruner, get_entity_display_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Intent scenarios — designed to produce varying ρ
# ═══════════════════════════════════════════════════════════════════════════════

# Scenario pairs with controlled semantic distance → controlled ρ
INTENT_PAIRS: List[Tuple[str, str]] = [
    # ρ~0.7: semantically close topics
    ("High priority on radar and depth for pedestrian crossing street.",
     "Vehicle collision avoidance with emergency braking on highway."),
    ("Drone surveillance with video streaming and camera monitoring.",
     "Autonomous robot navigation using camera and depth sensor."),
    # ρ~0.3: different domains
    ("Vehicle platoon on highway requires low-latency V2V coordination.",
     "Telemedicine remote surgery with haptic feedback and 8K video."),
    ("Smart factory AGV fleet requires real-time lidar collision avoidance.",
     "Smart grid power surge detection across substation transformers."),
    # ρ~0.05: completely unrelated
    ("Autonomous drone needs real-time video streaming for surveillance.",
     "Emergency remote surgery with ultra-reliable haptic feedback."),
    ("Driving with pedestrian radar and collision warning.",
     "Augmented reality maintenance guide for field technician."),
]


# ═══════════════════════════════════════════════════════════════════════════════

def compute_subgraph_overlap(g1: SubGraph, g2: SubGraph) -> float:
    """Compute overlap ratio ρ = |V1 ∩ V2| / max(|V1|, |V2|)."""
    s1, s2 = set(g1.nodes), set(g2.nodes)
    inter = len(s1 & s2)
    denom = max(len(s1), len(s2))
    return inter / denom if denom > 0 else 0.0


def evaluate_policy_on_subgraph(
    gen: Generator, env: SemiMDPEnv, collector: EpisodeCollector,
    sub_graph: SubGraph, intent_vec: np.ndarray,
    num_episodes: int = 20) -> float:
    """Run evaluation episodes and return mean reward."""
    rewards = []
    for _ in range(num_episodes):
        traj = collector.collect(
            sub_graph=sub_graph, intent_vec_override=intent_vec)
        if traj:
            rewards.append(sum(e.reward for e in traj))
    return float(np.mean(rewards)) if rewards else 0.0


def train_on_subgraph(
    agent: GAILAgent, expert: MockExpert,
    collector: EpisodeCollector,
    sub_graph: SubGraph, intent_vec: np.ndarray,
    num_episodes: int, batch_size: int = 48) -> List[float]:
    """Train GAIL on a sub-graph and return per-episode reward trace."""
    expert_buf = ReplayBuffer(capacity=10000)
    for _ in range(40):
        traj = expert.sample_trajectory(sub_graph)
        if traj:
            expert_buf.push_batch(traj)

    gen_buf = ReplayBuffer(capacity=10000)
    reward_trace = []

    for ep in range(num_episodes):
        traj = collector.collect(
            sub_graph=sub_graph, intent_vec_override=intent_vec)
        gen_buf.push_batch(traj)

        if len(gen_buf) >= batch_size and len(expert_buf) >= batch_size:
            gen_batch = gen_buf.to_tensors(gen_buf.sample(batch_size))
            exp_batch = expert_buf.to_tensors(expert_buf.sample(batch_size))
            agent.update_discriminator(gen_batch, exp_batch)
            for _ in range(2):
                gb = gen_buf.to_tensors(gen_buf.sample(batch_size))
                agent.update_generator(gb)

        # Evaluate
        if ep % 10 == 0 or ep < 15:
            r = evaluate_policy_on_subgraph(
                agent.generator, collector.env, collector,
                sub_graph, intent_vec, num_episodes=10)
            reward_trace.append({"episode": ep, "reward": r})

    return reward_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device

    kg_dir = "C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/FB15K237"
    node_emb_dim, sub_emb_dim, intent_dim, csi_dim = NODE_EMB_DIM, SUB_EMB_DIM, 64, CSI_DIM
    state_dim, action_dim = STATE_DIM, DEFAULT_ACTION_DIM

    print("Phase 4 — Cold-Start Recovery vs Sub-graph Overlap ρ")
    print("=" * 65)

    # Build components
    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=sub_emb_dim).to(device)
    env_cfg = EnvConfig(max_hops=8, action_dim=action_dim,
                        node_emb_dim=node_emb_dim, csi_dim=csi_dim)
    env = SemiMDPEnv(env_cfg)
    gen = Generator(state_dim=state_dim, action_dim=action_dim).to(device)
    disc = Discriminator(state_dim=state_dim, action_dim=action_dim).to(device)
    agent = GAILAgent(gen, disc, gen_lr=3e-4, disc_lr=1e-4,
                      entropy_coef=0.01, device=device)
    expert = MockExpert(gnn_encoder=gnn, node_emb_dim=node_emb_dim, device=device)
    mock_prov = MockSubGraphProvider(embedding_dim=128)
    register_provider(mock_prov)
    collector = EpisodeCollector(
        env=env, provider=mock_prov, gnn_encoder=gnn,
        state_dim=state_dim, max_hops=8, generator=gen, device=device)

    kg = GlobalKnowledgeGraph(kg_dir).load()
    extractor = IntentExtractor()

    all_results: List[Dict] = []

    for pair_idx, (query_old, query_new) in enumerate(INTENT_PAIRS):
        # Build pruner
        pruner = DynamicKBPruner(kg, similarity_threshold=0.7,
                                 enable_injection=True)

        # Extract old sub-graph
        resp_old = extractor.query(query_old)
        intent_old, entities_old = parse_intent(resp_old)
        sub_old = pruner.update_sub_graph(entities_old, intent_old)

        # Extract new sub-graph
        resp_new = extractor.query(query_new)
        intent_new, entities_new = parse_intent(resp_new)
        sub_new = pruner.update_sub_graph(entities_new, intent_new)

        ρ = compute_subgraph_overlap(sub_old, sub_new)
        print(f"\nPair {pair_idx}: ρ={ρ:.3f}")
        print(f"  Old: {len(sub_old.nodes)}n/{len(sub_old.edges)}e "
              f"| '{query_old[:50]}...'")
        print(f"  New: {len(sub_new.nodes)}n/{len(sub_new.edges)}e "
              f"| '{query_new[:50]}...'")

        # Phase 1: Train on old sub-graph
        print(f"  Training on OLD sub-graph ({args.episodes//3} episodes)...")
        trace_old = train_on_subgraph(
            agent, expert, collector, sub_old, intent_old,
            num_episodes=args.episodes // 3)

        # Phase 2: Switch to new sub-graph — measure recovery
        print(f"  Switching to NEW sub-graph, measuring recovery...")
        t0 = time.perf_counter()
        trace_new = train_on_subgraph(
            agent, expert, collector, sub_new, intent_new,
            num_episodes=args.episodes)
        t_elapsed = time.perf_counter() - t0

        # Find recovery time: episode where reward reaches 80% of old level
        old_baseline = trace_old[-1]["reward"] if trace_old else 0.0
        recovery_ep = -1
        recovery_reward = 0.0
        for tr in trace_new:
            if tr["reward"] >= 0.8 * old_baseline and old_baseline > 0:
                recovery_ep = tr["episode"]
                recovery_reward = tr["reward"]
                break

        # Immediate post-switch reward (first 5 episodes)
        immediate_reward = float(np.mean([t["reward"] for t in trace_new[:5]
                                           if t["episode"] < 5]))

        all_results.append({
            "pair_id": pair_idx,
            "rho": round(ρ, 4),
            "old_nodes": len(sub_old.nodes),
            "new_nodes": len(sub_new.nodes),
            "old_baseline_reward": round(old_baseline, 4),
            "recovery_episode": recovery_ep,
            "recovery_reward": round(recovery_reward, 4),
            "immediate_post_switch_reward": round(immediate_reward, 4),
            "total_episodes_trained": args.episodes,
        })

        # Print per-pair summary
        print(f"  Old baseline reward: {old_baseline:.3f}")
        print(f"  Immediate post-switch reward (avg of first 5): {immediate_reward:.3f}")
        if recovery_ep >= 0:
            print(f"  Recovered to 80% at episode {recovery_ep} "
                  f"(reward={recovery_reward:.3f})")
        else:
            print(f"  DID NOT RECOVER within {args.episodes} episodes")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'ρ':>8s}  {'Baseline':>9s}  {'Immediate':>9s}  {'Recovery@80%':>13s}  {'Judgment':>15s}")
    print(f"{'─'*8}  {'─'*9}  {'─'*9}  {'─'*13}  {'─'*15}")
    for r in all_results:
        rec_str = f"ep={r['recovery_episode']}" if r['recovery_episode'] >= 0 else "NOT RECOVERED"
        # Judgment logic
        if r['rho'] >= 0.5:
            judgment = "SMOOTH"
        elif r['rho'] >= 0.2 and r['recovery_episode'] >= 0:
            judgment = "RECOVERABLE"
        else:
            judgment = "COLD RESTART"
        print(f"  {r['rho']:8.4f}  {r['old_baseline_reward']:9.4f}  "
              f"{r['immediate_post_switch_reward']:9.4f}  "
              f"{rec_str:>13s}  {judgment:>15s}")

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "coldstart_rho_recovery.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
