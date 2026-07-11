"""Fast-slow dual-loop training entry point.

Slow loop:
  NL query -> keyword lookup -> intent_vec[64] + entities
  -> DynamicKBPruner.update_sub_graph()
  -> SubGraph(nodes, edges, intent_vec, node_embeddings)

Fast loop:
  EpisodeCollector.collect(sub_graph, intent_vec) -> trajectories
  -> GAIL discriminator/generator updates

Dimension contract:
  STATE_DIM = node_emb(256) + sub_emb(128) + intent(64) + csi(4) = 452
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from config.settings import load_config
from data.kg_interface import (
    INTENT_DIM,
    MockSubGraphProvider,
    SubGraph,
    register_provider,
    get_provider,
)
from data.dataloader import ReplayBuffer, EpisodeCollector, STATE_DIM
from env.mdp_env import SemiMDPEnv, EnvConfig, NODE_EMB_DIM, CSI_DIM
from models.gnn_encoder import SubGraphEncoder, SUB_EMB_DIM, IN_DIM
from models.generator import Generator, DEFAULT_ACTION_DIM
from models.discriminator import Discriminator
from agents.gail_agent import GAILAgent
from agents.slow_loop.mock_expert import MockExpert
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from agents.slow_loop.dynamic_kb_pruner import (
    DynamicKBPruner,
    GlobalKnowledgeGraph,
    get_entity_display_name,
)
from models.transe_embedder import create_transe_embedder  # [TransE Fix]
from utils.logger import setup_logger

# ── [Phase 3 Fix]: NL query pool — simulates dynamic user intent changes ────

QUERY_POOL: List[str] = [
    "The user is driving and suddenly a pedestrian crosses the street. "
    "High priority on radar and depth.",

    "Autonomous drone needs real-time video streaming for surveillance "
    "of a construction site with multiple workers.",

    "Vehicle platoon on highway requires low-latency V2V coordination "
    "for emergency braking and collision avoidance.",

    "Remote surgery robot needs ultra-reliable haptic feedback and "
    "high-definition video streaming with sub-millisecond latency.",

    "Smart factory AGV fleet requires real-time collision avoidance "
    "using lidar and camera fusion in a dynamic environment.",

    "Augmented reality maintenance guide for field technician "
    "repairing industrial equipment with remote expert assistance.",
]

DEFAULT_WN18_DIR = (
    "C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/WN18RR"
)


# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Semantic Communication - Fast-Slow Dual Loop (Phase 3)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "debug"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    # [Phase 3 Fix]: new CLI args
    parser.add_argument("--kg-dir", type=str, default=DEFAULT_WN18_DIR,
                        help="Path to WN18RR entity2id / relation2id / train2id directory")
    parser.add_argument("--slow-interval", type=int, default=10,
                        help="Trigger slow loop every N episodes (fast-slow metronome)")
    parser.add_argument("--sim-threshold", type=float, default=0.3,
                        help="Cosine similarity threshold for OOD injection [0, 1]")
    parser.add_argument("--max-subgraph-nodes", type=int, default=64,
                        help="Max nodes in pruned sub-graph")
    args = parser.parse_args()

    # Config
    cfg = load_config(args.config) if args.config else load_config()
    if args.mode == "debug":
        args.epochs = min(args.epochs, 8)
        args.slow_interval = min(args.slow_interval, 3)

    # Seed
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Logger
    log = setup_logger()

    # Device
    device = args.device
    log.info("=" * 60)
    log.info(f"Phase 3 - Fast-Slow Dual Loop  |  Device: {device}  |  Mode: {args.mode}")
    log.info(f"Epochs: {args.epochs}  |  Slow interval: {args.slow_interval}  |  "
             f"Sim threshold: {args.sim_threshold}")
    log.info("=" * 60)

    # ── Dimension contract ─────────────────────────────────────────────────
    dims = cfg.get("model_dims", {})
    node_emb_dim = dims.get("node_emb_dim", NODE_EMB_DIM)
    sub_emb_dim  = dims.get("sub_emb_dim", SUB_EMB_DIM)
    intent_dim   = dims.get("intent_dim", INTENT_DIM)
    csi_dim      = dims.get("csi_dim", CSI_DIM)
    state_dim    = dims.get("state_dim", STATE_DIM)
    action_dim   = dims.get("action_dim", DEFAULT_ACTION_DIM)
    assert state_dim == node_emb_dim + sub_emb_dim + intent_dim + csi_dim, \
        f"STATE_DIM {state_dim} != sum {node_emb_dim + sub_emb_dim + intent_dim + csi_dim}"
    log.info(f"[Contract] STATE_DIM = {node_emb_dim}+{sub_emb_dim}+{intent_dim}+{csi_dim} = {state_dim}")

    # ═════════════════════════════════════════════════════════════════════
    # Slow Loop Initialization [Phase 3 Fix]
    # ═════════════════════════════════════════════════════════════════════

    log.info("=" * 40 + "\n[Slow Loop] Initializing...")

    # 1. Keyword intent extractor. No LLM/LAM API is used.
    extractor = IntentExtractor()
    log.info("  Intent extractor: keyword lookup")

    # 2. Global Knowledge Graph
    log.info(f"  Loading global KG from: {args.kg_dir}")
    global_kg = GlobalKnowledgeGraph(args.kg_dir).load()
    log.info(f"  KG: {len(global_kg.entity2id):,} entities, "
             f"{len(global_kg.relation2id)} relations, "
             f"{len(global_kg.triples):,} triples, "
             f"{global_kg.graph.number_of_edges():,} edges")

    # 3. Dynamic KB Pruner [TransE Fix]: use semantic embeddings
    try:
        transe_emb = create_transe_embedder("checkpoints")
        log.info(f"  TransE embedder loaded: {transe_emb.entity_emb.shape[0]:,} entities")
    except FileNotFoundError:
        transe_emb = None
        log.info("  TransE embedder not found, falling back to char n-gram EntityEmbedder")

    pruner = DynamicKBPruner(
        global_kg=global_kg,
        max_subgraph_nodes=args.max_subgraph_nodes,
        max_hops=2,
        similarity_threshold=args.sim_threshold,
        hub_k=2,
        transe_embedder=transe_emb,          # [TransE Fix]
    )
    log.info(f"  Pruner: max_nodes={args.max_subgraph_nodes}, "
             f"sim_threshold={args.sim_threshold}")

    # 4. Also keep MockSubGraphProvider for fallback / expert buffer pre-fill
    slow_cfg = cfg.get("slow_loop", {})
    kg_cfg = slow_cfg.get("kg", {})
    emb_dim = kg_cfg.get("embedding_dim", IN_DIM)
    mock_provider = MockSubGraphProvider(embedding_dim=emb_dim)
    register_provider(mock_provider)

    # 5. Run initial slow-loop to get first sub-graph [Phase 3 Fix]
    initial_query = QUERY_POOL[0]
    intent_response = extractor.query(initial_query)
    current_intent_vec, current_entities = parse_intent(intent_response)
    current_sub_graph = pruner.update_sub_graph(current_entities, current_intent_vec)
    current_nl_query = initial_query
    log.info(f"  Initial slow-loop: intent='{intent_response['primary_intent']}', "
             f"entities={current_entities}, "
             f"sub_graph={len(current_sub_graph.nodes)} nodes / "
             f"{len(current_sub_graph.edges)} edges, "
             f"injected={pruner.injected_count}")

    log.info("[Slow Loop] Ready.\n" + "=" * 40)

    # ═════════════════════════════════════════════════════════════════════
    # Fast Loop Components (Phase 1 contract, unchanged)
    # ═════════════════════════════════════════════════════════════════════

    log.info("[Fast Loop] Initializing...")

    # GNN encoder
    gnn_encoder = SubGraphEncoder(in_dim=emb_dim, hidden_dim=256,
                                  out_dim=sub_emb_dim).to(device)
    log.info(f"  GNN encoder: {sum(p.numel() for p in gnn_encoder.parameters())} params")

    # Env
    fast_cfg = cfg.get("fast_loop", {})
    mdp_cfg = fast_cfg.get("mdp", {})
    env_cfg = EnvConfig(
        max_hops=mdp_cfg.get("max_hops", 8),
        action_dim=action_dim,
        node_emb_dim=node_emb_dim,
        csi_dim=csi_dim,
    )
    env = SemiMDPEnv(env_cfg)

    # Generator & Discriminator
    generator = Generator(state_dim=state_dim, action_dim=action_dim)
    discriminator = Discriminator(state_dim=state_dim, action_dim=action_dim)
    log.info(f"  Generator: {sum(p.numel() for p in generator.parameters())} params")
    log.info(f"  Discriminator: {sum(p.numel() for p in discriminator.parameters())} params")

    # GAIL agent
    gail_cfg = fast_cfg.get("gail", {})
    agent = GAILAgent(
        generator=generator, discriminator=discriminator,
        gen_lr=gail_cfg.get("generator_lr", 3e-4),
        disc_lr=gail_cfg.get("discriminator_lr", 1e-4),
        entropy_coef=gail_cfg.get("entropy_coef", 0.01),
        device=device,
    )

    # Mock expert (BFS-based, works on any SubGraph)
    expert = MockExpert(gnn_encoder=gnn_encoder, node_emb_dim=node_emb_dim,
                        device=device)

    # Episode collector [Phase 3 Fix]: generator attached
    collector = EpisodeCollector(
        env=env, provider=mock_provider, gnn_encoder=gnn_encoder,
        state_dim=state_dim, max_hops=env_cfg.max_hops,
        generator=generator, device=device,
    )

    # Buffers
    gen_buffer = ReplayBuffer(capacity=gail_cfg.get("buffer_capacity", 10000))
    expert_buffer = ReplayBuffer(capacity=gail_cfg.get("buffer_capacity", 10000))
    batch_size = gail_cfg.get("batch_size", 128)

    # [Phase 3 Fix]: Pre-fill expert buffer on the INITIAL sub-graph
    num_expert_episodes = 50
    for i in range(num_expert_episodes):
        exp_traj = expert.sample_trajectory(current_sub_graph)
        if exp_traj:
            expert_buffer.push_batch(exp_traj)
    log.info(f"  Expert buffer: {len(expert_buffer)} transitions "
             f"(on initial sub-graph, {num_expert_episodes} episodes)")

    log.info("[Fast Loop] Ready.\n" + "=" * 40)

    # ═════════════════════════════════════════════════════════════════════
    # Training Loop — Fast-Slow Metronome [Phase 3 Fix]
    # ═════════════════════════════════════════════════════════════════════

    disc_losses: list[float] = []
    gen_losses: list[float] = []
    slow_trigger_count: int = 0
    prev_intent: str = intent_response.get("primary_intent", "")

    log.info(f"\n{'='*60}\nTraining starts ({args.epochs} epochs)\n{'='*60}")

    for epoch in range(args.epochs):
        # ── [Phase 3 Fix]: Slow Loop — trigger every SLOW_INTERVAL episodes
        #     or when intent changes ──────────────────────────────────────
        slow_triggered = False
        if epoch % args.slow_interval == 0 or epoch == 0:
            # Cycle through query pool
            q_idx = (epoch // args.slow_interval) % len(QUERY_POOL)
            nl_text = QUERY_POOL[q_idx]

            # Only re-run slow loop if query changed (avoids redundant keyword lookup)
            if nl_text != current_nl_query or epoch == 0:
                try:
                    intent_response = extractor.query(nl_text)
                    new_intent_vec, new_entities = parse_intent(intent_response)
                    new_sub_graph = pruner.update_sub_graph(new_entities, new_intent_vec)

                    # Detect intent switch
                    new_intent = intent_response.get("primary_intent", "")
                    intent_switched = (new_intent != prev_intent)
                    old_intent = prev_intent     # save for logging
                    prev_intent = new_intent

                    # Update current state
                    current_sub_graph = new_sub_graph
                    current_intent_vec = new_intent_vec
                    current_entities = new_entities
                    current_nl_query = nl_text
                    prev_intent = new_intent
                    slow_triggered = True
                    slow_trigger_count += 1

                    # [Phase 3 Fix]: Refresh expert buffer for new sub-graph
                    expert_buffer = ReplayBuffer(
                        capacity=gail_cfg.get("buffer_capacity", 10000))
                    for i in range(num_expert_episodes):
                        exp_traj = expert.sample_trajectory(current_sub_graph)
                        if exp_traj:
                            expert_buffer.push_batch(exp_traj)

                    if intent_switched:
                        log.info(
                            f"[Slow Loop] Epoch {epoch+1:4d} | "
                            f"INTENT SWITCH: '{old_intent}' -> '{new_intent}' | "
                            f"query='{nl_text[:50]}...' | "
                            f"entities={new_entities} | "
                            f"sub_graph: {len(current_sub_graph.nodes)}n/"
                            f"{len(current_sub_graph.edges)}e | "
                            f"injected_total={pruner.injected_count}"
                        )
                except Exception as e:
                    log.warning(f"[Slow Loop] intent extraction failed, reusing previous sub-graph: {e}")

        # ── [Phase 3 Fix]: Fast Loop — collect trajectory on CURRENT sub-graph
        traj = collector.collect(
            query=f"ep_{epoch}",
            sub_graph=current_sub_graph,
            intent_vec_override=current_intent_vec,
        )
        gen_buffer.push_batch(traj)

        # ── GAIL updates ──────────────────────────────────────────────────
        if len(gen_buffer) < batch_size or len(expert_buffer) < batch_size:
            if (epoch + 1) % max(1, args.slow_interval) == 0 or epoch == 0:
                log.info(
                    f"Epoch {epoch+1:4d} | Buffering... "
                    f"(gen: {len(gen_buffer)}/{batch_size}, "
                    f"exp: {len(expert_buffer)}/{batch_size})"
                    + (f" [slow triggered]" if slow_triggered else "")
                )
            continue

        # Discriminator update
        gen_batch = gen_buffer.to_tensors(gen_buffer.sample(batch_size))
        exp_batch = expert_buffer.to_tensors(expert_buffer.sample(batch_size))
        d_loss = agent.update_discriminator(gen_batch, exp_batch)
        disc_losses.append(d_loss)

        # Generator update (3:1 ratio)
        for _ in range(3):
            gb = gen_buffer.to_tensors(gen_buffer.sample(batch_size))
            g_info = agent.update_generator(gb)
        gen_losses.append(g_info["total_loss"])

        # Logging
        log_interval = max(1, args.slow_interval)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            flag = " [SLOW]" if slow_triggered else ""
            log.info(
                f"Epoch {epoch+1:4d}/{args.epochs}{flag} | "
                f"D_loss: {d_loss:.4f} | "
                f"G_loss: {g_info['total_loss']:.4f} | "
                f"Entropy: {g_info['entropy']:.4f} | "
                f"GenBuf: {len(gen_buffer)} | "
                f"Injected: {pruner.injected_count}"
            )

    # ═════════════════════════════════════════════════════════════════════
    # Save & Summary
    # ═════════════════════════════════════════════════════════════════════

    log.info(f"\n{'='*60}")
    log.info("Phase 3 training complete.")
    log.info(f"  Slow-loop triggers: {slow_trigger_count}")
    log.info(f"  Total OOD nodes injected: {pruner.injected_count}")
    log.info(f"  Final KG size: {global_kg.graph.number_of_nodes():,} nodes, "
             f"{global_kg.graph.number_of_edges():,} edges")
    if disc_losses:
        log.info(f"  Final D_loss: {disc_losses[-1]:.4f}")
    if gen_losses:
        log.info(f"  Final G_loss: {gen_losses[-1]:.4f}")
    log.info(f"{'='*60}")

    # Checkpoint
    save_dir = Path(cfg.get("training", {}).get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "phase3_checkpoint.pt"
    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "gnn_encoder": gnn_encoder.state_dict(),
        "disc_losses": disc_losses,
        "gen_losses": gen_losses,
        "injected_count": pruner.injected_count,
        "slow_trigger_count": slow_trigger_count,
    }, ckpt_path)
    log.info(f"Checkpoint saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
