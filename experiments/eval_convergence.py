"""
eval_convergence.py — Production-grade multi-seed convergence evaluation
(FAIR shared-task protocol, cf. E3_FAIR_DESIGN_20260624.md).

Every config trains GAIL on its OWN pruned sub-graph, but ALL configs are
assigned the SAME neutral full-KG reference task (multi-seed 2-hop reasoning
hop, defined independently of any pruner). Convergence is therefore measured as
convergence-to-task-competence on an identical, method-independent task:

  - task_goal_rate : fraction of eval rollouts whose walked path reaches the
                     neutral target (headline convergence metric);
  - eval_reward    : raw Semi-MDP reward (diagnostic only — NOT comparable
                     across configs when a graph lacks the task);
  - g_loss/d_loss/entropy : GAIL adversarial-training stability.

Mechanism under test: a transmitted sub-graph that contains the reference task
lets MockExpert demonstrate the true path and GAIL imitate it; a graph without
the task cannot yield such demonstrations, so no amount of adversarial-loss
convergence produces task competence.

Output: experiments/results/convergence_data.csv

Usage:
  python experiments/eval_convergence.py --epochs 500 --seeds 3
  python experiments/eval_convergence.py --epochs 100 --seeds 3 --device cpu  # dry-run
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.kg_interface import MockSubGraphProvider, register_provider, INTENT_DIM
from data.dataloader import ReplayBuffer, EpisodeCollector, STATE_DIM
from env.mdp_env import SemiMDPEnv, EnvConfig, NODE_EMB_DIM, CSI_DIM
from models.gnn_encoder import SubGraphEncoder, SUB_EMB_DIM, IN_DIM
from models.generator import Generator, DEFAULT_ACTION_DIM
from models.discriminator import Discriminator
from agents.gail_agent import GAILAgent
from agents.slow_loop.mock_expert import MockExpert
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from agents.slow_loop.dynamic_kb_pruner import DynamicKBPruner, GlobalKnowledgeGraph, EntityEmbedder
from experiments.eval_end_to_end import (
    NoPruningProvider, HAKEBaselinePruner, match_seeds_full_kg,
    build_ground_truth_task, apply_reference_task, prune_with_method, has_path,
)
from models.transe_embedder import create_transe_embedder
from utils.logger import setup_logger

# ── Config ──────────────────────────────────────────────────────────────────

QUERY_POOL = [
    "The user is driving and suddenly a pedestrian crosses the street. "
    "High priority on radar and depth.",
    "Autonomous drone needs real-time video streaming for surveillance "
    "of a construction site with multiple workers.",
    "Vehicle platoon on highway requires low-latency V2V coordination "
    "for emergency braking and collision avoidance.",
]

DEFAULT_WN18 = "C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/WN18RR"
BATCH_SIZE = 64
NUM_EXPERT_EPISODES = 40
SLOW_INTERVAL = 15
CHECKPOINT_EVERY = 200      # save partial results every N epochs
EVAL_INTERVAL = 25          # task-level policy eval every N epochs
EVAL_EPISODES = 10          # rollouts per eval point

_TRANSE_CACHE: Dict[str, object] = {}


def load_transe_embedder_cached():
    """Load the WN18RR TransE embedder once (system config, cf. train.py/E5)."""
    if "wn18rr" not in _TRANSE_CACHE:
        try:
            _TRANSE_CACHE["wn18rr"] = create_transe_embedder("checkpoints", dataset="WN18RR")
        except FileNotFoundError:
            print("  WARNING: WN18RR TransE checkpoint not found; falling back to "
                  "char n-gram matching.")
            _TRANSE_CACHE["wn18rr"] = None
    return _TRANSE_CACHE["wn18rr"]


# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_seed(device: str, epochs: int, sim_threshold: float,
                 label: str, kg_dir: str, seed: int,
                 node_emb_dim: int, state_dim: int, action_dim: int,
                 sub_emb_dim: int, intent_dim: int, csi_dim: int,
                 out_dir: Path,
                 task_kg: GlobalKnowledgeGraph,
                 shared_embedder: EntityEmbedder,
                 pruner_builder=None,
                 reference_hops: int = 2,
                 ) -> List[Dict]:
    """
    Run ONE training session for a given seed and pruner configuration.
    Returns per-epoch metrics as a list of dicts.

    `pruner_builder(kg, seed)` constructs the sub-graph provider for this config.
    The reference task is built on the pristine `task_kg` (FULL KG, independent
    of any pruner — E3 fair protocol) and injected into every config's sub-graph
    via apply_reference_task, so training and evaluation aim at the SAME neutral
    start->target for all methods. task_goal_rate (policy walks to the neutral
    target) is the headline convergence metric; adversarial losses alone do not
    evidence policy convergence.

    All exceptions are caught; partial results are written to a per-seed CSV
    before re-raising, ensuring no data is lost on crash.
    """
    log = setup_logger()
    set_seed(seed)

    # Per-seed partial output
    partial_path = out_dir / f"_partial_{label}_seed{seed}.csv"

    try:
        # --- Build components ---
        global_kg = GlobalKnowledgeGraph(kg_dir).load()
        extractor = IntentExtractor()
        if pruner_builder is not None:
            pruner = pruner_builder(global_kg, seed)
        else:
            pruner = DynamicKBPruner(
                global_kg=global_kg, max_subgraph_nodes=64, max_hops=2,
                similarity_threshold=sim_threshold, hub_k=2, enable_injection=True,
            )
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
            state_dim=state_dim, max_hops=8, generator=gen, device=device,
        )

        def build_task(query: str):
            """Shared neutral full-KG task (E3 fair protocol) + this config's graph."""
            resp = extractor.query(query)
            intent_local, entities_local = parse_intent(resp)
            seeds = match_seeds_full_kg(entities_local, task_kg, shared_embedder)
            t_start, t_target, reference, t_dist, _mode = build_ground_truth_task(
                task_kg, seeds, intent_local, shared_embedder, max_hops=reference_hops)
            candidate = prune_with_method(pruner, entities_local, intent_local,
                                          seeds, reference_hops)
            apply_reference_task(candidate, reference, t_start, t_target, int(t_dist))
            return intent_local, candidate, t_start, t_target

        # --- Initial slow-loop (shared neutral task) ---
        intent_vec, sub_graph, task_start, task_target = build_task(QUERY_POOL[0])

        # --- Expert buffer ---
        expert_buf = ReplayBuffer(capacity=20000)
        for _ in range(NUM_EXPERT_EPISODES):
            traj = expert.sample_trajectory(sub_graph)
            if traj:
                expert_buf.push_batch(traj)

        gen_buf = ReplayBuffer(capacity=20000)
        records: List[Dict] = []

        for epoch in range(epochs):
            # Slow loop
            if epoch > 0 and epoch % SLOW_INTERVAL == 0:
                q_idx = (epoch // SLOW_INTERVAL) % len(QUERY_POOL)
                try:
                    intent_vec, sub_graph, task_start, task_target = build_task(
                        QUERY_POOL[q_idx])
                    expert_buf = ReplayBuffer(capacity=20000)
                    for _ in range(NUM_EXPERT_EPISODES):
                        traj = expert.sample_trajectory(sub_graph)
                        if traj:
                            expert_buf.push_batch(traj)
                except Exception:
                    pass

            # Fast loop
            traj = collector.collect(
                query=f"s{seed}_e{epoch}", sub_graph=sub_graph,
                intent_vec_override=intent_vec,
            )
            gen_buf.push_batch(traj)

            g_val, d_val, ent_val = float("nan"), float("nan"), float("nan")

            if len(gen_buf) >= BATCH_SIZE and len(expert_buf) >= BATCH_SIZE:
                gen_batch = gen_buf.to_tensors(gen_buf.sample(BATCH_SIZE))
                exp_batch = expert_buf.to_tensors(expert_buf.sample(BATCH_SIZE))
                d_val = agent.update_discriminator(gen_batch, exp_batch)
                for _ in range(2):
                    gb = gen_buf.to_tensors(gen_buf.sample(BATCH_SIZE))
                    info = agent.update_generator(gb)
                g_val = info["total_loss"]
                ent_val = info["entropy"]

            # Task-level policy eval on the SHARED neutral task, every EVAL_INTERVAL.
            # task_goal_rate counts rollouts whose walked path reaches the neutral
            # target — a config whose sub-graph lacks the target can never score.
            eval_reward = ""
            task_goal_rate = ""
            if epoch % EVAL_INTERVAL == 0 or epoch == epochs - 1:
                eval_rs = []
                goals = 0
                for _ in range(EVAL_EPISODES):
                    etraj = collector.collect(sub_graph=sub_graph,
                                              intent_vec_override=intent_vec)
                    if etraj:
                        eval_rs.append(sum(e.reward for e in etraj))
                        final_path = list(getattr(env, "_path_history", []))
                        if task_target and task_target in final_path:
                            goals += 1
                if eval_rs:
                    eval_reward = round(float(np.mean(eval_rs)), 4)
                    task_goal_rate = round(goals / len(eval_rs), 4)

            records.append({
                "epoch": epoch, "seed": seed, "model": label,
                "sim_threshold": sim_threshold,
                "g_loss": round(g_val, 6) if not np.isnan(g_val) else "",
                "d_loss": round(d_val, 6) if not np.isnan(d_val) else "",
                "entropy": round(float(ent_val), 6) if not np.isnan(ent_val) else "",
                "eval_reward": eval_reward,
                "task_goal_rate": task_goal_rate,
                "task_path_in_subgraph": int(has_path(sub_graph, task_start, task_target)),
                "task_start": task_start,
                "task_target": task_target,
                "gen_buffer_size": len(gen_buf),
                "expert_buffer_size": len(expert_buf),
                "sub_graph_nodes": len(sub_graph.nodes),
                "sub_graph_edges": len(sub_graph.edges),
                "injected_total": getattr(pruner, "injected_count", 0),
            })

            # Checkpoint partial results
            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                _write_partial(partial_path, records)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                log.info(f"[{label} seed={seed}] Epoch {epoch+1:4d}/{epochs} | "
                         f"G={g_val:.4f} | D={d_val:.4f} | Ent={ent_val:.4f} | "
                         f"inj={getattr(pruner, 'injected_count', 0)}")

        # Final write
        _write_partial(partial_path, records)
        return records

    except Exception:
        traceback.print_exc()
        # Try to save whatever we have
        try:
            if 'records' in locals() and records:
                _write_partial(partial_path, records)
        except Exception:
            pass
        raise


def _write_partial(path: Path, records: List[Dict]) -> None:
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation: align by epoch, compute mean ± std across seeds
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_across_seeds(all_records: List[Dict], label: str,
                           metric_cols: List[str]) -> List[Dict]:
    """
    Given flat records from multiple seeds (same model label),
    group by epoch and compute mean/std for each metric column.
    Only epochs where ALL seeds report non-empty values are included
    in the statistics.
    """
    from collections import defaultdict

    # Group: epoch → list of (seed, row)
    by_epoch: Dict[int, List[Dict]] = defaultdict(list)
    for r in all_records:
        if r["model"] == label:
            by_epoch[r["epoch"]].append(r)

    aggregated: List[Dict] = []
    num_seeds_total = len(set(r["seed"] for r in all_records if r["model"] == label))

    for epoch in sorted(by_epoch.keys()):
        rows = by_epoch[epoch]
        agg = {"epoch": epoch, "model": label, "num_seeds_reporting": len(rows)}

        for col in metric_cols:
            vals = []
            for r in rows:
                v = r.get(col, "")
                if v != "" and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            if len(vals) >= max(2, num_seeds_total // 2):  # at least half the seeds
                agg[f"{col}_mean"] = round(float(np.mean(vals)), 6)
                agg[f"{col}_std"] = round(float(np.std(vals)), 6) if len(vals) > 1 else 0.0
                agg[f"{col}_n"] = len(vals)
            else:
                agg[f"{col}_mean"] = ""
                agg[f"{col}_std"] = ""
                agg[f"{col}_n"] = 0

        aggregated.append(agg)

    return aggregated


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-seed Convergence Eval — Phase 4 Production")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=4,
                        help="Number of random seeds (3-5 recommended)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kg-dir", type=str, default=DEFAULT_WN18)
    parser.add_argument("--seeds-start", type=int, default=0,
                        help="Starting seed index (for parallel runs: 0, 4, 8, ...)")
    args = parser.parse_args()

    device = args.device
    seed_list = [args.seeds_start + i for i in range(args.seeds)]
    log = setup_logger()

    # Dimensions
    node_emb_dim, sub_emb_dim, intent_dim, csi_dim = NODE_EMB_DIM, SUB_EMB_DIM, INTENT_DIM, CSI_DIM
    state_dim, action_dim = STATE_DIM, DEFAULT_ACTION_DIM
    assert state_dim == node_emb_dim + sub_emb_dim + intent_dim + csi_dim

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info(f"Production Convergence Eval | {datetime.now():%Y-%m-%d %H:%M}")
    log.info(f"  Seeds: {seed_list}  |  Epochs per seed: {args.epochs}  |  Device: {device}")
    log.info(f"  STATE_DIM = {state_dim}  |  BATCH_SIZE = {BATCH_SIZE}")
    log.info("=" * 65)

    # Configs (E3 fair opponents): intent-driven dynamic pruning (E3 formal
    # config: char n-gram matching — the SAME embedder family as the neutral
    # task's match_seeds_full_kg, so pruner seeds align with task seeds;
    # threshold 0.30, N=2, <=64 nodes) vs seed-aware static-layer HAKE vs
    # budgeted no-pruning (intent-blind 512-node KG slice). All three are
    # trained identically and evaluated on the SAME neutral full-KG task; the
    # paper claim is convergence-to-task-competence, not raw-loss speed.
    # (The TransE embedder belongs to E5's OOD-detection setting, not here:
    # NLTK-resolved seeds diverge from the shared task seeds and break coverage.)

    # Pristine read-only KG for the neutral task definition (never mutated by any
    # pruner — Ours' OOD injection happens on its own KG instance).
    task_kg = GlobalKnowledgeGraph(args.kg_dir).load()
    shared_embedder = EntityEmbedder(dim=128)

    def build_dynamic(kg, seed):
        return DynamicKBPruner(
            global_kg=kg, max_subgraph_nodes=64, max_hops=2,
            similarity_threshold=0.30, hub_k=2, enable_injection=True,
        )

    def build_hake(kg, seed):
        return HAKEBaselinePruner(kg, num_layers=3, seed=seed)

    def build_noprune(kg, seed):
        return NoPruningProvider(kg, max_nodes=512, seed=seed)

    configs = [
        ("dynamic_pruned", 0.30, build_dynamic),
        ("hake_static_layer", "", build_hake),
        ("budgeted_no_pruning", "", build_noprune),
    ]

    all_raw: List[Dict] = []

    for cfg_label, cfg_threshold, cfg_builder in configs:
        log.info(f"\n{'─'*50}\n  Model: {cfg_label}  (threshold={cfg_threshold})\n{'─'*50}")
        for si, seed in enumerate(seed_list):
            log.info(f"  [{si+1}/{len(seed_list)}] Seed={seed} ...")
            try:
                records = run_one_seed(
                    device=device, epochs=args.epochs,
                    sim_threshold=cfg_threshold, label=cfg_label,
                    kg_dir=args.kg_dir, seed=seed,
                    node_emb_dim=node_emb_dim, state_dim=state_dim,
                    action_dim=action_dim, sub_emb_dim=sub_emb_dim,
                    intent_dim=intent_dim, csi_dim=csi_dim,
                    out_dir=out_dir,
                    task_kg=task_kg,
                    shared_embedder=shared_embedder,
                    pruner_builder=cfg_builder,
                )
                all_raw.extend(records)
                log.info(f"  [{si+1}/{len(seed_list)}] Seed={seed} DONE "
                         f"({len(records)} epochs)")
            except Exception as e:
                log.error(f"  [{si+1}/{len(seed_list)}] Seed={seed} FAILED: {e}")
                log.error(f"  Partial data saved in {out_dir}/_partial_*_seed{seed}.csv")

    # ── Aggregate ───────────────────────────────────────────────────────────
    metric_cols = ["g_loss", "d_loss", "entropy", "eval_reward", "task_goal_rate"]

    log.info(f"\n{'='*65}\nAggregating across seeds...\n{'='*65}")

    agg_rows: List[Dict] = []
    for cfg_label, _, _ in configs:
        agg = aggregate_across_seeds(all_raw, cfg_label, metric_cols)
        agg_rows.extend(agg)
        valid_epochs = sum(1 for r in agg if r.get("g_loss_mean", "") != "")
        log.info(f"  {cfg_label}: {len(agg)} epochs total, "
                 f"{valid_epochs} with valid GAIL stats across seeds")

    # ── Save aggregated ─────────────────────────────────────────────────────
    if agg_rows:
        agg_fieldnames = list(agg_rows[0].keys())
        agg_path = out_dir / "convergence_data.csv"
        with open(agg_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=agg_fieldnames)
            w.writeheader()
            w.writerows(agg_rows)
        log.info(f"\nAggregated data -> {agg_path}  ({len(agg_rows)} rows)")

    # ── Also save raw merged ────────────────────────────────────────────────
    if all_raw:
        raw_fieldnames = list(all_raw[0].keys())
        raw_path = out_dir / "convergence_raw_all_seeds.csv"
        with open(raw_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=raw_fieldnames)
            w.writeheader()
            w.writerows(all_raw)
        log.info(f"Raw merged data -> {raw_path}  ({len(all_raw)} rows)")

    # ── Verdict: convergence-to-task-competence on the SHARED neutral task ──
    log.info(f"\n{'='*65}\nFinal task competence (shared neutral task, last eval point)\n{'='*65}")
    verdict: Dict[str, Tuple[float, float, float]] = {}
    for cfg_label, _, _ in configs:
        rows = [r for r in all_raw if r["model"] == cfg_label and r["task_goal_rate"] != ""]
        if not rows:
            continue
        last_ep = max(int(r["epoch"]) for r in rows)
        last = [float(r["task_goal_rate"]) for r in rows if int(r["epoch"]) == last_ep]
        first_ep = min(int(r["epoch"]) for r in rows)
        first = [float(r["task_goal_rate"]) for r in rows if int(r["epoch"]) == first_ep]
        path_in = float(np.mean([int(r["task_path_in_subgraph"]) for r in all_raw
                                 if r["model"] == cfg_label]))
        verdict[cfg_label] = (float(np.mean(first)), float(np.mean(last)), path_in)
        log.info(f"  {cfg_label:22s}: task_goal_rate {np.mean(first):.3f} -> {np.mean(last):.3f} "
                 f"(epoch {first_ep} -> {last_ep}) | task-path-in-subgraph={path_in:.2f}")
    ours = verdict.get("dynamic_pruned")
    if ours:
        best_base = max((v[1] for k, v in verdict.items() if k != "dynamic_pruned"),
                        default=0.0)
        wins = ours[1] > best_base
        log.info(f"  VERDICT: dynamic_pruned trains to task competence "
                 f"{ours[1]:.3f} vs best baseline {best_base:.3f} -> "
                 f"{'Ours WINS (fair shared task)' if wins else 'Ours does NOT clearly win'}")

    log.info(f"\nDone. {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
