"""
eval_robustness.py — Production-grade OOD robustness & threshold ablation.

Experiment A (SNR sweep):  Injection ON vs OFF across SNR ∈ {-5, 0, 5, 10} dB.
Experiment B (Threshold ablation): Fixed SNR = 0 dB, sweep sim_threshold
  ∈ {0.30, 0.45, 0.60, 0.75, 0.90}. Records mean reward, fidelity, and
  injected node count per threshold. Identifies the optimal operating point
  balancing OOD injection against graph coherence.

Output:
  experiments/results/ood_robustness_snr.csv        (Experiment A raw)
  experiments/results/ood_robustness_snr_summary.csv (Experiment A aggregated)
  experiments/results/threshold_ablation.csv         (Experiment B)

Usage:
  python experiments/eval_robustness.py --episodes-per-config 30
  python experiments/eval_robustness.py --skip-snr  # only run threshold ablation
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.kg_interface import MockSubGraphProvider, register_provider, INTENT_DIM, SubGraph
from data.dataloader import EpisodeCollector, ReplayBuffer, STATE_DIM
from env.mdp_env import SemiMDPEnv, EnvConfig, NODE_EMB_DIM, CSI_DIM
from models.gnn_encoder import SubGraphEncoder, SUB_EMB_DIM
from models.generator import Generator, DEFAULT_ACTION_DIM
from models.discriminator import Discriminator
from agents.gail_agent import GAILAgent
from agents.slow_loop.mock_expert import MockExpert
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from agents.slow_loop.dynamic_kb_pruner import DynamicKBPruner, GlobalKnowledgeGraph
from models.transe_embedder import create_transe_embedder

# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_WN18 = "C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/WN18RR"
SNR_LEVELS = [-5, 0, 5, 10]
THRESHOLD_VALUES = [0.30, 0.45, 0.60, 0.75, 0.90]

_TRANSE_CACHE: Dict[str, object] = {}


def load_transe_embedder_cached():
    """Load the WN18RR TransE embedder once (system config, cf. train.py).

    With TransE, OOD detection is principled: keywords that NLTK cannot resolve
    to a WN18 entity are flagged is_ood and injected. The char n-gram fallback
    spuriously matches OOD words at low thresholds, which silently disables the
    injection ablation (injON == injOFF).
    """
    if "wn18rr" not in _TRANSE_CACHE:
        try:
            _TRANSE_CACHE["wn18rr"] = create_transe_embedder("checkpoints", dataset="WN18RR")
        except FileNotFoundError:
            print("  WARNING: WN18RR TransE checkpoint not found; falling back to "
                  "char n-gram matching (injection ablation may be vacuous).")
            _TRANSE_CACHE["wn18rr"] = None
    return _TRANSE_CACHE["wn18rr"]

OOD_QUERIES = [
    "Emergency telemedicine robot requires haptic feedback and 8K video "
    "with sub-millisecond latency for remote surgery.",
    "Autonomous underwater drone navigating using sonar and magnetic "
    "anomaly detection for pipeline inspection.",
    "Smart grid sensor network detects power surge and cascading failure "
    "across substation transformers in real time.",
]

OOD_SCENARIOS = [
    {"intent": "emergency_avoidance",
     "entities": ["telemedicine", "haptic_feedback", "surgery", "robot", "8k_video"]},
    {"intent": "sensor_query",
     "entities": ["underwater_drone", "sonar", "magnetic_anomaly", "pipeline", "navigation"]},
    {"intent": "telemetry_report",
     "entities": ["smart_grid", "power_surge", "cascading_failure", "substation", "transformer"]},
]


# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_gail_policy(env: SemiMDPEnv, collector: EpisodeCollector,
                      gnn: SubGraphEncoder, gen: Generator,
                      scenario_graphs: List[SubGraph],
                      scenario_intents: List[np.ndarray],
                      epochs: int, device: str,
                      batch_size: int = 48, num_expert: int = 40) -> None:
    """Train the GAIL policy on the scenario sub-graphs before evaluation.

    Robustness must be measured on a LEARNED policy, not a random-init network.
    Rotates through scenarios every 15 epochs (same pattern as E3/E6 training).
    """
    if epochs <= 0 or not scenario_graphs:
        return
    disc = Discriminator(state_dim=collector.state_dim,
                         action_dim=gen.action_dim).to(device)
    agent = GAILAgent(gen, disc, gen_lr=3e-4, disc_lr=1e-4,
                      entropy_coef=0.01, device=device)
    expert = MockExpert(gnn_encoder=gnn, node_emb_dim=env.cfg.node_emb_dim,
                        device=device)

    def refill(sub_graph: SubGraph) -> ReplayBuffer:
        buf = ReplayBuffer(capacity=10000)
        for _ in range(num_expert):
            traj = expert.sample_trajectory(sub_graph)
            if traj:
                buf.push_batch(traj)
        return buf

    idx = 0
    sub_graph = scenario_graphs[idx]
    intent_vec = scenario_intents[idx]
    expert_buf = refill(sub_graph)
    gen_buf = ReplayBuffer(capacity=10000)

    for ep in range(epochs):
        if ep > 0 and ep % 15 == 0:
            idx = (ep // 15) % len(scenario_graphs)
            sub_graph = scenario_graphs[idx]
            intent_vec = scenario_intents[idx]
            expert_buf = refill(sub_graph)

        traj = collector.collect(sub_graph=sub_graph, intent_vec_override=intent_vec)
        gen_buf.push_batch(traj)

        if len(gen_buf) >= batch_size and len(expert_buf) >= batch_size:
            gb = gen_buf.to_tensors(gen_buf.sample(batch_size))
            eb = expert_buf.to_tensors(expert_buf.sample(batch_size))
            agent.update_discriminator(gb, eb)
            for _ in range(2):
                gb2 = gen_buf.to_tensors(gen_buf.sample(batch_size))
                agent.update_generator(gb2)


def entity_coverage(sub_graph: SubGraph) -> float:
    """Fraction of intent entities semantically represented in the transmitted
    sub-graph (matched KG node present, or injected OOD node present).

    This is the representational benefit of OOD injection: with injection OFF the
    OOD concepts simply cannot appear in the sub-graph, regardless of reward.
    """
    trace = sub_graph.metadata.get("match_trace", [])
    if not trace:
        return 0.0
    nodes = set(sub_graph.nodes)
    covered = 0
    for row in trace:
        action = row.get("action")
        if action == "match" and row.get("best_match") in nodes:
            covered += 1
        elif action == "inject" and row.get("keyword") in nodes:
            covered += 1
    return covered / len(trace)


def run_episodes(env: SemiMDPEnv, collector: EpisodeCollector,
                 sub_graph: SubGraph, intent_vec: np.ndarray,
                 episodes: int) -> List[Dict]:
    """Run `episodes` independent rollouts and return per-episode metrics."""
    results: List[Dict] = []
    for ep in range(episodes):
        try:
            traj = collector.collect(
                query="eval", sub_graph=sub_graph,
                intent_vec_override=intent_vec,
            )
            if not traj:
                results.append({"cumulative_reward": 0.0, "hops": 0,
                                "reached_goal": False, "fidelity": 0.0})
            else:
                cum_r = sum(e.reward for e in traj)
                reached = any(e.reward > 5.0 for e in traj)
                fidelity = 1.0 if reached else max(0.0, min(1.0, cum_r / 10.0))
                results.append({
                    "cumulative_reward": round(cum_r, 4),
                    "hops": len(traj),
                    "reached_goal": reached,
                    "fidelity": round(fidelity, 4),
                })
        except Exception:
            results.append({"cumulative_reward": 0.0, "hops": 0,
                            "reached_goal": False, "fidelity": 0.0})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment A: SNR sweep
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_a_snr(device: str, kg_dir: str, episodes_per_config: int,
                     sim_threshold: float, max_hops: int,
                     node_emb_dim: int, state_dim: int, action_dim: int,
                     sub_emb_dim: int, intent_dim: int, csi_dim: int,
                     train_epochs: int = 150,
                     ) -> List[Dict]:
    """
    For each SNR and injection on/off, train a GAIL policy under that channel,
    then run evaluation episodes across all OOD scenarios.
    """
    all_rows: List[Dict] = []

    for snr in SNR_LEVELS:
        for inj_enabled in [True, False]:
            label = "injON" if inj_enabled else "injOFF"
            print(f"  SNR={snr:3d} dB, {label} ...", end=" ", flush=True)

            try:
                row_chunk = _evaluate_single_config(
                    device=device, kg_dir=kg_dir, snr_db=float(snr),
                    enable_injection=inj_enabled, sim_threshold=sim_threshold,
                    max_hops=max_hops,
                    episodes=episodes_per_config,
                    node_emb_dim=node_emb_dim, state_dim=state_dim,
                    action_dim=action_dim, sub_emb_dim=sub_emb_dim,
                    intent_dim=intent_dim, csi_dim=csi_dim,
                    train_epochs=train_epochs,
                )
                all_rows.extend(row_chunk)
                rewards = [r["cumulative_reward"] for r in row_chunk]
                goals = sum(1 for r in row_chunk if r["reached_goal"])
                print(f"avg_r={np.mean(rewards):.3f}  goals={goals}/{len(row_chunk)}")
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()

    return all_rows


def _evaluate_single_config(device: str, kg_dir: str, snr_db: float,
                            enable_injection: bool, sim_threshold: float,
                            max_hops: int,
                            episodes: int,
                            node_emb_dim: int, state_dim: int, action_dim: int,
                            sub_emb_dim: int, intent_dim: int, csi_dim: int,
                            train_epochs: int = 150,
                            ) -> List[Dict]:
    """Run all OOD scenarios under one (SNR, injection) config.

    The policy is GAIL-trained on the scenario sub-graphs first (train_epochs),
    then evaluated: robustness of a LEARNED policy, not a random-init network.
    """
    set_seed(42)
    global_kg = GlobalKnowledgeGraph(kg_dir).load()
    extractor = IntentExtractor()
    pruner = DynamicKBPruner(
        global_kg=global_kg, max_subgraph_nodes=64, max_hops=max_hops,
        similarity_threshold=sim_threshold, hub_k=2,
        enable_injection=enable_injection,
        transe_embedder=load_transe_embedder_cached(),
    )
    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=sub_emb_dim).to(device)
    gen = Generator(state_dim=state_dim, action_dim=action_dim).to(device)

    env_cfg = EnvConfig(max_hops=8, action_dim=action_dim,
                        node_emb_dim=node_emb_dim, csi_dim=csi_dim,
                        snr_db_low=snr_db, snr_db_high=snr_db + 0.1)
    env = SemiMDPEnv(env_cfg)

    mock_prov = MockSubGraphProvider(embedding_dim=128)
    register_provider(mock_prov)
    collector = EpisodeCollector(
        env=env, provider=mock_prov, gnn_encoder=gnn,
        state_dim=state_dim, max_hops=8, generator=gen, device=device,
    )

    # Build all scenario sub-graphs first (per-scenario injected delta, not the
    # pruner's cumulative counter).
    scenario_graphs: List[SubGraph] = []
    scenario_intents: List[np.ndarray] = []
    scenario_injected: List[int] = []
    for sidx, scenario in enumerate(OOD_SCENARIOS):
        intent_response = extractor.query(OOD_QUERIES[sidx])
        intent_vec, _ = parse_intent(intent_response)
        intent_vec = np.asarray(intent_vec, dtype=np.float32)
        injected_before = pruner.injected_count
        sub_graph = pruner.update_sub_graph(scenario["entities"], intent_vec)
        scenario_graphs.append(sub_graph)
        scenario_intents.append(intent_vec)
        scenario_injected.append(pruner.injected_count - injected_before)

    # Train the policy under this config's channel before measuring robustness.
    train_gail_policy(env, collector, gnn, gen, scenario_graphs, scenario_intents,
                      epochs=train_epochs, device=device)

    rows: List[Dict] = []
    for sidx, scenario in enumerate(OOD_SCENARIOS):
        sub_graph = scenario_graphs[sidx]
        intent_vec = scenario_intents[sidx]
        ep_results = run_episodes(env, collector, sub_graph, intent_vec, episodes)
        for ep_r in ep_results:
            ep_r.update({
                "snr_db": snr_db,
                "injection_enabled": enable_injection,
                "sim_threshold": sim_threshold,
                "matcher": "transe" if pruner._use_transe else "char_ngram_fallback",
                "scenario_id": sidx,
                "intent": scenario["intent"],
                "num_entities": len(scenario["entities"]),
                "sub_graph_nodes": len(sub_graph.nodes),
                "sub_graph_edges": len(sub_graph.edges),
                "injected_count": scenario_injected[sidx],
                "entity_coverage": round(entity_coverage(sub_graph), 4),
                "train_epochs": train_epochs,
            })
            rows.append(ep_r)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment B: Threshold ablation at SNR = 0 dB
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_b_threshold(device: str, kg_dir: str, episodes_per_config: int,
                           max_hops: int,
                           node_emb_dim: int, state_dim: int, action_dim: int,
                           sub_emb_dim: int, intent_dim: int, csi_dim: int,
                           train_epochs: int = 150,
                           ) -> List[Dict]:
    """
    Fixed SNR = 0 dB. Sweep sim_threshold across THRESHOLD_VALUES.

    NOTE: this ablation deliberately uses the CHAR N-GRAM FALLBACK matcher.
    With the TransE matcher the threshold is inert by construction (NLTK-resolved
    keywords score sim≈1.0, OOD keywords bypass the threshold entirely), so the
    sim_threshold is only load-bearing for the fallback matcher — this experiment
    documents the fallback's operating point (cf. the 0.30 default used elsewhere).
    Each threshold trains its own GAIL policy, then records mean reward,
    fidelity, per-scenario injected node count, and sub-graph size.
    """
    fixed_snr = 0.0
    all_rows: List[Dict] = []

    print(f"\n  Fixed SNR = {fixed_snr:.0f} dB  (char n-gram fallback matcher: "
          f"the only path where sim_threshold is load-bearing)")
    print(f"  {'Threshold':>10s}  {'Mean Reward':>12s}  {'Mean Fidelity':>13s}  "
          f"{'Injected':>9s}  {'Sub-graph N':>12s}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*13}  {'─'*9}  {'─'*12}")

    for thresh in THRESHOLD_VALUES:
        # Run with injection ON (the threshold controls matching vs injection)
        try:
            set_seed(42)
            global_kg = GlobalKnowledgeGraph(kg_dir).load()
            extractor = IntentExtractor()
            pruner = DynamicKBPruner(
                global_kg=global_kg, max_subgraph_nodes=64, max_hops=max_hops,
                similarity_threshold=thresh, hub_k=2, enable_injection=True,
                transe_embedder=None,   # fallback matcher: threshold is load-bearing here
            )
            gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=sub_emb_dim).to(device)
            gen = Generator(state_dim=state_dim, action_dim=action_dim).to(device)

            env_cfg = EnvConfig(max_hops=8, action_dim=action_dim,
                                node_emb_dim=node_emb_dim, csi_dim=csi_dim,
                                snr_db_low=fixed_snr, snr_db_high=fixed_snr + 0.1)
            env = SemiMDPEnv(env_cfg)
            mock_prov = MockSubGraphProvider(embedding_dim=128)
            register_provider(mock_prov)
            collector = EpisodeCollector(
                env=env, provider=mock_prov, gnn_encoder=gnn,
                state_dim=state_dim, max_hops=8, generator=gen, device=device,
            )

            scenario_graphs: List[SubGraph] = []
            scenario_intents: List[np.ndarray] = []
            scenario_injected: List[int] = []
            for sidx, scenario in enumerate(OOD_SCENARIOS):
                intent_response = extractor.query(OOD_QUERIES[sidx])
                intent_vec, _ = parse_intent(intent_response)
                intent_vec = np.asarray(intent_vec, dtype=np.float32)
                injected_before = pruner.injected_count
                sub_graph = pruner.update_sub_graph(scenario["entities"], intent_vec)
                scenario_graphs.append(sub_graph)
                scenario_intents.append(intent_vec)
                scenario_injected.append(pruner.injected_count - injected_before)

            train_gail_policy(env, collector, gnn, gen, scenario_graphs,
                              scenario_intents, epochs=train_epochs, device=device)

            config_rewards: List[float] = []
            config_fidelities: List[float] = []
            config_n_nodes: List[int] = []
            config_n_edges: List[int] = []

            for sidx in range(len(OOD_SCENARIOS)):
                ep_results = run_episodes(env, collector, scenario_graphs[sidx],
                                          scenario_intents[sidx], episodes_per_config)
                for ep_r in ep_results:
                    config_rewards.append(ep_r["cumulative_reward"])
                    config_fidelities.append(ep_r["fidelity"])
                config_n_nodes.append(len(scenario_graphs[sidx].nodes))
                config_n_edges.append(len(scenario_graphs[sidx].edges))

            row = {
                "sim_threshold": thresh,
                "snr_db": fixed_snr,
                "matcher": "char_ngram_fallback",
                "episodes_per_scenario": episodes_per_config,
                "num_scenarios": len(OOD_SCENARIOS),
                "train_epochs": train_epochs,
                "mean_reward": round(float(np.mean(config_rewards)), 4),
                "std_reward": round(float(np.std(config_rewards)), 4),
                "mean_fidelity": round(float(np.mean(config_fidelities)), 4),
                "std_fidelity": round(float(np.std(config_fidelities)), 4),
                "mean_injected_count": round(float(np.mean(scenario_injected)), 1),
                "mean_entity_coverage": round(float(np.mean(
                    [entity_coverage(g) for g in scenario_graphs])), 4),
                "mean_subgraph_nodes": round(float(np.mean(config_n_nodes)), 1),
                "mean_subgraph_edges": round(float(np.mean(config_n_edges)), 1),
            }
            all_rows.append(row)

            print(f"  {thresh:10.2f}  {row['mean_reward']:12.4f}  "
                  f"{row['mean_fidelity']:13.4f}  {row['mean_injected_count']:9.1f}  "
                  f"{row['mean_subgraph_nodes']:12.1f}")

        except Exception as e:
            print(f"  {thresh:10.2f}  FAILED: {e}")
            traceback.print_exc()

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OOD Robustness — Phase 4 Production")
    parser.add_argument("--episodes-per-config", type=int, default=30)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kg-dir", type=str, default=DEFAULT_WN18)
    parser.add_argument("--snr-sim-threshold", type=float, default=0.30,
                        help="Threshold used by SNR sweep. Default follows measured ablation optimum.")
    parser.add_argument("--max-hops", type=int, default=2,
                        help="Pruning depth used by robustness runs. Default follows the fair E2 N*=2 result.")
    parser.add_argument("--train-epochs", type=int, default=150,
                        help="GAIL training epochs per config before evaluation. 0 = random-init policy "
                             "(diagnostic only; robustness claims require a trained policy).")
    parser.add_argument("--skip-snr", action="store_true",
                        help="Skip Experiment A (SNR sweep), only run threshold ablation")
    parser.add_argument("--skip-threshold", action="store_true",
                        help="Skip Experiment B (threshold ablation), only run SNR sweep")
    args = parser.parse_args()

    device = args.device
    ep_per = args.episodes_per_config

    node_emb_dim, sub_emb_dim = NODE_EMB_DIM, SUB_EMB_DIM
    intent_dim, csi_dim = INTENT_DIM, CSI_DIM
    state_dim, action_dim = STATE_DIM, DEFAULT_ACTION_DIM
    assert state_dim == node_emb_dim + sub_emb_dim + intent_dim + csi_dim

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 4 Production — OOD Robustness & Threshold Ablation")
    print(f"  STATE_DIM = {state_dim}")
    print(f"  Episodes per config: {ep_per}")
    print(f"  Device: {device}")
    print(f"  Tuned config: max_hops={args.max_hops}, "
          f"snr_sim_threshold={args.snr_sim_threshold}, train_epochs={args.train_epochs}")

    # ── Experiment A: SNR sweep ────────────────────────────────────────────
    if not args.skip_snr:
        print(f"\n{'='*65}")
        print("Experiment A: OOD Robustness vs SNR")
        print(f"{'='*65}")

        rows_a = experiment_a_snr(
            device, args.kg_dir, ep_per, args.snr_sim_threshold, args.max_hops,
            node_emb_dim, state_dim, action_dim,
            sub_emb_dim, intent_dim, csi_dim,
            train_epochs=args.train_epochs,
        )

        # Raw
        if rows_a:
            fieldnames = list(rows_a[0].keys())
            csv_a = out_dir / "ood_robustness_snr.csv"
            with open(csv_a, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows_a)
            print(f"\n  Raw data -> {csv_a}  ({len(rows_a)} rows)")

        # Aggregated summary
        summary: Dict[str, Dict] = defaultdict(lambda: {"rewards": [], "fids": [], "covs": [], "goals": 0, "n": 0})
        for r in rows_a:
            k = f"SNR={r['snr_db']:.0f}dB_{'injON' if r['injection_enabled'] else 'injOFF'}"
            summary[k]["rewards"].append(r["cumulative_reward"])
            summary[k]["fids"].append(r["fidelity"])
            summary[k]["covs"].append(r.get("entity_coverage", 0.0))
            summary[k]["goals"] += 1 if r["reached_goal"] else 0
            summary[k]["n"] += 1

        summary_rows = []
        for k, v in sorted(summary.items()):
            summary_rows.append({
                "config": k,
                "snr_db": int(float(k.split("dB")[0].split("=")[1])),
                "injection_enabled": "injON" in k,
                "mean_reward": round(float(np.mean(v["rewards"])), 4),
                "std_reward": round(float(np.std(v["rewards"])), 4),
                "mean_fidelity": round(float(np.mean(v["fids"])), 4),
                "std_fidelity": round(float(np.std(v["fids"])), 4),
                "mean_entity_coverage": round(float(np.mean(v["covs"])), 4),
                "goal_rate": round(v["goals"] / max(v["n"], 1), 4),
                "num_episodes": v["n"],
            })
        if summary_rows:
            csv_as = out_dir / "ood_robustness_snr_summary.csv"
            with open(csv_as, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)
            print(f"  Summary -> {csv_as}")

    # ── Experiment B: Threshold ablation ────────────────────────────────────
    if not args.skip_threshold:
        print(f"\n{'='*65}")
        print("Experiment B: Threshold Ablation (SNR = 0 dB)")
        print(f"{'='*65}")

        rows_b = experiment_b_threshold(
            device, args.kg_dir, ep_per, args.max_hops,
            node_emb_dim, state_dim, action_dim,
            sub_emb_dim, intent_dim, csi_dim,
            train_epochs=args.train_epochs,
        )

        if rows_b:
            csv_b = out_dir / "threshold_ablation.csv"
            with open(csv_b, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows_b[0].keys()))
                w.writeheader()
                w.writerows(rows_b)
            print(f"\n  Saved -> {csv_b}  ({len(rows_b)} rows)")

            # Identify optimal threshold
            best = max(rows_b, key=lambda r: r["mean_reward"])
            print(f"\n  Optimal threshold: {best['sim_threshold']:.2f} "
                  f"(reward={best['mean_reward']:.4f}, "
                  f"injected={best['mean_injected_count']:.1f})")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
