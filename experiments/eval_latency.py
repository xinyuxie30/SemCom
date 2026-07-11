"""
eval_latency.py — Measured latency of the fast-slow dual loop.

All SYSTEM components are timed on real PyTorch modules (no fabricated
distributions): deterministic intent keyword lookup, GNN sub-graph encode, GAIL
policy inference, and the channel-state op.

Experiment A: real end-to-end per-decision latency breakdown + amortization, with
  an ARCHITECTURAL comparison against a learned NL intent extractor whose latency
  is a cited LITERATURE REFERENCE (--reference-extractor-ms), NOT a component of
  this system (which uses deterministic keyword lookup by hard constraint). The
  reference is labelled `measured=NO` and never presented as a measured value.

Experiment B: real GNN + policy fast-loop latency vs sub-graph node count N,
  demonstrating that pruning keeps the fast-loop cost bounded.

Output:
  experiments/results/latency_breakdown.csv        (trial-level, Experiment A)
  experiments/results/latency_summary.csv          (component stats, measured flag)
  experiments/results/latency_subgraph_scaling.csv (Experiment B: N vs latency)

Usage:
  python experiments/eval_latency.py --trials 500 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.kg_interface import MockSubGraphProvider
from data.dataloader import STATE_DIM
from models.gnn_encoder import SubGraphEncoder, SUB_EMB_DIM
from models.generator import Generator, DEFAULT_ACTION_DIM
from env.channel import channel_snapshot
from env.mdp_env import CSI_DIM
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent

QUERIES = [
    "emergency vehicle collision avoidance on highway",
    "drone surveillance with video streaming and camera",
    "pedestrian crossing street radar depth priority",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Real timing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _time_ms(fn: Callable[[], None], device: str, n_warmup: int, n_trials: int,
             cuda_sync: bool) -> List[float]:
    """Return per-call wall-clock latency (ms) of `fn`, with warmup + optional
    CUDA synchronization so GPU kernels are actually awaited."""
    for _ in range(n_warmup):
        fn()
    if cuda_sync and device == "cuda":
        torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        if cuda_sync and device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _make_subgraph_tensors(num_nodes: int, device: str) -> Dict[str, object]:
    """Build a real N-node sub-graph tensor dict for the GNN encoder."""
    prov = MockSubGraphProvider(embedding_dim=128, num_nodes=num_nodes)
    sub = prov.fetch("latency")
    data = sub.to_tensors()
    data["edge_index"] = data["edge_index"].to(device)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment A: real per-decision latency breakdown + literature-reference compare
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_a(device: str, trials: int, amortization: int,
                 reference_extractor_ms: float) -> Tuple[List[Dict], Dict, Dict]:
    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=SUB_EMB_DIM).to(device).eval()
    gen = Generator(state_dim=STATE_DIM, action_dim=DEFAULT_ACTION_DIM).to(device).eval()
    extractor = IntentExtractor()
    sub_data = _make_subgraph_tensors(64, device)
    state = torch.randn(1, STATE_DIM, device=device)

    def intent_fn() -> None:
        parse_intent(extractor.query(QUERIES[0]))

    def gnn_fn() -> None:
        with torch.no_grad():
            gnn(sub_data)

    def pol_fn() -> None:
        with torch.no_grad():
            gen.get_action(state)

    def ch_fn() -> None:
        channel_snapshot(dim=CSI_DIM)

    t_intent = _time_ms(intent_fn, device, 20, trials, cuda_sync=False)
    t_gnn = _time_ms(gnn_fn, device, 20, trials, cuda_sync=True)
    t_pol = _time_ms(pol_fn, device, 20, trials, cuda_sync=True)
    t_ch = _time_ms(ch_fn, device, 20, trials, cuda_sync=False)

    rows: List[Dict] = []
    for i in range(trials):
        rows.append({
            "trial_id": i,
            "intent_lookup_ms": round(t_intent[i], 6),
            "gnn_encode_ms": round(t_gnn[i], 6),
            "policy_inference_ms": round(t_pol[i], 6),
            "channel_ms": round(t_ch[i], 6),
            "fast_loop_ms": round(t_pol[i] + t_ch[i], 6),
            "slow_loop_ms": round(t_intent[i] + t_gnn[i], 6),
        })

    mi, mg, mp, mc = (float(np.mean(x)) for x in (t_intent, t_gnn, t_pol, t_ch))
    fast = mp + mc                       # per fast-loop decision (real)
    slow = mi + mg                       # per intent/slow update (real)
    system_amortized = fast + slow / max(1, amortization)
    # Architectural reference: if the deterministic keyword lookup were replaced by
    # a learned NL intent extractor (literature latency), the slow update would cost
    # reference_extractor_ms + GNN instead of intent_lookup + GNN.
    learned_slow = reference_extractor_ms + mg
    learned_amortized = fast + learned_slow / max(1, amortization)

    stats = {
        "component": [
            "intent_lookup", "gnn_encode", "policy_inference", "channel",
            "fast_loop_per_decision", "slow_loop_per_update",
            "system_amortized_per_decision",
            "learned_extractor_amortized_per_decision",
        ],
        "mean_ms": [
            round(mi, 6), round(mg, 6), round(mp, 6), round(mc, 6),
            round(fast, 6), round(slow, 6), round(system_amortized, 6),
            round(learned_amortized, 6),
        ],
        "p99_ms": [
            round(float(np.percentile(t_intent, 99)), 6),
            round(float(np.percentile(t_gnn, 99)), 6),
            round(float(np.percentile(t_pol, 99)), 6),
            round(float(np.percentile(t_ch, 99)), 6),
            "", "", "", "",
        ],
        "measured": ["yes", "yes", "yes", "yes", "yes", "yes", "yes",
                     "NO_literature_reference"],
    }
    extra = {
        "amortization": amortization,
        "reference_extractor_ms": reference_extractor_ms,
        "reference_source": "literature (learned NL/LLM intent extractor); NOT this system",
        "system_vs_learned_speedup": round(learned_amortized / max(system_amortized, 1e-9), 1),
    }
    return rows, stats, extra


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment B: real fast-loop latency vs sub-graph node count N
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_b(device: str, node_counts: List[int], trials_per_n: int) -> List[Dict]:
    gnn = SubGraphEncoder(in_dim=128, hidden_dim=256, out_dim=SUB_EMB_DIM).to(device).eval()
    gen = Generator(state_dim=STATE_DIM, action_dim=DEFAULT_ACTION_DIM).to(device).eval()
    state = torch.randn(1, STATE_DIM, device=device)

    rows: List[Dict] = []
    for N in node_counts:
        sub_data = _make_subgraph_tensors(N, device)

        def gnn_fn() -> None:
            with torch.no_grad():
                gnn(sub_data)

        def pol_fn() -> None:
            with torch.no_grad():
                gen.get_action(state)

        t_gnn = _time_ms(gnn_fn, device, 10, trials_per_n, cuda_sync=True)
        t_pol = _time_ms(pol_fn, device, 10, trials_per_n, cuda_sync=True)
        total = [g + p for g, p in zip(t_gnn, t_pol)]
        rows.append({
            "num_nodes": N,
            "trials": trials_per_n,
            "gnn_mean_ms": round(float(np.mean(t_gnn)), 6),
            "gnn_std_ms": round(float(np.std(t_gnn)), 6),
            "gnn_p99_ms": round(float(np.percentile(t_gnn, 99)), 6),
            "policy_mean_ms": round(float(np.mean(t_pol)), 6),
            "total_mean_ms": round(float(np.mean(total)), 6),
            "total_std_ms": round(float(np.std(total)), 6),
            "total_p99_ms": round(float(np.percentile(total, 99)), 6),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Latency Evaluation - real measurements")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--amortization", type=int, default=200,
                        help="Fast-loop decisions per slow/intent update (slow cost is amortized).")
    parser.add_argument("--reference-extractor-ms", type=float, default=1200.0,
                        help="LITERATURE latency of a learned NL/LLM intent extractor for the "
                             "architectural comparison. This is NOT measured and NOT a component of "
                             "this system (which uses deterministic keyword lookup). Cite accordingly.")
    parser.add_argument("--subgraph-sizes", type=str, default="20,40,60,80",
                        help="Comma-separated node counts for the scaling experiment")
    parser.add_argument("--scaling-trials", type=int, default=200,
                        help="Trials per sub-graph size in Experiment B")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Latency Evaluation (real measurements)")
    print(f"  Device: {device} | Trials (Exp A): {args.trials}")
    print(f"  Reference extractor: {args.reference_extractor_ms:.0f} ms "
          f"[LITERATURE reference, NOT measured, NOT a system component]")
    print()

    # ── Experiment A ────────────────────────────────────────────────────────
    print("=" * 72)
    print("Experiment A: Real per-decision latency breakdown")
    print("-" * 72)

    rows_a, stats_a, extra_a = experiment_a(device, args.trials, args.amortization,
                                            args.reference_extractor_ms)
    for i in range(len(stats_a["component"])):
        flag = "" if stats_a["measured"][i] == "yes" else "  [literature ref, NOT measured]"
        p99 = stats_a["p99_ms"][i]
        p99s = f"  p99={p99:8.4f} ms" if p99 != "" else ""
        print(f"  {stats_a['component'][i]:<42s}  mean={stats_a['mean_ms'][i]:9.4f} ms{p99s}{flag}")
    print(f"\n  Real system amortized/decision: {stats_a['mean_ms'][6]:.4f} ms")
    print(f"  If a learned extractor were used: {stats_a['mean_ms'][7]:.4f} ms "
          f"(reference, NOT measured)")
    print(f"  Architectural speedup vs learned extractor: "
          f"{extra_a['system_vs_learned_speedup']:.0f}x "
          f"(deterministic keyword lookup avoids the learned-extractor cost)")

    csv_a = out_dir / "latency_breakdown.csv"
    with open(csv_a, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_a[0].keys()))
        w.writeheader()
        w.writerows(rows_a)
    print(f"  Saved -> {csv_a}")

    summary_a = out_dir / "latency_summary.csv"
    with open(summary_a, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats_a.keys()))
        w.writeheader()
        for i in range(len(stats_a["component"])):
            w.writerow({k: v[i] for k, v in stats_a.items()})
    print(f"  Saved -> {summary_a}")

    # ── Experiment B ────────────────────────────────────────────────────────
    node_counts = [int(x.strip()) for x in args.subgraph_sizes.split(",")]
    print(f"\n{'='*72}")
    print(f"Experiment B: Real fast-loop latency vs sub-graph size  (N in {node_counts})")
    print("-" * 72)

    rows_b = experiment_b(device, node_counts, args.scaling_trials)
    print(f"  {'N':>5s}  {'GNN mean':>10s}  {'GNN p99':>10s}  {'Total mean':>11s}  {'Total p99':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*10}")
    for r in rows_b:
        print(f"  {r['num_nodes']:5d}  {r['gnn_mean_ms']:8.4f} ms  "
              f"{r['gnn_p99_ms']:8.4f} ms  {r['total_mean_ms']:9.4f} ms  "
              f"{r['total_p99_ms']:8.4f} ms")

    if len(rows_b) >= 2:
        n_small, t_small = rows_b[0]["num_nodes"], rows_b[0]["total_mean_ms"]
        n_large, t_large = rows_b[-1]["num_nodes"], rows_b[-1]["total_mean_ms"]
        ratio_n = n_large / n_small
        ratio_t = t_large / max(t_small, 1e-9)
        print(f"\n  N scaling: {ratio_n:.1f}x nodes -> {ratio_t:.2f}x latency "
              f"({'sub-linear' if ratio_t < ratio_n else 'super-linear'})")

    csv_b = out_dir / "latency_subgraph_scaling.csv"
    with open(csv_b, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_b[0].keys()))
        w.writeheader()
        w.writerows(rows_b)
    print(f"  Saved -> {csv_b}")

    print("\nDone.")


if __name__ == "__main__":
    main()
