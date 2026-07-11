# E4 Latency Redesign — Real Measurements — 2026-06-24

## Goal

Make E4 (latency) report **real measured** numbers instead of fabricated ones, so
the latency claims survive a reviewer.

## Why the old E4 was dishonest

`eval_latency.py` produced **no real measurements** — every latency was a
hard-coded random distribution:

- `simulate_online_extractor_inference()` = `np.random.normal(1200, 200)`
- `simulate_gnn_encode()` = `0.3 + 0.003*N`
- `simulate_policy_inference()` = `np.random.normal(0.2, 0.05)`
- `simulate_channel_tx()` = `np.random.uniform(0.5, 3.0)`

Worse, it compared against a **1200 ms "online extractor"** (the renamed
`pure_lam`) — but the project **forbids any LLM/LAM by hard constraint**, so that
baseline is a system component that does not exist. The resulting "~5000x speedup"
was fabricated data + a strawman baseline.

(`profile_bfs_gnn.py` was already honest — real `perf_counter` timing of BFS +
GNN with `cuda.synchronize` — and is kept.)

## What the fair/real E4 does now

- **Every system component is measured** on real PyTorch modules with warmup +
  CUDA sync (`eval_latency.py::_time_ms`): deterministic intent keyword lookup,
  `SubGraphEncoder` GNN forward, `Generator` policy forward, `channel_snapshot`.
- **The learned-extractor comparison is a labelled literature reference**
  (`--reference-extractor-ms`, default 1200 ms), with `measured=NO` in the CSV and
  "NOT measured" in every print. It quantifies the architectural cost a learned
  NL/LLM extractor *would* add — it is not presented as this system's latency.
- `profile_bfs_gnn.py` keeps the real BFS+GNN profile; its Exp 2 framing was fixed
  (it no longer claims the slow-loop "fails Tc" — the slow loop is not per-coherence
  -window), and its Exp 3 energy figures are explicitly labelled illustrative.

## Results (measured, RTX-class GPU, 500 trials)

Experiment A — real per-component latency:

| Component | mean | p99 | measured |
|---|---|---|---|
| intent_lookup (keyword dict) | **0.062 ms** | 0.215 ms | yes |
| gnn_encode | 2.86 ms | 4.42 ms | yes |
| policy_inference | 2.04 ms | 3.72 ms | yes |
| channel | 0.060 ms | 0.131 ms | yes |
| fast_loop / decision | 2.10 ms | — | yes |
| system amortized / decision | **2.12 ms** | — | yes |
| learned-extractor amortized / decision | 8.12 ms | — | **NO (literature ref)** |

- Real **architectural speedup ~4x** vs a learned extractor (labelled reference),
  driven by the deterministic keyword lookup (0.062 ms) replacing a ~1200 ms
  learned NL extractor. Honest, vs the old fabricated 5000x.

Experiment B — fast-loop latency vs sub-graph size N (real):

| N | GNN mean | total mean | total p99 |
|---|---|---|---|
| 20 | 2.75 ms | 4.86 ms | 7.57 ms |
| 40 | 2.69 ms | 4.74 ms | 6.87 ms |
| 60 | 2.72 ms | 4.80 ms | 7.21 ms |
| 80 | 2.81 ms | 4.91 ms | 9.50 ms |

- **Sub-linear**: 4.0x nodes -> ~1.01x latency. Pruning keeps the fast loop bounded
  even as the global KG grows (kernel-launch-dominated for small N).

Slow-loop (profile_bfs_gnn, FB15K237):

- BFS match + GNN ≈ **214 ms / intent update** (char n-gram matching over the full
  14,541-entity KG is the dominant cost). This runs **once per intent change**, not
  per coherence window.
- Amortized over ~200 fast-loop decisions: **~1.09 ms / decision ≈ 22% of Tc=5 ms**.
  This is the fast-slow architecture's point: the expensive slow loop is infrequent.
- Future speedup: the E3 path uses a cached TransE embedding matrix for matching
  (`_entity_embedding_matrix`); wiring it into the pruner would cut the 214 ms.

## Figure

`experiments/figures/fig2_latency.pdf` (via `python experiments/plot_results.py`):
left = measured per-component latency (log); right = measured system vs the
learned-extractor literature reference, annotated "literature reference, NOT
measured".

## Reproduce

```powershell
python experiments/eval_latency.py --trials 500 --device cuda
python experiments/profile_bfs_gnn.py --samples 200 --device cuda
python experiments/plot_results.py
```

Outputs: `experiments/results/latency_breakdown.csv`, `latency_summary.csv`
(with a `measured` column), `latency_subgraph_scaling.csv`,
`bfs_gnn_latency_profile.csv`, `bfs_gnn_scaling.csv`; figure `fig2_latency.pdf`.
