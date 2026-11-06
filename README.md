# Cognitive Semantic Communication

Cognitive Semantic Communication is a research workspace for exploring how a sender and receiver can share task-relevant knowledge without transmitting an entire knowledge graph every time.

The project selects a connected subgraph using intent keywords, simulates a receiver with a stale semantic knowledge base, and compares ways to align that knowledge. Its experiments report task success, semantic consistency, payload and latency alongside communication decisions. Cloud and edge behavior is simulated locally; this is an experimental platform, not a deployed service.

The implementation uses deterministic keyword matching rather than an LLM API. Results apply to the datasets, tasks and evaluation settings documented below; the project does not claim that the individual mechanisms are independently novel.

## Core Idea

This project studies semantic communication over a shared knowledge graph
(KG). The transmitter sends only the intent-relevant KG sub-graph. The receiver
keeps a stale edge-side semantic knowledge base (SKB) cache and requests
incremental cloud-edge alignment only when needed.

No LLM/LAM API is used. Intent extraction is deterministic keyword lookup in
`agents/slow_loop/intent_extractor.py`.

## Pipeline

```text
intent keywords
  -> TransE cosine matching
  -> BFS dynamic pruning
  -> SubGraph <= 64 nodes with semantic node_embeddings
  -> GCN encoder -> sub_emb[128]
  -> state = node_emb(256) + sub_emb(128) + intent(64) + csi(4)
  -> STATE_DIM = 452
  -> GAIL policy over path hop + bandwidth + MCS
```

Cloud-edge SKB alignment:

```text
TX reference SubGraph
  -> stale RX EdgeSKB cache
  -> compare no_align / full_retransmit / incremental_align
  -> KG semantic consistency + payload + latency + kappa
```

## Core Components

- Intent-driven dynamic connected sub-graph pruning.
- OOD entity semantic injection into the KG.
- KG-SKB cloud-edge alignment without real server deployment.
- KG semantic consistency metric, analogous to generation similarity in
  generative SemCom.
- CSI-aware Semi-MDP action with path hop + BW + MCS.

## Key Constraints

- `STATE_DIM = 452` is fixed.
- Do not add LLM/LAM API calls.
- Cloud/edge server behavior is simulated locally for reproducibility.
- External projects are baseline references only, not direct dependencies.

## Datasets

Dataset root:

`C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/`

Datasets:

- `FB15K237`: 14,541 entities, 237 relations.
- `WN18RR`: 40,943 entities, 11 relations.
- `NELL-995`: 75,492 entities, 200 relations.

## Baseline References

- `C:/Users/xinyuxie/Desktop/iSAC-main`: original iSAC/G-RML behavior.
- `C:/Users/xinyuxie/Desktop/iRML-main`: multi-layer and collaborative
  reasoning reference.
- `D:/Paper/语义入门资料/code+paper（写代码重点看）`: image SemCom mechanisms
  such as Dynamic-JSCC, OFDM-JSCC, PADC/OraNet, and WITT.

## Common Commands

Debug train:

```powershell
python train.py --mode debug --epochs 8
```

TransE training:

```powershell
python scripts/train_transe_unified.py --dataset FB15K237 --epochs 3000
```

SKB alignment experiment:

```powershell
python experiments/eval_skb_alignment.py --trials 40
```

Visible terminal run for E8:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_skb_alignment.ps1
```

Full visible E1-E8 suite:

```powershell
scripts\run_full_experiments.cmd
```

Publication figures and full-suite summary:

```powershell
python experiments/plot_results.py
python experiments/summarize_full_suite.py
```

Task-aware E2 N* scan (FAIR protocol):

```powershell
python experiments/eval_nstar_scan.py --dataset WN18RR --seeds 3 --device cuda
```

The N* scan defines a NEUTRAL multi-seed 2-hop task on the FULL KG (independent of
any pruning radius), then compares candidate radii `N in {1,2,3}` on that same
task with symmetric metrics (path_success / semantic_consistency / payload). N* is
the smallest radius that covers the task at lowest complexity. Result (WN18RR):
N=1 `path_success=0.00`, N=2 `1.00` (N*, 46 nodes), N=3 `1.00` (64 nodes,
redundant) — see `experiments/results/nstar_scan_fair.csv`. The old circular scan
(task defined by an N=2 DynamicKBPruner) is removed.

## Experiment Map

- E1: `scripts/train_transe_unified.py`
- E2: `experiments/measure_psi_transE.py`, `experiments/eval_nstar_scan.py`
  for `Psi(d)` semantic decay and FAIR task-aware `N*=2` validation (neutral
  full-KG task, no self-reference).
- E3: `experiments/eval_end_to_end.py` (FAIR protocol, see
  `E3_FAIR_DESIGN_20260624.md`). The start-target task is defined on the FULL KG
  (multi-seed 2-hop reasoning hop), not by any pruner. Every method is scored
  identically with multi-metric (`path_success`, `semantic_consistency`,
  `payload`, `task_goal_rate`, `raw_policy_reward`) — no composite, no Ours-only
  step, seed-aware non-strawman HAKE. Default dataset WN18RR (FB15K237 mids do
  not support intent matching). Formal: Ours `path_success=1.00`, consistency
  `0.800` vs HAKE `0.00`/`0.169`.
- E4: `experiments/eval_latency.py`, `experiments/profile_bfs_gnn.py` (real measured
  latency; the learned-extractor baseline is a labelled literature reference, real
  speedup ~4x; see `E4_LATENCY_DESIGN_20260624.md`)
- E5: `experiments/eval_robustness.py`
- E6: `experiments/eval_convergence.py`
- E7: `experiments/profile_lipschitz.py`
- E8: `experiments/eval_skb_alignment.py`

All existing CSV files should be treated as smoke-test artifacts unless they
were explicitly rerun for paper use.

The latest completed paper-scale run is:

`experiments/results/full_suite_20260621_220711/`

It finished all E1-E8 steps with `failed_steps=[]`. The generated summary is:

`experiments/results/full_suite_20260621_220711/full_suite_summary.md`
