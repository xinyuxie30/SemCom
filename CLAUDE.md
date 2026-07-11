# CognitiveSemCom

This is an independent project. Do not import code or claims from other
projects unless the reuse is explicitly documented as a baseline or reference.

## Project Direction

KG-based semantic communication with GAIL + Semi-MDP:

- Intent-driven dynamic KG sub-graph pruning.
- TransE semantic matching for intent keywords.
- OOD entity injection for open-world KG updates.
- Cloud-edge SKB alignment simulation for receiver-side knowledge mismatch.
- No LLM/LAM API. Intent extraction is deterministic keyword lookup in
  `agents/slow_loop/intent_extractor.py`.

The current paper direction is:

> Intent-driven KG-SKB semantic communication with dynamic pruning, OOD
> injection, and cloud-edge knowledge alignment.

## Hard Constraints

- `STATE_DIM = 452` must not change.
- State composition is fixed:
  `node_emb(256) + sub_emb(128) + intent(64) + csi(4) = 452`.
- Do not add LLM/LAM API calls.
- Do not require real cloud/edge server deployment for experiments.
- Cloud/edge behavior should be simulated locally with reproducible Python
  objects, cache states, payload bytes, latency, and knowledge mismatch ratio.

## Data

Dataset root:

`C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/`

Main datasets:

- `FB15K237`: 14,541 entities, 237 relations.
- `WN18RR`: 40,943 entities, 11 relations.
- `NELL-995`: 75,492 entities, 200 relations.

## Core Pipeline

```text
intent keywords
  -> TransE cosine matching
  -> BFS dynamic pruning
  -> SubGraph <= 64 nodes with semantic node_embeddings
  -> GCN encoder -> sub_emb[128]
  -> Semi-MDP state[452]
  -> GAIL policy with spectral-normalized generator
```

Cloud-edge SKB alignment:

```text
TX reference SubGraph
  -> stale EdgeSKB cache at RX
  -> compare no_align / full_retransmit / incremental_align
  -> KG semantic consistency + payload + latency + kappa
```

## New SKB Alignment Components

- `agents/slow_loop/skb_server.py`
  - `CloudSKBServer`: authoritative SKB payload and latency model.
  - `EdgeSKBServer`: receiver-side cache with controlled misses.
  - Strategies: `no_align`, `full_retransmit`, `incremental_align`.

- `utils/semantic_consistency.py`
  - KG analogue of generation similarity.
  - Combines node coverage, edge coverage, start-target connectivity,
    embedding coverage, and seed coverage.

- `experiments/eval_skb_alignment.py`
  - Reproducible cloud-edge SKB alignment experiment.
  - Outputs:
    - `experiments/results/skb_alignment.csv`
    - `experiments/results/skb_alignment_summary.csv`

- `experiments/eval_nstar_scan.py`
  - Task-aware E2 N* scan (FAIR: neutral full-KG task, no self-reference).
  - Defines a neutral multi-seed 2-hop task on the FULL KG (independent of any
    radius), then compares N in {1, 2, 3} on the same task; N* is the smallest
    radius that covers the task at lowest complexity (N*=2 on WN18RR).

## Baseline Guidance

Use external code as conceptual baselines, not direct dependencies:

- `C:/Users/xinyuxie/Desktop/iSAC-main`
  - Best baseline source for original G-RML / iSAC behavior.
  - Reproduce as `NoPruning` or original BFS reasoning inside this project.

- `C:/Users/xinyuxie/Desktop/iRML-main`
  - Best reference for multi-layer reasoning and collaborative reasoning.
  - Reuse experiment ideas such as number of layers/devices, not raw code.

- `D:/Paper/语义入门资料/code+paper（写代码重点看）`
  - Image SemCom projects are not same-task KG baselines.
  - Borrow mechanisms only:
    - Dynamic-JSCC: adaptive rate control.
    - OFDM-JSCC: explicit CSI/pilot/equalization modeling.
    - PADC/OraNet: quality prediction before transmission.
    - WITT: attention/Transformer semantic encoder ideas.

## Experiment Plan

Existing CSV files are smoke-test artifacts unless explicitly rerun.

E1 TransE:

```powershell
python scripts/train_transe_unified.py --dataset FB15K237 --epochs 3000 --seed 42
```

E2 Psi(d) and N*:

```powershell
python experiments/measure_psi_transE.py --num-seeds 300
python experiments/eval_nstar_scan.py --dataset WN18RR --seeds 3 --device cuda
```

Latest formal E2 N* rerun (FAIR protocol, neutral full-KG task, no self-reference):

- Result (WN18RR, `experiments/results/nstar_scan_fair.csv`): `N*=2` emerges from
  real task coverage. N=1 path_success `0.00` (1-hop cannot cover the 2-hop task),
  N=2 `1.00` (covers, 46 nodes), N=3 `1.00` but 64 nodes (redundant). N* = smallest
  covering radius at lowest complexity.
- The old circular scan (task defined by an N=2 DynamicKBPruner + node_overlap-vs-
  self score, "N*=2 score 9.1477") is removed as a deprecated artifact.

E3 End-to-end comparison (FAIR protocol, see `E3_FAIR_DESIGN_20260624.md`):

```powershell
python experiments/eval_end_to_end.py --epochs 500 --seeds 5 --dataset WN18RR
```

Latest E3 design note:

- The reference start-target task is defined on the FULL KG (a multi-seed 2-hop
  reasoning hop), NOT by any pruner. No circular task, no self-referential score.
- All methods are scored identically with multi-metric (no composite):
  `path_success`, `semantic_consistency` (KG analogue of generation similarity),
  `payload`, `task_goal_rate`, `raw_policy_reward`. No Ours-only step; HAKE is a
  seed-aware (non-strawman) static-hierarchy opponent.
- Default dataset is WN18RR: its synset entities support meaningful intent-keyword
  matching. FB15K237 Freebase mids do not, so intent matching degenerates there
  (every method `path_success=0`); diagnose with
  `scripts/diagnose_intent_match.py`.
- NELL-995 is a valid second dataset on `semantic_consistency` (Ours `0.271` vs
  HAKE `0.118`), but its dense graph + noisy `concept_*` matching degenerate
  `path_success` to 0 for all methods. Report `path_success` on WN18RR and
  `semantic_consistency` on both. `--node-budget` (symmetric for Ours/HAKE/
  reference) and the `seed_to_seed` task definition were added to probe dense KGs;
  both leave WN18RR unchanged (regression-verified).
- Latest formal fair rerun (WN18RR, 500 epochs, 5 seeds,
  `experiments/results/end_to_end_summary_fair_formal.csv`):
  `DynamicPruned_Ours` path_success `1.00 +/- 0.00`, semantic_consistency
  `0.800 +/- 0.000`, task_goal_rate `0.10`; HAKE path_success `0.00`,
  consistency `0.169 +/- 0.021`; NoPruning path_success `0.00`, consistency
  `0.009`. Ours fairly wins on every task metric.
- The old `mean_reward` composite and the `-5.0` strict floor are removed; pre-fix
  unfair CSVs (without a `fair_` tag) are deprecated artifacts.

E4 Latency (real measurements, see `E4_LATENCY_DESIGN_20260624.md`):

```powershell
python experiments/eval_latency.py --trials 500 --device cuda
python experiments/profile_bfs_gnn.py --samples 200 --device cuda
```

E4 design note:

- All system components are REAL measurements (intent keyword lookup ~0.06 ms,
  GNN encode ~2.9 ms, policy ~2.0 ms, channel ~0.06 ms). The old fabricated
  `np.random.normal` latencies and the fictional 1200 ms LAM baseline are removed.
- The learned-extractor comparison (`--reference-extractor-ms`, default 1200 ms) is
  a LITERATURE reference labelled `measured=NO`, not a system component (this project
  uses deterministic keyword lookup). Real architectural speedup ~4x, not 5000x.
- Sub-graph GNN latency is sub-linear in N (pruning keeps the fast loop bounded).
  Slow-loop BFS+matching is ~214 ms/intent-update (char n-gram over the full KG),
  runs once per intent change, amortized to ~1 ms/decision (~22% of Tc=5 ms).

E5 Robustness (see `E5_ROBUSTNESS_DESIGN_20260630.md`):

```powershell
python experiments/eval_robustness.py --episodes-per-config 30 --train-epochs 150
```

E5 design note:

- Robustness is measured on a GAIL-TRAINED policy (`--train-epochs 150` per
  config), not a random-init network. `--train-epochs 0` is a diagnostic only.
- Experiment A uses the WN18RR TransE embedder so OOD detection is principled;
  injection is reported as a coverage-vs-reward trade-off: injON
  `entity_coverage=1.00` vs injOFF `0.20` (the receiver cannot represent absent
  OOD concepts), while injOFF's higher raw reward is a smaller-graph navigation
  artifact. Reward rises monotonically with SNR under both settings.
- Experiment B (threshold ablation) deliberately uses the char n-gram FALLBACK
  matcher — the only path where `sim_threshold` is load-bearing (TransE-resolved
  keywords score sim≈1.0 and OOD bypasses the threshold). Operating point 0.30.
- `--max-hops` defaults to 2, following the fair E2 `N*=2`.

E6 Convergence:

```powershell
python experiments/eval_convergence.py --epochs 500 --seeds 3
```

E6 design note (formal rerun 500 epochs x 3 seeds, WN18RR, FAIR shared-task
protocol, `experiments/results/convergence_data.csv`):

- All three configs — `dynamic_pruned` (E3 formal config: char n-gram matching,
  threshold 0.30, N=2, <=64 nodes), `hake_static_layer` (seed-aware), and
  `budgeted_no_pruning` (intent-blind 512-node slice) — train GAIL on their OWN
  pruned sub-graph but are assigned the SAME neutral full-KG task (E3 protocol,
  `build_ground_truth_task` on a pristine KG). Convergence is measured as
  convergence-to-task-competence: `task_goal_rate` = fraction of eval rollouts
  whose walked path reaches the neutral target (every 25 epochs).
- Result: Ours `task_goal_rate` mean-over-eval-points `0.416` (task in sub-graph
  67% of query rotations; 1.00 at the final eval point) vs HAKE `0.000` and
  NoPruning `0.008`. VERDICT: Ours wins on the fair shared task.
- Mechanism (three-fold, honest): (1) only Ours' sub-graph contains the task
  path (0.67 vs 0.00/0.00); (2) HAKE's fragmented in-layer graphs (median 11
  nodes) cannot even yield MockExpert demonstrations (expert buffer median 0, so
  its GAIL barely updates) — static-layer pruning fails imitation learning at
  the demonstration stage; (3) NoPruning trains normally (G->0.0007) yet earns
  ~0 task competence: adversarial-loss convergence without task semantics.
- Do NOT compare raw eval_reward across configs (self-graph rewards are not
  task-comparable); the cross-config metric is `task_goal_rate` on the shared
  neutral task. Ours' pruner deliberately uses char n-gram matching here (same
  embedder family as the shared seed matching) — the TransE embedder belongs to
  E5's OOD setting and would misalign pruner seeds from task seeds.

E7 Injection stability:

```powershell
python experiments/profile_lipschitz.py --train-epochs 150 --device cuda
```

E7 design note (formal rerun, 6 configs x 5 seeds,
`experiments/results/lipschitz_comparison.csv`):

- Measures exact SVD spectral bounds + finite-difference logits/TV Lipschitz
  under three conditions: random init, synthetic x3 weight stress, and REAL
  GAIL training (150 epochs, adversarial loop with MockExpert demos).
- Result: real GAIL training pushes the unconstrained generator's spectral
  bound to `31.7` (logits-L `1.65`) — adversarial training genuinely inflates
  weight norms, this is measured, not assumed. Spectral normalization keeps the
  bound at exactly `1.0000` through training (and `1.0000` under x3 stress,
  110.9x logits-L reduction), so the policy-jump bound
  `||pi_new - pi_old||_TV <= L_pi * eta * (1-rho)` stays non-vacuous.

E8 Cloud-edge SKB alignment:

```powershell
python experiments/eval_skb_alignment.py --trials 40
```

E8 design note (formal rerun, WN18RR, threshold 0.30 — aligned with E3/E5/E6;
`experiments/results/skb_alignment_summary.csv`):

- Same-system comparison of `no_align` / `full_retransmit` /
  `incremental_align` on identical stale EdgeSKB states (kappa `0.483`, i.e.
  ~half the required knowledge missing), intent-matched WN18RR references
  (35-61 nodes). Deterministic local simulation per the project constraint.
- Result: `incremental_align` restores the SAME semantic consistency as full
  retransmission (`0.800` vs `0.800`; `no_align` `0.415`) at `9.4 KB` vs
  `25.5 KB` payload (63% saving) and slightly lower latency (20.2 vs 21.5 ms).
  The old FB15K237/threshold-0.55 default produced semantically meaningless
  references and is deprecated.

## PyCharm Run Configurations

The project includes shared PyCharm run configurations under
`.idea/runConfigurations/` for visible execution in the PyCharm Run panel:

- `Full Experiments E1-E8`: runs `scripts/run_full_experiments.py`.
- `Plot Results`: runs `experiments/plot_results.py`.
- `Summarize Full Suite`: runs `experiments/summarize_full_suite.py`.
- `E8 SKB Alignment`: runs `experiments/eval_skb_alignment.py --trials 40`.
- `Unit Tests`: runs `scripts/run_unittest_discover.py`.
- `Compile Checks`: runs `scripts/run_compile_checks.py`.

Use the project SDK named `pytorch` and keep `PYTHONUNBUFFERED=1` so progress is
printed immediately in the Run panel.

For a separate visible Windows console, run:

```powershell
scripts\run_full_experiments.cmd
```

The full-suite runner executes E1-E8 with paper-scale defaults, writes a
timestamped `experiments/results/full_suite_*` directory, tees each step to its
own log, and publishes the E1 seed42 embedding to the legacy checkpoint path
used by `experiments/measure_psi_transE.py`.

Latest completed full-suite run:

- `experiments/results/full_suite_20260621_220711/`
- `failed_steps=[]`
- Summary: `experiments/results/full_suite_20260621_220711/full_suite_summary.md`
- Key metrics CSV: `experiments/results/full_suite_20260621_220711/full_suite_key_metrics.csv`
- Figures: `experiments/figures/fig1_convergence.pdf`,
  `experiments/figures/fig2_latency.pdf`,
  `experiments/figures/fig3_ood_robustness.pdf`

## Verification Commands

```powershell
D:\anconda\envs\pytorch\python.exe -m unittest discover -s tests
D:\anconda\envs\pytorch\python.exe -m py_compile agents\slow_loop\skb_server.py utils\semantic_consistency.py experiments\eval_skb_alignment.py
```

## Notes For Future Agents

- Do not edit `scripts/train_transe_unified.py` while a multi-seed E1 launcher
  is still running, because later seeds will import the current file version.
- Keep the cloud-edge server layer as local deterministic simulation unless
  the user explicitly asks for real deployment.
- When adding a new mechanism, add a small experiment script and a focused unit
  test first. The project is experiment-driven.
