# CognitiveSemCom Agent Notes

This is an independent project. Do not depend on external code unless it is
explicitly documented as a baseline reference.

## Current Research Direction

Intent-driven KG-SKB semantic communication with:

- TransE semantic matching.
- Dynamic connected KG sub-graph pruning.
- OOD entity injection.
- GAIL + Semi-MDP reasoning.
- Cloud-edge SKB alignment simulation.

No LLM/LAM API is allowed. Intent extraction is keyword lookup only:

`agents/slow_loop/intent_extractor.py`

## Hard Constraints

- `STATE_DIM = 452` must not change.
- Fixed state layout:
  `node_emb(256) + sub_emb(128) + intent(64) + csi(4)`.
- Do not require real cloud or edge server deployment.
- Cloud/edge SKB behavior must be reproducible local simulation.
- Existing CSV files are smoke-test artifacts unless explicitly rerun.

## Data Root

`C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks/`

Datasets:

- `FB15K237`
- `WN18RR`
- `NELL-995`

## Main Components

- `agents/slow_loop/dynamic_kb_pruner.py`
  - Intent keyword matching, BFS pruning, OOD injection.

- `agents/slow_loop/skb_server.py`
  - Local cloud-edge SKB simulation.
  - Strategies: `no_align`, `full_retransmit`, `incremental_align`.

- `utils/semantic_consistency.py`
  - KG semantic consistency score for receiver-side SKB quality.

- `experiments/eval_skb_alignment.py`
  - E8 cloud-edge SKB alignment experiment.

- `experiments/eval_nstar_scan.py`
  - E2 task-aware N* scan (FAIR: neutral full-KG task, no self-reference).
  - Defines a neutral multi-seed 2-hop task on the FULL KG (independent of any
    radius), then compares N in {1, 2, 3}; N* = smallest covering radius at
    lowest complexity (N*=2 on WN18RR).

- `env/mdp_env.py`
  - Semi-MDP state and action.
  - Action is path hop + bandwidth + MCS.

## External Baseline Interpretation

- `C:/Users/xinyuxie/Desktop/iSAC-main`
  - Reference for original iSAC/G-RML and no-pruning behavior.

- `C:/Users/xinyuxie/Desktop/iRML-main`
  - Reference for multi-layer reasoning and collaborative reasoning.

- `D:/Paper/语义入门资料/code+paper（写代码重点看）`
  - Mechanism references only:
    - Dynamic-JSCC for adaptive rate control.
    - OFDM-JSCC for explicit CSI modeling.
    - PADC/OraNet for quality prediction.
    - WITT for attention/Transformer semantic encoding.

## Experiment Commands

```powershell
python scripts/train_transe_unified.py --dataset FB15K237 --epochs 3000
python experiments/measure_psi_transE.py --num-seeds 300
python experiments/eval_nstar_scan.py --dataset WN18RR --seeds 3 --device cuda
python experiments/eval_end_to_end.py --epochs 500 --seeds 5
python experiments/eval_convergence.py --epochs 500 --seeds 3
python experiments/eval_latency.py --trials 500
python experiments/eval_robustness.py --episodes-per-config 30
python experiments/profile_lipschitz.py
python experiments/eval_skb_alignment.py --trials 40
```

Visible E8 terminal run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_skb_alignment.ps1
```

PyCharm visible runs:

- `Full Experiments E1-E8`: `scripts/run_full_experiments.py`.
- `Plot Results`: `experiments/plot_results.py`.
- `Summarize Full Suite`: `experiments/summarize_full_suite.py`.
- `E8 SKB Alignment`: `experiments/eval_skb_alignment.py --trials 40`.
- `Unit Tests`: `scripts/run_unittest_discover.py`.
- `Compile Checks`: `scripts/run_compile_checks.py`.

These configurations live in `.idea/runConfigurations/` and use the `pytorch`
project SDK with `PYTHONUNBUFFERED=1`.

Visible full-suite command:

```powershell
scripts\run_full_experiments.cmd
```

The full-suite runner creates a timestamped directory under
`experiments/results/full_suite_*`, stores one log per step, writes
`manifest.json`, and saves E1 TransE checkpoints under
`checkpoints/full_suite_transe_fb15k237_*`.

Latest completed run:

- `experiments/results/full_suite_20260621_220711/`
- Summary: `experiments/results/full_suite_20260621_220711/full_suite_summary.md`
- Figures: `experiments/figures/fig1_convergence.pdf`,
  `experiments/figures/fig2_latency.pdf`,
  `experiments/figures/fig3_ood_robustness.pdf`

Latest formal E2 N* rerun (FAIR protocol, neutral full-KG task, no self-reference):

- Command: `python experiments/eval_nstar_scan.py --dataset WN18RR --seeds 3 --device cuda`
- Result (WN18RR, `experiments/results/nstar_scan_fair.csv`): `N*=2` emerges from
  real task coverage. N=1 path_success `0.00` (1-hop cannot cover the 2-hop task),
  N=2 `1.00` (46 nodes), N=3 `1.00` (64 nodes, redundant). N* = smallest covering
  radius at lowest complexity. The old circular "score 9.1477" scan is deprecated.

Latest formal E3 FAIR rerun (see `E3_FAIR_DESIGN_20260624.md`):

- Command: `python experiments/eval_end_to_end.py --epochs 500 --seeds 5 --dataset WN18RR --device cuda`
- Summary: `experiments/results/end_to_end_summary_fair_formal.csv`
- Design: the start-target task is defined on the FULL KG (multi-seed 2-hop
  reasoning hop), NOT by any pruner — no circular task, no self-referential score.
  All methods are scored identically with multi-metric (`path_success`,
  `semantic_consistency`, `payload`, `task_goal_rate`, `raw_policy_reward`); no
  Ours-only step; HAKE is a seed-aware non-strawman static-hierarchy opponent.
  Default dataset WN18RR (FB15K237 Freebase mids cannot be intent-matched).
- Result (5 seeds): `DynamicPruned_Ours` path_success `1.00 +/- 0.00`,
  semantic_consistency `0.800 +/- 0.000`, task_goal_rate `0.10`; HAKE
  path_success `0.00`, consistency `0.169 +/- 0.021`; NoPruning path_success
  `0.00`, consistency `0.009`. Ours fairly wins on every task metric.

## Verification

```powershell
D:\anconda\envs\pytorch\python.exe -m unittest discover -s tests
D:\anconda\envs\pytorch\python.exe -m py_compile agents\slow_loop\skb_server.py utils\semantic_consistency.py experiments\eval_skb_alignment.py
```

## Safety Notes

- Do not edit `scripts/train_transe_unified.py` while a multi-seed E1 launcher
  is running, because later seeds import the current file version.
- Prefer adding focused tests with each mechanism.
- Keep the implementation local and reproducible unless the user explicitly
  requests real service deployment.
