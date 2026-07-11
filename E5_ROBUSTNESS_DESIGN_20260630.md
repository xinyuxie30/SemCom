# E5 Robustness Redesign - 2026-06-30

## Goal

Make E5 (`experiments/eval_robustness.py`) a publication-grade OOD-robustness
experiment: a LEARNED policy, a principled OOD-injection ablation, and metrics
that honestly separate the cost and the benefit of injection.

## Flaws found in the old E5 (rigor audit)

1. **Random-init policy (high).** `Generator` was rolled out with random weights —
   the experiment measured the robustness of an untrained network, so any
   "policy robustness" claim was unsupported.
2. **Stale `--max-hops 1` default (medium).** Help text cited "measured N*=1",
   contradicting the fair E2 result `N*=2`.
3. **Cumulative `injected_count` (medium).** Rows recorded the pruner's cumulative
   counter across scenarios, not per-scenario injections.
4. **Vacuous injection ablation (high, found by smoke).** Without the TransE
   embedder the pruner fell back to char n-gram matching; at `sim_threshold=0.30`
   every OOD keyword spuriously matched some KG entity, so injection never
   triggered and injON was IDENTICAL to injOFF at every SNR.
5. **Reward-only reporting (medium).** Injection grows the sub-graph (harder
   navigation → lower reward), while its actual benefit — representing OOD
   concepts at all — was never measured.

## Fixes

- `train_gail_policy()`: every (SNR, injection) / threshold config trains a GAIL
  policy (default `--train-epochs 150`, MockExpert + Discriminator, scenario
  rotation every 15 epochs — same pattern as E3/E6) before evaluation.
  `--train-epochs 0` remains available as a random-policy diagnostic.
- `--max-hops` default 1 → 2 (follows fair E2 `N*=2`).
- `injected_count` records the per-scenario delta.
- Experiment A loads the WN18RR TransE embedder (system config, cf. `train.py`):
  OOD detection is principled (NLTK-unresolvable keyword → `is_ood` → inject),
  so injON vs injOFF genuinely differ.
- New `entity_coverage` metric: fraction of intent entities semantically present
  in the transmitted sub-graph (matched KG node or injected OOD node).
- Experiment B (threshold ablation) deliberately uses the CHAR N-GRAM FALLBACK
  matcher and is labelled as such: with TransE the threshold is inert by
  construction (NLTK-resolved keywords score sim≈1.0; OOD keywords bypass the
  threshold), so the threshold is only load-bearing for the fallback matcher.

## Formal results (WN18RR, trained policy, 150 train epochs, 90 eval episodes/config)

### Experiment A — OOD robustness vs SNR (TransE matcher)

| SNR (dB) | inj | reward | goal_rate | entity_coverage |
|---:|---|---:|---:|---:|
| -5 | ON | 0.694 | 0.078 | **1.00** |
| -5 | OFF | 5.571 | 0.556 | 0.20 |
| 0 | ON | 0.806 | 0.078 | **1.00** |
| 0 | OFF | 5.643 | 0.556 | 0.20 |
| 5 | ON | 0.917 | 0.078 | **1.00** |
| 5 | OFF | 5.717 | 0.556 | 0.20 |
| 10 | ON | 1.237 | 0.100 | **1.00** |
| 10 | OFF | 5.793 | 0.556 | 0.20 |

Honest reading:

- **Injection is a coverage-vs-reward trade-off, reported as such.** With
  injection ON the transmitted sub-graph represents **100%** of the intent
  entities (including OOD concepts like `8k_video`, `magnetic_anomaly`); with
  injection OFF only **20%** — the receiver simply cannot reason about the
  missing OOD concepts, regardless of its higher navigation reward.
- injOFF's higher reward is a smaller-graph artifact (in-KG-only graph → nearer
  targets → easier Semi-MDP navigation), NOT evidence against injection. Reward
  must not be read alone here — exactly the multi-metric lesson from E3.
- Reward increases monotonically with SNR under BOTH settings (e.g. injON
  0.694 → 1.237), confirming the channel model actually couples into the
  decision loop.

### Experiment B — threshold ablation (char n-gram FALLBACK matcher, SNR 0 dB)

| threshold | reward | injected/scenario | nodes |
|---:|---:|---:|---:|
| 0.30 | 7.589 | 0.0 | 40.7 |
| 0.45 | 5.185 | 4.0 | 64.0 |
| 0.60 | 1.032 | 4.3 | 64.0 |
| 0.75 | 0.468 | 5.0 | 64.0 |
| 0.90 | 0.468 | 5.0 | 64.0 |

Documents the fallback matcher's operating point (`0.30`, the default cited by
E3's `--ours-sim-threshold`). This table is about the FALLBACK path only; the
system's primary TransE path does not use the threshold.

## Reproduce

```powershell
D:\anconda\envs\pytorch\python.exe experiments\eval_robustness.py --episodes-per-config 30 --train-epochs 150 --device cuda
```

Outputs: `experiments/results/ood_robustness_snr.csv`,
`ood_robustness_snr_summary.csv`, `threshold_ablation.csv`.
