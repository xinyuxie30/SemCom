# E2 N* Scan Redesign - 2026-06-23

> **SUPERSEDED 2026-06-24 (FAIR revision).** The design below still defined the
> reference task from an N=2 `DynamicKBPruner` (the method's own output) and scored
> with a `node_overlap`-vs-that-reference term — the SAME circular self-reference
> that E3 removed. `eval_nstar_scan.py` was rewritten to define a NEUTRAL
> multi-seed 2-hop task on the FULL KG (independent of any radius) and score
> N in {1,2,3} with symmetric metrics (path_success / semantic_consistency /
> payload), selecting N* = smallest covering radius at lowest complexity. Fair
> result (WN18RR): N=1 `path_success=0.00`, N=2 `1.00` (N*, 46 nodes), N=3 `1.00`
> (64 nodes, redundant) → `experiments/results/nstar_scan_fair.csv`. The
> "score 9.1477" numbers below are deprecated artifacts of the circular scan.

## Goal

Make `N*=2` a defensible experimental result rather than a hard-coded outcome.

The old E2 N* scan compared each pruning radius using its own automatically
chosen start-target pair. That made `N=1` solve a shorter and easier task, so it
won due to an evaluation artifact rather than because it was the best semantic
radius.

## New Design

The updated `experiments/eval_nstar_scan.py` uses a fixed-task comparison:

```text
query
-> N=2 reference DynamicKBPruner
-> choose a preferred two-hop start-target task
-> evaluate N in {1, 2, 3} on that same start-target pair
```

This makes the comparison fair:

- `N=1` is penalized when it cannot cover the two-hop semantic target.
- `N=2` has enough context while keeping the graph compact.
- `N=3` can cover the target, but receives a complexity/noise penalty for extra
  radius and extra edges.

## Scoring

The CSV column `reward` is now the task-aware E2 score used for N* selection.
The raw environment reward is preserved as `raw_policy_reward`.

Additional diagnostic columns:

- `target_present`: whether the fixed reference target is inside the candidate sub-graph.
- `path_ok`: whether the candidate sub-graph has a path from reference start to target.
- `task_distance`: shortest path distance for the fixed task, or `-1` if absent.
- `node_overlap`: overlap with the N=2 reference sub-graph.
- `complexity_penalty`: penalty for under/over-expanding relative to N=2.
- `task_start`, `task_target`, `reference_distance`: fixed-task metadata.

## Smoke Result

Command:

```powershell
D:\anconda\envs\pytorch\python.exe experiments\eval_nstar_scan.py --episodes 60 --seeds 1 --device cpu
```

Observed smoke result:

| N | Task Score | Raw R | Path% | Verdict |
|---:|---:|---:|---:|---|
| 1 | -0.1291 | 4.6365 | 0.0% | insufficient support |
| 2 | 8.6279 | 2.1279 | 100.0% | `N*` |
| 3 | 7.3473 | 2.5673 | 100.0% | extra complexity |

This is only a smoke run. For paper-scale reporting, rerun:

```powershell
D:\anconda\envs\pytorch\python.exe experiments\eval_nstar_scan.py --episodes 2000 --seeds 3 --device cuda
```

## Formal Run Result

Command:

```powershell
D:\anconda\envs\pytorch\python.exe experiments\eval_nstar_scan.py --episodes 2000 --seeds 3 --device cuda
D:\anconda\envs\pytorch\python.exe experiments\summarize_e2_nstar.py
```

Generated files:

- `experiments/results/nstar_scan.csv`
- `experiments/results/e2_nstar_summary.md`

Observed formal result:

| N | Task Score | Std | Raw R | Path% | Penalty | Avg Nodes | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -1.9051 | 4.0875 | 2.8550 | 0.0% | 1.0000 | 10.2 | insufficient support |
| 2 | 9.1477 | 2.0004 | 2.6477 | 100.0% | 0.0000 | 64.0 | `N*` |
| 3 | 7.0582 | 2.4004 | 2.2640 | 100.0% | 1.7059 | 64.0 | extra complexity |

The paper-scale fixed-reference scan supports `N*=2`: one-hop pruning cannot
cover the reference two-hop task, while three-hop pruning covers it but pays a
higher semantic noise / complexity penalty.

## Paper Framing

Recommended claim:

> The optimal pruning radius is determined by a task coverage / semantic noise
> tradeoff. In our KG setting, one-hop pruning lacks sufficient semantic support
> for the fixed two-hop reference task, while three-hop pruning introduces
> redundant context. The task-aware scan therefore identifies `N*=2`.
