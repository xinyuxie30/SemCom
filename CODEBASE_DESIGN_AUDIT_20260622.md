# CognitiveSemCom Codebase Design Audit - 2026-06-22

## Verdict

The active code path is compliant with the current CognitiveSemCom design
constraints after the fixes in this audit.

Key result: the supported pipeline is local and reproducible:

```text
keyword intent lookup
-> parse_intent() intent_vec[64]
-> DynamicKBPruner connected sub-graph + OOD injection
-> SubGraph with semantic node embeddings
-> GNN sub_emb[128]
-> Semi-MDP/GAIL state[452]
-> local cloud-edge SKB alignment simulation
```

## Compliance Matrix

| Requirement | Status | Evidence |
|---|---|---|
| `STATE_DIM = 452` is fixed | Pass | `data/dataloader.py`, `env/mdp_env.py`, unit tests |
| Fixed layout `node_emb(256)+sub_emb(128)+intent(64)+csi(4)` | Pass | state builder assertion and `EnvConfig` contract |
| No LLM/LAM API | Pass | code scan has no active LAM/GPT/OpenAI usage |
| Intent extraction is keyword lookup only | Pass | `agents/slow_loop/intent_extractor.py` |
| No real cloud/edge deployment required | Pass | `agents/slow_loop/skb_server.py` is local object simulation |
| Cloud-edge SKB behavior reproducible locally | Pass | E8 components use deterministic local payload/cache models |
| External projects are references only | Pass | no active imports from external baseline repos |
| Existing CSVs treated as smoke artifacts unless rerun | Pass with note | latest latency CSVs were refreshed by a 5-trial smoke in this audit |

## Issues Fixed

- Fixed `SemiMDPEnv.reset()` fallback intent dimension. It now uses
  `SubGraph.intent_vec` and rejects non-64-dimensional intent vectors.
- Added `SubGraph.__post_init__()` validation for `INTENT_DIM == 64`.
- Changed `SemiMDPEnv.state_dim` to report the full policy input dimension
  `452`, and added `observation_dim` for the local env-only observation size.
- Removed the misleading `MockLAMProvider` export and cleaned LAM-style naming
  from active code paths.
- Renamed E4 "pure LAM" latency baseline to `simulated_online_extractor`.
- Replaced unstable `hash(query)` usage with SHA-256 stable IDs/seeds.
- Made `MockExpert` random-walk fallback deterministic from SubGraph content.
- Fixed nondeterministic `list(visited)` ordering in the E3 HAKE baseline.
- Expanded `scripts/run_compile_checks.py` to cover the core slow-loop,
  state, experiment, plotting, summary, and training files.
- Added unit tests for the intent dimension contract and env reset fallback.
- Cleaned main logger and train/E4 visible output to ASCII for readable
  PowerShell/PyCharm progress.

## Verification

All commands below passed on 2026-06-22:

```powershell
D:\anconda\envs\pytorch\python.exe scripts\run_compile_checks.py
D:\anconda\envs\pytorch\python.exe scripts\run_unittest_discover.py
D:\anconda\envs\pytorch\python.exe test_slow_loop.py --device cpu
D:\anconda\envs\pytorch\python.exe train.py --mode debug --epochs 2 --device cpu
D:\anconda\envs\pytorch\python.exe experiments\eval_latency.py --trials 5 --scaling-trials 3
D:\anconda\envs\pytorch\python.exe -m compileall -q agents data env models experiments scripts config tests train.py eval.py check_env.py test_slow_loop.py
```

Important observed outputs:

- Unit tests: 11 tests passed.
- Slow-loop integration: WN18RR loaded, SubGraph `64` nodes / `64` edges,
  GNN `sub_emb` shape `(128,)`, `STATE_DIM` contract OK at `452`.
- Debug train: completed on CPU and saved `checkpoints\phase3_checkpoint.pt`.
- E4 smoke: wrote local latency CSVs with `simulated_online_extractor`.

## Static Scans

Passed scans:

```powershell
rg -n -e "\blam\b|lam_resp|lam_response|MockLAMProvider|Mockobject|pure_lam|lam_inference" -S -g "!experiments/results/**" ...
rg -n "np\.random\.default_rng\(\)|hash\(query\)|str\(hash\(" -S -g "!experiments/results/**" ...
```

Network/API scan result:

```text
agents/slow_loop/skb_server.py: "This module deliberately does not open sockets or deploy real services."
```

This is a negative/safety statement, not an active network dependency.

## Remaining Notes

- `.git/HEAD` is missing, so `git status` cannot be used in this directory even
  though a `.git` folder exists. This is project metadata hygiene, not a runtime
  design violation.
- `experiments/results/**` contains historical logs/CSVs with old labels such as
  `pure_lam_online`. Those are generated artifacts and were excluded from code
  compliance scans.
- The E4 latency CSVs were overwritten by a 5-trial smoke during this audit.
  Treat them as smoke artifacts unless E4 is rerun at paper scale.
- This audit did not rerun repaired paper-scale E3/E5/full-suite experiments.
  Use `scripts\run_full_experiments.cmd` for a fresh paper-scale result set.
- Some source comments still contain Unicode arrows/math symbols. The main
  visible train and E4 outputs are now ASCII; remaining Unicode is cosmetic.

## Recommended Next Step

Run a fresh visible full suite after this audit if paper numbers are needed:

```powershell
scripts\run_full_experiments.cmd
```
