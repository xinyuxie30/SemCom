# TransE Code Review - 2026-06-21

Scope: TransE/RotatE training, evaluation, checkpoint export, `TransEEmbedder`,
and the dynamic KB pruner integration. The running E1 experiment was not
modified after launch, so seed results stay on one code version.

## Current Verdict

The current TransE direction is viable for this project. It is being used as a
stable 128-d KG geometry for intent-to-entity matching and subgraph pruning,
not as a standalone state-of-the-art KGE benchmark. The latest training script
has the important alignment properties:

- TransE train score, filtered evaluation, and exported downstream embeddings
  all use L2-normalized entity/relation vectors.
- Checkpoint tags can include seed suffixes, avoiding multi-seed overwrite.
- E1 is running with conda `pytorch` CUDA, not the CPU-only `.venv`.

Running E1:

- Results: `experiments/results/e1_transe_fb15k237_20260621_191300`
- Checkpoints: `checkpoints/e1_transe_fb15k237_20260621_191300`
- Launcher PID at review time: `11764`
- Training PID at review time: `32296`
- Status check at 2026-06-21 19:33: no completed seed JSON yet; GPU process
  `D:\anconda\envs\pytorch\python.exe` is still active.
- Status check at 2026-06-21 20:45: no completed seed JSON yet; launcher
  PID `11764` and training PID `32296` are still active. Because the launcher
  starts later seeds from the current `scripts/train_transe_unified.py`, do not
  edit that script until the 3-seed E1 run completes unless a new mixed-version
  run is intentional.

## Patch Status

Applied during this review, without modifying the running E1 training script:

- `models/transe.py`: aligned default TransE scoring with E1 semantics
  (`margin=5.0`, L2-normalized entity/relation vectors before distance).
- `models/transe_embedder.py`: added dataset-fingerprinted WordNet index cache
  and safe `tag_suffix` checkpoint resolution for multi-seed E1 outputs.
- `agents/slow_loop/dynamic_kb_pruner.py`: cached TransE entity matrices,
  used `np.argpartition` for semantic top-k, bypassed KG argmax for known OOD
  keywords, added `match_trace` metadata, and restored old `prune()` /
  `resolve_entity_name()` aliases. The duplicate `get_top_hub_nodes()` method
  was also removed.
- `tests/test_transe_pipeline.py`: added six focused tests for normalized
  scoring, embedding row alignment, dataset fingerprints, suffix resolution,
  OOD injection, and pruner matrix caching.

Verification:

```powershell
D:\anconda\envs\pytorch\python.exe -m unittest tests.test_transe_pipeline
D:\anconda\envs\pytorch\python.exe -m py_compile models\transe.py models\transe_embedder.py agents\slow_loop\dynamic_kb_pruner.py tests\test_transe_pipeline.py
```

Both commands passed.

## Paper Anchors

- Bordes et al., "Translating Embeddings for Modeling Multi-relational Data",
  NeurIPS 2013. Core TransE idea: score triples by making `h + r` close to `t`.
  Source: https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html
- Sun et al., "RotatE: Knowledge Graph Embedding by Relational Rotation in
  Complex Space", 2019. Useful reference for relation patterns and
  self-adversarial negative sampling.
  Source: https://arxiv.org/abs/1902.10197
- Wang et al., "Knowledge Graph Embedding by Translating on Hyperplanes",
  AAAI 2014. Useful reference for TransE limitations on 1-to-N/N-to-1/N-to-N
  relations and Bernoulli corruption motivation.

Zotero note: no Zotero MCP/tool was exposed in the current Codex tool surface.
This review therefore uses local code plus public primary sources.

## Findings

### P1 - Duplicate TransE implementations can diverge [FIXED]

Files:

- `scripts/train_transe_unified.py`
- `models/transe.py`

`scripts/train_transe_unified.py` has the experiment-critical TransE class with
normalized scoring and margin 5.0. `models/transe.py` has a separate minimal
TransE with unnormalized scoring and margin 1.0. If future code imports
`models.TransE`, it will not match E1 checkpoint semantics.

Recommendation:

- `models/transe.py` now exposes the same normalized TransE scoring semantics
  used by the training script.
- Keep `scripts/train_transe_unified.py` as an experiment driver only.
- A tiny synthetic test now verifies normalized score behavior.

### P1 - RotatE implementation is incomplete relative to the paper

File: `scripts/train_transe_unified.py`

The current RotatE relation embedding is used as an unconstrained phase via
`cos(R)`/`sin(R)`, but entity modulus is not constrained and the training loss is
still the TransE-style hinge loss. RotatE's paper emphasizes rotation in complex
space and self-adversarial negative sampling. If RotatE is used as a baseline,
this should be marked as "lightweight RotatE-like baseline", not full RotatE.

Recommendation:

- For paper claims, either remove RotatE as a serious KGE baseline or implement
  a closer RotatE training objective with gamma/adversarial negative sampling.
- If kept, record `model_variant` in eval JSON.

### P1 - Global WordNet caches are unsafe across datasets [FIXED]

File: `models/transe_embedder.py`

`_WN_LEMMA_TO_ENTITIES` and `_WORD_TO_BEST_ENTITY` are module-level globals.
`build_wn18_nltk_index()` returns the existing cache as soon as it is non-empty,
without checking whether the current `entity2id` belongs to the same dataset.
This can silently leak WN18RR mappings into a later FB15K237/NELL embedder in
the same Python process.

Recommendation:

- WordNet index caching is now keyed by a stable `entity2id` fingerprint.
- Tests verify that same-size datasets with different entity names do not share
  a fingerprint.

### P2 - Downstream TransE entity matrix is rebuilt per pruner call [FIXED]

File: `agents/slow_loop/dynamic_kb_pruner.py`

`DynamicKBPruner._match_and_inject()` rebuilds `eid_list` and stacks all entity
embeddings every call. That is acceptable for smoke tests but wasteful for
FB15K237 and worse for NELL-995.

Recommendation:

- `DynamicKBPruner` now caches the original KG `(eid_list, ent_matrix)` on
  first use.
- Dynamically injected nodes are intentionally excluded from the TransE semantic
  index, because they do not have trained KG embeddings.
- Semantic top-k now uses `np.argpartition`.

### P2 - OOD hash vectors can create false high-similarity matches [FIXED]

Files:

- `models/transe_embedder.py`
- `agents/slow_loop/dynamic_kb_pruner.py`

For keywords not resolved by WordNet, `embed_keyword()` returns a random
deterministic unit vector and `is_ood=True`. The pruner computes matrix cosine
anyway and only injects if `best_sim < threshold`; for large entity sets, random
unit vectors can occasionally exceed a loose threshold by chance. Since
`is_ood` is known, unresolved terms should usually inject directly.

Recommendation:

- If `is_ood=True`, the pruner bypasses KG argmax and injects directly.
- `SubGraph.metadata["match_trace"]` records keyword, OOD flag, action,
  best match, and similarity where applicable.

### P2 - Multi-seed checkpoint loading can become ambiguous [FIXED]

File: `models/transe_embedder.py`

E1 writes checkpoint files with seed suffixes, e.g.
`transe_fb15k237_transe_seed42_entity_emb.npy`. The old loader only searched
for unsuffixed defaults, which made post-E1 seed-specific loading awkward.

Recommendation:

- `create_transe_embedder(..., tag_suffix="seed42")` is now supported.
- If multiple suffixed checkpoints exist and no suffix is specified, the loader
  raises a clear error instead of silently choosing one.

### P2 - Evaluation history uses a fixed prefix, not sampled validation/test

File: `scripts/train_transe_unified.py`

Fast eval uses `test[:fast_eval_size]` and `train[:fast_eval_size]`. If triples
are ordered by relation or generation process, MRR curves can be biased.

Recommendation:

- At startup, create deterministic shuffled train/valid/test eval subsets using
  the experiment seed.
- Include validation MRR in history and optionally track best validation epoch.

### P2 - Full filtered evaluation is loop-based and may dominate runtime

File: `scripts/train_transe_unified.py`

The implementation scores all entities per triple in a Python loop. It is
correct and simple, but full FB15K237 eval is expensive and WN18RR/NELL can be
much slower.

Recommendation:

- Keep current evaluator for correctness.
- Add an optional chunked/batched evaluator for tail/head prediction.
- Record `eval_time_sec` separately from training time.

### P3 - Training sampler is Python-heavy

File: `scripts/train_transe_unified.py`

`BernoulliSampler.sample_batch()` uses Python loops, `random.choices`, and
creates CPU tensors each batch. GPU training is therefore partly input-bound.

Recommendation:

- For the current E1, do not change this.
- Next version: pre-store triples in NumPy arrays, vectorize positive sampling,
  and use batched corruption with rejection repair.

### P3 - Duplicate `get_top_hub_nodes()` definition [FIXED]

File: `agents/slow_loop/dynamic_kb_pruner.py`

`GlobalKnowledgeGraph.get_top_hub_nodes()` was defined twice with identical
implementation. This was harmless but noisy.

Recommendation:

- One duplicate definition was removed. The public behavior is unchanged.

### P3 - Encoding/comment corruption reduces maintainability

Files:

- `CLAUDE.md`
- `scripts/train_transe_unified.py`
- `models/transe_embedder.py`
- `agents/slow_loop/dynamic_kb_pruner.py`

Several comments/docstrings show mojibake. This does not affect runtime, but it
does make review and paper artifact tracing harder.

Recommendation:

- Convert source files to clean UTF-8 after E1 completes.
- Prefer ASCII in code comments unless Chinese comments are needed and verified
  in UTF-8.

## Suggested Next Patch Order After E1 Completes

1. Freeze the E1 code version and copy completed summaries into
   `experiments/results`.
2. Add deterministic shuffled eval subsets and validation MRR in
   `scripts/train_transe_unified.py`.
3. Add optional chunked/batched filtered evaluation and record `eval_time_sec`.
4. Decide whether RotatE is removed, renamed as lightweight RotatE-like, or
   upgraded to a closer paper-faithful baseline.
5. Vectorize the Python-heavy negative sampler if E1 runtime is too slow.
6. Clean mojibake comments/docstrings and remove the duplicate hub method.

## Commands

Monitor the running E1:

```powershell
D:\anconda\envs\pytorch\python.exe scripts\summarize_transe_e1.py --run-dir experiments\results\e1_transe_fb15k237_20260621_191300
```

Check GPU process:

```powershell
nvidia-smi
```
