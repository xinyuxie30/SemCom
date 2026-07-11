"""Diagnostic: is intent keyword -> KG entity matching meaningful on a dataset?

Char n-gram `EntityEmbedder` matching only carries signal when entity surface
forms contain natural-language words. This prints, per dataset, the best full-KG
match + similarity for each intent keyword, the neutral E3 task, and whether Ours'
pruned sub-graph actually contains that task.

It justifies WN18RR as the fair-E3 default: WN18 synsets resolve (via NLTK) to
English lemmas that match keywords, NELL `concept_*_word` ids match partially, and
FB15K237 Freebase mids (`/m/0c73z`) carry no lexical signal so matching degenerates
to noise (every method scores path=0). See E3_FAIR_DESIGN_20260624.md.

Usage:
  python scripts/diagnose_intent_match.py WN18RR NELL-995 FB15K237
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from agents.slow_loop.dynamic_kb_pruner import EntityEmbedder, DynamicKBPruner, get_entity_display_name
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.intent_parser import parse_intent
from experiments.eval_end_to_end import (
    load_base_kg, clone_kg, match_seeds_full_kg, build_ground_truth_task,
    has_path, _entity_embedding_matrix,
)

QUERY = "High priority on radar and depth for pedestrian crossing street."


def diag(ds: str) -> None:
    print(f"\n===== {ds} =====")
    kg = load_base_kg(ds)
    print(f"entities={len(kg.entity2id):,}  sample_ids={list(kg.entity2id)[:3]}")
    emb = EntityEmbedder(dim=128)
    ext = IntentExtractor()
    intent, ents = parse_intent(ext.query(QUERY))
    print(f"intent core_entities = {ents}")

    eid_list, mat = _entity_embedding_matrix(kg, emb)
    print("keyword -> best full-KG match (similarity):")
    for ent in ents:
        q = emb.embed(str(ent).lower().strip().replace(" ", "_"))
        sims = mat @ q
        i = int(np.argmax(sims))
        print(f"  {ent:14s} -> {get_entity_display_name(eid_list[i]):24s} (id={eid_list[i]}) sim={sims[i]:.3f}")

    seeds = match_seeds_full_kg(ents, kg, emb)
    start, target, ref, dist, mode = build_ground_truth_task(kg, seeds, intent, emb)
    print(f"task: {get_entity_display_name(start)} -> {get_entity_display_name(target)} "
          f"dist={dist} mode={mode} ref_nodes={len(ref.nodes)}")

    pr = DynamicKBPruner(clone_kg(kg), max_hops=2, similarity_threshold=0.3, enable_injection=True)
    cand = pr.update_sub_graph(ents, intent)
    print(f"Ours: nodes={len(cand.nodes)} edges={len(cand.edges)} injected={pr.injected_count}")
    print(f"  start in Ours? {start in cand.nodes} | target in Ours? {target in cand.nodes} | "
          f"path? {has_path(cand, start, target)}")
    trace = cand.metadata.get("match_trace", [])
    actions: dict = {}
    for row in trace:
        actions[row.get("action", "?")] = actions.get(row.get("action", "?"), 0) + 1
    print(f"  match_trace actions = {actions}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["WN18RR"]:
        diag(ds)
