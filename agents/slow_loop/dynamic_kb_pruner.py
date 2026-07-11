"""
Dynamic Knowledge Base Pruner — Phase 3.

[Phase 3 Fix]: Complete rewrite of entity matching and graph evolution.
  1. EntityEmbedder: character n-gram hashing → deterministic embedding [128].
  2. Cosine similarity matching (replaces brittle NLTK exact match).
  3. Dynamic OOD injection: when similarity < threshold, add new node to global
     graph + edges to top-K hub nodes. This IS the "dynamic KB update" mechanism.

Pipeline:
  intent entities -> embed -> cosine-sim vs KG entity embeddings
    - sim >= threshold -> matched seed node for BFS
    - sim < threshold -> inject new node into global KG and use it as seed
  Multi-source BFS from all seeds -> SubGraph (<= max_nodes)
"""
from __future__ import annotations

import hashlib
import re
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from data.kg_interface import INTENT_DIM, SubGraph

# ── Dimension ───────────────────────────────────────────────────────────────
EMBEDDING_DIM = 128            # entity embedding dimension (matches TransE input)


# ═══════════════════════════════════════════════════════════════════════════════
# Entity Embedder — character n-gram hashing
# ═══════════════════════════════════════════════════════════════════════════════

class EntityEmbedder:
    """
    [Phase 3 Fix]: Deterministic character n-gram embedding.

    Words sharing substrings ("pedestrian" / "pedestrians") get higher cosine
    similarity than unrelated words. No external data or pre-trained model needed.

    Algorithm:
      1. Extract all char n-grams (n = 2, 3, 4) from the input string.
      2. Hash each n-gram → seed a random Gaussian vector.
      3. Sum all vectors → L2-normalize.
      4. Result: deterministic embedding where substring overlap → higher cosine sim.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        """Return [dim] L2-normalized embedding for `text`."""
        key = text.lower().strip()
        if key in self._cache:
            return self._cache[key].copy()

        ngrams: Set[str] = set()
        for n in (2, 3, 4):
            for i in range(len(key) - n + 1):
                ngrams.add(key[i:i + n])
        if not ngrams:                    # text shorter than 2 chars
            ngrams.add(key)

        vec = np.zeros(self.dim, dtype=np.float32)
        for ng in ngrams:
            seed = int(hashlib.md5(ng.encode()).hexdigest()[:8], 16) % (2**31)
            rng = np.random.default_rng(seed)
            vec += rng.normal(0, 1, (self.dim,)).astype(np.float32)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        self._cache[key] = vec
        return vec.copy()

    def cosine_sim(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between embeddings of text_a and text_b."""
        a = self.embed(text_a)
        b = self.embed(text_b)
        return float(np.dot(a, b))         # both are L2-normalized


# ═══════════════════════════════════════════════════════════════════════════════
# NLTK helpers (optional, used only for entity name readability)
# ═══════════════════════════════════════════════════════════════════════════════

_SYNSET_NAME_CACHE: Dict[str, str] = {}
_NLTK_AVAILABLE: Optional[bool] = None


def _nltk_available() -> bool:
    global _NLTK_AVAILABLE
    if _NLTK_AVAILABLE is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from nltk.corpus import wordnet as wn
                _ = wn.synsets('dog')
            _NLTK_AVAILABLE = True
        except Exception:
            _NLTK_AVAILABLE = False
    return _NLTK_AVAILABLE


def get_entity_display_name(entity_id: str) -> str:
    """
    [Phase 3 Fix]: Best-effort human-readable name for an entity.
    - WN18 numeric ID + NLTK → "lemma.n.01" → extract "lemma"
    - Falls back to raw entity_id.
    Used ONLY to generate a string for the EntityEmbedder; matching is purely
    embedding-based, not string-exact.
    """
    if entity_id in _SYNSET_NAME_CACHE:
        return _SYNSET_NAME_CACHE[entity_id]

    # FB15K237-style
    if entity_id.startswith('/m/'):
        _SYNSET_NAME_CACHE[entity_id] = entity_id
        return entity_id

    # WN18-style numeric ID
    if re.match(r'^\d{8}$', entity_id) and _nltk_available():
        try:
            from nltk.corpus import wordnet as wn
            offset = int(entity_id[:-1])
            pos_digit = int(entity_id[-1])
            pos_map = {1: 'n', 2: 'v', 3: 'a', 4: 'r'}
            pos = pos_map.get(pos_digit, 'n')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                synset = wn.synset_from_pos_and_offset(pos, offset)
            # Extract lemma from "lemma.n.01"
            name = synset.name().split('.')[0] if '.' in synset.name() else synset.name()
            _SYNSET_NAME_CACHE[entity_id] = name
            return name
        except Exception:
            pass

    _SYNSET_NAME_CACHE[entity_id] = entity_id
    return entity_id


def resolve_entity_name(entity_id: str) -> str:
    """Backward-compatible alias used by older slow-loop tests."""
    return get_entity_display_name(entity_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Global Knowledge Graph (extended with dynamic injection)
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalKnowledgeGraph:
    """
    [Phase 3 Fix]: Extended with inject_node() / inject_edge() for dynamic
    OOD knowledge evolution.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        self.triples: List[Tuple[int, int, int]] = []
        self.graph: nx.DiGraph = nx.DiGraph()
        self._next_dynamic_id: int = -1        # negative IDs for injected nodes
        self._loaded = False

    def load(self) -> "GlobalKnowledgeGraph":
        self._load_entities()
        self._load_relations()
        self._load_triples()
        self._loaded = True
        return self

    def _load_entities(self) -> None:
        path = self.data_dir / "entity2id.txt"
        if not path.exists():
            raise FileNotFoundError(f"entity2id.txt not found at {path}")
        with open(path, "r", encoding="utf-8") as fh:
            total = int(fh.readline().strip())
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_name, eid = parts[0], int(parts[1])
                    self.entity2id[entity_name] = eid
                    self.id2entity[eid] = entity_name

    def _load_relations(self) -> None:
        path = self.data_dir / "relation2id.txt"
        if not path.exists():
            raise FileNotFoundError(f"relation2id.txt not found at {path}")
        with open(path, "r", encoding="utf-8") as fh:
            total = int(fh.readline().strip())
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rel_name, rid = parts[0], int(parts[1])
                    self.relation2id[rel_name] = rid
                    self.id2relation[rid] = rel_name

    def _load_triples(self) -> None:
        path = self.data_dir / "train2id.txt"
        if not path.exists():
            raise FileNotFoundError(f"train2id.txt not found at {path}")
        with open(path, "r", encoding="utf-8") as fh:
            total = int(fh.readline().strip())
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 3:
                    h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                    self.triples.append((h, t, r))
        for h, t, r in self.triples:
            h_name = self.id2entity.get(h, str(h))
            t_name = self.id2entity.get(t, str(t))
            r_name = self.id2relation.get(r, str(r))
            self.graph.add_edge(h_name, t_name, relation=r_name, relation_id=r)

    # ── [Phase 3 Fix]: Dynamic injection methods ──────────────────────────

    def inject_node(self, node_name: str) -> str:
        """
        Add a new entity to the global KG. Idempotent — returns existing ID
        if the node already exists.
        """
        if node_name in self.entity2id:
            return node_name
        self._next_dynamic_id -= 1
        eid = self._next_dynamic_id
        self.entity2id[node_name] = eid
        self.id2entity[eid] = node_name
        self.graph.add_node(node_name)
        return node_name

    def get_semantic_neighbors(self, query_emb: np.ndarray,
                               eid_list: List[str], ent_matrix: np.ndarray,
                               k: int = 2, exclude: set = None) -> List[str]:
        """
        [Semantic Injection Fix]: Find the K entities in the KG with highest
        cosine similarity to `query_emb`, excluding nodes in `exclude`.
        Uses pre-computed entity embedding matrix for one-shot batch similarity.
        """
        if k <= 0 or len(eid_list) == 0:
            return []
        exclude = exclude or set()
        sims = np.dot(ent_matrix, query_emb)  # [E]
        valid_idx = np.fromiter(
            (i for i, eid in enumerate(eid_list) if eid not in exclude),
            dtype=np.int64,
        )
        if valid_idx.size == 0:
            return []

        valid_sims = sims[valid_idx]
        k_eff = min(k, valid_idx.size)
        if k_eff < valid_idx.size:
            top_local = np.argpartition(-valid_sims, k_eff - 1)[:k_eff]
            top_local = top_local[np.argsort(-valid_sims[top_local])]
        else:
            top_local = np.argsort(-valid_sims)
        return [eid_list[int(valid_idx[i])] for i in top_local]

    def inject_edge(self, src: str, dst: str, relation: str = "dynamic_related_to") -> None:
        """Add an edge between two nodes if it doesn't already exist."""
        if src not in self.graph or dst not in self.graph:
            return
        if not self.graph.has_edge(src, dst):
            if relation not in self.relation2id:
                self.relation2id[relation] = len(self.relation2id)
                self.id2relation[self.relation2id[relation]] = relation
            self.graph.add_edge(src, dst, relation=relation)

    def get_top_hub_nodes(self, k: int = 2) -> List[str]:
        """Return the k nodes with highest out-degree in the current graph."""
        degrees = sorted(self.graph.out_degree(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in degrees[:k]]


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic KB Pruner (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicKBPruner:
    """
    [Phase 3 Fix]: Embedding-based entity matching + OOD dynamic injection.

    Core algorithm (update_sub_graph):
      1. Embed all intent entities using EntityEmbedder.
      2. For each intent entity:
         a. Compute cosine similarity vs ALL KG entity display names.
         b. If max_sim ≥ threshold → best-match KG node is a seed.
         c. If max_sim < threshold → INJECT new node, connect to top-K hubs.
      3. Multi-source BFS from all seeds → sub-graph.
      4. Return SubGraph compatible with Phase 1 contract.
    """

    def __init__(self, global_kg: GlobalKnowledgeGraph,
                 max_subgraph_nodes: int = 64,
                 max_hops: int = 2,
                 similarity_threshold: float = 0.3,
                 hub_k: int = 2,
                 enable_injection: bool = True,       # [Phase 4 Fix]: toggle for ablation
                 embedding_dim: int = INTENT_DIM,
                 transe_embedder=None):               # [TransE Fix]: semantic embedder
        self.kg = global_kg
        self.max_subgraph_nodes = max_subgraph_nodes
        self.max_hops = max_hops
        self.similarity_threshold = similarity_threshold
        self.hub_k = hub_k
        self.embedding_dim = embedding_dim
        # [TransE Fix]: Use TransE semantic embeddings when available
        if transe_embedder is not None:
            self.embedder = transe_embedder
            self._use_transe = True
        else:
            self.embedder = EntityEmbedder(dim=EMBEDDING_DIM)
            self._use_transe = False
        self.enable_injection = enable_injection  # [Phase 4 Fix]: ablation toggle
        self.injected_count: int = 0          # [Phase 3 Fix]: track OOD injections
        self._transe_eid_list: Optional[List[str]] = None
        self._transe_ent_matrix: Optional[np.ndarray] = None
        self._match_trace: List[Dict[str, object]] = []

    # ── Main entry point ──────────────────────────────────────────────────

    def update_sub_graph(self, core_entities: List[str],
                         intent_vec: np.ndarray) -> SubGraph:
        """
        [Phase 3 Fix]: Full pipeline — match, inject, prune.

        Args:
            core_entities: deterministic intent keywords such as ["pedestrian", "radar"]
            intent_vec:    [INTENT_DIM] intent embedding

        Returns:
            SubGraph (Phase 1 contract: nodes, edges, intent_vec)
        """
        # Step 1: Match + Inject → determine seed nodes
        seed_nodes, injected = self._match_and_inject(core_entities)
        self.injected_count += injected
        match_trace = list(self._match_trace)

        # [Phase 3 Fix]: If too few seeds, add top hub nodes to ensure connectivity
        if len(seed_nodes) < 3:
            hubs = self.kg.get_top_hub_nodes(k=3)
            for h in hubs:
                if h not in seed_nodes:
                    seed_nodes.append(h)
            if len(seed_nodes) < 2:
                # Extreme fallback: pick any 2 nodes
                all_nodes = list(self.kg.graph.nodes())
                for n in all_nodes[:5]:
                    if n not in seed_nodes:
                        seed_nodes.append(n)
                    if len(seed_nodes) >= 3:
                        break

        # Step 2: Multi-source BFS from seeds
        visited_nodes, visited_edges, layer_map = self._multi_source_bfs(seed_nodes)

        # Step 3: Rank & truncate to max_subgraph_nodes
        ranked = self._rank_nodes(visited_nodes, seed_nodes, layer_map)
        kept = ranked[:self.max_subgraph_nodes]
        kept_set = set(kept)

        # Step 4: Filter edges among kept nodes
        kept_edges = [
            (h, r, t) for (h, r, t) in visited_edges
            if h in kept_set and t in kept_set
        ]

        preferred_start, preferred_target = self._choose_preferred_pair(kept, kept_edges, seed_nodes)

        return SubGraph(
            nodes=kept,
            edges=kept_edges,
            intent_vec=intent_vec.astype(np.float32),
            metadata={
                "seed_entities": core_entities,
                "preferred_start": preferred_start,
                "preferred_target": preferred_target,
                "num_seeds": len(seed_nodes),
                "num_injected": injected,
                "total_visited": len(visited_nodes),
                "max_hops": self.max_hops,
                "similarity_threshold": self.similarity_threshold,
                "match_trace": match_trace,
            },
            node_embeddings=self._build_node_embeddings(kept),
        )

    def prune(self, core_entities: List[str], intent_vec: np.ndarray) -> SubGraph:
        """Backward-compatible alias for older slow-loop callers."""
        return self.update_sub_graph(core_entities, intent_vec)

    # ── [TransE Fix]: Semantic embedding-based match + OOD injection ──────────

    def _get_transe_entity_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Cache original KG entity embeddings for TransE cosine retrieval."""
        if self._transe_eid_list is None or self._transe_ent_matrix is None:
            eid_list = [
                eid for eid, idx in self.kg.entity2id.items()
                if isinstance(idx, int) and idx >= 0
            ]
            if not eid_list:
                raise ValueError("No non-dynamic KG entities available for TransE matching")
            ent_matrix = np.stack([self.embedder.embed_entity(eid) for eid in eid_list]).astype(np.float32)
            norms = np.linalg.norm(ent_matrix, axis=1, keepdims=True)
            self._transe_eid_list = eid_list
            self._transe_ent_matrix = ent_matrix / (norms + 1e-8)
        return self._transe_eid_list, self._transe_ent_matrix

    def _inject_seed(
        self,
        ent_clean: str,
        query_emb: np.ndarray,
        seen: Set[str],
        seed_nodes: List[str],
        semantic_index: Optional[Tuple[List[str], np.ndarray]] = None,
    ) -> bool:
        """Inject an OOD node and connect it to semantic neighbors or hubs."""
        if not self.enable_injection:
            return False

        new_id = self.kg.inject_node(ent_clean)
        if semantic_index is not None:
            eid_list, ent_matrix = semantic_index
            neighbors = self.kg.get_semantic_neighbors(
                query_emb, eid_list, ent_matrix, k=self.hub_k, exclude={new_id}
            )
        else:
            neighbors = self.kg.get_top_hub_nodes(k=self.hub_k)

        for nb in neighbors:
            if nb != new_id:
                self.kg.inject_edge(nb, new_id, relation="dynamic_related_to")
                self.kg.inject_edge(new_id, nb, relation="dynamic_related_to")

        if new_id not in seen:
            seed_nodes.append(new_id)
            seen.add(new_id)
        return True

    def _build_node_embeddings(self, nodes: List[str]) -> Dict[str, np.ndarray]:
        """Attach semantic node features to the pruned sub-graph."""
        node_embeddings: Dict[str, np.ndarray] = {}
        for node in nodes:
            if self._use_transe:
                try:
                    node_embeddings[node] = self.embedder.embed_entity(node).astype(np.float32)
                except Exception:
                    node_embeddings[node] = self._hash_embedding(node).astype(np.float32)
            else:
                display_name = get_entity_display_name(node)
                node_embeddings[node] = self.embedder.embed(display_name).astype(np.float32)
        return node_embeddings

    def _hash_embedding(self, text: str) -> np.ndarray:
        """Deterministic fallback for injected nodes when an embedder lacks OOD support."""
        dim = int(getattr(self.embedder, "dim", EMBEDDING_DIM))
        seed = int(hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()[:16], 16) % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _match_and_inject(self, core_entities: List[str]) -> Tuple[List[str], int]:
        """
        For each intent entity:
          - If TransE available: NLTK-resolve keyword → semantic embedding →
            matrix cosine similarity vs ALL KG entity TransE embeddings.
            OOD keywords get near-zero similarity → correctly injected.
          - If TransE unavailable: fallback char n-gram EntityEmbedder.
          - If best match >= threshold: use that KG node as seed.
          - Else: inject new node → connect to hubs → use as seed.

        Returns:
            (seed_node_ids, num_injected)
        """
        seed_nodes: List[str] = []
        injected = 0
        seen: Set[str] = set()
        self._match_trace = []

        if self._use_transe:
            semantic_index = self._get_transe_entity_matrix()
            eid_list, ent_matrix = semantic_index

            for ent_keyword in core_entities:
                ent_clean = ent_keyword.lower().strip().replace(' ', '_')
                if not ent_clean:
                    continue
                kw_emb, is_ood = self.embedder.embed_keyword(ent_clean)  # [D]

                if is_ood:
                    was_injected = self._inject_seed(ent_clean, kw_emb, seen, seed_nodes)
                    if was_injected:
                        injected += 1
                    self._match_trace.append({
                        "keyword": ent_clean,
                        "is_ood": True,
                        "action": "inject" if was_injected else "skip_injection",
                        "best_match": None,
                        "best_sim": None,
                    })
                    continue

                # Matrix cosine similarity against all KG entities
                sims = np.dot(ent_matrix, kw_emb)  # [E]
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_match = eid_list[best_idx]

                if best_sim >= self.similarity_threshold:
                    if best_match not in seen:
                        seed_nodes.append(best_match)
                        seen.add(best_match)
                    self._match_trace.append({
                        "keyword": ent_clean,
                        "is_ood": False,
                        "action": "match",
                        "best_match": best_match,
                        "best_sim": best_sim,
                    })
                elif self.enable_injection:
                    if self._inject_seed(ent_clean, kw_emb, seen, seed_nodes, semantic_index):
                        injected += 1
                    self._match_trace.append({
                        "keyword": ent_clean,
                        "is_ood": False,
                        "action": "inject",
                        "best_match": best_match,
                        "best_sim": best_sim,
                    })
                else:
                    self._match_trace.append({
                        "keyword": ent_clean,
                        "is_ood": False,
                        "action": "skip_injection",
                        "best_match": best_match,
                        "best_sim": best_sim,
                    })
        else:
            # Legacy char n-gram path (Phase 3 fallback)
            kg_names: Dict[str, str] = {}
            kg_embs: Dict[str, np.ndarray] = {}

            for ent_keyword in core_entities:
                ent_clean = ent_keyword.lower().strip().replace(' ', '_')
                if not ent_clean:
                    continue
                ent_emb = self.embedder.embed(ent_clean)

                best_sim = -1.0
                best_match: Optional[str] = None
                for eid in self.kg.entity2id:
                    if eid not in kg_names:
                        kg_names[eid] = get_entity_display_name(eid)
                    if eid not in kg_embs:
                        kg_embs[eid] = self.embedder.embed(kg_names[eid])
                    sim = float(np.dot(ent_emb, kg_embs[eid]))
                    if sim > best_sim:
                        best_sim = sim
                        best_match = eid

                if best_match is not None and best_sim >= self.similarity_threshold:
                    if best_match not in seen:
                        seed_nodes.append(best_match)
                        seen.add(best_match)
                    self._match_trace.append({
                        "keyword": ent_clean,
                        "is_ood": False,
                        "action": "match",
                        "best_match": best_match,
                        "best_sim": best_sim,
                    })
                elif self.enable_injection:
                    if self._inject_seed(ent_clean, ent_emb, seen, seed_nodes):
                        injected += 1
                    self._match_trace.append({
                        "keyword": ent_clean,
                        "is_ood": True,
                        "action": "inject",
                        "best_match": best_match,
                        "best_sim": best_sim,
                    })
                else:
                    self._match_trace.append({
                        "keyword": ent_clean,
                        "is_ood": True,
                        "action": "skip_injection",
                        "best_match": best_match,
                        "best_sim": best_sim,
                    })

        return seed_nodes, injected

    # ── Multi-source BFS ──────────────────────────────────────────────────

    def _multi_source_bfs(self, seed_nodes: List[str]
                          ) -> Tuple[List[str], List[Tuple[str, str, str]], Dict[str, int]]:
        queue: deque = deque()
        visited: Set[str] = set()
        layer_map: Dict[str, int] = {}
        edge_order: List[Tuple[str, str, str]] = []

        for seed in seed_nodes:
            if seed in self.kg.graph:
                queue.append(seed)
                visited.add(seed)
                layer_map[seed] = 0

        node_order = list(seed_nodes)

        while queue:
            current = queue.popleft()
            cur_layer = layer_map[current]
            if cur_layer >= self.max_hops:
                continue
            for _, neighbour in self.kg.graph.out_edges(current):
                edge_data = self.kg.graph.get_edge_data(current, neighbour)
                r_name = edge_data.get("relation", "unknown") if edge_data else "unknown"
                edge_order.append((current, r_name, neighbour))   # (head, relation, tail)
                if neighbour not in visited:
                    visited.add(neighbour)
                    layer_map[neighbour] = cur_layer + 1
                    node_order.append(neighbour)
                    queue.append(neighbour)

        return node_order, edge_order, layer_map

    # ── Node ranking ──────────────────────────────────────────────────────

    @staticmethod
    def _rank_nodes(nodes: List[str], seeds: List[str],
                    layer_map: Dict[str, int]) -> List[str]:
        seed_set = set(seeds)
        return sorted(nodes, key=lambda n: (
            0 if n in seed_set else 1,
            layer_map.get(n, 999),
        ))

    @staticmethod
    def _choose_preferred_pair(
        nodes: List[str],
        edges: List[Tuple[str, str, str]],
        seeds: List[str],
    ) -> Tuple[str, str]:
        if not nodes:
            return "", ""
        start = next((seed for seed in seeds if seed in nodes), nodes[0])
        adj: Dict[str, List[str]] = {node: [] for node in nodes}
        for h, _, t in edges:
            if h in adj and t in adj and t not in adj[h]:
                adj[h].append(t)
        queue: deque = deque([(start, 0)])
        seen = {start}
        farthest = start
        farthest_dist = 0
        while queue:
            node, dist = queue.popleft()
            if dist > farthest_dist:
                farthest = node
                farthest_dist = dist
            for nb in adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, dist + 1))
        if farthest != start:
            return start, farthest
        return start, nodes[-1]
