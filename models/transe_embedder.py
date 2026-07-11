"""
transe_embedder.py - semantic entity embedder bridging intent keywords to WN18
TransE embeddings via NLTK WordNet.

Replaces the Phase 2/3 character n-gram EntityEmbedder with genuine semantic
embeddings trained on the knowledge graph.

Workflow:
  1. English intent keyword (e.g. "pedestrian") -> NLTK WordNet synsets
  2. Synset offset -> check if exists in WN18 entity2id
  3. If yes -> retrieve pre-trained TransE embedding [128]
  4. If no -> OOD flag (triggers dynamic injection in the pruner)
  5. Cosine similarity between keyword embedding and KG entity embedding
     is now semantically meaningful (not random).

Dimension contract: all embeddings are [128] float32, L2-normalized.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── NLTK WordNet bridge ────────────────────────────────────────────────────

_NLTK_OK: Optional[bool] = None
_WN_INDEX_CACHE: Dict[str, Tuple[Dict[str, List[str]], Dict[str, str]]] = {}


def _entity2id_fingerprint(entity2id: Dict[str, int]) -> str:
    """Stable cache key so WordNet indexes cannot leak across datasets."""
    h = hashlib.sha256()
    h.update(str(len(entity2id)).encode("utf-8"))
    for entity, idx in sorted(entity2id.items(), key=lambda item: (int(item[1]), item[0])):
        h.update(b"\0")
        h.update(str(int(idx)).encode("utf-8"))
        h.update(b"\t")
        h.update(entity.encode("utf-8", errors="surrogatepass"))
    return h.hexdigest()
_WN_LEMMA_TO_ENTITIES: Dict[str, List[str]] = {}   # lemma → [entity_id, ...]
_WORD_TO_BEST_ENTITY: Dict[str, str] = {}           # keyword → best entity_id


def _init_nltk() -> bool:
    global _NLTK_OK
    if _NLTK_OK is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from nltk.corpus import wordnet as wn
                _ = wn.synsets('dog')
            _NLTK_OK = True
        except Exception:
            _NLTK_OK = False
    return _NLTK_OK


def build_wn18_nltk_index(entity2id: Dict[str, int]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Build a reverse index: English word/lemma → WN18 entity IDs.

    For each WN18 entity (8-digit offset), attempts NLTK name resolution.
    If resolved, extracts lemma names and indexes them.
    """
    cache_key = _entity2id_fingerprint(entity2id)
    if cache_key in _WN_INDEX_CACHE:
        lemma_to_entities, word_to_best_entity = _WN_INDEX_CACHE[cache_key]
        return {k: v.copy() for k, v in lemma_to_entities.items()}, word_to_best_entity.copy()

    if not _init_nltk():
        return {}, {}

    from nltk.corpus import wordnet as wn

    pos_map = {1: 'n', 2: 'v', 3: 'a', 4: 'r'}
    hit_count = 0
    lemma_to_entities: Dict[str, List[str]] = {}
    word_to_best_entity: Dict[str, str] = {}

    for eid in entity2id:
        if not re.match(r'^\d{8}$', eid):
            continue
        try:
            offset = int(eid[:-1])
            pos_digit = int(eid[-1])
            pos = pos_map.get(pos_digit, 'n')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                synset = wn.synset_from_pos_and_offset(pos, offset)
            for lemma in synset.lemma_names():
                key = lemma.lower().replace(' ', '_')
                lemma_to_entities.setdefault(key, []).append(eid)
                # Store the first match as the primary
                if key not in word_to_best_entity:
                    word_to_best_entity[key] = eid
            hit_count += 1
        except Exception:
            pass

    print(f"  [TransEEmbedder] NLTK index: {hit_count}/{len(entity2id)} entities resolved, "
          f"{len(lemma_to_entities)} unique lemmas indexed")
    _WN_INDEX_CACHE[cache_key] = (lemma_to_entities, word_to_best_entity)
    return {k: v.copy() for k, v in lemma_to_entities.items()}, word_to_best_entity.copy()


def resolve_keyword_to_entity(keyword: str,
                              entity2id: Dict[str, int],
                              word_to_entity: Dict[str, str],
                              lemma_to_entities: Dict[str, List[str]],
                              ) -> Optional[str]:
    """
    Map an English keyword to a WN18 entity ID via NLTK WordNet.

    Returns the best-matching entity ID, or None if no match (OOD).
    """
    if not _init_nltk():
        return None

    from nltk.corpus import wordnet as wn

    key = keyword.lower().strip().replace(' ', '_')

    # 1. Direct lemma index lookup
    if key in word_to_entity:
        return word_to_entity[key]

    # 2. NLTK synset search
    for syn in wn.synsets(key):
        offset = syn.offset()
        pos = syn.pos()
        pos_map_inv = {'n': 1, 'v': 2, 'a': 3, 'r': 4, 's': 3}
        pos_digit = pos_map_inv.get(pos, 1)
        eid = f'{offset:08d}'[:-1] + str(pos_digit)

        if eid in entity2id:
            word_to_entity[key] = eid  # cache
            return eid

    # 3. Check lemma names of the synset
    for syn in wn.synsets(key):
        for lemma in syn.lemma_names():
            lk = lemma.lower().replace(' ', '_')
            if lk in word_to_entity:
                word_to_entity[key] = word_to_entity[lk]  # cache
                return word_to_entity[lk]

    return None


# ═══════════════════════════════════════════════════════════════════════════════

class TransEEmbedder:
    """
    Semantic embedder using pre-trained TransE entity embeddings.

    - For KG entities with trained TransE vectors -> use those.
    - For intent keywords -> resolve to WN18 entity via NLTK -> use TransE vector.
    - For OOD keywords -> return the OOD flag; caller triggers dynamic injection.

    All embeddings are [128] float32, L2-normalized, with true semantic structure.
    """

    def __init__(self, entity_embeddings: np.ndarray,
                 entity2id: Dict[str, int],
                 embedding_dim: Optional[int] = None):
        """
        Args:
            entity_embeddings: [num_entities, dim] pre-trained TransE embeddings
            entity2id: {entity_name: idx} mapping
        """
        if entity_embeddings.ndim != 2:
            raise ValueError(f"entity_embeddings must be [num_entities, dim], got {entity_embeddings.shape}")
        self.dim = int(embedding_dim or entity_embeddings.shape[1])
        if entity_embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embedding_dim={self.dim} does not match checkpoint dim={entity_embeddings.shape[1]}"
            )
        self.entity_emb = entity_embeddings.astype(np.float32)
        self.entity2id = entity2id
        self.id2idx = {eid: int(idx) for eid, idx in entity2id.items()}
        max_idx = max(self.id2idx.values(), default=-1)
        min_idx = min(self.id2idx.values(), default=0)
        if min_idx < 0 or max_idx >= self.entity_emb.shape[0]:
            raise ValueError(
                f"entity2id indices [{min_idx}, {max_idx}] are incompatible with "
                f"embedding rows={self.entity_emb.shape[0]}"
            )

        # L2-normalize all embeddings
        norms = np.linalg.norm(self.entity_emb, axis=1, keepdims=True)
        self.entity_emb = self.entity_emb / (norms + 1e-8)

        # Build NLTK → WN18 bridge index
        self._lemma_to_entities, self._word_to_entity = build_wn18_nltk_index(entity2id)

        # OOD cache: for keywords that don't map to WN18, generate deterministic hash embedding
        self._ood_cache: Dict[str, np.ndarray] = {}
        # KG embedding cache
        self._kg_cache: Dict[str, np.ndarray] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def embed_entity(self, entity_id: str) -> np.ndarray:
        """Return [dim] TransE embedding for a KG entity ID."""
        if entity_id in self._kg_cache:
            return self._kg_cache[entity_id].copy()

        if entity_id in self.id2idx:
            idx = self.id2idx[entity_id]
            vec = self.entity_emb[idx].copy()
            self._kg_cache[entity_id] = vec
            return vec

        # Entity not in KG (dynamically injected) — deterministic hash embedding
        return self._embed_ood(entity_id)

    def embed_keyword(self, keyword: str) -> Tuple[np.ndarray, bool]:
        """
        Embed an intent keyword.

        Returns:
            (embedding [dim], is_ood: bool)
            If is_ood=True, the keyword has no WN18 match and caller should inject.
        """
        entity_id = resolve_keyword_to_entity(
            keyword, self.entity2id, self._word_to_entity, self._lemma_to_entities)

        if entity_id is not None:
            return self.embed_entity(entity_id), False
        else:
            return self._embed_ood(keyword), True

    def cosine_sim_keyword_vs_entity(self, keyword: str, entity_id: str) -> float:
        """Cosine similarity between an intent keyword and a KG entity."""
        kw_emb, is_ood = self.embed_keyword(keyword)
        ent_emb = self.embed_entity(entity_id)
        return float(np.dot(kw_emb, ent_emb))

    # ── OOD fallback ──────────────────────────────────────────────────────

    def _embed_ood(self, text: str) -> np.ndarray:
        """
        Deterministic embedding for OOD terms (not in WN18).
        Uses SHA-256 hash → seed → random Gaussian → L2-normalize.
        Consistent across calls for the same text.
        """
        if text in self._ood_cache:
            return self._ood_cache[text].copy()

        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:8], 'big') % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, (self.dim,)).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        self._ood_cache[text] = vec
        return vec.copy()


# ── Factory ─────────────────────────────────────────────────────────────────

def _checkpoint_tag(dataset: str, model: str) -> str:
    dataset_key = dataset.lower().replace("-", "")
    model_key = model.lower()
    if model_key == "transe":
        model_key = "transe"
    return f"{dataset_key}_{model_key}"


def _resolve_checkpoint_paths(
    checkpoint_dir: str,
    dataset: str,
    model: str,
    tag_suffix: str = "",
) -> Tuple[Path, Path]:
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_absolute():
        ckpt = Path(__file__).resolve().parent.parent / ckpt

    tag = _checkpoint_tag(dataset, model)
    if tag_suffix:
        tag = f"{tag}_{tag_suffix.strip().replace(' ', '_')}"
    candidates = [
        (ckpt / f"transe_{tag}_entity_emb.npy", ckpt / f"transe_{tag}_entity2id.json"),
    ]
    if not tag_suffix and dataset.upper() == "WN18RR":
        candidates.append((ckpt / "transe_wn18_entity_emb.npy", ckpt / "transe_wn18_entity2id.json"))
    if not tag_suffix and dataset.upper() == "FB15K237" and model.lower() == "transe":
        candidates.append((ckpt / "transe_fb15k237_entity_emb.npy", ckpt / "transe_fb15k237_entity2id.json"))

    for emb_path, e2id_path in candidates:
        if emb_path.exists() and e2id_path.exists():
            return emb_path, e2id_path

    if not tag_suffix:
        pattern = f"transe_{_checkpoint_tag(dataset, model)}_*_entity_emb.npy"
        suffixed = sorted(ckpt.glob(pattern))
        valid = [
            (p, p.with_name(p.name.replace("_entity_emb.npy", "_entity2id.json")))
            for p in suffixed
        ]
        valid = [(emb, e2id) for emb, e2id in valid if e2id.exists()]
        if len(valid) == 1:
            return valid[0]
        if len(valid) > 1:
            names = "\n    ".join(p.name for p, _ in valid)
            raise FileNotFoundError(
                "Multiple TransE checkpoint suffixes found; pass tag_suffix explicitly.\n"
                f"Available:\n    {names}"
            )

    tried = "\n    ".join(str(p[0]) for p in candidates)
    raise FileNotFoundError(
        f"TransE embeddings not found. Tried:\n    {tried}\n"
        f"Run: python scripts/train_transe_unified.py --dataset {dataset} --model {model}"
    )


def create_transe_embedder(checkpoint_dir: str = "checkpoints",
                           dataset: str = "WN18RR",
                           model: str = "transE",
                           tag_suffix: str = "") -> TransEEmbedder:
    """
    Load pre-trained TransE embeddings and build a TransEEmbedder.
    """
    emb_path, e2id_path = _resolve_checkpoint_paths(checkpoint_dir, dataset, model, tag_suffix)

    entity_emb = np.load(emb_path)
    with open(e2id_path, "r") as f:
        entity2id = json.load(f)

    print(f"  [TransEEmbedder] Loaded {entity_emb.shape[0]:,} entity embeddings "
          f"dim={entity_emb.shape[1]}")
    return TransEEmbedder(entity_emb, entity2id)
