"""
Intent Parser - converts deterministic extractor response dict into:
  1. A 64-dim intent embedding tensor
  2. A list of core entity strings

[Phase 2]: Intent embedding is derived from the primary_intent label using
pre-defined intent category embeddings. For novel/unseen intents, a
deterministic hash-based embedding is used as fallback.

Output shape contract (MUST match Phase 1):
  intent_vec: np.ndarray [INTENT_DIM] = [64]
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np

from data.kg_interface import INTENT_DIM  # [Phase 2]: reuse Phase 1 dimension contract

# ── Pre-defined intent category embeddings ──────────────────────────────────
# Each is a [64]-dim normalized vector. Generated once with a fixed seed to
# ensure reproducibility. In production these could be learned end-to-end.

_INTENT_CATEGORIES: Dict[str, np.ndarray] = {}

_RNG = np.random.default_rng(42)


def _init_intent_embeddings() -> None:
    """One-time initialization of intent category embeddings."""
    if _INTENT_CATEGORIES:
        return
    categories = [
        "emergency_avoidance",
        "video_streaming",
        "surveillance",
        "sensor_query",
        "navigation_update",
        "v2x_platooning",
        "telemetry_report",
        "ar_collaboration",
        "general_communication",
    ]
    for cat in categories:
        vec = _RNG.normal(0, 1, (INTENT_DIM,)).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8          # L2-normalize
        _INTENT_CATEGORIES[cat] = vec


_init_intent_embeddings()


# ── Public API ──────────────────────────────────────────────────────────────

def parse_intent(response: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Parse an intent-extractor response into intent vector + entity list.

    Args:
        response: dict from IntentExtractor.query(), must contain:
            - "primary_intent": str
            - "core_entities":  list[str]

    Returns:
        intent_vec:   np.ndarray [INTENT_DIM] = [64], L2-normalized
        core_entities: list[str] — entity keywords for KG pruning
    """
    primary_intent = response.get("primary_intent", "general_communication")
    raw_entities = response.get("core_entities", [])

    # 1. Intent embedding
    intent_vec = _lookup_intent_embedding(primary_intent)

    # 2. Entity list — deduplicate, lowercase, replace spaces with underscores
    core_entities: List[str] = []
    for ent in raw_entities:
        cleaned = str(ent).lower().strip().replace(" ", "_")
        if cleaned and cleaned not in core_entities:
            core_entities.append(cleaned)

    return intent_vec, core_entities


def _lookup_intent_embedding(intent_label: str) -> np.ndarray:
    """
    Map intent label → 64-dim vector.
    1. Exact match in pre-defined categories
    2. Fuzzy match (substring)
    3. Deterministic hash-based fallback
    """
    label_lower = intent_label.lower().strip()

    # Exact match
    if label_lower in _INTENT_CATEGORIES:
        return _INTENT_CATEGORIES[label_lower].copy()

    # Fuzzy: check if any known category is a substring
    for cat, vec in _INTENT_CATEGORIES.items():
        if cat in label_lower or label_lower in cat:
            return vec.copy()

    # Fallback: deterministic hash-based embedding
    h = hashlib.sha256(label_lower.encode()).digest()
    seed = int.from_bytes(h[:8], 'big') % (2**31)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0, 1, (INTENT_DIM,)).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec
