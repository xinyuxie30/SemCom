"""Minimal deterministic keyword-based entity extractor.

The project intentionally does not call LLM/LAM APIs. This module uses a
pre-defined keyword-to-entity dictionary to produce entity lists for dynamic
KG pruning in a fully reproducible local pipeline.
"""
from __future__ import annotations

from typing import Dict, List


# ── Keyword → entity mapping ────────────────────────────────────────────────

_KEYWORD_TO_ENTITIES: Dict[str, List[str]] = {
    # Emergency / driving
    "collision":   ["vehicle", "collision", "brake", "emergency"],
    "crash":       ["vehicle", "collision", "emergency"],
    "brake":       ["brake", "vehicle", "emergency"],
    "pedestrian":  ["pedestrian", "street", "crossing", "safety"],
    "accident":    ["accident", "vehicle", "emergency"],
    "emergency":   ["emergency", "safety", "response"],
    "driving":     ["vehicle", "driving", "road"],
    "highway":     ["highway", "vehicle", "road"],
    "street":      ["street", "road", "crossing"],

    # Surveillance / drone
    "drone":       ["drone", "surveillance", "camera", "video"],
    "surveillance":["surveillance", "camera", "monitoring", "video"],
    "camera":      ["camera", "video", "imaging"],
    "monitor":     ["monitoring", "camera", "surveillance"],
    "video":       ["video", "streaming", "camera"],
    "streaming":   ["video", "streaming", "transmission"],

    # Sensors
    "sensor":      ["sensor", "detection", "monitoring"],
    "lidar":       ["lidar", "sensor", "depth"],
    "radar":       ["radar", "detection", "sensor"],
    "depth":       ["depth", "sensor", "measurement"],
    "sonar":       ["sonar", "underwater", "detection"],

    # Medical
    "surgery":     ["surgery", "medical", "robot"],
    "telemedicine":["telemedicine", "remote", "medical"],
    "haptic":      ["haptic", "feedback", "robot"],

    # Industrial
    "robot":       ["robot", "autonomous", "navigation"],
    "platoon":     ["platoon", "vehicle", "V2V"],
    "navigation":  ["navigation", "path", "autonomous"],
    "pipeline":    ["pipeline", "inspection", "industrial"],

    # Power / grid
    "grid":        ["grid", "power", "substation"],
    "substation":  ["substation", "power", "grid"],
    "transformer": ["transformer", "power", "electrical"],
}


class IntentExtractor:
    """
    Deterministic keyword-to-entity extractor.
    """

    def query(self, text: str) -> dict:
        """Extract entities from NL text using keyword matching."""
        text_lower = text.lower()
        entities: List[str] = []
        seen: set = set()

        # Match keywords in text
        for keyword, ents in _KEYWORD_TO_ENTITIES.items():
            if keyword in text_lower:
                for e in ents:
                    if e not in seen:
                        entities.append(e)
                        seen.add(e)

        # Fallback: extract individual words that match known keywords
        words = text_lower.replace('.', '').replace(',', '').split()
        for w in words:
            if w in _KEYWORD_TO_ENTITIES:
                for e in _KEYWORD_TO_ENTITIES[w]:
                    if e not in seen:
                        entities.append(e)
                        seen.add(e)

        if not entities:
            entities = ["general", "communication"]

        return {
            "primary_intent": entities[0] if entities else "general",
            "core_entities": entities,
            "provider": "keyword_extractor",
        }
