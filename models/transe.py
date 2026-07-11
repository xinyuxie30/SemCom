"""Canonical TransE module used by downstream KG components.

The experiment driver in ``scripts/train_transe_unified.py`` trains TransE with
L2-normalized entity/relation vectors before scoring.  This module mirrors that
semantics so imported model code does not drift away from checkpoint behavior.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    """Translation-based KG embedding with normalized scoring by default."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        margin: float = 5.0,
        p_norm: int = 1,
        normalize: bool = True,
    ):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.p_norm = p_norm
        self.normalize = normalize

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def _maybe_normalize(self, emb: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return emb
        return F.normalize(emb, p=2, dim=-1)

    def score_triples(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """Return TransE distance scores. Lower means more plausible."""
        h = self._maybe_normalize(self.entity_emb(heads))
        r = self._maybe_normalize(self.relation_emb(relations))
        t = self._maybe_normalize(self.entity_emb(tails))
        return torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for ``score_triples`` to preserve the old API."""
        return self.score_triples(heads, relations, tails)

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_rels: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_rels: torch.Tensor,
        neg_tails: torch.Tensor,
    ) -> torch.Tensor:
        """Margin ranking loss: max(0, margin + score_pos - score_neg)."""
        pos_score = self.score_triples(pos_heads, pos_rels, pos_tails)
        neg_score = self.score_triples(neg_heads, neg_rels, neg_tails)
        return F.relu(self.margin + pos_score - neg_score).mean()

    def get_entity_embeddings(
        self,
        indices: torch.Tensor,
        normalized: bool | None = None,
    ) -> torch.Tensor:
        """Retrieve entity embeddings, normalized by default to match export."""
        emb = self.entity_emb(indices)
        use_normalized = self.normalize if normalized is None else normalized
        return F.normalize(emb, p=2, dim=-1) if use_normalized else emb

    def export_entity_embeddings(self) -> torch.Tensor:
        """Return all entity embeddings in checkpoint/export semantics."""
        return self._maybe_normalize(self.entity_emb.weight.detach())

    def export_relation_embeddings(self) -> torch.Tensor:
        """Return all relation embeddings in checkpoint/export semantics."""
        return self._maybe_normalize(self.relation_emb.weight.detach())
