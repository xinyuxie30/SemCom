"""
GNN sub-graph encoder — converts the (dynamic) pruned sub-graph into a
fixed-size embedding vector consumed by the policy network.

This is the key architectural bridge between slow and fast loops:
  SubGraph (JSON)  →  GNNEncoder  →  sub_emb [SUB_EMB_DIM]  →  concatenated into state.

[Phase 1 Fix]: Explicitly documents the input/output dimension contract.
  Input:  node_features [N, IN_DIM] + edge_index [2, E]
  Output: sub_emb [SUB_EMB_DIM] (128)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ── [Phase 1 Fix]: Dimension constants — single source of truth ─────────────
IN_DIM = 128          # input node feature dim (from TransE or random init)
HIDDEN_DIM = 256      # GCN hidden dim
SUB_EMB_DIM = 128     # output sub-graph embedding dim


class SubGraphEncoder(nn.Module):
    """
    Encodes a dynamic sub-graph into a single vector.

    Input dict (from SubGraph.to_tensors):
      - 'node_embeddings': [N, IN_DIM]  — node features (TransE or random)
      - 'edge_index':       [2, E]      — COO format edge indices
      - 'num_nodes':        int

    Output: [SUB_EMB_DIM]  — fixed-size sub-graph embedding
    """

    def __init__(self, in_dim: int = IN_DIM, hidden_dim: int = HIDDEN_DIM,
                 out_dim: int = SUB_EMB_DIM, num_layers: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Dict[str, object]) -> torch.Tensor:
        """
        Args:
            data: dict with keys:
                'node_embeddings': [N, IN_DIM] or None (random init if absent)
                'edge_index':      [2, E]
                'num_nodes':       int
        Returns:
            sub_graph_embedding: [SUB_EMB_DIM]  (128,)
        """
        edge_index: torch.Tensor = data["edge_index"]       # [2, E]
        num_nodes: int = data["num_nodes"]

        # [Phase 1 Fix]: node features — use provided embeddings or random init
        if "node_embeddings" in data and data["node_embeddings"] is not None:
            x: torch.Tensor = data["node_embeddings"].to(edge_index.device)  # [N, IN_DIM]
        else:
            x = torch.randn(num_nodes, self.in_dim, device=edge_index.device)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # Global mean pool → single vector [1, HIDDEN_DIM]
        g = global_mean_pool(x, batch=torch.zeros(num_nodes, dtype=torch.long, device=x.device))
        # Project → [1, SUB_EMB_DIM], then squeeze to [SUB_EMB_DIM] for single-graph input
        out = self.proj(g)                                     # [1, SUB_EMB_DIM]
        return out.squeeze(0)                                  # [SUB_EMB_DIM]


# Alias for backward-compat
GNNEncoder = SubGraphEncoder
