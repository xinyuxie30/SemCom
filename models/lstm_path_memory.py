"""
LSTM path memory — encodes the traversal history (sequence of visited nodes)
into a fixed-size vector, used as part of the agent's state representation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PathMemoryEncoder(nn.Module):
    """
    Encodes the sequence of visited entity embeddings into a path summary vector.

    Input:  [B, L, D]  — sequence of node embeddings along the traversal so far.
    Output: [B, H]     — final hidden state of LSTM.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_dim = hidden_dim

    def forward(self, path_embeddings: torch.Tensor,
                hidden: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            path_embeddings: [B, L, D] — padded sequence of node embeddings.
            hidden: optional (h, c) tuple for stateful encoding.
        Returns:
            (path_vec [B, H], (h, c)).
        """
        output, (h, c) = self.lstm(path_embeddings, hidden)
        path_vec = h[-1]                      # last layer's hidden state
        return path_vec, (h, c)
