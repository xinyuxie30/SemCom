"""
Evaluation metrics for the fast-slow dual-loop system.

Phase 4 will use these for the four key figures:
  1. Convergence speed (pruning benefit)
  2. End-to-end latency decomposition
  3. Semantic fidelity under OOD / adverse channels
  4. Ablation study
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def semantic_fidelity(pred_path: List[str], target_path: List[str]) -> float:
    """
    Overlap-based semantic fidelity.
    Returns Jaccard similarity between predicted and target entity paths.
    """
    if not target_path:
        return 0.0
    inter = len(set(pred_path) & set(target_path))
    union = len(set(pred_path) | set(target_path))
    return inter / union if union > 0 else 0.0


def path_hit_rate(pred_path: List[str], target_path: List[str]) -> float:
    """Fraction of target entities reached."""
    if not target_path:
        return 0.0
    return len(set(pred_path) & set(target_path)) / len(set(target_path))


def convergence_speed(loss_curve: List[float], threshold: float = 0.1) -> int:
    """
    Number of epochs until loss stabilizes below `threshold` * max_loss.
    Lower = faster convergence.
    """
    if not loss_curve:
        return 0
    target = max(loss_curve) * threshold + min(loss_curve) * (1 - threshold)
    for i, v in enumerate(loss_curve):
        if v <= target:
            return i
    return len(loss_curve)


def latency_breakdown(slow_loop_ms: float, gnn_enc_ms: float,
                      policy_inf_ms: float, channel_ms: float) -> Dict[str, float]:
    """End-to-end latency decomposition."""
    total = slow_loop_ms + gnn_enc_ms + policy_inf_ms + channel_ms
    return {
        "slow_loop": slow_loop_ms,
        "gnn_encode": gnn_enc_ms,
        "policy_inference": policy_inf_ms,
        "channel": channel_ms,
        "total": total,
    }


def average_return(rewards: List[float], gamma: float = 0.99) -> float:
    """Discounted cumulative return."""
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G
