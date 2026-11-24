"""Ranking metrics and calibrated statistics written from scratch."""

from __future__ import annotations

import math

import numpy as np


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    gains = [rel.get(d, 0) for d in ranked[:k]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    pos = {d for d, g in rel.items() if g > 0}
    return len(pos & set(ranked[:k])) / len(pos) if pos else 0.0


def overlap_at_k(a: list[str], b: list[str], k: int = 10) -> float:
    """Top-k set overlap between two rankings (analogue of CandOverlap in the hook paper)."""
    return len(set(a[:k]) & set(b[:k])) / k


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over queries."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) over the same queries with a percentile CI (paired resampling)."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n_boot, seed)


def relative_retention_ci(
    clean: np.ndarray, pert: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Retention = mean(pert)/mean(clean), bootstrap CI resampling queries jointly (pairs kept together)."""
    rng = np.random.default_rng(seed)
    c, p = np.asarray(clean, float), np.asarray(pert, float)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    r = p[idx].mean(axis=1) / np.maximum(c[idx].mean(axis=1), 1e-12)
    return float(p.mean() / c.mean()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))
