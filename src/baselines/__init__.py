"""Baseline methods for water leak localization."""

from .residual_baseline import ResidualRankingBaseline
from .graph_distance_baseline import GraphDistanceBaseline
from .mlp_baseline import MLPBaseline

__all__ = [
    "ResidualRankingBaseline",
    "GraphDistanceBaseline",
    "MLPBaseline",
]
