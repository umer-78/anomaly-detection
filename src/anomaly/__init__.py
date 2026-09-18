"""Anomaly detection for metrics: statistical detectors, an isolation forest, and honest scoring."""

from .data import Metric, read_csv
from .detectors import (
    EWMA,
    IQR,
    Detector,
    IsolationForest,
    RandomDetector,
    RobustZScore,
    SeasonalResidual,
    ZScore,
    default_detectors,
)
from .evaluate import Result, evaluate, header, leaderboard
from .metrics import Counts, best_f1, point_adjusted, pr_auc, segments, strict, sweep

__all__ = [
    "Counts", "Detector", "EWMA", "IQR", "IsolationForest", "Metric", "RandomDetector", "Result",
    "RobustZScore", "SeasonalResidual", "ZScore", "best_f1", "default_detectors",
    "evaluate", "header", "leaderboard", "point_adjusted", "pr_auc", "read_csv",
    "segments", "strict", "sweep",
]
__version__ = "1.0.0"
