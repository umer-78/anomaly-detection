"""Scoring detectors on the same data, the same way."""

from __future__ import annotations

from dataclasses import dataclass

from .data import Metric
from .detectors import Detector
from .metrics import Counts, best_f1, point_adjusted, pr_auc, strict


@dataclass(frozen=True)
class Result:
    detector: str
    threshold: float
    strict: Counts
    adjusted: Counts
    pr_auc: float
    ceiling: float          # the best strict F1 any threshold could have reached
    alerts: int

    def row(self) -> str:
        return (f"{self.detector:<20}{self.strict.precision:>10.3f}{self.strict.recall:>9.3f}"
                f"{self.strict.f1:>9.3f}{self.pr_auc:>10.3f}{self.adjusted.f1:>12.3f}"
                f"{self.alerts:>9}")


def header() -> str:
    return (f"{'detector':<20}{'precision':>10}{'recall':>9}{'F1':>9}{'PR-AUC':>10}"
            f"{'adjusted F1':>12}{'alerts':>9}")


def evaluate(detector: Detector, train: Metric, test: Metric, *, quantile: float = 0.98) -> Result:
    """Fit on the first half, choose a threshold there, score on the second.

    The threshold is the given quantile of the *training* scores — chosen from
    data the detector was fitted on, never from the labels it is about to be
    judged against. Picking the threshold that maximises F1 on the test set is
    the most common way an anomaly detector's published number stops meaning
    anything; that figure is still reported, as a ceiling, clearly labelled.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")

    detector.fit(train.values)
    training_scores = sorted(detector.score(train.values))
    index = min(int(len(training_scores) * quantile), len(training_scores) - 1)
    threshold = training_scores[index]

    scores = detector.score(test.values)
    flags = [1 if s > threshold else 0 for s in scores]

    ceiling = best_f1(test.labels, scores)[1].f1
    return Result(
        detector=detector.name,
        threshold=threshold,
        strict=strict(test.labels, flags),
        adjusted=point_adjusted(test.labels, flags),
        pr_auc=pr_auc(test.labels, scores),
        ceiling=ceiling,
        alerts=sum(flags),
    )


def leaderboard(detectors: list[Detector], train: Metric, test: Metric, **kwargs) -> list[Result]:
    return sorted((evaluate(d, train, test, **kwargs) for d in detectors),
                  key=lambda r: -r.pr_auc)
