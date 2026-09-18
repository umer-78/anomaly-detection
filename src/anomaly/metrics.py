"""Scoring an anomaly detector.

Two ways of counting are implemented, and the gap between them is the point.

Strict counting asks, for every timestamp, whether the detector flagged it.
Point-adjusted counting says that if a detector flagged *any* point inside a true
anomalous segment, the whole segment counts as correctly detected. Point
adjustment is the convention in a large share of published time-series anomaly
detection, and it inflates F1 so far that a detector firing at random can score
above 0.9. Both numbers are reported here, always, side by side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Counts:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def _check(labels: Sequence[int], flags: Sequence[int]) -> None:
    if len(labels) != len(flags):
        raise ValueError(f"{len(labels)} labels against {len(flags)} flags")
    if not labels:
        raise ValueError("nothing to score")


def segments(labels: Sequence[int]) -> list[tuple[int, int]]:
    """The [start, end) runs of consecutive anomalous points."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, label in enumerate(labels):
        if label and start is None:
            start = i
        elif not label and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(labels)))
    return runs


def strict(labels: Sequence[int], flags: Sequence[int]) -> Counts:
    """Point by point. Every timestamp is judged on its own."""
    _check(labels, flags)
    tp = sum(1 for a, b in zip(labels, flags, strict=True) if a and b)
    fp = sum(1 for a, b in zip(labels, flags, strict=True) if not a and b)
    fn = sum(1 for a, b in zip(labels, flags, strict=True) if a and not b)
    return Counts(tp, fp, fn)


def point_adjusted(labels: Sequence[int], flags: Sequence[int]) -> Counts:
    """One hit inside a true segment marks the whole segment as detected.

    Reported so the inflation is visible, never on its own. A detector that fires
    once inside a long outage is credited with having found every minute of it.
    """
    _check(labels, flags)
    adjusted = list(flags)
    for start, end in segments(labels):
        if any(flags[start:end]):
            for i in range(start, end):
                adjusted[i] = 1
    return strict(labels, adjusted)


def sweep(labels: Sequence[int], scores: Sequence[float], steps: int = 200) -> list[tuple[float, Counts]]:
    """Counts at every threshold, from flagging nothing to flagging everything."""
    _check(labels, scores)
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(hi, strict(labels, [1] * len(labels)))]
    return [
        (threshold, strict(labels, [1 if s >= threshold else 0 for s in scores]))
        for threshold in (lo + (hi - lo) * i / steps for i in range(steps + 1))
    ]


def best_f1(labels: Sequence[int], scores: Sequence[float], steps: int = 200) -> tuple[float, Counts]:
    """The best strict F1 any threshold could reach, and that threshold.

    This is an upper bound, not a result: it is chosen with the labels in hand.
    Quoting it as a detector's score is threshold selection on the test set, and
    it is reported here only to separate "the score is bad" from "the threshold
    is bad".
    """
    return max(sweep(labels, scores, steps), key=lambda pair: pair[1].f1)


def pr_auc(labels: Sequence[int], scores: Sequence[float], steps: int = 200) -> float:
    """Area under precision-recall, which does not need a threshold at all.

    Precision-recall rather than ROC: anomalies are rare, and ROC's false
    positive rate divides by a huge number of normal points, so a detector
    drowning the user in false alarms still looks excellent.
    """
    # Several thresholds can land on the same recall with different precisions.
    # Keeping the best of them is what the standard curve does: a point that is
    # dominated — same recall, lower precision — is never the operating point
    # anyone would choose, and averaging it in understates the detector.
    best: dict[float, float] = {}
    for _threshold, counts in sweep(labels, scores, steps):
        recall = counts.recall
        best[recall] = max(best.get(recall, 0.0), counts.precision)

    area = 0.0
    previous_recall, previous_precision = 0.0, 1.0
    for recall in sorted(best):
        precision = best[recall]
        area += (recall - previous_recall) * (precision + previous_precision) / 2
        previous_recall, previous_precision = recall, precision
    return area
