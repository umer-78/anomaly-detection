"""Detectors. Each one scores every point; higher means more anomalous.

Scoring and thresholding are kept apart on purpose. A detector that returns a
score can be tuned for the alert budget you actually have; one that returns
booleans has already made that decision for you, usually with a hard-coded 3.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

Series = Sequence[float]


class Detector:
    name = "detector"

    def fit(self, series: Series) -> Detector:
        """Learn from data believed to be mostly normal."""
        raise NotImplementedError

    def score(self, series: Series) -> list[float]:
        raise NotImplementedError

    def fit_score(self, series: Series) -> list[float]:
        return self.fit(series).score(series)


def _mean(values: Series) -> float:
    return sum(values) / len(values)


def _std(values: Series) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _tiny(scale: float, magnitude: float) -> bool:
    """Is this scale estimate too small to divide by?

    Dividing by a spread that is floating-point dust turns rounding error into
    scores in the thousands.
    """
    return scale <= 1e-9 * max(magnitude, 1.0)


def _scale(deviations: Series) -> float:
    """A usable spread for a set of absolute deviations, or 0 if there is none.

    The median absolute deviation first, because it is what makes these detectors
    robust. But the MAD is exactly zero whenever more than half the points are
    identical — a clean signal with one spike in it, which is precisely the case
    that must still be caught — so it falls back to the mean absolute deviation
    before giving up. Only when both are dust does the series genuinely have no
    variation to judge against, and the honest answer is then zero.
    """
    magnitude = sum(deviations) / len(deviations) if deviations else 0.0

    mad = _median(deviations) if deviations else 0.0
    if not _tiny(mad, magnitude):
        return mad
    if not _tiny(magnitude, magnitude):
        return magnitude
    return 0.0


def _median(values: Series) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


@dataclass
class RandomDetector(Detector):
    """Uniform random scores. Carries no information about the series at all.

    Here to make one number impossible to ignore: on point-adjusted F1 — the
    convention in a large share of published time-series anomaly detection — this
    scores respectably, because firing at random eventually lands one point
    inside each long anomalous segment and point adjustment then credits the
    whole segment.
    """

    seed: int = 3
    name: str = "random"

    def fit(self, series: Series) -> RandomDetector:
        return self

    def score(self, series: Series) -> list[float]:
        rng = random.Random(self.seed)
        return [rng.random() for _ in series]


@dataclass
class ZScore(Detector):
    """Distance from the mean in standard deviations.

    Included to be beaten. The mean and the standard deviation are both dragged
    by the very points being looked for: one large spike inflates the standard
    deviation enough that the spike itself stops looking unusual. That is called
    masking, and it is why this is the wrong default despite being everyone's
    first idea.
    """

    name: str = "z-score"
    _mu: float = 0.0
    _sigma: float = 0.0

    def fit(self, series: Series) -> ZScore:
        self._mu, self._sigma = _mean(series), _std(series)
        return self

    def score(self, series: Series) -> list[float]:
        if not self._sigma:
            return [0.0] * len(series)
        return [abs(v - self._mu) / self._sigma for v in series]


@dataclass
class RobustZScore(Detector):
    """Distance from the median, scaled by the median absolute deviation.

    The median and the MAD both have a breakdown point of 50%: it takes half the
    data being contaminated to move them. The 0.6745 factor makes the MAD
    comparable to a standard deviation for normally distributed data, so the
    familiar "over 3" intuition still applies.
    """

    name: str = "robust z-score"
    _median: float = 0.0
    _mad: float = 0.0

    def fit(self, series: Series) -> RobustZScore:
        self._median = _median(series)
        self._mad = _scale([abs(v - self._median) for v in series])
        return self

    def score(self, series: Series) -> list[float]:
        if not self._mad:
            return [0.0] * len(series)
        return [abs(v - self._median) * 0.6745 / self._mad for v in series]


@dataclass
class IQR(Detector):
    """How far outside the quartiles a point sits, in interquartile ranges."""

    name: str = "iqr"
    _q1: float = 0.0
    _q3: float = 0.0

    def fit(self, series: Series) -> IQR:
        ordered = sorted(series)
        self._q1 = ordered[len(ordered) // 4]
        self._q3 = ordered[(3 * len(ordered)) // 4]
        return self

    def score(self, series: Series) -> list[float]:
        spread = self._q3 - self._q1
        if not spread:
            return [0.0] * len(series)
        return [max(self._q1 - v, v - self._q3, 0.0) / spread for v in series]


@dataclass
class EWMA(Detector):
    """A control chart: how far each point is from a moving expectation.

    Unlike the three above, this adapts. A series that drifts upward over a month
    stops being flagged, because the expectation drifts with it — which is what
    you want for a metric with a trend, and not what you want if the drift is the
    problem.
    """

    alpha: float = 0.1
    name: str = "ewma"
    _centre: float = 0.0
    _sigma: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

    def fit(self, series: Series) -> EWMA:
        # The residuals are centred before their spread is taken. On a steadily
        # rising series the moving average sits a constant distance behind, and
        # that constant lag is a property of the smoothing, not an anomaly — score
        # the raw residual and every point of a clean trend looks alarming.
        residuals = self._residuals(series)
        self._centre = _median(residuals)
        centred = [r - self._centre for r in residuals]
        self._sigma = _scale([abs(r) for r in centred])
        return self

    def _residuals(self, series: Series) -> list[float]:
        level = series[0]
        out = []
        for value in series:
            out.append(value - level)
            level = self.alpha * value + (1 - self.alpha) * level
        return out

    def score(self, series: Series) -> list[float]:
        if not self._sigma:
            return [0.0] * len(series)
        return [abs(r - self._centre) / self._sigma for r in self._residuals(series)]


@dataclass
class SeasonalResidual(Detector):
    """Subtract the daily shape first, then look at what is left.

    On a metric with a strong daily rhythm, every quiet night looks like an
    anomaly to a detector that does not know about nights. This removes the
    per-position median before scoring, so a value is judged against the same
    hour on other days rather than against the whole week.
    """

    period: int = 24
    name: str = "seasonal residual"
    _profile: list[float] = field(default_factory=list)
    _mad: float = 0.0

    def __post_init__(self) -> None:
        if self.period < 2:
            raise ValueError("period must be at least 2")

    def fit(self, series: Series) -> SeasonalResidual:
        self._profile = [
            _median([series[i] for i in range(position, len(series), self.period)])
            for position in range(self.period)
        ]
        residuals = self._residuals(series)
        centre = _median(residuals)
        self._mad = _scale([abs(r - centre) for r in residuals])
        return self

    def _residuals(self, series: Series) -> list[float]:
        return [v - self._profile[i % self.period] for i, v in enumerate(series)]

    def score(self, series: Series) -> list[float]:
        if not self._mad:
            return [0.0] * len(series)
        residuals = self._residuals(series)
        centre = _median(residuals)
        return [abs(r - centre) * 0.6745 / self._mad for r in residuals]


@dataclass
class IsolationForest(Detector):
    """Isolation forest over a sliding window of features.

    The idea is that an anomaly is easy to cut off: pick a feature, pick a split
    at random, and an outlier ends up alone after very few cuts. The score is the
    average path length to isolation, normalised so that 0.5 means "as hard to
    isolate as an average point" and values near 1 mean "cut off almost at once".

    A single reading carries no context, so each point becomes a small feature
    vector: its value, its distance from the local median, and how much it moved.
    """

    trees: int = 100
    sample_size: int = 256
    window: int = 12
    seed: int = 5
    name: str = "isolation forest"
    _forest: list[tuple] = field(default_factory=list)
    _height: float = 0.0

    def _features(self, series: Series) -> list[list[float]]:
        rows = []
        for i, value in enumerate(series):
            start = max(0, i - self.window)
            local = series[start:i + 1]
            centre = _median(local)
            previous = series[i - 1] if i else value
            rows.append([value, value - centre, value - previous])
        return rows

    @staticmethod
    def _expected_path(n: int) -> float:
        """Average path length of an unsuccessful search in a binary tree."""
        if n <= 1:
            return 0.0
        return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def _grow(self, rows: list[list[float]], depth: int, limit: int, rng: random.Random):
        if depth >= limit or len(rows) <= 1:
            return ("leaf", len(rows))

        columns = len(rows[0])
        for _ in range(columns):                      # a few tries to find a usable split
            column = rng.randrange(columns)
            values = [row[column] for row in rows]
            low, high = min(values), max(values)
            if low == high:
                continue
            split = rng.uniform(low, high)
            left = [row for row in rows if row[column] < split]
            right = [row for row in rows if row[column] >= split]
            if left and right:
                return ("node", column, split,
                        self._grow(left, depth + 1, limit, rng),
                        self._grow(right, depth + 1, limit, rng))
        return ("leaf", len(rows))

    def _path(self, tree, row: list[float], depth: int = 0) -> float:
        if tree[0] == "leaf":
            return depth + self._expected_path(tree[1])
        _, column, split, left, right = tree
        return self._path(left if row[column] < split else right, row, depth + 1)

    def fit(self, series: Series) -> IsolationForest:
        rng = random.Random(self.seed)
        rows = self._features(series)
        size = min(self.sample_size, len(rows))
        limit = max(1, math.ceil(math.log2(size))) if size > 1 else 1

        self._forest = [
            self._grow(rng.sample(rows, size), 0, limit, rng) for _ in range(self.trees)
        ]
        self._height = self._expected_path(size) or 1.0
        return self

    def score(self, series: Series) -> list[float]:
        rows = self._features(series)
        out = []
        for row in rows:
            average = sum(self._path(tree, row) for tree in self._forest) / len(self._forest)
            out.append(2 ** (-average / self._height))
        return out


def default_detectors(period: int = 24) -> list[Detector]:
    return [
        RandomDetector(),
        ZScore(),
        RobustZScore(),
        IQR(),
        EWMA(alpha=0.1),
        SeasonalResidual(period=period),
        IsolationForest(trees=100, window=period // 2 or 1),
    ]
