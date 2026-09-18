import math

import pytest

from anomaly import (
    EWMA,
    IQR,
    IsolationForest,
    RandomDetector,
    RobustZScore,
    SeasonalResidual,
    ZScore,
    default_detectors,
    evaluate,
    leaderboard,
)


def flat(n: int = 200, value: float = 100.0) -> list[float]:
    return [value] * n


def noisy(n: int = 400, seed: int = 1) -> list[float]:
    import random
    rng = random.Random(seed)
    return [100 + rng.gauss(0, 5) for _ in range(n)]


def daily(n: int = 480, period: int = 24) -> list[float]:
    return [100 + 40 * math.sin(2 * math.pi * i / period) for i in range(n)]


def test_every_detector_scores_one_number_per_point():
    series = noisy(120)
    for detector in default_detectors(period=24):
        assert len(detector.fit_score(series)) == len(series), detector.name


def test_a_constant_series_produces_no_anomalies_instead_of_dividing_by_zero():
    series = flat()
    for detector in (ZScore(), RobustZScore(), IQR(), SeasonalResidual(period=24)):
        assert set(detector.fit_score(series)) == {0.0}, detector.name


def test_an_obvious_spike_scores_highest():
    series = noisy(300)
    series[150] = 400.0

    for detector in (ZScore(), RobustZScore(), IQR(), EWMA(), IsolationForest(trees=40)):
        scores = detector.fit_score(series)
        assert scores.index(max(scores)) == 150, detector.name


def test_the_mean_is_dragged_by_the_very_points_it_is_looking_for():
    """Masking, the reason the plain z-score is the wrong default.

    One enormous outlier inflates the standard deviation enough that a second,
    smaller outlier stops looking unusual at all. The median and the MAD are not
    moved by it.
    """
    series = noisy(300)
    series[100] = 250.0        # a moderate outlier
    series[200] = 5000.0       # one huge one

    plain = ZScore().fit_score(series)
    robust = RobustZScore().fit_score(series)

    assert plain[100] < 1.0, "the huge outlier hid the moderate one"
    assert robust[100] > 5.0, "the robust score still sees it"


def test_a_seasonal_detector_ignores_the_daily_rhythm_a_flat_one_flags():
    """Every quiet night looks like an anomaly to a detector that knows nothing
    about nights."""
    series = daily()

    plain = ZScore().fit_score(series)
    seasonal = SeasonalResidual(period=24).fit_score(series)

    assert max(plain) > 1.4, "the trough reads as unusual without the seasonal profile"
    assert max(seasonal) < 0.5, "with it, nothing here is unusual"


def test_the_seasonal_detector_still_catches_a_spike_inside_the_rhythm():
    series = daily()
    series[300] += 120

    scores = SeasonalResidual(period=24).fit_score(series)
    assert scores.index(max(scores)) == 300


def test_ewma_follows_a_drift_and_still_catches_a_step():
    drifting = [100 + 0.5 * i for i in range(300)]
    scores = EWMA(alpha=0.2).fit_score(drifting)
    assert max(scores[50:]) < 3.0, "a steady trend should stop being news"

    stepped = drifting[:]
    stepped[200:] = [v + 80 for v in stepped[200:]]
    stepped_scores = EWMA(alpha=0.2).fit_score(stepped)
    assert stepped_scores[200] > 5.0


def test_the_isolation_forest_is_deterministic_for_a_seed():
    series = noisy(200)
    first = IsolationForest(trees=30, seed=9).fit_score(series)
    second = IsolationForest(trees=30, seed=9).fit_score(series)

    assert first == second


def test_isolation_scores_sit_between_zero_and_one():
    scores = IsolationForest(trees=40).fit_score(noisy(200))
    assert all(0 < s < 1 for s in scores)


def test_parameters_are_validated():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        EWMA(alpha=0)
    with pytest.raises(ValueError, match="period must be at least 2"):
        SeasonalResidual(period=1)


# ------------------------------------------------------------ on the sample data

def test_the_seasonal_detector_wins_on_a_metric_with_a_daily_shape(halves):
    train, test = halves
    results = leaderboard(default_detectors(24), train, test)

    assert results[0].detector == "seasonal residual"
    assert results[0].strict.f1 > 0.5


def test_noise_scores_near_zero_on_strict_f1(halves):
    train, test = halves
    result = evaluate(RandomDetector(), train, test)

    assert result.strict.f1 < 0.05
    assert result.pr_auc < 0.1


def test_point_adjustment_flatters_pure_noise(halves):
    """The number this project exists to make impossible to ignore.

    A detector with no information about the series at all reaches a
    point-adjusted F1 several times its real one, because firing often enough
    eventually lands a flag inside each long anomalous segment — and point
    adjustment then credits the whole segment.
    """
    train, test = halves

    for quantile in (0.98, 0.90):
        result = evaluate(RandomDetector(), train, test, quantile=quantile)
        assert result.adjusted.f1 > result.strict.f1 * 4, (
            f"at q={quantile}: strict {result.strict.f1:.3f}, "
            f"adjusted {result.adjusted.f1:.3f}"
        )


def test_noise_can_out_score_a_real_detector_on_the_adjusted_metric(halves):
    train, test = halves

    noise = evaluate(RandomDetector(), train, test, quantile=0.90)
    ewma = evaluate(EWMA(alpha=0.1), train, test)

    assert noise.strict.f1 < ewma.strict.f1 / 3, "honestly, the real detector is far better"
    assert noise.adjusted.f1 > ewma.adjusted.f1 * 0.6, (
        "and yet on the adjusted metric the noise is in the same league"
    )


def test_the_threshold_comes_from_training_scores_not_from_the_labels(halves):
    """Picking the threshold that maximises F1 on the test set is the most common
    way a published number stops meaning anything. The ceiling is reported, but
    it is not the score."""
    train, test = halves
    result = evaluate(SeasonalResidual(period=24), train, test)

    assert result.ceiling >= result.strict.f1
    assert result.threshold == pytest.approx(
        sorted(SeasonalResidual(period=24).fit_score(train.values))[int(len(train) * 0.98)]
    )


def test_a_tighter_quantile_raises_precision_and_lowers_recall(halves):
    train, test = halves

    loose = evaluate(SeasonalResidual(period=24), train, test, quantile=0.90)
    tight = evaluate(SeasonalResidual(period=24), train, test, quantile=0.99)

    assert tight.alerts < loose.alerts
    assert tight.strict.precision > loose.strict.precision


def test_an_impossible_quantile_is_rejected(halves):
    train, test = halves
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate(ZScore(), train, test, quantile=1.5)
