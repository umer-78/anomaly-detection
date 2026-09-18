import pytest

from anomaly import Counts, best_f1, point_adjusted, pr_auc, segments, strict, sweep


def test_counts_derive_precision_recall_and_f1():
    counts = Counts(true_positives=3, false_positives=1, false_negatives=1)

    assert counts.precision == 0.75
    assert counts.recall == 0.75
    assert counts.f1 == 0.75


def test_a_detector_that_flags_nothing_scores_zero_rather_than_dividing_by_zero():
    counts = Counts(0, 0, 5)
    assert counts.precision == 0.0
    assert counts.f1 == 0.0


def test_segments_finds_runs_including_one_at_the_very_end():
    assert segments([0, 1, 1, 0, 0, 1]) == [(1, 3), (5, 6)]
    assert segments([1, 1, 1]) == [(0, 3)]
    assert segments([0, 0]) == []


def test_strict_counting_judges_every_timestamp_on_its_own():
    labels = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    flags = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]

    counts = strict(labels, flags)

    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (1, 1, 2)
    assert counts.f1 == pytest.approx(0.4)


def test_point_adjustment_credits_a_whole_segment_for_one_hit():
    """The same single hit, counted the way most published work counts it."""
    labels = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    flags = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]

    adjusted = point_adjusted(labels, flags)

    assert adjusted.true_positives == 3, "one flag inside the run marked all three"
    assert adjusted.false_negatives == 0
    assert adjusted.f1 == pytest.approx(0.857, abs=0.001)
    assert adjusted.f1 > strict(labels, flags).f1 * 2


def test_point_adjustment_does_not_invent_hits_in_untouched_segments():
    labels = [1, 1, 0, 0, 1, 1]
    flags = [1, 0, 0, 0, 0, 0]

    adjusted = point_adjusted(labels, flags)

    assert adjusted.true_positives == 2, "only the segment that was actually hit"
    assert adjusted.false_negatives == 2


def test_the_sweep_runs_from_flagging_everything_to_flagging_nothing():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    results = sweep(labels, scores, steps=10)

    assert results[0][1].recall == 1.0, "the lowest threshold flags everything"
    assert results[-1][1].false_positives == 0


def test_best_f1_finds_a_threshold_that_separates_them():
    labels = [0, 0, 0, 1, 1]
    scores = [0.1, 0.1, 0.2, 0.9, 0.95]

    threshold, counts = best_f1(labels, scores)

    assert counts.f1 == 1.0
    assert 0.2 < threshold <= 0.9


def test_pr_auc_is_one_for_a_perfect_ordering_and_low_for_noise():
    labels = [0, 0, 0, 0, 1, 1]
    perfect = [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]
    inverted = [1.0, 0.9, 0.4, 0.3, 0.2, 0.1]

    assert pr_auc(labels, perfect) > 0.99
    assert pr_auc(labels, inverted) < 0.5


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="labels against"):
        strict([1, 0], [1])
