# anomaly

[![CI](https://github.com/umer-78/anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/umer-78/anomaly-detection/actions/workflows/ci.yml)

**Live demo:** https://umer-78.github.io/anomaly-detection/

Anomaly detection for metrics, in Python with no dependencies: statistical
detectors, an isolation forest written from scratch, and — the part that matters —
scoring that does not flatter itself.

```
$ anomaly evaluate
fitted on 1,344 points, scored on 1,344 (63 anomalous), threshold at the 98% quantile of training scores

detector             precision   recall       F1    PR-AUC adjusted F1   alerts
seasonal residual        0.649    0.587    0.617     0.662       0.742       57
z-score                  0.127    0.556    0.206     0.503       0.316      276
iqr                      0.130    0.556    0.211     0.502       0.322      269
robust z-score           0.125    0.556    0.203     0.502       0.311      281
isolation forest         0.340    0.556    0.422     0.418       0.649      103
ewma                     0.302    0.460    0.365     0.324       0.653       96
random                   0.038    0.016    0.022     0.052       0.275       26
```

The bottom row is a detector that returns uniform random numbers. It knows
nothing about the series. Its real F1 is **0.022** — and its point-adjusted F1 is
**0.275**, twelve times higher. Loosen its threshold a little and it reaches
**0.463**, which is better than the isolation forest's honest 0.422.

Point-adjusted F1 is the convention in a large share of published time-series
anomaly detection. This is why the random detector is shipped in the box.

- **27 tests**, Python 3.10–3.12, no runtime dependencies
- 2,688 hours of labelled metric data with four different kinds of fault

## Quick start

```bash
git clone https://github.com/umer-78/anomaly-detection.git
cd anomaly-detection
pip install -e ".[dev]"
pytest -q                     # 27 tests

anomaly describe
anomaly evaluate
anomaly detect --detector seasonal --limit 10
```

## Point adjustment, concretely

Point adjustment says: if a detector flagged *any* point inside a true anomalous
segment, treat the whole segment as correctly detected. The argument for it is
that an operator only needs one alert to start investigating. The problem is what
it does to the arithmetic:

```
labels  0 0 1 1 1 0 0 0 0 0      three anomalous points
flags   0 0 0 1 0 0 1 0 0 0      one hit, one false alarm

strict            precision 0.50   recall 0.33   F1 0.40
point-adjusted    precision 0.75   recall 1.00   F1 0.86
```

One flag became three true positives and erased two misses. The longer the
segment, the bigger the gift — and real outages are long. Fire often enough and
you are guaranteed to land inside each one:

```
 quantile  alerts  strict F1  adjusted F1
     0.98      26      0.022        0.275
     0.95      67      0.031        0.197
     0.90     134      0.081        0.463
     0.85     201      0.098        0.401
     0.80     268      0.091        0.332
```

Both numbers are always printed, side by side, and two tests pin the
relationship so it cannot quietly stop being true.

## Three more decisions

**The threshold comes from the training scores, never from the labels.** It is
the 98th percentile of the scores the detector produced on the half it was fitted
on. Choosing instead the threshold that maximises F1 on the test set is the most
common way a published number stops meaning anything — that figure is computed
too, reported as `best_possible_f1` in the JSON output and clearly labelled a
ceiling rather than a score.

**PR-AUC, not ROC-AUC.** Anomalies are 2.9% of this data. ROC's false positive
rate divides by the 97% that are normal, so a detector burying an operator in
false alarms still looks excellent. Precision-recall does not have that
comfort.

**Fitting and scoring happen on different halves.** Fit on the same points you
score and the detector ends up tuned to the very anomalies it is meant to find.

## The detectors

| Detector | Idea | Where it fails |
| --- | --- | --- |
| `RandomDetector` | uniform noise | everywhere — it is the control |
| `ZScore` | distance from the mean in standard deviations | masking, below |
| `RobustZScore` | distance from the median, scaled by the MAD | still blind to seasonality |
| `IQR` | distance outside the quartiles | same |
| `EWMA` | distance from a moving expectation | follows a drift, so it will not report one |
| `SeasonalResidual` | subtract the daily shape, then judge | needs to know the period |
| `IsolationForest` | how few random cuts isolate a point | no notion of time beyond its window |

**Masking** is why the plain z-score is the wrong default despite being everyone's
first idea. The mean and the standard deviation are both dragged by the very
points being looked for:

```python
series[100] = 250.0        # a moderate outlier
series[200] = 5000.0       # one huge one

ZScore().fit_score(series)[100]        # 0.47 — the huge one hid the moderate one
RobustZScore().fit_score(series)[100]  # 30.85 — the robust score still sees it
```

The median and the MAD have a breakdown point of 50%: half the data must be
contaminated before they move at all.

**Seasonality** is why the winner wins. On a metric with a daily rhythm, every
quiet night looks like an anomaly to a detector that knows nothing about nights —
which is exactly what the three flat detectors above are doing when they raise
276 alerts to the seasonal detector's 57, for the same recall.

## What it finds

```
$ anomaly detect --detector seasonal --limit 8
seasonal residual: 127 of 2,688 points over 3.330

when                    requests    score   labelled
2025-09-02 04:00         5,998.4    14.73   spike
2025-09-02 02:00         5,922.5    14.50   spike
2025-09-02 03:00         5,831.6    14.29   spike
2025-08-20 14:00         6,234.4    10.40   level_shift
2025-08-21 16:00         6,180.1    10.00   level_shift
2025-06-28 18:00         5,900.2     9.96   spike
2025-08-20 15:00         6,163.5     9.92   level_shift
2025-08-21 18:00         5,700.0     9.34   level_shift
... 119 more
```

## The data

```
$ anomaly describe
requests: 2,688 points, 2025-06-02 to 2025-09-21
  min 325.1   mean 2,060.2   max 6,234.4
  77 labelled anomalous points (2.9%) in 6 segment(s)
    dip              15 points
    level_shift      40 points
    spike             8 points
    variance         14 points
```

Sixteen weeks of hourly request rate with a daily rhythm, a weekend dip and a
slow climb, generated from a fixed seed. Four kinds of fault are injected and
labelled: a spike, an outage, a step change that persists, and a stretch where
the mean holds but the noise triples. They are deliberately different, because a
detector that catches spikes and nothing else is easy to write and easy to
mistake for a good one.

CI regenerates the file and fails if it differs from what is committed.

## A detail worth knowing about

Every robust detector here divides by a spread estimate, and there are two ways
that goes wrong. A perfectly regular signal leaves residuals that are floating
point dust, and dividing by their spread turns rounding error into scores in the
thousands. But the median absolute deviation is *also* exactly zero whenever more
than half the points are identical — a clean signal with one spike in it, which
is precisely the case that must still be caught.

`_scale` handles both: the MAD first, falling back to the mean absolute deviation
when the MAD is dust, and returning zero only when both are. Two tests cover the
pair — a clean season must score nothing, and a clean season with one spike must
score that spike.

## As a library

```python
from anomaly import read_csv, SeasonalResidual, evaluate

metric = read_csv("data/server-requests.csv")
train, test = metric.split(0.5)

result = evaluate(SeasonalResidual(period=24), train, test, quantile=0.98)
result.strict.f1      # 0.617
result.adjusted.f1    # 0.742  — always read next to the one above
result.pr_auc         # 0.662
```

## Layout

```
src/anomaly/detectors.py   random, z-score, robust z-score, IQR, EWMA, seasonal, isolation forest
src/anomaly/metrics.py     strict and point-adjusted counting, threshold sweep, PR-AUC
src/anomaly/data.py        loading and the temporal split
src/anomaly/evaluate.py    the leaderboard
src/anomaly/cli.py         the `anomaly` command
data/                      2,688 labelled hours, from a fixed seed
tools/make_data.py         the generator; CI fails if the output drifts
tests/                     27 tests
```

## Not included

Multivariate detection, changepoint algorithms, forecast-residual detectors,
streaming/online updates, alert deduplication. What is here is the part that has
to be right first: scoring you can believe, and a control that tells you when you
should not.

## Licence

MIT — see [LICENSE](LICENSE).
