"""anomaly: score detectors on a labelled metric, or flag points in an unlabelled one."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .data import read_csv
from .detectors import (
    EWMA,
    IQR,
    IsolationForest,
    RandomDetector,
    RobustZScore,
    SeasonalResidual,
    ZScore,
    default_detectors,
)
from .evaluate import header, leaderboard
from .metrics import segments

DETECTORS = {
    "random": RandomDetector,
    "zscore": ZScore,
    "robust": RobustZScore,
    "iqr": IQR,
    "ewma": EWMA,
    "seasonal": SeasonalResidual,
    "forest": IsolationForest,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anomaly", description=__doc__)
    parser.add_argument("--version", action="version", version=f"anomaly {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("csv", type=Path, nargs="?", default=Path("data/server-requests.csv"))
        p.add_argument("--value-column", default=None)
        p.add_argument("--period", type=int, default=24, help="seasonal period, in points")

    describe = sub.add_parser("describe", help="what is in the file")
    common(describe)

    evaluate = sub.add_parser("evaluate", help="score every detector on the labelled half")
    common(evaluate)
    evaluate.add_argument("--quantile", type=float, default=0.98,
                          help="threshold quantile, taken from the training scores")
    evaluate.add_argument("--json", action="store_true")

    detect = sub.add_parser("detect", help="flag points with one detector")
    common(detect)
    detect.add_argument("--detector", choices=sorted(DETECTORS), default="seasonal")
    detect.add_argument("--quantile", type=float, default=0.98)
    detect.add_argument("--limit", type=int, default=20)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Wraps the real work so that piping into `head` — which closes
    the pipe early — ends quietly instead of printing a BrokenPipeError."""
    try:
        return _run(argv)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as error:
        print(f"anomaly: {error}", file=sys.stderr)
        return 2


def _run(argv: list[str] | None) -> int:
    args = build_parser().parse_args(argv)
    metric = read_csv(args.csv, value_column=args.value_column)

    if args.cmd == "describe":
        print(f"{metric.name}: {len(metric):,} points, "
              f"{metric.timestamps[0]:%Y-%m-%d} to {metric.timestamps[-1]:%Y-%m-%d}")
        print(f"  min {min(metric.values):,.1f}   "
              f"mean {sum(metric.values) / len(metric):,.1f}   max {max(metric.values):,.1f}")
        if metric.anomalies:
            runs = segments(metric.labels)
            print(f"  {metric.anomalies} labelled anomalous points ({metric.rate:.1%}) "
                  f"in {len(runs)} segment(s)")
            for kind, count in sorted(metric.fault_kinds().items()):
                print(f"    {kind:<14} {count:>4} points")
        else:
            print("  no labels in this file")
        return 0

    train, test = metric.split(0.5)

    if args.cmd == "evaluate":
        if not test.anomalies:
            print("anomaly: the second half carries no labels to score against", file=sys.stderr)
            return 2
        results = leaderboard(default_detectors(args.period), train, test, quantile=args.quantile)
        if args.json:
            print(json.dumps([{"detector": r.detector, "precision": round(r.strict.precision, 4),
                               "recall": round(r.strict.recall, 4), "f1": round(r.strict.f1, 4),
                               "pr_auc": round(r.pr_auc, 4),
                               "point_adjusted_f1": round(r.adjusted.f1, 4),
                               "best_possible_f1": round(r.ceiling, 4),
                               "alerts": r.alerts} for r in results], indent=2))
            return 0
        print(f"fitted on {len(train):,} points, scored on {len(test):,} "
              f"({test.anomalies} anomalous), threshold at the {args.quantile:.0%} "
              f"quantile of training scores\n")
        print(header())
        for result in results:
            print(result.row())
        print("\nadjusted F1 credits a whole segment when any point in it is flagged;")
        print("compare it with the random detector's before trusting one.")
        return 0

    detector = DETECTORS[args.detector]()
    if hasattr(detector, "period"):
        detector.period = args.period
    detector.fit(train.values)

    training_scores = sorted(detector.score(train.values))
    index = min(int(len(training_scores) * args.quantile), len(training_scores) - 1)
    threshold = training_scores[index]

    scores = detector.score(metric.values)
    flagged = [(i, s) for i, s in enumerate(scores) if s > threshold]
    flagged.sort(key=lambda pair: -pair[1])

    print(f"{detector.name}: {len(flagged)} of {len(metric):,} points over "
          f"{threshold:.3f}\n")
    print(f"{'when':<20}{metric.name:>12}{'score':>9}   labelled")
    for index_, score in flagged[:args.limit]:
        mark = metric.faults[index_] or ("yes" if metric.labels[index_] else "")
        print(f"{metric.timestamps[index_]:%Y-%m-%d %H:%M}{metric.values[index_]:>16,.1f}"
              f"{score:>9.2f}   {mark}")
    if len(flagged) > args.limit:
        print(f"... {len(flagged) - args.limit} more")
    return 0
