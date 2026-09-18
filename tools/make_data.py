"""Writes the labelled sample metric in data/.

Hourly server request rate for sixteen weeks, with a daily rhythm, a weekend dip
and a slow climb — then four kinds of injected fault, each labelled so a detector
can be scored honestly:

  spike          a sudden burst over a few hours
  dip            an outage: traffic falls away and returns
  level shift    a step change that persists
  variance burst the mean holds but the noise triples

The four are deliberately different, because a detector that catches spikes and
nothing else is easy to write and easy to mistake for a good one.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20260918
DATA = Path(__file__).resolve().parent.parent / "data"
HOURS = 24 * 7 * 16

FAULTS = [
    ("spike", 640, 5, 3.4),
    ("dip", 1180, 9, 0.25),
    ("variance", 1520, 14, 1.0),
    ("level_shift", 1900, 40, 1.6),
    ("spike", 2210, 3, 4.1),
    ("dip", 2560, 6, 0.3),
]


def build() -> list[dict]:
    rng = random.Random(SEED)
    start = datetime(2025, 6, 2, 0, 0)     # a Monday
    rows = []

    for i in range(HOURS):
        when = start + timedelta(hours=i)
        # Daily shape: quiet at 04:00, busiest mid-afternoon.
        daily = 1 + 0.45 * math.sin(2 * math.pi * (when.hour - 9) / 24)
        weekend = 0.62 if when.weekday() >= 5 else 1.0
        trend = 1 + 0.0002 * i
        value = 1800 * daily * weekend * trend
        noise = 0.05
        label = 0
        kind = ""

        for fault, begin, length, factor in FAULTS:
            if begin <= i < begin + length:
                label = 1
                kind = fault
                if fault == "variance":
                    noise = 0.18
                elif fault == "level_shift":
                    value *= factor
                else:
                    value *= factor

        value *= 1 + rng.gauss(0, noise)
        rows.append({
            "timestamp": when.isoformat(timespec="seconds"),
            "requests": round(max(value, 0), 1),
            "is_anomaly": label,
            "fault": kind,
        })

    return rows


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rows = build()
    path = DATA / "server-requests.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    anomalies = sum(r["is_anomaly"] for r in rows)
    print(f"server-requests.csv: {len(rows)} hours, {anomalies} anomalous "
          f"({anomalies / len(rows) * 100:.1f}%), {len(FAULTS)} faults")


if __name__ == "__main__":
    main()
