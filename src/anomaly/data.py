"""Reading a labelled metric out of a CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Metric:
    timestamps: list[datetime]
    values: list[float]
    labels: list[int]
    faults: list[str] = field(default_factory=list)
    name: str = "value"

    def __post_init__(self) -> None:
        if not (len(self.timestamps) == len(self.values) == len(self.labels)):
            raise ValueError("timestamps, values and labels are different lengths")
        if not self.faults:
            self.faults = [""] * len(self.values)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def anomalies(self) -> int:
        return sum(self.labels)

    @property
    def rate(self) -> float:
        return self.anomalies / len(self) if self else 0.0

    def fault_kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label, kind in zip(self.labels, self.faults, strict=True):
            if label and kind:
                counts[kind] = counts.get(kind, 0) + 1
        return counts

    def split(self, fraction: float = 0.5) -> tuple[Metric, Metric]:
        """Cut in two by time. The first half is for fitting, the second for scoring.

        Fitting and scoring on the same points is how a detector ends up tuned to
        the very anomalies it is meant to find.
        """
        if not 0 < fraction < 1:
            raise ValueError("fraction must be between 0 and 1")
        cut = int(len(self) * fraction)
        if cut < 2 or len(self) - cut < 2:
            raise ValueError("the series is too short to split")
        return (
            Metric(self.timestamps[:cut], self.values[:cut], self.labels[:cut],
                   self.faults[:cut], self.name),
            Metric(self.timestamps[cut:], self.values[cut:], self.labels[cut:],
                   self.faults[cut:], self.name),
        )


def read_csv(path: str | Path, *, value_column: str | None = None) -> Metric:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{path.name} holds no rows")

    columns = list(rows[0])
    value_column = value_column or next(
        (c for c in columns if c not in {"timestamp", "is_anomaly", "fault"}), None
    )
    if value_column is None or value_column not in columns:
        raise ValueError(f"no value column found in {', '.join(columns)}")

    timestamps, values, labels, faults = [], [], [], []
    for line, row in enumerate(rows, start=2):
        try:
            timestamps.append(datetime.fromisoformat(row["timestamp"]))
            values.append(float(row[value_column]))
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path.name} line {line}: {error}") from None
        labels.append(int(row.get("is_anomaly", 0) or 0))
        faults.append(row.get("fault", "") or "")

    return Metric(timestamps, values, labels, faults, value_column)
