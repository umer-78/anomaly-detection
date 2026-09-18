from pathlib import Path

import pytest

from anomaly import read_csv

DATA = Path(__file__).resolve().parent.parent / "data" / "server-requests.csv"


@pytest.fixture(scope="session")
def metric():
    return read_csv(DATA)


@pytest.fixture(scope="session")
def halves(metric):
    return metric.split(0.5)
