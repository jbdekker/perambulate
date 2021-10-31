from datetime import datetime
from datetime import time

import pandas as pd

from perambulate import Condition


def test_before(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))
    B = A.before(datetime(2021, 1, 10))

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 7), datetime(2021, 1, 9)),
        ],
        closed="left",
    )

    assert B.interval_index.equals(idx)


def test_at_or_before(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))
    B = A.at_or_before(datetime(2021, 1, 7))

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 7), datetime(2021, 1, 8)),
        ],
        closed="left",
    )

    assert B.interval_index.equals(idx)


def test_after(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))
    B = A.after(datetime(2021, 2, 1))

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 2, 3), datetime(2021, 2, 5)),
        ],
        closed="left",
    )

    assert B.interval_index.equals(idx)


def test_at_or_after(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))
    B = A.at_or_after(datetime(2021, 2, 4))

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 2, 4), datetime(2021, 2, 5)),
        ],
        closed="left",
    )

    assert B.interval_index.equals(idx)


def test_time(sinusoid_h):
    A = Condition(index=sinusoid_h.index)
    B = A.at_or_after(time(9, 0, 0)).before(time(17, 0, 0))

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 1, 9), datetime(2021, 1, 1, 17)),
            (datetime(2021, 1, 2, 9), datetime(2021, 1, 2, 15)),
        ],
        closed="left",
    )

    assert B.interval_index.equals(idx)
