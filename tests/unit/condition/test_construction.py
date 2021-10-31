from datetime import datetime

import pandas as pd
import pytest

from perambulate import Condition


def test_init_none():
    with pytest.raises(ValueError):
        Condition()


def test_init_reproduce(sinusoid_d):
    assert Condition(sinusoid_d > 0) == Condition.reproduce(sinusoid_d > 0)


def test_init_all():
    with pytest.raises(ValueError):
        Condition(42, 42)


def test_init_with_series(sinusoid_d):
    with pytest.raises(TypeError):
        Condition(condition=sinusoid_d)

    with pytest.raises(TypeError):
        Condition(condition=42)

    A = Condition(condition=sinusoid_d < 1)

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 1), datetime(2021, 1, 24)),
            (datetime(2021, 2, 1), datetime(2021, 2, 8)),
        ],
        closed="left",
    )

    assert A.interval_index.equals(idx)
    assert len(A) == 2


def test_init_with_index(sinusoid_d):
    with pytest.raises(TypeError):
        Condition(index=sinusoid_d)

    c = Condition(index=sinusoid_d.index)
    assert c.index.equals(sinusoid_d.index)


def test_init_with_empty_condition(sinusoid_d):
    c = Condition(condition=sinusoid_d > 100)
    assert len(c.interval_index) == 0


def test_repr(sinusoid_d):
    A = Condition(condition=sinusoid_d < 1)
    assert A.__repr__() == (
        "        left      right closed  length\n0 2021-01-01 2021-01-24   "
        "left 23 days\n1 2021-02-01 2021-02-08   left  7 days"
    )


def test_to_frame(sinusoid_d):
    A = Condition(condition=sinusoid_d < 1)

    assert A.to_frame().shape == (2, 4)
