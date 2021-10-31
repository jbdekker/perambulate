from datetime import datetime

import pandas as pd
import pytest

from perambulate import Condition


def test_equality(sinusoid_d):
    A = Condition(condition=sinusoid_d < 1)
    B = Condition(condition=sinusoid_d < 1)

    assert A == B


def test_and(sinusoid_d):
    A = Condition(condition=sinusoid_d < 1)
    B = Condition(condition=sinusoid_d > -1)

    C = A & B

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 4), datetime(2021, 1, 11)),
            (datetime(2021, 1, 19), datetime(2021, 1, 24)),
            (datetime(2021, 2, 1), datetime(2021, 2, 8)),
        ],
        closed="left",
    )

    with pytest.raises(AssertionError):
        A & (sinusoid_d.head(5) < 1)  # index mismatch

    assert C.interval_index.equals(idx)
    assert (A & B) == (B & A)
    assert C == A._and(B) == B._and(A)
    assert C == A.intersect(B) == B.intersect(A)


def test_or(sinusoid_d):
    A = Condition(condition=sinusoid_d > 1)
    B = Condition(condition=sinusoid_d < -1)

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 1), datetime(2021, 1, 4)),
            (datetime(2021, 1, 11), datetime(2021, 1, 19)),
            (datetime(2021, 1, 24), datetime(2021, 2, 1)),
            (datetime(2021, 2, 8), datetime(2021, 2, 9)),
        ],
        closed="left",
    )

    C = A | B

    with pytest.raises(AssertionError):
        A | (sinusoid_d.head(5) < 1)  # index mismatch

    assert C.interval_index.equals(idx)
    assert (A | B) == (B | A)
    assert C == A._or(B) == B._or(A)
    assert C == A.union(B) == B.union(A)


def test_not(sinusoid_d):
    A = Condition(condition=sinusoid_d > 1)

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 1), datetime(2021, 1, 24)),
            (datetime(2021, 2, 1), datetime(2021, 2, 8)),
        ],
        closed="left",
    )

    C = ~A

    assert C.interval_index.equals(idx)
    assert C == A._not() == A.inverse()


def test_xor(sinusoid_d):
    A = Condition(condition=sinusoid_d > -0.75)
    B = Condition(condition=sinusoid_d < 0.75)

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 1), datetime(2021, 1, 5)),
            (datetime(2021, 1, 10), datetime(2021, 1, 20)),
            (datetime(2021, 1, 23), datetime(2021, 2, 2)),
            (datetime(2021, 2, 7), datetime(2021, 2, 9)),
        ],
        closed="left",
    )

    C = A ^ B

    assert C.interval_index.equals(idx)
    assert C == A._xor(B) == B._xor(A) == (A ^ (sinusoid_d < 0.75))


def test_move(sinusoid_d):
    A = Condition(condition=sinusoid_d > -1)

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 6), datetime(2021, 1, 13)),
            (datetime(2021, 1, 21), datetime(2021, 2, 9)),
        ],
        closed="left",
    )

    C = A.move("2d")

    assert C.interval_index.equals(idx)
    assert A == A.move("-2d").move("1d").move("1d")
    assert A != A.move("10d").move("-10d")  # overflows the index


def test_grow_shrink(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 6), datetime(2021, 1, 10)),
            (datetime(2021, 1, 19), datetime(2021, 1, 24)),
            (datetime(2021, 2, 2), datetime(2021, 2, 6)),
        ],
        closed="left",
    )

    with pytest.raises(ValueError):
        A.shrink("10d")  # intervals collapse

    assert A.grow("2d").interval_index.equals(idx)
    assert A == A.grow("2d").grow("-2d")
    assert A == A.shrink("1d").shrink("-1d")
    assert A.grow("2d") == A.shrink("-2d")


def test_grow_end(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))
    B = A.grow_end()
    C = Condition(index=sinusoid_d.index).grow_end()

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 7), datetime(2021, 1, 20)),
            (datetime(2021, 1, 20), datetime(2021, 2, 3)),
            (datetime(2021, 2, 3), datetime(2021, 2, 9)),
        ],
        closed="left",
    )

    assert B.interval_index.equals(idx)
    assert C.interval_index.equals(C._empty_interval_index)


def test_reduce(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 1), datetime(2021, 2, 9)),
        ],
        closed="left",
    )

    assert A.grow("100d").reduce().interval_index.equals(idx)


def test_mask(sinusoid_d):
    mask = (sinusoid_d > -0.5) & (sinusoid_d < 0.5)
    A = Condition(condition=mask)
    assert mask.equals(A.mask())


def test_stack(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))

    with pytest.raises(ValueError):  # no intervals to stack
        Condition(sinusoid_d > 100).stack(sinusoid_d)

    df = A.stack(sinusoid_d)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (7, 2)
