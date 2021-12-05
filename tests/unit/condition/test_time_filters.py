from datetime import datetime

import pandas as pd

from perambulate import Condition


def test_filter(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))

    B = A.filter(">2d").filter("<=3d")

    idx = pd.IntervalIndex.from_tuples(
        [
            (datetime(2021, 1, 20), datetime(2021, 1, 23)),
        ],
        closed="left",
    )

    assert B.interval_index.equals(idx)

    assert len(A.filter("<2d")) == 0
    assert len(A.filter("<=2d")) == 2
    assert len(A.filter(">2d")) == 1
    assert len(A.filter(">=3d")) == 1
    assert len(A.filter("==3d")) == 1
    assert len(A.filter("!=3d")) == 2
