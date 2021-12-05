from datetime import datetime
from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd

from perambulate.core.condition import Condition


def Periodic(
    index: pd.Index,
    freq: Union[pd.Timedelta, timedelta, np.timedelta64, str, int],
    unit: str = None,
    start: datetime = None,
    end: datetime = None,
    peg: datetime = None,
    normalize: bool = False,
    closed: str = "left",
    partial_periods: bool = False,
):
    """
    You can construct a Timedelta scalar through various arguments, including
    ISO 8601 Duration strings.
    """
    freq = pd.Timedelta(value=freq, unit=unit)

    start = start or index.min()
    end = end or index.max()
    periods = np.ceil((index.max() - index.min()) / freq) + 1

    if peg is not None:
        start = peg - pd.Timedelta((np.ceil((peg - start) / freq) * freq))
        periods += 1
        end = None
    else:
        if closed == "left":
            end = None
        elif closed == "right":
            start = None
        else:
            raise ValueError(f"`{closed}` is not a valid value for closed")

    date_range = pd.date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        normalize=normalize,
    )

    if not partial_periods:
        date_range = date_range[
            (date_range > index.min()) & (date_range < index.max())
        ]

    result = Condition()
    result.interval_index = pd.IntervalIndex.from_breaks(
        date_range, closed="left"
    )

    return result
