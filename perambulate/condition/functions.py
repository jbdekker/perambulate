from datetime import timedelta
from typing import Union

import pandas as pd


def derivative(
    s: pd.Series, dx: Union[str, timedelta] = None, order: int = 1
) -> pd.Series:

    result = s
    for _ in range(order):
        dy = result.diff()
        x = result.index.to_series()
        dx = x.diff().dt.total_seconds() / pd.Timedelta(dx).total_seconds()

        result = dy / dx

    return result


def _range(s: pd.Series, window: Union[str, timedelta]) -> pd.Series:
    def _calc_range(x: pd.Series):
        w = [x.idxmin(), x.idxmax()]
        if w[0] < w[1]:
            return x.loc[w[1]] - x.loc[w[0]]
        return x.loc[w[0]] - x.loc[w[1]]

    return s.rolling(pd.Timedelta(window)).apply(_calc_range)
