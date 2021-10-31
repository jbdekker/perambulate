from datetime import timedelta

import pandas as pd


def derivative(s: pd.Series) -> pd.Series:
    return s.diff() / s.index.to_series().diff().dt.total_seconds()


def moves(s: pd.Series, window: timedelta) -> pd.Series:
    def calc_move(x: pd.Series):
        w = [x.idxmin(), x.idxmax()]
        if w[0] < w[1]:
            return x.loc[w[1]] - x.loc[w[0]]
        return x.loc[w[0]] - x.loc[w[1]]

    return s.rolling(pd.Timedelta(window)).apply(calc_move)
