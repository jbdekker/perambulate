import io
import pkgutil
from datetime import datetime
from datetime import timezone

import numpy as np
import pandas as pd


def load_sinusoid():
    periods = 256
    index = np.arange(-2 * np.pi, 2 * np.pi, np.pi / periods)

    df = pd.Series(
        data=np.sin(index) + index / np.pi,
        index=pd.date_range(
            start=datetime(2021, 1, 1, tzinfo=timezone.utc),
            periods=len(index),
            freq="1h",
        ),
    ).to_frame(name="sinusoid")

    return df


def load_ecg():
    fn = "ecg.csv"
    data = pkgutil.get_data(__name__, fn)
    df = pd.read_csv(
        io.StringIO(data.decode("utf-8")), header=None, names=["mV"]
    )

    dt_idx = pd.date_range(
        start=datetime(2021, 1, 1, tzinfo=timezone.utc),
        periods=len(df),
        freq="10ms",
    )

    df.index = dt_idx

    return df
