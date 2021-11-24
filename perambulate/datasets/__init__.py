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
