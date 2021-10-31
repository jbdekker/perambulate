from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def sinusoid_d():
    index = np.arange(-2 * np.pi, 2 * np.pi, np.pi / 10)

    return pd.Series(
        data=np.sin(index) + index / np.pi,
        index=pd.date_range(
            start=datetime(2021, 1, 1), periods=len(index), freq="1d"
        ),
    )


@pytest.fixture(autouse=True)
def sinusoid_h():
    index = np.arange(-2 * np.pi, 2 * np.pi, np.pi / 10)

    return pd.Series(
        data=np.sin(index) + index / np.pi,
        index=pd.date_range(
            start=datetime(2021, 1, 1), periods=len(index), freq="1h"
        ),
    )
