import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def SG(
    s: pd.Series,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    mode: str = "interp",
    cval: float = 0.0,
) -> pd.Series:
    """
    Apply a Savitzky-Golay filter to a signal

    Parameters
    ----------
    s : pd.Series
        Input signal
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        window_length must be an odd positive integer.
    deriv : int, optional
        The order of the derivative to compute. This must be a nonnegative
        integer. The default is 0, which means to filter the data without
        differentiating. Default is 0
    delta : float, optional
        The spacing of the samples to which the filter will be applied. This
        is only used if deriv > 0. Default is 1.0.
    mode : str, optional
        Must be ‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’. This
        determines the type of extension to use for the padded signal to which
        the filter is applied. When mode is ‘constant’, the padding value is
        given by cval. See the Notes for more details on ‘mirror’, ‘constant’,
        ‘wrap’, and ‘nearest’. When the ‘interp’ mode is selected
        (the default), no extension is used. Instead, a degree polyorder
        polynomial is fit to the last window_length values of the edges,
        and this polynomial is used to evaluate the last window_length // 2
        output values.
    cval : scalar, optional
        Value to fill past the edges of the input if mode is ‘constant’.
        Default is 0.0.

    Returns
    -------
    ps.Series
        Filtered input signal
    """
    y = savgol_filter(
        s.values,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        mode=mode,
        cval=cval,
    )

    return pd.Series(y, index=s.index, name="SG")


def MovingAverage(s: pd.Series, window_length: int) -> pd.Series:
    """
    Savitzky-Golay filter with polyorder of 0

    Parameters
    ----------
    s : pd.Series
        Input signal
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        window_length must be an odd positive integer.

    Returns
    -------
    ps.Series
        Filtered input signal
    """
    y = SG(
        s=s,
        window_length=window_length,
        polyorder=0,
        mode="constant",
        cval=np.nan,
    )

    return pd.Series(y, index=s.index, name="MovingAverage")


def WindowedSincLP(fc: float = 0.0006, b: float = 0.0002):
    N = int(np.ceil((4 / b)))
    if not N % 2:
        N += 1

    n = np.arange(N)
    h = np.sinc(2 * fc * (n - (N - 1) / 2.0))
    w = np.blackman(N)
    h = h * w
    h = h / np.sum(h)

    return N, h


def WindowedSincHP(fc: float = 0.0006, b: float = 0.0002):
    N, h = WindowedSincLP(fc, b)
    h = -h
    h[(N - 1) // 2] += 1

    return N, h


def LowPass(s: pd.Series, fc: float = 0.5, b: float = 0.5) -> pd.Series:
    """
    Scaled and windowed sinc filter to turn it into a digital filter

    Background:
        tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
        tomroelandts.com/articles/how-to-create-a-simple-high-pass-filter

    Parameters
    ----------
    s : pd.Series
        Input signal
    fc : float
        Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b : float
        Transition band, as a fraction of the sampling rate (in (0, 0.5)).

    Returns
    -------
    ps.Series
        Filtered input signal
    """

    _, h = WindowedSincLP(fc, b)
    y = np.convolve(s.values, h, "valid")

    pad = h.shape[0] - 1
    y = np.pad(y, (pad, 0), "constant", constant_values=np.nan)

    return pd.Series(y, index=s.index, name="LowPass")


def HighPass(s: pd.Series, fc: float = 0.1, b: float = 0.08) -> pd.Series:
    """
    Scaled and windowed sinc filter to turn it into a digital filter

    Background:
        tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
        tomroelandts.com/articles/how-to-create-a-simple-high-pass-filter

    Parameters
    ----------
    s : pd.Series
        Input signal
    fc : float
        Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b : float
        Transition band, as a fraction of the sampling rate (in (0, 0.5)).

    Returns
    -------
    ps.Series
        Filtered input signal
    """
    _, h = WindowedSincHP(fc, b)
    y = np.convolve(s.values, h, "valid")

    pad = h.shape[0] - 1
    print(h.shape)

    y = np.pad(y, (pad, 0), "constant", constant_values=np.nan)

    return pd.Series(y, index=s.index, name="HighPass")


def BandPass(
    s: pd.Series, fc_l: float = 0.1, fc_h: float = 0.4, b: float = 0.08
) -> pd.Series:
    """
    Scaled and windowed sinc filter to turn it into a digital filter

    Background:
        tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
        tomroelandts.com/articles/how-to-create-a-simple-high-pass-filter

    Parameters
    ----------
    s : pd.Series
        Input signal
    fc_l : float
        Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    fc_h : float
        Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b : float
        Transition band, as a fraction of the sampling rate (in (0, 0.5)).

    Returns
    -------
    ps.Series
        Filtered input signal
    """
    _, h_lpf = WindowedSincLP(fc_l, b)
    _, h_hpf = WindowedSincHP(fc_h, b)

    h = np.convolve(h_lpf, h_hpf)
    y = np.convolve(s.values, h, "valid")

    pad = h.shape[0] - 1
    y = np.pad(y, (pad, 0), "constant", constant_values=np.nan)

    return pd.Series(y, index=s.index, name="BandReject")


def BandReject(
    s: pd.Series, fc_l: float = 0.1, fc_h: float = 0.4, b: float = 0.08
) -> pd.Series:
    """
    Scaled and windowed sinc filter to turn it into a digital filter

    Background:
        tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
        tomroelandts.com/articles/how-to-create-a-simple-high-pass-filter

    Parameters
    ----------
    s : pd.Series
        Input signal
    fc_l : float
        Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    fc_h : float
        Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b : float
        Transition band, as a fraction of the sampling rate (in (0, 0.5)).

    Returns
    -------
    ps.Series
        Filtered input signal
    """
    _, h_lpf = WindowedSincLP(fc_l, b)
    _, h_hpf = WindowedSincHP(fc_h, b)

    h = h_lpf + h_hpf
    y = np.convolve(s.values, h, "valid")

    pad = h.shape[0] - 1
    y = np.pad(y, (pad, 0), "constant", constant_values=np.nan)

    return pd.Series(y, index=s.index, name="BandReject")


def Agile(
    data: pd.Series,
    f: float = 2 / 3,
    pts: int = None,
    itn: int = 3,
    order: int = 1,
) -> pd.Series:
    """Fits a nonparametric regression curve to a scatterplot.

    Parameters
    ----------
    data : pandas.Series
        Data points in the scatterplot. The
        function returns the estimated (smooth) values of y.
    f : float, optional
        The fraction of the data set to use for smoothing. A
        larger value for f will result in a smoother curve.
    pts : int, optional
        The explicit number of data points to be used for
        smoothing instead of f.
    itn : int, optional
        The number of robustifying iterations. The function will run
        faster with a smaller number of iterations.
    order : int, optional
        The order of the polynomial used for fitting. Defaults to 1
        (straight line). Values < 1 are made 1. Larger values should be
        chosen based on shape of data (# of peaks and valleys + 1)

    Returns
    -------
    pandas.Series
        containing the smoothed data.
    """

    x = np.array(data.index, dtype=float)
    # condition x-values to be between 0 and 1 to reduce errors in linalg
    x = x - x.min()
    x = x / x.max()
    y = data.values
    n = len(data)
    if pts is None:
        f = np.min([f, 1.0])
        r = int(np.ceil(f * n))
    else:  # allow use of number of points to determine smoothing
        r = int(np.min([pts, n]))
    r = min([r, n - 1])
    order = max([1, order])
    # Create matrix of 1, x, x**2, x**3, etc, by row
    xm = np.array([x ** j for j in range(order + 1)])
    # Create weight matrix, one column per data point
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    # Set up output
    yEst = np.zeros(n)
    delta = np.ones(n)  # Additional weights for iterations

    for _ in range(itn):
        for i in range(n):
            weights = delta * w[:, i]
            xw = np.array([weights * x ** j for j in range(order + 1)])
            b = xw.dot(y)
            a = xw.dot(xm.T)
            beta = np.linalg.solve(a, b)
            yEst[i] = sum([beta[j] * x[i] ** j for j in range(order + 1)])

        # Set up weights to reduce effect of outlier points on next iteration
        residuals = y - yEst
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return pd.Series(yEst, index=data.index, name="AgileFilter")
