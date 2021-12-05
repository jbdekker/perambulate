from typing import Union

import pandas as pd

from perambulate.core.condition import Condition
from perambulate.core.utils import extract_operator


def ValueSearch(
    entry_condition: Union[Condition, pd.Series],
    entry_filter: str,
    exit_condition: Union[Condition, pd.Series],
    exit_filter: str,
    exclude=None,
) -> Condition:
    """
    Create a condition with intervals that match the given entry and exit
    criteria.

    Parameters
    ----------
    entry_condition : Condition, pd.Series
        Entry condition
    entry_filter : str
        timedelta (in)equality
    exit_condition : Condition, pd.Series
        Exit condition
    exit_filter : str
        timedelta (in)equality
    exclude : {{None, 'entry', 'exit', 'both'}}, default None
        If not None, excludes the entry or exit (or both) from the matched
        intervals

    Returns
    -------
    Condition
        Condition object with its intervals matching the entry and exit
        criteria

    Examples
    --------
    >>> df = pr.datasets.load_sinusoid()
    >>> pr.ValueSearch(df.sinusoid > -1, "1d", df.sinusoid > 1, "1d")
                     left               right closed           length
    0 2021-01-03 20:00:00 2021-01-26 04:00:00   left 22 days 08:00:00
    """
    _, entry_value = extract_operator(entry_filter)
    EntryCondition = (
        Condition(entry_condition)
        .filter(entry_filter)
        .clip(upper=entry_value, side="right")
    )

    _, exit_value = extract_operator(exit_filter)
    ExitCondition = (
        Condition(exit_condition)
        .filter(exit_filter)
        .clip(upper=exit_value, side="right")
    )

    entries = sorted(EntryCondition.interval_index, key=lambda x: x.left)
    exits = sorted(ExitCondition.interval_index, key=lambda x: x.left)

    r = []
    last_exit = None
    for entry in entries:
        if last_exit is None or entry.left >= last_exit:
            for exit in exits:
                if entry.left < exit.right:
                    r += [pd.Interval(entry.left, exit.right, exit.closed)]
                    last_exit = exit.right
                    break

    result = Condition()
    result.interval_index = pd.IntervalIndex(r)

    if exclude in ["both", "exit"]:
        result = result.shrink(exit_value, side="right")

    if exclude in ["both", "entry"]:
        result = result.shrink(entry_value, side="left")

    return result
