from typing import Union

import pandas as pd

from .condition import Condition
from .utils import extract_operator


def ValueSearch(
    entry_condition: Union[Condition, pd.Series],
    entry_filter: str,
    exit_condition: Union[Condition, pd.Series],
    exit_filter: str,
    exclude=None,
) -> Condition:
    """
    Trim interval length at the input threshold(s)

    Parameters
    ----------
    lower : object
        Minimum interval length, all intervals having a length below this
        value will be extended to it.
    upper : object
        Maximum interval length, all intervals having a length above this
        value will be shortened to it.
    side : {{'both', 'left', 'right'}}, default 'right'
        Side on which to adjust the interval

    Returns
    -------
    Condition
        Same type as calling object with the intervals clipped according to
        the clip threshold(s)
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
