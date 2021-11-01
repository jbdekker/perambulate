import copy
import re
from datetime import datetime
from datetime import time
from itertools import compress
from operator import eq
from operator import ge
from operator import gt
from operator import le
from operator import lt
from operator import ne
from typing import Callable
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex


__all__ = ["Condition"]


class Condition:
    def __init__(
        self,
        condition: Union[pd.Index, pd.Series] = None,
        index: pd.Index = None,
    ):
        self.index = None
        self.interval_index = self._empty_interval_index

        if self.all_nons([condition, index]):
            raise ValueError("either a condition or index is required")
        if self.no_nons([condition, index]):
            raise ValueError(
                "provide either a condition or an index, not both"
            )
        if condition is not None:
            if not isinstance(condition, pd.Series):
                raise TypeError(
                    "condition must be of type `pd.Series`, "
                    f"`{type(condition)}` was passed"
                )
            self.validate_series(condition)
            self.index = condition.index
            self.interval_index = self.mask_to_intervals(condition)
        if index is not None:
            if not isinstance(index, pd.Index):
                raise TypeError(
                    "index must be of type `pd.Index`, "
                    f"`{type(index)}` was passed"
                )
            self.validate_index(index)
            self.index = index

    @property
    def _empty_interval_index(self):
        return pd.IntervalIndex(
            [], closed="left", dtype="interval[datetime64[ns]]"
        )

    def _is_datetime_type(self) -> bool:
        return self.interval_index.dtype.subtype == np.dtype(
            "datetime64[ns]"
        ) or isinstance(self.index, pd.DatetimeIndex)

    def _is_none(self, obj) -> List[bool]:
        return [x is None for x in list(obj)]

    def all_nons(self, obj) -> bool:
        return all(self._is_none(obj))

    def no_nons(self, obj) -> bool:
        return not any(self._is_none(obj))

    @classmethod
    def reproduce(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def validate_series(self, obj: pd.Series) -> None:
        self.validate_index(obj.index)

        if not pd.api.types.is_bool_dtype(obj.dtype):
            raise TypeError("condition must be as pd.Series of dtype `bool`")

    def validate_index(self, obj: pd.Index) -> None:
        assert obj.is_unique, "index is not unique"
        assert not obj.hasnans, "index contains NaN values"
        assert not obj.is_mixed(), "index has mixed values"
        assert (
            obj.is_monotonic_increasing
        ), "index must be is monotonic increasing"

        if self.index is not None:
            assert self.index.equals(obj), "indices must be equal"

    def mask_to_intervals(self, obj: pd.Series) -> pd.IntervalIndex:
        mask = (obj != obj.shift(1)).cumsum()

        datetime_mask = False
        freq = None
        if isinstance(obj.index, DatetimeIndex):
            datetime_mask = True
            freq = pd.Timedelta(obj.index.freq)

        def func(x):
            left = x.index[0]
            i = mask.index.get_loc(x.index[-1])
            try:
                right = mask.index[i + 1]
            except IndexError:
                if datetime_mask and freq is not None:
                    right = mask.index[i] + freq
                else:
                    right = mask.index[i] + (mask.index[i] - mask.index[i - 1])
            return (left, right)

        intervals = mask.groupby(mask[obj]).apply(func)

        if intervals.empty:
            return self._empty_interval_index

        return pd.IntervalIndex.from_tuples(intervals, closed="left")

    def mask(self, s: Union[pd.Series, pd.DataFrame] = None) -> pd.Series:
        idx = self.index if s is None else s.index
        cat = pd.cut(idx, self.interval_index)
        return pd.Series(data=~cat.isna(), index=self.index)

    @staticmethod
    def _reduce_intervals(s: pd.IntervalIndex) -> pd.IntervalIndex:
        s = sorted(s, key=lambda x: x.left)

        m = 0
        for t in s:
            if not t.overlaps(s[m]):
                m += 1
                s[m] = t
            else:
                s[m] = pd.Interval(
                    s[m].left, max(s[m].right, t.right), t.closed
                )
        return pd.IntervalIndex(s[: m + 1])

    def reduce(self):
        result = self.copy()
        result.interval_index = Condition._reduce_intervals(
            self.interval_index
        )
        return result

    def _or(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        if isinstance(other, pd.Series):
            other = self.reproduce(condition=other)
            self.validate_index(other.index)
        s = self.interval_index.append(other.interval_index)

        result = self.copy()
        result.index = self.index if other.index is None else other.index
        result.interval_index = Condition._reduce_intervals(s)

        return result

    def __or__(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        return self._or(other)

    def union(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        return self._or(other)

    def _and(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        if isinstance(other, pd.Series):
            other = self.reproduce(condition=other)
            self.validate_index(other.index)
        intervals = sorted(other.interval_index, key=lambda x: x.left)

        result = self.copy()
        result.index = self.index if other.index is None else other.index

        r = []
        for i in intervals:
            overlaps = self.interval_index.overlaps(i)
            if any(overlaps):
                r += [
                    pd.Interval(
                        max(j.left, i.left),
                        min(j.right, i.right),
                        i.closed,
                    )
                    for j in list(compress(self.interval_index, overlaps))
                ]
        result.interval_index = pd.IntervalIndex(r)

        return result

    def __and__(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        return self._and(other)

    def intersect(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        return self._and(other)

    def _xor(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        """Exclusive disjunction"""
        # p XOR q = ( p AND NOT q ) OR ( NOT p AND q )
        p = self

        if isinstance(other, pd.Series):
            q = self.reproduce(condition=other)
            self.validate_index(q.index)
        else:
            q = other
        return p._and(q._not())._or(p._not()._and(q))

    def __xor__(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        return self._xor(other)

    def _not(self) -> "Condition":
        intervals = sorted(self.interval_index, key=lambda x: x.left)
        start, end = self.index.min(), self.index.max()

        r = []
        if start < intervals[0].left:
            r.append(
                pd.Interval(start, intervals[0].left, intervals[0].closed)
            )
        for n, i in enumerate(intervals[:-1]):
            if i.right >= start and intervals[n + 1].left <= end:
                r.append(pd.Interval(i.right, intervals[n + 1].left, i.closed))
        if end > intervals[-1].right:
            r.append(
                pd.Interval(intervals[-1].right, end, intervals[-1].closed)
            )
        result = self.copy()
        result.interval_index = pd.IntervalIndex(r)

        return result

    def inverse(self) -> "Condition":
        return self._not()

    def __invert__(self) -> "Condition":
        return self._not()

    def shift_intervals(self, left, right) -> pd.IntervalIndex:
        s = pd.IntervalIndex(
            [
                pd.Interval(
                    x.left + left,
                    x.right + right,
                    x.closed,
                )
                for x in self.interval_index
            ]
        )

        return s

    def move(self, value: object) -> "Condition":
        result = self.copy()

        if self._is_datetime_type():
            value = pd.Timedelta(value)
        result.interval_index = self.shift_intervals(value, value)

        return result

    def shrink(self, value: object, side="both") -> "Condition":
        result = self.copy()
        try:
            left = value if side in ["left", "both"] else 0
            right = value if side in ["right", "both"] else 0

            if self._is_datetime_type():
                left = pd.Timedelta(left)
                right = pd.Timedelta(right)

            result.interval_index = self.shift_intervals(left, -right)
        except (ValueError, IndexError):
            raise ValueError("some intervals are to short, filter first")
        return result

    def grow(self, value: object, side="both") -> "Condition":
        if self._is_datetime_type():
            value = pd.Timedelta(value)
        return self.shrink(-value, side)

    def grow_end(self) -> "Condition":
        max_value = max(self.index)
        s = sorted(self.interval_index, key=lambda x: x.left)

        result = self.copy()
        if self.__len__() == 0:
            result.interval_index = self._empty_interval_index
            return result

        r = []
        for i in range(len(s) - 1):
            r.append((s[i].left, s[i + 1].left))
        r.append((s[-1].left, max_value))
        result.interval_index = pd.IntervalIndex.from_tuples(
            r, closed=self.interval_index.closed
        )
        return result

    def copy(self):
        return copy.deepcopy(self)

    def plot(
        self, data: pd.Series, *args, figsize=(16, 4), **kwargs
    ) -> None:  # pragma: no cover
        from matplotlib import pyplot as plt

        _, ax = plt.subplots(*args, figsize=figsize, **kwargs)

        data.plot(ax=ax)

        ax.fill_between(
            data.index,
            0,
            1,
            where=self.mask(data),
            alpha=0.3,
            transform=ax.get_xaxis_transform(),
        )

    def stack(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if len(self.interval_index) == 0:
            raise ValueError("No intervals to stack")

        df = self._cut(data)

        dfs = []
        for _, data in df.groupby("interval"):
            data.index = data.index - data.index.min()
            dfs.append(data)
        return pd.concat(dfs)

    def stack_plot(
        self, s: pd.Series, *args, figsize=(16, 4), **kwargs
    ) -> None:  # pragma: no cover
        if not isinstance(s, pd.Series):
            raise TypeError("stack_plot only supports `pd.Series`")
        df = self.stack(s)

        from matplotlib import pyplot as plt

        _, ax = plt.subplots(*args, figsize=figsize, **kwargs)

        for key, grp in df.groupby("interval"):
            grp = grp.drop(columns=["interval"])
            grp.columns = [str(key)]
            grp.plot(ax=ax)
        ax.legend(loc="best")

    def extract_operator(self, value) -> object:
        regex = r"^[><=!]{1,2}"

        mapping = {
            ">=": ge,
            "<=": le,
            ">": gt,
            "<": lt,
            "==": eq,
            "!=": ne,
        }

        try:
            token = re.findall(regex, value.strip())[0]
            op = mapping[token]
            value = value.replace(token, "")
        except (KeyError, IndexError):
            raise KeyError(value)
        return op, value

    def filter(self, value: object) -> "Condition":
        result = self.copy()

        op, value = self.extract_operator(value)

        if self._is_datetime_type():
            value = pd.Timedelta(value)
        result.interval_index = pd.IntervalIndex(
            [x for x in self.interval_index if op(x.length, value)]
        )

        return result

    def to_frame(self):
        return pd.DataFrame(
            {
                "left": [i.left for i in self.interval_index],
                "right": [i.right for i in self.interval_index],
                "closed": [i.closed for i in self.interval_index],
                "length": [i.length for i in self.interval_index],
            }
        )

    def __repr__(self):
        return self.to_frame().__repr__()

    def __len__(self):
        return len(self.interval_index)

    def _index_filter(
        self, operator, value: Union[int, float, time, datetime]
    ) -> "Condition":
        result = self.copy()

        if isinstance(value, time):
            ref = self.index.time
        else:
            ref = self.index
        mask = pd.Series(operator(ref, value), self.index)
        result.interval_index = self.mask_to_intervals(mask)

        if len(self.interval_index) == 0:
            return result
        return self._and(result)

    def before(self, value: Union[time, datetime]) -> "Condition":
        return self._index_filter(lt, value)

    def at_or_before(self, value: Union[time, datetime]) -> "Condition":
        return self._index_filter(le, value)

    def after(self, value: Union[time, datetime]) -> "Condition":
        return self._index_filter(gt, value)

    def at_or_after(self, value: Union[time, datetime]) -> "Condition":
        return self._index_filter(ge, value)

    def __eq__(self, other: "Condition") -> bool:
        return (
            isinstance(other, type(self))
            & self.index.equals(other.index)
            & self.interval_index.equals(other.interval_index)
        )

    def touches(self, other: "Condition") -> "Condition":
        """Filters condition intervals based on whether it touches an
        interval in other"""
        result = self.copy()

        r = []
        for s in self.interval_index:
            if any(other.interval_index.overlaps(s)):
                r.append(s)

        result.interval_index = pd.IntervalIndex(r)

        return result

    def encloses(self, other: "Condition") -> "Condition":
        """Filters condition intervals based on whether it encloses an
        interval in other"""

        result = self.copy()

        r = []
        for s in self.interval_index:
            overlaps = other.interval_index.overlaps(s)

            if any(
                [
                    (s.left <= j.left) and (s.right >= j.right)
                    for j in list(compress(other.interval_index, overlaps))
                ]
            ):
                r.append(s)

        result.interval_index = pd.IntervalIndex(r)

        return result

    def inside(self, other: "Condition") -> "Condition":
        """Filters condition intervals based on whether it is enclosed by an
        interval in other"""

        result = self.copy()

        r = []
        for s in self.interval_index:
            overlaps = other.interval_index.overlaps(s)

            if any(
                [
                    (s.left >= j.left) and (s.right <= j.right)
                    for j in list(compress(other.interval_index, overlaps))
                ]
            ):
                r.append(s)

        result.interval_index = pd.IntervalIndex(r)

        return result

    def _cut(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df["interval"] = pd.cut(df.index, self.interval_index)
        return df.dropna(subset=["interval"])

    def series_from_condition(
        self,
        data: Union[pd.Series, pd.DataFrame],
        func: Union[Callable, dict],
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.Series, pd.DataFrame
        func : function
            Function to apply to each column
        *args : tuple
            Positional arguments to pass to `func` in addition to the
            array/series.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        Series or DataFrame
            Result of applying ``func`` over each Interval along the columns
            of the DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame([1, 3, 0, 2, 4], columns=["A"])
        >>> C = Condition(df.A >= 1)
        >>> df["B"] = C.series_from_condition(df.A, np.mean)
           A  B
        0  1  2
        1  3  2
        2  0  NaN
        3  2  3
        4  4  3
        """

        df = self._cut(data)
        df["interval"] = pd.Categorical(df.interval).codes

        df_agg = df.groupby(df.interval, as_index=False).agg(
            func, *args, **kwargs
        )

        if isinstance(df_agg.columns, pd.MultiIndex):
            df_agg.columns = ["_".join(c).strip("_") for c in df_agg.columns]

        df = df["interval"].reset_index()

        result = (
            pd.merge(df, df_agg, on="interval", how="left")
            .reset_index(drop=True)
            .drop(columns=["interval"])
            .set_index("index")
        )

        return result

    def bin_data(
        self, data: Union[pd.Series, pd.DataFrame]
    ) -> List[Union[pd.Series, pd.DataFrame]]:
        return [grp for _, grp in self._cut(data).groupby("interval")]

    def describe(
        self, data: Union[pd.Series, pd.DataFrame], func: Union[Callable, dict]
    ):
        pass
