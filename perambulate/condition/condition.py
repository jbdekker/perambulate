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
from typing import List
from typing import Union

import numpy as np
import pandas as pd


__all__ = ["Condition"]


class Condition:
    def __init__(
        self,
        condition: Union[pd.Index, pd.Series] = None,
        index: pd.Index = None,
    ):
        self.index = None
        self.intervals = pd.IntervalIndex([])

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
            self.intervals = self.calc_intervals(condition)
        if index is not None:
            if not isinstance(index, pd.Index):
                raise TypeError(
                    "condition must be of type `pd.Index`, "
                    f"`{type(index)}` was passed"
                )
            self.validate_index(index)
            self.index = index

    def _is_datetime_type(self) -> bool:
        return self.intervals.dtype.subtype == np.dtype(
            "datetime64[ns]"
        ) or isinstance(self.index, pd.DatetimeIndex)

    def _is_none(self, obj) -> List[bool]:
        return [x is None for x in list(obj)]

    def any_nons(self, obj) -> bool:
        return any(self._is_none(obj))

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

    def validate_condition(self, obj: "Condition") -> None:
        self.validate_index(obj.index)

    def calc_intervals(self, obj: pd.Series) -> pd.IntervalIndex:
        groups = (obj != obj.shift(1)).cumsum()

        intervals = groups.groupby(groups[obj]).apply(
            lambda x: (x.index.min(), x.index.max())
        )

        if intervals.empty:
            return pd.IntervalIndex([])

        return pd.IntervalIndex.from_tuples(intervals, closed="left")

    def mask(self, s: Union[pd.Series, pd.DataFrame] = None) -> pd.Series:
        idx = self.index if s is None else s.index
        cat = pd.cut(idx, self.intervals)
        return pd.Series(data=~cat.isna(), index=self.index)

    @staticmethod
    def _reduce_intervals(s: pd.IntervalIndex) -> pd.IntervalIndex:
        s = sorted(s, key=lambda x: x.left)

        m = 0
        for t in s:
            if t.left > s[m].right:
                m += 1
                s[m] = t
            else:
                s[m] = pd.Interval(
                    s[m].left, max(s[m].right, t.right), t.closed
                )
        return pd.IntervalIndex(s[: m + 1])

    def _or(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        if isinstance(other, pd.Series):
            other = self.reproduce(condition=other)
            self.validate_index(other.index)
        s = self.intervals.append(other.intervals)

        result = self.copy()
        result.index = self.index if other.index is None else other.index
        result.intervals = Condition._reduce_intervals(s)

        return result

    def __or__(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        return self._or(other)

    def union(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        return self._or(other)

    def _and(self, other: Union[pd.Series, "Condition"]) -> "Condition":
        if isinstance(other, pd.Series):
            other = self.reproduce(condition=other)
            self.validate_index(other.index)
        intervals = sorted(other.intervals, key=lambda x: x.left)

        result = self.copy()
        result.index = self.index if other.index is None else other.index

        if self.intervals is None or other.intervals is None:
            result.intervals = pd.IntervalIndex([])
        else:
            r = []
            for i in intervals:
                overlaps = self.intervals.overlaps(i)
                if any(overlaps):
                    r += [
                        pd.Interval(
                            max(j.left, i.left),
                            min(j.right, i.right),
                            i.closed,
                        )
                        for j in list(compress(self.intervals, overlaps))
                    ]
            result.intervals = pd.IntervalIndex(r)
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
        intervals = sorted(self.intervals, key=lambda x: x.left)
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
        result.intervals = pd.IntervalIndex(r)

        return result

    def __invert__(self) -> "Condition":
        return self._not()

    def shift_intervals(self, left, right) -> pd.IntervalIndex:
        s = pd.IntervalIndex(
            [
                pd.Interval(x.left + left, x.right + right, x.closed)
                for x in self.intervals
            ]
        )

        return Condition._reduce_intervals(s)

    def move(self, value: object) -> "Condition":
        result = self.copy()

        if self._is_datetime_type():
            value = pd.Timedelta(value)
        result.intervals = self.shift_intervals(value, value)

        return result

    def shrink(self, value: object) -> "Condition":
        result = self.copy()
        try:
            if self._is_datetime_type():
                value = pd.Timedelta(value)
            result.intervals = self.shift_intervals(value, -value)
        except ValueError:
            raise ValueError("some intervals are to short, filter first")
        return result

    def grow(self, value: object) -> "Condition":
        if self._is_datetime_type():
            value = pd.Timedelta(value)
        return self.shrink(-value)

    def copy(self):
        return copy.deepcopy(self)

    def plot(self, s: pd.Series, *args, figsize=(16, 4), **kwargs) -> None:
        from matplotlib import pyplot as plt

        _, ax = plt.subplots(*args, figsize=figsize, **kwargs)

        s.plot(ax=ax)

        ax.fill_between(
            s.index,
            0,
            1,
            where=self.mask(s),
            alpha=0.3,
            transform=ax.get_xaxis_transform(),
        )

    def stack(self, df: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if len(self.intervals) == 0:
            raise ValueError("No intervals to stack")
        print(self.intervals)

        df["interval"] = pd.cut(df.index, self.intervals)
        df = df.dropna(subset=["interval"])

        dfs = []
        for _, data in df.groupby("interval"):
            data.index = data.index - data.index.min()
            dfs.append(data)
        return pd.concat(dfs)

    def stack_plot(
        self, s: pd.Series, *args, figsize=(16, 4), **kwargs
    ) -> None:
        if not isinstance(s, pd.Series):
            raise TypeError("stack_plot only supports `pd.Series1`")
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
        result.intervals = pd.IntervalIndex(
            [x for x in self.intervals if op(x.length, value)]
        )

        return result

    def to_frame(self):
        return pd.DataFrame(
            {
                "left": [i.left for i in self.intervals],
                "right": [i.right for i in self.intervals],
                "closed": [i.closed for i in self.intervals],
                "length": [i.length for i in self.intervals],
            }
        )

    def __repr__(self):
        return self.to_frame().__repr__()

    def __len__(self):
        return len(self.intervals)

    def _index_filter(
        self, operator, value: Union[int, float, time, datetime]
    ) -> "Condition":
        result = self.copy()

        if isinstance(value, time):
            ref = self.index.time
        else:
            ref = self.index
        mask = pd.Series(operator(ref, value), self.index)
        result.intervals = self.calc_intervals(mask)

        if len(self.intervals) == 0:
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
