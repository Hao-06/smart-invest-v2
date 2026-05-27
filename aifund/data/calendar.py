"""交易日历工具。

封装 A股交易日的常用查询：是否交易日、前/后第 N 个交易日、区间交易日列表。
回测引擎按交易日推进时间，决策模块据此确定「最近可用交易日」。
"""
from __future__ import annotations

import bisect
from datetime import date, datetime
from typing import Iterable

from aifund.data.sources import get_trade_calendar


def as_date(d: str | date | datetime) -> date:
    """把字符串 / datetime 统一转成 date。"""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d).replace("/", "-")[:10], "%Y-%m-%d").date()


class TradeCalendar:
    """A股交易日历。"""

    def __init__(self, dates: Iterable[date] | None = None) -> None:
        if dates is None:
            cal = get_trade_calendar()
            dates = list(cal["date"]) if not cal.empty else []
        self._dates: list[date] = sorted({as_date(d) for d in dates})
        self._set = set(self._dates)

    @property
    def dates(self) -> list[date]:
        return list(self._dates)

    def __len__(self) -> int:
        return len(self._dates)

    def is_trade_day(self, d: str | date) -> bool:
        return as_date(d) in self._set

    def latest(self, on_or_before: str | date) -> date | None:
        """不晚于给定日期的最近一个交易日（含当日）。"""
        d = as_date(on_or_before)
        i = bisect.bisect_right(self._dates, d)
        return self._dates[i - 1] if i > 0 else None

    def prev(self, d: str | date, n: int = 1) -> date | None:
        """给定日期之前的第 n 个交易日（不含当日）。"""
        d = as_date(d)
        i = bisect.bisect_left(self._dates, d)
        j = i - n
        return self._dates[j] if 0 <= j < len(self._dates) else None

    def next(self, d: str | date, n: int = 1) -> date | None:
        """给定日期之后的第 n 个交易日（不含当日）。"""
        d = as_date(d)
        i = bisect.bisect_right(self._dates, d)
        j = i + n - 1
        return self._dates[j] if 0 <= j < len(self._dates) else None

    def range(self, start: str | date, end: str | date) -> list[date]:
        """闭区间 [start, end] 内的全部交易日。"""
        s, e = as_date(start), as_date(end)
        lo = bisect.bisect_left(self._dates, s)
        hi = bisect.bisect_right(self._dates, e)
        return self._dates[lo:hi]
