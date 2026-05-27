"""加载 Hao 前期多因子打分 CSV，提供按日期的 top-N 查询。

CSV 来自 ``~/量化金融/多因子策略优化-re/5.2 沪深300_剔除垃圾股_含分数_最终版.csv``
覆盖沪深300 成分股 2024-01 ~ 2025-12 的日频因子与综合 score。

这是「**学生本人前期研究成果的资产复用**」—— Hao 之前算好的 13 个量化因子
（BP/SP/CFP/REV/MOM/VOL/Turn/MaxRet/VSTD 等）+ 综合 score 直接作为选股漏斗的第 1 层粗筛。
"""
from __future__ import annotations

import bisect
from datetime import date
from pathlib import Path

import pandas as pd

# 默认 CSV 路径（指向 Hao 量化金融文件夹）
DEFAULT_CSV = (
    Path.home() / "量化金融" / "多因子策略优化-re"
    / "5.2 沪深300_剔除垃圾股_含分数_最终版.csv"
)

#: 只读这些列，避免加载整个 55MB CSV 浪费内存
_USED_COLS = [
    "代码", "简称", "日期", "score", "is_junk",
    # 13 个原始因子，便于自定义复合打分
    "Factor_BP", "Factor_SP", "Factor_CFP",
    "Factor_REV_5", "Factor_REV_10",
    "Factor_MOM_60", "Factor_MOM_120",
    "Factor_VOL_20", "Factor_VOL_60",
    "Factor_MaxRet_20", "Factor_Turn_20", "Factor_Turn_60",
    "Factor_VSTD_20",
]


class FactorLoader:
    """多因子打分 CSV 的内存索引。"""

    def __init__(self, csv_path: Path | str | None = None) -> None:
        path = Path(csv_path) if csv_path else DEFAULT_CSV
        if not path.exists():
            raise FileNotFoundError(
                f"多因子 CSV 不存在：{path}\n"
                f"请确认 Hao 的量化金融文件夹存在并包含该文件。"
            )
        self.csv_path: Path = path
        self._df: pd.DataFrame = self._load()
        self._dates: list[date] = sorted(self._df["日期"].unique().tolist())

    # ------------------------------------------------------------------
    # 内部加载
    # ------------------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, usecols=_USED_COLS, low_memory=False)
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.date
        df = df.dropna(subset=["日期"]).copy()
        df = df[~df["is_junk"]]  # 剔除垃圾股
        # 代码归一化：'000001.SZ' / '600000.SH' → '000001' / '600000'
        df["symbol"] = df["代码"].astype(str).str[:6]
        df = df.dropna(subset=["score"])
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 对外查询
    # ------------------------------------------------------------------
    @property
    def date_range(self) -> tuple[date, date]:
        """数据覆盖的日期范围。"""
        return self._dates[0], self._dates[-1]

    @property
    def n_stocks(self) -> int:
        """覆盖的去重股票数量。"""
        return self._df["symbol"].nunique()

    def _nearest_date(self, target: date) -> date | None:
        """取 <= target 的最近交易日；超出范围返回 None。"""
        if not self._dates or target < self._dates[0]:
            return None
        if target >= self._dates[-1]:
            return self._dates[-1]
        # 二分查找：bisect_right - 1
        i = bisect.bisect_right(self._dates, target)
        return self._dates[i - 1] if i > 0 else None

    def top_n_by_score(self, as_of: date, n: int = 20) -> pd.DataFrame:
        """取 as_of（或最近一日）的 top-N 标的，按综合 score 降序。

        Returns:
            DataFrame，含 ``symbol``、``简称``、``日期``、``score`` 列。
            找不到合适日期时返回空 DataFrame。
        """
        d = self._nearest_date(as_of)
        if d is None:
            return pd.DataFrame(columns=["symbol", "简称", "日期", "score"])
        day_df = self._df[self._df["日期"] == d]
        return day_df.nlargest(n, "score")[["symbol", "简称", "日期", "score"]].reset_index(drop=True)

    def pit_top_n(self, as_of: date, n: int = 80,
                  min_complete: int = 100) -> pd.DataFrame:
        """PIT 正确地取 universe：找 **≤ as_of 且当日有效股票数 ≥ min_complete**
        的最近一个交易日，返回该日 top-N（按 score 降序）。

        比 ``top_n_by_score`` 更稳健 —— **跳过 CSV 边界的稀疏日**（数据末端常出现
        每天只有个别票的脏日，直接取会得到残缺的 1 只票 universe）。
        找不到合适日期（如 as_of 早于数据起点）时返回空 DataFrame。
        """
        counts = self._df.groupby("日期").size()
        complete = sorted(d for d, c in counts.items()
                          if c >= min_complete and d <= as_of)
        if not complete:
            return pd.DataFrame(columns=["symbol", "简称", "日期", "score"])
        day_df = self._df[self._df["日期"] == complete[-1]]
        return day_df.nlargest(n, "score")[["symbol", "简称", "日期", "score"]].reset_index(drop=True)

    def factors_by_date(self, as_of: date) -> pd.DataFrame:
        """取 as_of（或最近一日）的全部标的 + 13 个原始因子。

        供后续做**自定义复合打分**用 —— 不强依赖 Hao 简单的 PE+EP score。
        """
        d = self._nearest_date(as_of)
        if d is None:
            return pd.DataFrame()
        return self._df[self._df["日期"] == d].reset_index(drop=True)
