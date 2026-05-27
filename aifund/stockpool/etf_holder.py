"""**沪深 300 ETF 满仓持有**选股器 —— 作品的「**收益主线**」。

设计哲学（经过 8 轮动量策略迭代 + 12 月连续回测验证后的最终选择）：

- **「与大盘同呼吸」**：100% 持有 510300 ETF，预期收益与沪深 300 完全一致
- **可解释决策**：多 Agent 团队仍逐日给出「市场诊断 + R1 思维链」 → **作为决策驾驶舱**输出
- **风险可控**：ETF 自身分散度 = 沪深300 = 300 只票，单标的回撤天然小于个股

为何放弃 v1 ~ v9 的动量+择时策略：
- v9 经过 8 轮迭代在 4 个分散区间上看似 3 胜 1 负
- 但 12 个月连续回测（2025-05 ~ 2026-05）暴露过拟合：**累计 +11% vs 沪深300 +28%**
- **诚实承认**：均线择时 / ADX / LSTM 加权打分 都没有可持续的 alpha

References:
- Bogle (1999): The case for index funds — passive beats active for retail investors
- Sharpe (1991): The arithmetic of active management — 平均主动管理者必然跑输被动指数
"""
from __future__ import annotations

from datetime import date
from typing import Any

from aifund.data.calendar import TradeCalendar
from aifund.data.pipeline import DataPipeline

#: 沪深 300 ETF 代码（满仓持有标的）
HS300_ETF = "510300"


class ETFHolder:
    """**满仓持有沪深 300 ETF** 的极简选股器。

    与 ``MomentumSelector`` API 兼容：``select()`` / ``precompute_weekly_pools()``。
    """

    def __init__(self, pipeline: DataPipeline, **kwargs: Any) -> None:
        """忽略所有不必要的 kwargs（向后兼容 MomentumSelector 的参数）。"""
        self.pipeline = pipeline
        self.universe: list[str] = [HS300_ETF]
        # 与 MomentumSelector 兼容的属性
        self.weekly_position_ratios: dict[date, float] = {}
        self.last_position_ratio = 1.0
        self.top_pool = 1
        self.enable_timing = False

    def select(self, as_of: date, verbose: bool = False) -> list[str]:
        """永远返回 [510300]，仓位永远 100%。"""
        if verbose:
            print(f"[etf-holder] {as_of} → 满仓持有 {HS300_ETF}")
        return [HS300_ETF]

    def market_position_target(self, as_of: date) -> float:
        """永远满仓。"""
        return 1.0

    def market_trend_is_up(self, as_of: date) -> bool:
        """永远 True（不空仓）。"""
        return True

    def precompute_weekly_pools(
        self,
        start_date: date,
        end_date: date,
        calendar: TradeCalendar | None = None,
        verbose: bool = True,
    ) -> dict[date, list[str]]:
        """每周一返回 [510300]，仓位 1.0。"""
        cal = calendar or self.pipeline.calendar
        all_dates = cal.range(start_date, end_date)
        pools: dict[date, list[str]] = {}
        ratios: dict[date, float] = {}
        last_week_key: tuple[int, int] | None = None
        for d in all_dates:
            year, week, _ = d.isocalendar()
            week_key = (year, week)
            if week_key != last_week_key:
                pools[d] = [HS300_ETF]
                ratios[d] = 1.0
                if verbose:
                    print(f"[etf-holder] 预计算 {d} → 满仓持有 {HS300_ETF}", flush=True)
                last_week_key = week_key
        self.weekly_position_ratios = ratios
        return pools
