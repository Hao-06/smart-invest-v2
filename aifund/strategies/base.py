"""**策略统一接口** —— 所有策略遵守同样 API，方便 Meta-Agent 动态切换。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass
class StrategyOutput:
    """策略输出 = 持仓清单 + 仓位比例 + 解释。"""
    symbols: list[str]                    # 持仓股票代码（满仓 100% / 半仓 50% / 空仓 0%）
    position_ratio: float                 # 总仓位（0.0 = 空仓 / 1.0 = 满仓）
    reasoning: str                        # 为何这么选（给评委 / 用户看）
    extras: dict[str, Any] | None = None  # 任意扩展字段


class BaseStrategy(ABC):
    """所有策略的基类。每个具体策略实现 4 个方法 + 3 个属性。"""

    #: 策略名（与 Meta-Agent 沟通用的 ID）
    name: str = "base"

    #: 给 Strategy Selector Agent 看的一句话描述
    description: str = "Base strategy interface"

    #: 适用市场场景标签（让 Selector Agent 知道什么时候选这个策略）
    #: 可选值：trending_bull / trending_bear / range_bound / volatile / breakout / unknown
    suitable_regimes: list[str] = []

    def __init__(self, pipeline, **kwargs: Any) -> None:
        self.pipeline = pipeline
        # 子类自行接收剩余 kwargs

    @abstractmethod
    def select(self, as_of: date) -> StrategyOutput:
        """返回 as_of 日的持仓建议。"""
        raise NotImplementedError

    def precompute_weekly_pools(
        self,
        start_date: date,
        end_date: date,
        verbose: bool = True,
    ) -> tuple[dict[date, list[str]], dict[date, float], dict[date, str]]:
        """预计算 [start, end] 区间内每周首个交易日的：
        - pools: dict[date, list[symbol]]
        - ratios: dict[date, float]
        - reasoning: dict[date, str]
        """
        cal = self.pipeline.calendar
        all_dates = cal.range(start_date, end_date)
        pools: dict[date, list[str]] = {}
        ratios: dict[date, float] = {}
        reasoning: dict[date, str] = {}
        last_week_key: tuple[int, int] | None = None
        for d in all_dates:
            year, week, _ = d.isocalendar()
            week_key = (year, week)
            if week_key != last_week_key:
                out = self.select(d)
                pools[d] = out.symbols
                ratios[d] = out.position_ratio
                reasoning[d] = out.reasoning
                if verbose:
                    print(f"[{self.name}] {d} → {out.symbols[:3]}... "
                          f"(ratio={out.position_ratio:.2f}) {out.reasoning[:50]}",
                          flush=True)
                last_week_key = week_key
        return pools, ratios, reasoning
