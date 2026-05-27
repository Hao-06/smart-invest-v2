"""**策略库** —— 多策略库 + 统一接口，供 Meta-Strategy Agent 动态选择。

设计哲学：
- 每个策略都遵守 ``BaseStrategy`` 接口（``select`` / ``precompute_weekly_pools`` / ``description``）
- 每个策略有明确的「适用场景」标签，给 Strategy Selector Agent 看
- Meta-Agent 在每周决策时调用 Regime Agent 识别市场状态，然后选最匹配策略

策略列表（v1）：
1. ``MomentumStrategy``      —— 动量+择时（适合趋势 / 牛市主线）
2. ``ETFDefenseStrategy``    —— 沪深 300 ETF 满仓（适合震荡市 / 不确定）
3. ``SectorRotationStrategy``—— 行业轮动（适合主线明确的牛市）
4. ``HighDividendStrategy``  —— 高股息防御（适合熊市 / 长期震荡）
"""
from aifund.strategies.base import BaseStrategy, StrategyOutput
from aifund.strategies.momentum import MomentumStrategy
from aifund.strategies.etf_defense import ETFDefenseStrategy
from aifund.strategies.sector_rotation import SectorRotationStrategy
from aifund.strategies.high_dividend import HighDividendStrategy
from aifund.strategies.reversal import ReversalStrategy
from aifund.strategies.cash_hold import CashHoldStrategy
from aifund.strategies.bond_etf import BondETFStrategy

__all__ = [
    "BaseStrategy",
    "StrategyOutput",
    "MomentumStrategy",
    "ETFDefenseStrategy",
    "SectorRotationStrategy",
    "HighDividendStrategy",
    "ReversalStrategy",
    "CashHoldStrategy",
    "BondETFStrategy",
]


def all_strategies(pipeline, universe: list[str] | None = None,
                   include_experimental: bool = False) -> dict[str, BaseStrategy]:
    """返回**核心 4 策略**字典 {name: instance}，供 Meta-Agent / All-Weather 使用。

    - **默认**返回 4 个**经过验证的核心策略**（v1 配置，已验证 6 区间 4 胜 2 负 + 2025 +45.97%）：
      momentum / etf_defense / sector_rotation / high_dividend
    - ``include_experimental=True`` 时**额外**返回 3 个实验策略（reversal / cash_hold / bond_etf）
      —— v5 实验证明这 3 个会让 Selector 误判，**默认不启用**

    Args:
        pipeline: 共享数据管道。
        universe: 可选；个股策略只从这个列表选股。
        include_experimental: 是否包含实验策略（默认 False，保守稳健）。
    """
    base = {
        "momentum": MomentumStrategy(pipeline=pipeline, universe=universe),
        "etf_defense": ETFDefenseStrategy(pipeline=pipeline),
        "sector_rotation": SectorRotationStrategy(pipeline=pipeline),
        "high_dividend": HighDividendStrategy(pipeline=pipeline, universe=universe),
    }
    if include_experimental:
        base["reversal"] = ReversalStrategy(pipeline=pipeline, universe=universe)
        base["cash_hold"] = CashHoldStrategy(pipeline=pipeline)
        base["bond_etf"] = BondETFStrategy(pipeline=pipeline)
    return base
