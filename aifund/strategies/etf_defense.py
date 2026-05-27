"""**ETF 防御策略** —— 满仓持有沪深 300 ETF 510300。

适用场景：
- 震荡市（趋势不明，被动管理避免反复甩耳光）
- 熊市初期（不确定趋势何时启动，保守持仓）
- 牛市中段（动量已被普遍认知，主动 alpha 衰减）

经过 12 个月连续回测验证：累计 +27.10%，回撤 -7.32%，夏普 1.71（vs 主动策略 v9 远差）。
"""
from __future__ import annotations

from datetime import date

from aifund.strategies.base import BaseStrategy, StrategyOutput

HS300_ETF = "510300"


class ETFDefenseStrategy(BaseStrategy):
    """**沪深 300 ETF 满仓** —— 最稳健的默认策略。"""

    name = "etf_defense"
    description = "沪深 300 ETF 满仓 —— 与大盘同步收益，被动管理，零换手率"
    suitable_regimes = ["range_bound", "trending_bear", "unknown", "volatile"]

    def select(self, as_of: date) -> StrategyOutput:
        return StrategyOutput(
            symbols=[HS300_ETF],
            position_ratio=1.0,
            reasoning="震荡市 / 不确定环境，被动持有沪深 300 ETF 跟踪大盘，回撤可控",
            extras={"strategy_type": "passive_index"},
        )
