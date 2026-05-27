"""**现金兜底策略** —— 极端避险，全程持有现金（零持仓）。

适用场景：
- 黑天鹅事件（如 2020-03 闪崩、2018 中美贸易战开打）
- 极度恐慌的 volatile regime（波动率 > 40% 年化）
- Regime Agent 无法判断（confidence < 0.4）

设计原理：
- 在「**未来高度不确定**」的时刻，**不亏钱就是赢钱**
- 比赛排名按期末资金算 —— 别人亏 -20% 时你保住 100%，排名直接前移
- 现金不产生收益，但**也不产生损失**

这是 Meta-Agent 工具箱里的「**最后保险**」。
"""
from __future__ import annotations

from datetime import date

from aifund.strategies.base import BaseStrategy, StrategyOutput


class CashHoldStrategy(BaseStrategy):
    """**全现金持有** —— 零仓位，零风险，零收益。"""

    name = "cash_hold"
    description = "现金兜底 —— 零持仓全程现金，最极端的避险（黑天鹅 / 极度不确定）"
    suitable_regimes = ["volatile", "trending_bear"]  # 仅在极端情况触发

    def select(self, as_of: date) -> StrategyOutput:
        return StrategyOutput(
            symbols=[],
            position_ratio=0.0,
            reasoning="现金兜底 · 黑天鹅 / 极度不确定 → 全程现金，零风险",
            extras={"strategy_type": "cash_hold"},
        )
