"""**国债 ETF 防御策略** —— 在长期熊市 / 利率下行时持有国债 ETF。

适用场景：
- 长期熊市（A 股普跌，债券作为避风港）
- 利率下行预期（货币宽松周期）
- 极端避险（比现金多 2-3% 年化收益）

候选标的（按规模 / 流动性排序）：
- 511010 国债 ETF（华夏 5 年期国债 ETF，最主流）
- 511260 十年国债 ETF（鹏华，长久期，更敏感）
- 511180 短债 ETF（汇添富，短久期，最稳）

默认用 511010（流动性最好，A 股个人投资者最熟悉）。

设计原理：
- 「**股债跷跷板**」—— A 股大跌时债券倾向于上涨（避险资金涌入）
- 国债 ETF 年化波动率 2-3%（远低于股票 ETF 的 20%）
- 在比赛 30+ 天里若遇到熊市，国债 ETF 至少**保住本金 + 小幅正收益**
"""
from __future__ import annotations

from datetime import date

from aifund.strategies.base import BaseStrategy, StrategyOutput

#: 国债 ETF 标的（华夏 5 年期国债 ETF）
BOND_ETF_CODE = "511010"


class BondETFStrategy(BaseStrategy):
    """**国债 ETF 满仓** —— 长期熊市 / 利率下行的避风港。"""

    name = "bond_etf"
    description = "国债 ETF 满仓 —— 511010 五年期国债，长期熊市/利率下行的避险选择"
    suitable_regimes = ["trending_bear"]  # 仅在确定熊市趋势时启用

    def select(self, as_of: date) -> StrategyOutput:
        return StrategyOutput(
            symbols=[BOND_ETF_CODE],
            position_ratio=1.0,
            reasoning=f"国债 ETF 防御 · 长期熊市/利率下行 → 满仓 {BOND_ETF_CODE} 避险",
            extras={"strategy_type": "bond_defense", "etf_code": BOND_ETF_CODE},
        )
