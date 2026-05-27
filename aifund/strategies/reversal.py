"""**反转/抄底策略** —— 在超跌/低估时反向操作。

适用场景：
- 熊市末段（连续下跌后超跌反弹概率大）
- 震荡市底部（情绪极度悲观时）
- 单日暴跌后（V 形反弹）

设计原理：
- **行为金融学**：散户在恐慌时过度卖出（DeBondt & Thaler 1985 "Does the Stock Market Overreact?"）
- **均值回归**：短期超跌的股票未来 1-3 个月跑赢市场（Jegadeesh 1990 短期反转因子）
- **价值锚定**：估值低（高 BP）+ 短期超跌（REV_5 极负）= 真正的抄底机会

选股逻辑：
1. 沪深 300 成分股 + 因子完整
2. 综合分 = α × 短期反转分（-REV_5）+ β × 价值分（BP）+ γ × 低波分（1 - VOL_60）
3. 取 top 8 等权持有

**风险声明**：反转策略在「真正的熊市趋势」里会持续亏损（接飞刀）。
**Meta-Agent** 应该只在 ``trending_bear`` 末段 / ``range_bound`` 超跌 / ``volatile`` 后启用，
而不是在 ``trending_bull`` 牛市里用（违背动量趋势）。
"""
from __future__ import annotations

from datetime import date

from aifund.stockpool.factor_loader import FactorLoader
from aifund.strategies.base import BaseStrategy, StrategyOutput


class ReversalStrategy(BaseStrategy):
    """反转/抄底 —— 选超跌 + 低估 + 低波动的 top 8。"""

    name = "reversal"
    description = "反转/抄底 —— 短期超跌 + 估值低 + 低波动的龙头股（适合熊市末/震荡底/暴跌后）"
    suitable_regimes = ["trending_bear", "range_bound", "volatile"]

    def __init__(self, pipeline, top_n: int = 8,
                 reversal_weight: float = 0.5,
                 value_weight: float = 0.3,
                 lowvol_weight: float = 0.2,
                 universe: list[str] | None = None,
                 **kwargs) -> None:
        super().__init__(pipeline, **kwargs)
        self.top_n = top_n
        self.weights = (reversal_weight, value_weight, lowvol_weight)
        self.universe = set(universe) if universe else None
        self.loader = FactorLoader()

    def select(self, as_of: date) -> StrategyOutput:
        factor_df = self.loader.factors_by_date(as_of)
        if factor_df.empty:
            return StrategyOutput(
                symbols=[], position_ratio=0.0,
                reasoning="反转策略 · CSV 中无 as_of 日数据，跳过",
            )

        # 需要 3 个因子：REV_5 (短期反转) + BP (价值) + VOL_60 (波动率)
        needed = {"Factor_REV_5", "Factor_BP", "Factor_VOL_60"}
        if not needed.issubset(factor_df.columns):
            return StrategyOutput(
                symbols=[], position_ratio=0.0,
                reasoning=f"反转策略 · CSV 缺少必要因子（需 {needed}）",
            )
        sub = factor_df[["symbol", "Factor_REV_5", "Factor_BP", "Factor_VOL_60"]].dropna()
        if self.universe:
            sub = sub[sub["symbol"].isin(self.universe)]
        if sub.empty:
            return StrategyOutput(
                symbols=[], position_ratio=0.0,
                reasoning="反转策略 · 当日因子全空 / universe 过滤后为空",
            )

        # 综合分 = α×(-REV_5 排名) + β×(BP 排名) + γ×(低波动率排名)
        # -REV_5 高 = 跌得多 → 反转分高
        # BP 高 = PB 低 → 价值分高
        # 1 - VOL_60 排名 = 波动率低 → 防御性强
        a, b, g = self.weights
        sub["reversal_rank"] = (-sub["Factor_REV_5"]).rank(pct=True)
        sub["value_rank"] = sub["Factor_BP"].rank(pct=True)
        sub["lowvol_rank"] = 1 - sub["Factor_VOL_60"].rank(pct=True)
        sub["score"] = (
            a * sub["reversal_rank"]
            + b * sub["value_rank"]
            + g * sub["lowvol_rank"]
        )
        sub = sub.sort_values("score", ascending=False)
        picks = sub.head(self.top_n)["symbol"].tolist()

        return StrategyOutput(
            symbols=picks,
            position_ratio=1.0,
            reasoning=f"反转/抄底 · 综合分（{a:.1f}反转 + {b:.1f}价值 + {g:.1f}低波）选 top {self.top_n}: "
                      f"{','.join(picks)}",
            extras={"strategy_type": "reversal"},
        )
