"""**高股息防御策略** —— 选股息率最高的 8 只稳健蓝筹。

适用场景：
- 熊市（高股息提供安全垫，下跌阻力大）
- 长期震荡市（股息复利效应放大）
- 利率下行（高股息相对债券吸引力增加）

数据源：
- 优先用 Hao 多因子 CSV 里的股息率字段
- 若 CSV 无此字段，回退到「市值大 + PB 低」的代理（蓝筹倾向）
"""
from __future__ import annotations

from datetime import date

from aifund.stockpool.factor_loader import FactorLoader
from aifund.strategies.base import BaseStrategy, StrategyOutput


class HighDividendStrategy(BaseStrategy):
    """高股息防御 —— 选股息率 top 8 的蓝筹股。"""

    name = "high_dividend"
    description = "高股息防御 —— 选股息率 top 8 稳健蓝筹（熊市/长期震荡的避风港）"
    suitable_regimes = ["trending_bear", "range_bound", "volatile"]

    def __init__(self, pipeline, top_n: int = 8,
                 universe: list[str] | None = None, **kwargs) -> None:
        super().__init__(pipeline, **kwargs)
        self.top_n = top_n
        self.universe = set(universe) if universe else None
        self.loader = FactorLoader()

    def select(self, as_of: date) -> StrategyOutput:
        factor_df = self.loader.factors_by_date(as_of)
        if factor_df.empty:
            return StrategyOutput(
                symbols=[], position_ratio=0.0,
                reasoning="高股息策略 · CSV 中无 as_of 日数据，跳过",
            )

        # CSV 没有股息率字段，用三因子代理蓝筹（v5 验证版本）：
        # - 高 BP（账面市值比高 = PB 低 = 价值股）
        # - 高 CFP（现金流市值比高 = 能分红）
        # - 低 VOL_60（60 日波动率低 = 稳健蓝筹）
        #
        # 注：曾尝试 9A 加入 SP + Turn_20 + MaxRet_20 等共 6 因子，但 2026 H1 退化 -0.73 pp，故回退此版本。
        needed = {"Factor_BP", "Factor_CFP", "Factor_VOL_60"}
        if not needed.issubset(factor_df.columns):
            return StrategyOutput(
                symbols=[], position_ratio=0.0,
                reasoning=f"高股息策略 · CSV 缺少必要因子字段（需 {needed}）",
            )
        sub = factor_df[["symbol", "Factor_BP", "Factor_CFP", "Factor_VOL_60"]].dropna()
        if self.universe:
            sub = sub[sub["symbol"].isin(self.universe)]
        if sub.empty:
            return StrategyOutput(
                symbols=[], position_ratio=0.0,
                reasoning="高股息策略 · 当日所有股票因子均缺失 / universe 过滤后为空",
            )
        # 综合分 = BP 排名 + CFP 排名 + (1 - VOL 排名)  → 越大越「蓝筹+价值+低波动」
        sub["bp_rank"] = sub["Factor_BP"].rank(pct=True)
        sub["cfp_rank"] = sub["Factor_CFP"].rank(pct=True)
        sub["lowvol_rank"] = 1 - sub["Factor_VOL_60"].rank(pct=True)
        sub["score"] = sub["bp_rank"] + sub["cfp_rank"] + sub["lowvol_rank"]
        sub = sub.sort_values("score", ascending=False)
        picks = sub.head(self.top_n)["symbol"].tolist()

        return StrategyOutput(
            symbols=picks,
            position_ratio=1.0,
            reasoning=f"高股息防御 · 三因子代理（高 BP + 高 CFP + 低波动）选 top {self.top_n}: "
                      f"{','.join(picks)}",
            extras={"strategy_type": "dividend_defense"},
        )
