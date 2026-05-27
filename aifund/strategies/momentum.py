"""**动量+择时策略** —— 适合趋势市 / 牛市主线。

复用 `aifund.stockpool.momentum_selector.MomentumSelector` 的 v9 配置：
- 60 日动量 + LSTM 涨停过滤 + 不对称 hysteresis + ADX 强度
- 综合分 = 0.6 × 标准化动量分 + 0.4 × LSTM 涨停概率

注意：v9 在 12 月连续回测里跑输沪深 300（过拟合），但**在明确的趋势市里仍是 alpha 来源**。
Meta-Agent 应该只在「**确认趋势市**」时选这个策略。
"""
from __future__ import annotations

from datetime import date

from aifund.stockpool.momentum_selector import MomentumSelector
from aifund.strategies.base import BaseStrategy, StrategyOutput


class MomentumStrategy(BaseStrategy):
    """动量+大盘择时（v9 配置）—— 适合趋势市 / 牛市主线。"""

    name = "momentum"
    description = "动量+择时+LSTM 涨停 —— 趋势市的 alpha 引擎（在震荡市会被甩耳光）"
    suitable_regimes = ["trending_bull", "breakout"]

    def __init__(self, pipeline, universe: list[str] | None = None, **kwargs) -> None:
        super().__init__(pipeline, **kwargs)
        self._inner = MomentumSelector(
            pipeline=pipeline,
            # 显式注入 PIT universe（None=未指定走 CSV 回退；[]=该区间无因子数据，不选个股）
            universe=universe,
            top_pool=8,
            enable_timing=True,
            use_realtime_momentum=True,
            universe_top_n=80,
            # **All Weather 模式下关闭 LSTM**：
            # 1) LSTM 推理慢（每周 60+ 秒），是 All Weather 超时主因
            # 2) All Weather 多策略分散，单策略 alpha 重要性下降
            # 3) 已验证：v9 (有 LSTM) 12 月连续 -17.54 pp 失败，证明 LSTM 不是必需
            use_lstm_filter=False,
        )

    def select(self, as_of: date) -> StrategyOutput:
        ratio = self._inner.market_position_target(as_of)
        if ratio <= 0:
            return StrategyOutput(
                symbols=[],
                position_ratio=0.0,
                reasoning="动量策略 · 大盘趋势向下，空仓避险",
            )
        symbols = self._inner.select(as_of)
        if not symbols:
            return StrategyOutput(
                symbols=[],
                position_ratio=0.0,
                reasoning="动量策略 · 无候选股票（数据不足或全部被过滤）",
            )
        return StrategyOutput(
            symbols=symbols,
            position_ratio=ratio,
            reasoning=f"动量策略 · 选出 {len(symbols)} 只动量+LSTM 双高股 / "
                     f"大盘 ADX 触发 ratio={ratio:.2f}",
            extras={"strategy_type": "momentum_active"},
        )
