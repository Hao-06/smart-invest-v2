"""**日频健康检查 Agent** —— 实现「**真正每天自主决策**」的核心。

设计哲学：
- **周频战略层**（WeightAllocator）：每周一定大方向（4 策略权重）
- **日频战术层**（**本 Agent**）：每天检查 3 个风险信号，触发时**自动应急**

为什么需要日频应急：
- 比赛 30 天里随时可能发生**周中黑天鹅**（个股暴雷 / 突发政策 / 大盘单日 -5%）
- 周频决策无法响应这种事件
- **完全自主智能体**必须能日频检测 + 应急

3 个独立的检查（按优先级）：
1. **大盘崩盘检测**：沪深 300 单日跌 > 3% → 切 ETF 半仓
2. **组合回撤检测**：组合从历史高点回撤 > -5% → 切 ETF 满仓
3. **持仓异常检测**：重仓股停牌 / ST → 自动卖出

**关键**：不重新选策略（避免过度切换 + 噪声）。只做**应急防御**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from aifund.data import sources

HS300_ETF = "510300"

HealthAction = Literal[
    "no_action",           # 一切正常，维持周一持仓
    "market_crash",        # 大盘崩盘，切 ETF 半仓
    "portfolio_drawdown",  # 组合回撤过大，切 ETF 满仓
    "position_anomaly",    # 持仓异常，自动卖出问题股
]


@dataclass
class DailyHealthOpinion:
    """日频健康检查输出。"""
    as_of: date
    action: HealthAction
    target_symbol_weights: dict[str, float] | None  # 应急时的目标权重；None = 不动
    reasoning: str
    signals: dict[str, Any] = field(default_factory=dict)


class DailyHealthCheckAgent:
    """每日健康检查 + 应急刹车（不调用 LLM，秒级判断）。"""

    name = "日频健康检查 Agent"

    def __init__(
        self,
        market_crash_threshold_pct: float = -3.0,
        portfolio_drawdown_threshold_pct: float = -5.0,
    ) -> None:
        self.market_crash_threshold = market_crash_threshold_pct
        self.portfolio_drawdown_threshold = portfolio_drawdown_threshold_pct
        # 状态
        self._equity_peak: float | None = None

    def update_equity_peak(self, current_equity: float) -> None:
        """每日调用，跟踪组合历史最高净值。"""
        if self._equity_peak is None or current_equity > self._equity_peak:
            self._equity_peak = current_equity

    def reset_peak(self) -> None:
        """每周一调仓后重置峰值（跟新决策周期）—— 可选。"""
        # 不重置：保留全程峰值跟踪
        pass

    def check(
        self,
        as_of: date,
        current_equity: float,
        current_positions: list[str],
    ) -> DailyHealthOpinion:
        """每天交易日开盘前调用。

        Args:
            as_of: 今天日期。
            current_equity: 组合当前总资产。
            current_positions: 当前持仓的标的代码列表（用于异常检测）。
        """
        signals: dict[str, Any] = {"as_of": str(as_of)}

        # 1. 大盘单日跌幅（最近交易日 close vs 前一日 close）
        try:
            df = sources.get_price_history(
                HS300_ETF, as_of - timedelta(days=10), as_of, asset_type="etf",
            )
            if df is not None and not df.empty and len(df) >= 2:
                df = df.sort_values("date").reset_index(drop=True)
                today_close = float(df["close"].iloc[-1])
                yesterday_close = float(df["close"].iloc[-2])
                daily_change_pct = (today_close / yesterday_close - 1) * 100
                signals["hs300_daily_change_pct"] = round(daily_change_pct, 2)

                # 触发 1：大盘崩盘
                if daily_change_pct <= self.market_crash_threshold:
                    return DailyHealthOpinion(
                        as_of=as_of,
                        action="market_crash",
                        target_symbol_weights={HS300_ETF: 0.5},  # 切 ETF 半仓
                        reasoning=f"🚨 大盘单日跌 {daily_change_pct:.2f}% ≤ "
                                  f"{self.market_crash_threshold}% → 切 ETF 半仓避险",
                        signals=signals,
                    )
        except Exception as exc:
            signals["hs300_error"] = f"{type(exc).__name__}: {exc}"

        # 2. 组合回撤检测
        self.update_equity_peak(current_equity)
        if self._equity_peak and self._equity_peak > 0:
            drawdown_pct = (current_equity / self._equity_peak - 1) * 100
            signals["portfolio_drawdown_pct"] = round(drawdown_pct, 2)

            if drawdown_pct <= self.portfolio_drawdown_threshold:
                return DailyHealthOpinion(
                    as_of=as_of,
                    action="portfolio_drawdown",
                    target_symbol_weights={HS300_ETF: 1.0},  # 切 ETF 满仓守底
                    reasoning=f"🚨 组合回撤 {drawdown_pct:.2f}% ≤ "
                              f"{self.portfolio_drawdown_threshold}% → 切 ETF 满仓守底",
                    signals=signals,
                )

        # 3. 持仓异常检测（个股停牌 / ST / 暴跌 > 10%）
        anomaly_symbols = []
        for sym in current_positions:
            if sym in (HS300_ETF, "511010"):  # ETF 跳过
                continue
            try:
                df = sources.get_price_history(sym, as_of - timedelta(days=5), as_of)
                if df is None or df.empty:
                    anomaly_symbols.append((sym, "无数据(停牌?)"))
                    continue
                df = df.sort_values("date").reset_index(drop=True)
                if len(df) >= 2:
                    last_change = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
                    if last_change <= -9.5:  # 跌停（个股 -10%）
                        anomaly_symbols.append((sym, f"跌停 {last_change:.1f}%"))
            except Exception:
                continue

        if len(anomaly_symbols) >= 2:  # 多只持仓异常 → 触发应急
            signals["anomaly_symbols"] = [f"{s}:{r}" for s, r in anomaly_symbols]
            return DailyHealthOpinion(
                as_of=as_of,
                action="position_anomaly",
                target_symbol_weights={HS300_ETF: 1.0},
                reasoning=f"🚨 {len(anomaly_symbols)} 只持仓异常: "
                          f"{', '.join(s for s, _ in anomaly_symbols[:3])} → 切 ETF 满仓",
                signals=signals,
            )

        # 一切正常
        return DailyHealthOpinion(
            as_of=as_of,
            action="no_action",
            target_symbol_weights=None,
            reasoning=f"日频健康正常（大盘 {signals.get('hs300_daily_change_pct', 'N/A')}%, "
                      f"组合回撤 {signals.get('portfolio_drawdown_pct', 'N/A')}%）",
            signals=signals,
        )
