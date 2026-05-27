"""回测绩效指标。

输入：``Portfolio.equity_curve`` 与 ``Portfolio.trades``。
输出：标准量化绩效指标 —— 总收益、年化收益、最大回撤、夏普、胜率、盈亏比、
换手率等。所有指标用百分数/无单位表达，便于设计书直接呈现。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from aifund.backtest.portfolio import Trade


# ---------------------------------------------------------------------------
# 基础指标
# ---------------------------------------------------------------------------


def equity_series(equity_curve: list[tuple[date, float]]) -> pd.Series:
    """把净值曲线转为以日期为索引的 Series。"""
    if not equity_curve:
        return pd.Series(dtype=float)
    dates, values = zip(*equity_curve)
    return pd.Series(values, index=pd.to_datetime(list(dates)), name="equity").sort_index()


def total_return(equity_curve: list[tuple[date, float]], initial_capital: float) -> float:
    """累计收益率（小数）。"""
    if not equity_curve or initial_capital <= 0:
        return 0.0
    return equity_curve[-1][1] / initial_capital - 1.0


def annualized_return(equity_curve: list[tuple[date, float]], initial_capital: float) -> float:
    """年化收益率（小数）。按日历日折算。"""
    if len(equity_curve) < 2 or initial_capital <= 0:
        return 0.0
    days = (equity_curve[-1][0] - equity_curve[0][0]).days
    if days <= 0:
        return 0.0
    total = total_return(equity_curve, initial_capital)
    return (1.0 + total) ** (365.0 / days) - 1.0


def max_drawdown(
    equity_curve: list[tuple[date, float]],
) -> tuple[float, date | None, date | None]:
    """最大回撤（负小数）及其顶部、谷底日期。"""
    if len(equity_curve) < 2:
        return 0.0, None, None
    series = equity_series(equity_curve)
    cummax = series.cummax()
    drawdown = series / cummax - 1.0
    trough_ts = drawdown.idxmin()
    peak_ts = series.loc[:trough_ts].idxmax()
    return float(drawdown.min()), peak_ts.date(), trough_ts.date()


def sharpe_ratio(
    equity_curve: list[tuple[date, float]],
    risk_free_annual: float = 0.02,
) -> float:
    """年化夏普比率。基于日度收益率与 252 交易日年化。"""
    if len(equity_curve) < 3:
        return 0.0
    returns = equity_series(equity_curve).pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    excess = returns - risk_free_annual / 252.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def volatility(equity_curve: list[tuple[date, float]]) -> float:
    """年化波动率。"""
    if len(equity_curve) < 3:
        return 0.0
    returns = equity_series(equity_curve).pct_change().dropna()
    return float(returns.std() * np.sqrt(252))


def calmar_ratio(equity_curve: list[tuple[date, float]], initial_capital: float) -> float:
    """Calmar = 年化收益 / |最大回撤|。"""
    ar = annualized_return(equity_curve, initial_capital)
    mdd, _, _ = max_drawdown(equity_curve)
    if mdd >= 0:
        return 0.0
    return ar / abs(mdd)


# ---------------------------------------------------------------------------
# 交易侧指标
# ---------------------------------------------------------------------------


def win_rate_and_profit_factor(trades: list[Trade]) -> tuple[float, float, int, int]:
    """基于卖出交易的已实现盈亏，返回 (胜率, 盈亏比, 盈利笔数, 亏损笔数)。

    胜率 = 盈利笔数 / 总笔数；盈亏比 = 总盈利 / |总亏损|。
    """
    sells = [t for t in trades if t.side == "SELL"]
    if not sells:
        return 0.0, 0.0, 0, 0
    wins = [t for t in sells if t.realized_pnl > 0]
    losses = [t for t in sells if t.realized_pnl < 0]
    total_win = sum(t.realized_pnl for t in wins)
    total_loss = sum(-t.realized_pnl for t in losses)
    rate = len(wins) / len(sells) if sells else 0.0
    pf = total_win / total_loss if total_loss > 0 else float("inf") if total_win > 0 else 0.0
    return rate, pf, len(wins), len(losses)


def turnover_ratio(trades: list[Trade], initial_capital: float) -> float:
    """换手率（双边成交额 / 初始资金）。"""
    if initial_capital <= 0:
        return 0.0
    total = sum(t.amount for t in trades)
    return total / initial_capital


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    start_date: date | None
    end_date: date | None
    days: int
    initial_capital: float
    final_equity: float
    total_return: float  # 小数
    annualized_return: float
    max_drawdown: float
    drawdown_peak: date | None
    drawdown_trough: date | None
    sharpe: float
    calmar: float
    volatility: float
    win_rate: float
    profit_factor: float
    win_count: int
    loss_count: int
    trade_count: int
    turnover: float

    def to_dict(self) -> dict[str, object]:
        return {
            "区间": f"{self.start_date} ~ {self.end_date}（{self.days} 个自然日）",
            "初始资金": round(self.initial_capital, 2),
            "期末权益": round(self.final_equity, 2),
            "累计收益率": f"{self.total_return * 100:.2f}%",
            "年化收益率": f"{self.annualized_return * 100:.2f}%",
            "最大回撤": f"{self.max_drawdown * 100:.2f}%",
            "回撤峰谷": f"{self.drawdown_peak} → {self.drawdown_trough}",
            "夏普比率": round(self.sharpe, 3),
            "Calmar 比率": round(self.calmar, 3),
            "年化波动率": f"{self.volatility * 100:.2f}%",
            "胜率": f"{self.win_rate * 100:.2f}%",
            "盈亏比": round(self.profit_factor, 3) if self.profit_factor != float("inf") else "∞",
            "盈利/亏损笔数": f"{self.win_count} / {self.loss_count}",
            "总交易笔数": self.trade_count,
            "换手率": round(self.turnover, 3),
        }


def compute_performance(
    equity_curve: list[tuple[date, float]],
    trades: list[Trade],
    initial_capital: float,
) -> PerformanceMetrics:
    """一次性计算全部绩效指标。"""
    mdd, peak, trough = max_drawdown(equity_curve)
    win_rate, pf, wins, losses = win_rate_and_profit_factor(trades)
    start = equity_curve[0][0] if equity_curve else None
    end = equity_curve[-1][0] if equity_curve else None
    days = (end - start).days if (start and end) else 0
    return PerformanceMetrics(
        start_date=start,
        end_date=end,
        days=days,
        initial_capital=initial_capital,
        final_equity=equity_curve[-1][1] if equity_curve else initial_capital,
        total_return=total_return(equity_curve, initial_capital),
        annualized_return=annualized_return(equity_curve, initial_capital),
        max_drawdown=mdd,
        drawdown_peak=peak,
        drawdown_trough=trough,
        sharpe=sharpe_ratio(equity_curve),
        calmar=calmar_ratio(equity_curve, initial_capital),
        volatility=volatility(equity_curve),
        win_rate=win_rate,
        profit_factor=pf,
        win_count=wins,
        loss_count=losses,
        trade_count=len(trades),
        turnover=turnover_ratio(trades, initial_capital),
    )
