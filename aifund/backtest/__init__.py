"""回测引擎：日频回测、A股交易约束模拟与绩效指标。

对外主要接口：
    BacktestEngine    —— 日频回测主引擎
    BacktestResult    —— 回测结果（含组合终态、绩效指标、逐日日志）
    Portfolio         —— 组合状态（现金 / 持仓 / 成交 / 净值曲线）
    Order             —— 决策订单
    PerformanceMetrics—— 标准量化绩效指标
"""
from aifund.backtest.engine import BacktestEngine, BacktestResult, DailyLog
from aifund.backtest.metrics import PerformanceMetrics, compute_performance
from aifund.backtest.portfolio import Order, Portfolio, Position, Trade

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "DailyLog",
    "Portfolio",
    "Position",
    "Order",
    "Trade",
    "PerformanceMetrics",
    "compute_performance",
]
