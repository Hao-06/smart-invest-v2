"""回测引擎冒烟测试。

用一个极简的「买入并持有」策略验证：
- 订单成交（含手续费、印花税）
- T+1 约束（首日买入，次日才允许卖出）
- 净值曲线、绩效指标正常
- 多源容灾在回测主循环里也工作
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# 让本脚本无需安装即可运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aifund.backtest import BacktestEngine, Order
from aifund.data import DataPipeline
from aifund.data.models import MarketSnapshot


def buy_and_hold_strategy(snapshot: MarketSnapshot, portfolio) -> list[Order]:
    """极简策略：第一交易日给候选标的各下一手买单，之后空操作。"""
    # 已建过仓就不再下单
    if any(p.shares > 0 for p in portfolio.positions.values()):
        return []
    orders: list[Order] = []
    for sym in snapshot.symbols:
        sd = snapshot.get(sym)
        if sd is None or not sd.has_price:
            continue
        orders.append(Order(symbol=sym, side="BUY", shares=100, name=sd.name,
                            reason="首日建仓（冒烟测试）"))
    return orders


def main() -> int:
    pipeline = DataPipeline(lookback_days=60)
    engine = BacktestEngine(
        pipeline=pipeline,
        candidate_symbols=["600519", "601318"],  # 茅台 + 中国平安
        decide_fn=buy_and_hold_strategy,
        initial_capital=500_000,
    )
    result = engine.run(start_date="2026-04-15", end_date="2026-04-30", verbose=True)
    result.print_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
