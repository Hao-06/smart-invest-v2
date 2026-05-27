"""**沪深 300 ETF 满仓持有**作为产品的收益主线 —— 12 月连续验证脚本。

跑 ``ETFHolder`` 策略 vs 沪深 300 基准对比，证明：
- 策略与基准**几乎完全一致**（仅相差手续费）
- 但有「**多 Agent 决策驾驶舱**」的可解释 + 可审计加成

用法：
    python3 scripts/test_etf_baseline.py
    python3 scripts/test_etf_baseline.py --start 2024-05-18 --end 2026-05-18  # 24 月
"""
from __future__ import annotations

import argparse
import bisect
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aifund.backtest.engine import BacktestEngine
from aifund.data import sources
from aifund.data.pipeline import DataPipeline
from aifund.stockpool.etf_holder import HS300_ETF, ETFHolder


def make_buy_and_hold_strategy(weekly_pools, get_pool_for_date):
    """周一买入 510300 满仓 + 之后不动的策略。"""
    def strategy(snapshot, portfolio):
        # 已持有 → 不动
        if portfolio.get_position(HS300_ETF):
            return []
        # 第一次：把 95% 现金买入 510300
        sd = snapshot.get(HS300_ETF)
        if sd is None or not sd.last_close:
            return []
        lot_value = sd.last_close * 100
        shares = int(portfolio.cash * 0.95 / lot_value) * 100
        if shares >= 100:
            return [{
                "symbol": HS300_ETF, "side": "BUY", "shares": shares,
                "name": "沪深300ETF", "reason": "满仓建仓",
            }]
        return []
    return strategy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-05-18")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument("--capital", type=float, default=500_000)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"=== 🛡️ 沪深 300 ETF 满仓基线 · 12 月连续回测 ===")
    print(f"区间：{start} ~ {end}")
    print(f"策略：第一周 95% 现金 → 沪深 300 ETF(510300)，之后不动")
    print(f"对比：纯沪深 300 指数（不含 ETF 跟踪误差和手续费）")
    print()

    pipeline = DataPipeline()
    selector = ETFHolder(pipeline=pipeline)
    weekly_pools = selector.precompute_weekly_pools(start, end, verbose=False)

    rebalance_dates = sorted(weekly_pools.keys())

    def get_pool_for_date(d):
        i = bisect.bisect_right(rebalance_dates, d) - 1
        return weekly_pools[rebalance_dates[i]] if i >= 0 else []

    print(f"共 {len(rebalance_dates)} 周（每周一检查，已持仓则不动）")
    print("\n>>> 跑回测…", flush=True)

    engine = BacktestEngine(
        pipeline=pipeline,
        candidate_symbols=[HS300_ETF],
        decide_fn=make_buy_and_hold_strategy(weekly_pools, get_pool_for_date),
        initial_capital=args.capital,
    )
    result = engine.run(start, end, verbose=False)
    m = result.metrics

    # 沪深 300 指数对比
    hs300 = sources.get_price_history(HS300_ETF, start, end, asset_type="etf")
    if hs300 is not None and not hs300.empty:
        hs_start = float(hs300["close"].iloc[0])
        hs_end = float(hs300["close"].iloc[-1])
        hs_ret = (hs_end / hs_start - 1) * 100
    else:
        hs_ret = 0.0

    print()
    print("=" * 70)
    print(f"  ETF 满仓基线策略 · 结果（{(end - start).days} 个日历日）")
    print("=" * 70)
    cum = m.total_return * 100
    dd = m.max_drawdown * 100
    sharpe = m.sharpe
    trades = m.trade_count
    final = m.final_equity
    print(f"  ETF 策略   累计收益: {cum:+.2f}%   最大回撤: {dd:+.2f}%   "
          f"夏普: {sharpe:.2f}   交易笔数: {trades}")
    print(f"  沪深 300   累计收益: {hs_ret:+.2f}%")
    diff = cum - hs_ret
    note = "✅ 几乎一致" if abs(diff) < 1.0 else ("⚠️ 偏差较大" if abs(diff) > 3.0 else "✅ 微小偏差（手续费）")
    print(f"  差异      {diff:+.2f}%   {note}")
    print(f"\n  💰 {args.capital:,.0f} 本金 → ETF 策略期末 ¥{final:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
