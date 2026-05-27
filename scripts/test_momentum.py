"""快速验证「动量 + 大盘择时」策略能否跑赢沪深300。

不调 LLM、不调多 Agent —— 纯量化 baseline，验证策略**本身**的有效性。
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aifund.backtest import BacktestEngine, Portfolio
from aifund.data import DataPipeline
from aifund.stockpool import MomentumSelector


def make_weekly_rebalance_strategy(weekly_pools, get_pool_for_date):
    """每周一重新调仓到当周池子的等权策略。"""

    def strategy(snapshot, portfolio):
        current_pool = get_pool_for_date(snapshot.as_of)
        if not current_pool:
            # 大盘择时空仓 → 卖光所有持仓
            orders = []
            for sym, pos in list(portfolio.positions.items()):
                if pos.shares > 0:
                    sellable = pos.sellable_shares(snapshot.as_of)
                    if sellable > 0:
                        orders.append({
                            "symbol": sym, "side": "SELL",
                            "shares": (sellable // 100) * 100,
                            "name": pos.name, "reason": "大盘趋势向下，清仓"
                        })
            return orders

        # 当周首日（rebalance）：先卖掉不在池中的，再等权买入新的
        is_rebalance_day = snapshot.as_of in weekly_pools
        if not is_rebalance_day:
            return []

        orders = []
        # 卖掉不在新池子里的
        for sym, pos in list(portfolio.positions.items()):
            if pos.shares > 0 and sym not in current_pool:
                sellable = pos.sellable_shares(snapshot.as_of)
                if sellable > 0:
                    orders.append({
                        "symbol": sym, "side": "SELL",
                        "shares": (sellable // 100) * 100,
                        "name": pos.name, "reason": "调出池"
                    })

        # 等权买入新池子里的（粗略估算可用现金）
        existing_in_pool = {s for s in current_pool if portfolio.get_position(s)
                            and portfolio.get_position(s).shares > 0}
        new_targets = [s for s in current_pool if s not in existing_in_pool]
        if not new_targets:
            return orders

        # 估算可用现金（保留 5% buffer + 减去本次卖出后的现金，简化用当前现金 / N 估算）
        cash_per_stock = portfolio.cash * 0.95 / len(new_targets)
        for sym in new_targets:
            sd = snapshot.get(sym)
            if sd is None or not sd.last_close:
                continue
            shares = int(cash_per_stock / (sd.last_close * 100)) * 100
            if shares >= 100:
                orders.append({
                    "symbol": sym, "side": "BUY", "shares": shares,
                    "name": sd.name, "reason": "动量入池"
                })
        return orders

    return strategy


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--end", default="2025-07-31")
    parser.add_argument("--capital", type=float, default=500_000)
    parser.add_argument("--top", type=int, default=8, help="持仓数")
    parser.add_argument("--no-timing", action="store_true", help="禁用大盘择时")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"=== 动量+大盘择时 策略回测 ===")
    print(f"区间：{start} ~ {end}")
    print(f"持仓：{args.top} 只等权")
    print(f"大盘择时：{'禁用' if args.no_timing else '启用（510300 ETF 5MA/20MA）'}")
    print()

    pipeline = DataPipeline()
    selector = MomentumSelector(
        pipeline=pipeline,
        top_pool=args.top,
        enable_timing=not args.no_timing,
    )

    print(f"标的池大小: {len(selector.universe)}")
    print(">>> 预计算周频池子（首次拉数据约 5-10 分钟）…")
    t0 = time.time()
    weekly_pools = selector.precompute_weekly_pools(start, end, verbose=True)
    print(f"预计算耗时 {time.time() - t0:.1f}s")

    if not weekly_pools:
        print("✗ 没有产出周池子，退出")
        return 1

    import bisect
    rebalance_dates = sorted(weekly_pools.keys())

    def get_pool_for_date(d):
        i = bisect.bisect_right(rebalance_dates, d) - 1
        return weekly_pools[rebalance_dates[i]] if i >= 0 else []

    all_symbols = sorted({s for pool in weekly_pools.values() for s in pool})
    print(f"\n去重候选标的总数: {len(all_symbols)}")

    # 跑回测
    print("\n>>> 跑回测…")
    engine = BacktestEngine(
        pipeline=pipeline,
        candidate_symbols=all_symbols,
        decide_fn=make_weekly_rebalance_strategy(weekly_pools, get_pool_for_date),
        initial_capital=args.capital,
    )
    result = engine.run(start, end, verbose=False)

    # 拉沪深300 基准对比
    from aifund.data import sources as _sources
    hs300 = _sources.get_price_history("510300", start, end, asset_type="etf")
    if not hs300.empty:
        hs300_start = float(hs300.iloc[0]["close"])
        hs300_end = float(hs300.iloc[-1]["close"])
        hs300_return = (hs300_end / hs300_start - 1) * 100
    else:
        hs300_return = None

    m = result.metrics
    print()
    print("=" * 60)
    print("策略对比")
    print("=" * 60)
    print(f"  动量策略  累计收益: {m.total_return*100:+.2f}%   最大回撤: {m.max_drawdown*100:.2f}%   夏普: {m.sharpe:.2f}   交易笔数: {m.trade_count}")
    if hs300_return is not None:
        diff = m.total_return * 100 - hs300_return
        winner = "✅ 动量赢" if diff > 0 else "❌ 动量输"
        print(f"  沪深300   累计收益: {hs300_return:+.2f}%")
        print(f"  超额收益: {diff:+.2f}%  {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
