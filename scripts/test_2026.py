"""**2026 年数据回测** —— 跑「真正样本外」的 forward test。

CSV 覆盖到 2025-12，2026 年数据 CSV 完全没有 —— 用实时行情拉取 + 自己算动量因子，
检验策略在「**前所未见**」的市场环境下是否仍然有效。

这是策略稳健性的**终极检验**：
- 数据 = AkShare 实时拉取（不在任何训练 / 调参样本里）
- 动量计算 = 自己算（不用 CSV 现成因子）
- 大盘择时 = 沪深300 ETF 510300 的 5MA/20MA（同一规则）

为加速首次数据拉取，universe 缩到 CSV 最后一日 score top 80 只（其中很多已缓存）。
"""
from __future__ import annotations

import bisect
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aifund.backtest import BacktestEngine
from aifund.data import DataPipeline, sources
from aifund.stockpool import MomentumSelector


def make_weekly_rebalance_strategy(weekly_pools, get_pool_for_date,
                                   get_ratio_for_date=None):
    """周一调仓「目标市值匹配」策略（支持震荡市半仓）。"""
    def strategy(snapshot, portfolio):
        current = get_pool_for_date(snapshot.as_of)
        target_ratio = get_ratio_for_date(snapshot.as_of) if get_ratio_for_date else 1.0
        orders = []

        # 池外清仓
        for sym, pos in list(portfolio.positions.items()):
            if pos.shares > 0 and sym not in (current or []):
                sellable = pos.sellable_shares(snapshot.as_of)
                if sellable > 0:
                    orders.append({
                        "symbol": sym, "side": "SELL",
                        "shares": (sellable // 100) * 100,
                        "name": pos.name, "reason": "调出池/趋势向下",
                    })

        if target_ratio <= 0.0 or not current:
            return orders
        if snapshot.as_of not in weekly_pools:
            return []

        equity = portfolio.equity()
        target_value_per = equity * target_ratio * 0.95 / len(current)
        for sym in current:
            pos = portfolio.get_position(sym)
            current_value = pos.market_value if pos else 0.0
            sd = snapshot.get(sym)
            if sd is None or not sd.last_close:
                continue
            diff = target_value_per - current_value
            lot_value = sd.last_close * 100
            if diff > lot_value:
                shares = int(diff / lot_value) * 100
                max_by_cash = int(portfolio.cash * 0.95 / lot_value) * 100
                shares = min(shares, max_by_cash)
                if shares >= 100:
                    orders.append({
                        "symbol": sym, "side": "BUY", "shares": shares,
                        "name": sd.name, "reason": f"动量入池(ratio={target_ratio:.2f})",
                    })
            elif diff < -lot_value and pos:
                shares = int(-diff / lot_value) * 100
                sellable = pos.sellable_shares(snapshot.as_of)
                shares = min(shares, (sellable // 100) * 100)
                if shares >= 100:
                    orders.append({
                        "symbol": sym, "side": "SELL", "shares": shares,
                        "name": pos.name, "reason": f"震荡市减仓(ratio={target_ratio:.2f})",
                    })
        return orders
    return strategy


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-05-15")
    parser.add_argument("--capital", type=float, default=500_000)
    parser.add_argument("--top", type=int, default=8, help="持仓数")
    parser.add_argument("--universe-top-n", type=int, default=80,
                        help="universe 大小（按 CSV 最后一日 score 取前 N）")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"=== 🚀 2026 年「样本外」forward test ===")
    print(f"区间：{start} ~ {end}")
    print(f"动量计算：**实时行情计算**（不依赖 CSV）")
    print(f"Universe：CSV 最后一日 score top {args.universe_top_n}")
    print(f"持仓：{args.top} 只等权")
    print(f"大盘择时：启用（510300 ETF 5MA/20MA）")
    print()

    pipeline = DataPipeline()
    selector = MomentumSelector(
        pipeline=pipeline,
        top_pool=args.top,
        enable_timing=True,
        use_realtime_momentum=True,  # 关键：不读 CSV
        universe_top_n=args.universe_top_n,
        use_lstm_filter=True,
        lstm_weight=0.4,
    )
    print(f"Universe 实际大小: {len(selector.universe)}")

    print("\n>>> 预计算周频池子（首次 ~60-90 分钟拉数据；后续秒过）…", flush=True)
    t0 = time.time()
    weekly_pools = selector.precompute_weekly_pools(start, end, verbose=True)
    print(f"\n预计算耗时 {time.time() - t0:.1f}s")
    if not weekly_pools:
        print("✗ 没有产出周池子，退出")
        return 1

    rebalance_dates = sorted(weekly_pools.keys())
    weekly_ratios = getattr(selector, "weekly_position_ratios", {}) or {}
    def get_pool_for_date(d):
        i = bisect.bisect_right(rebalance_dates, d) - 1
        return weekly_pools[rebalance_dates[i]] if i >= 0 else []
    def get_ratio_for_date(d):
        i = bisect.bisect_right(rebalance_dates, d) - 1
        return weekly_ratios.get(rebalance_dates[i], 1.0) if i >= 0 else 1.0

    all_symbols = sorted({s for pool in weekly_pools.values() for s in pool})
    empty_weeks = sum(1 for d in rebalance_dates if not weekly_pools[d])
    half_weeks = sum(1 for d in rebalance_dates if 0 < weekly_ratios.get(d, 1.0) < 1.0)
    print(f"\n共 {len(rebalance_dates)} 周，{empty_weeks} 周空仓 + {half_weeks} 周半仓，候选 {len(all_symbols)} 只")

    if not all_symbols:
        print("⚠ 全程空仓，无回测可跑")
        return 0

    print("\n>>> 跑回测…", flush=True)
    engine = BacktestEngine(
        pipeline=pipeline,
        candidate_symbols=all_symbols,
        decide_fn=make_weekly_rebalance_strategy(
            weekly_pools, get_pool_for_date, get_ratio_for_date,
        ),
        initial_capital=args.capital,
    )
    result = engine.run(start, end, verbose=False)

    # 沪深300 基准
    hs300 = sources.get_price_history("510300", start, end, asset_type="etf")
    if not hs300.empty:
        hs300_start = float(hs300.iloc[0]["close"])
        hs300_end = float(hs300.iloc[-1]["close"])
        hs300_return = (hs300_end / hs300_start - 1) * 100
    else:
        hs300_return = None

    m = result.metrics
    print()
    print("=" * 70)
    print(f"  2026 年 forward test 结果（{(end - start).days} 个日历日）")
    print("=" * 70)
    print(f"  动量策略  累计收益: {m.total_return*100:+.2f}%   最大回撤: {m.max_drawdown*100:.2f}%   夏普: {m.sharpe:.2f}   交易笔数: {m.trade_count}")
    if hs300_return is not None:
        diff = m.total_return * 100 - hs300_return
        winner = "✅ 动量赢" if diff > 0 else "❌ 动量输"
        print(f"  沪深300   累计收益: {hs300_return:+.2f}%")
        print(f"  超额收益: {diff:+.2f}%  {winner}")
        print(f"\n  💰 50 万本金 → 动量策略期末 ¥{result.metrics.final_equity:,.0f}, "
              f"沪深300 期末 ¥{args.capital * (1 + hs300_return/100):,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
