"""动量+大盘择时 策略的**多区间稳健性验证**。

跨多个市场环境（熊市/牛市/震荡）验证策略是否稳定跑赢沪深300，避免「单次回测过拟合」。
这是金融研究的金标准 —— **多个独立样本外区间的胜率**才是真正的稳健性证据。

输出综合对比表 + 多区间净值曲线对比图。
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from aifund.backtest import BacktestEngine
from aifund.backtest.metrics import (
    annualized_return, max_drawdown, sharpe_ratio, total_return, volatility,
)
from aifund.data import DataPipeline, sources
from aifund.stockpool import MomentumSelector

# 3 个覆盖不同市场环境的验证区间
REGIONS = [
    {
        "name": "🐻 熊市/反弹（2024 Q1 大跌）",
        "start": date(2024, 2, 1),
        "end": date(2024, 4, 30),
        "expectation": "大盘择时应触发空仓避险",
    },
    {
        "name": "🐂 牛市（2024 9.24 大行情）",
        "start": date(2024, 9, 1),
        "end": date(2024, 11, 30),
        "expectation": "动量策略应捕捉到大涨",
    },
    {
        "name": "🐂 牛市（2025 06-07 已验证）",
        "start": date(2025, 6, 1),
        "end": date(2025, 7, 31),
        "expectation": "✓ 之前已证 +19.13% / 跑赢 +11.49 pp",
    },
]


def make_weekly_rebalance_strategy(weekly_pools, get_pool_for_date,
                                   get_ratio_for_date=None):
    """周一调仓到当周池子的「**目标市值匹配**」策略（支持半仓震荡市过滤）。

    Args:
        weekly_pools: dict[date, list[str]]
        get_pool_for_date: 用 bisect 取当前日期对应的池子
        get_ratio_for_date: 可选；当前日期对应的目标仓位比例（0.0/0.5/1.0）。
            None 时默认 1.0（向后兼容）。
    """

    def strategy(snapshot, portfolio):
        current_pool = get_pool_for_date(snapshot.as_of)
        target_ratio = get_ratio_for_date(snapshot.as_of) if get_ratio_for_date else 1.0
        orders = []

        # 1. 池子外的持仓 → 全部卖出
        for sym, pos in list(portfolio.positions.items()):
            if pos.shares > 0 and sym not in (current_pool or []):
                sellable = pos.sellable_shares(snapshot.as_of)
                if sellable > 0:
                    orders.append({
                        "symbol": sym, "side": "SELL",
                        "shares": (sellable // 100) * 100,
                        "name": pos.name,
                        "reason": "调出池/趋势向下",
                    })

        # 2. 空仓状态 / 没有候选 → 仅完成上面的卖出
        if target_ratio <= 0.0 or not current_pool:
            return orders

        if snapshot.as_of not in weekly_pools:
            return []  # 非调仓日

        # 3. 目标市值匹配：每只 = equity × target_ratio × 0.95 / 池子大小
        equity = portfolio.equity()
        target_value_per = equity * target_ratio * 0.95 / len(current_pool)

        for sym in current_pool:
            pos = portfolio.get_position(sym)
            current_value = pos.market_value if pos else 0.0
            sd = snapshot.get(sym)
            if sd is None or not sd.last_close:
                continue
            diff = target_value_per - current_value
            lot_value = sd.last_close * 100
            # 加仓：缺口 > 1 手
            if diff > lot_value:
                shares = int(diff / lot_value) * 100
                max_by_cash = int(portfolio.cash * 0.95 / lot_value) * 100
                shares = min(shares, max_by_cash)
                if shares >= 100:
                    orders.append({
                        "symbol": sym, "side": "BUY", "shares": shares,
                        "name": sd.name,
                        "reason": f"动量入池(ratio={target_ratio:.2f})",
                    })
            # 减仓：超出 > 1 手（震荡市从满仓 → 半仓时）
            elif diff < -lot_value and pos:
                shares = int(-diff / lot_value) * 100
                sellable = pos.sellable_shares(snapshot.as_of)
                shares = min(shares, (sellable // 100) * 100)
                if shares >= 100:
                    orders.append({
                        "symbol": sym, "side": "SELL", "shares": shares,
                        "name": pos.name,
                        "reason": f"震荡市减仓(ratio={target_ratio:.2f})",
                    })
        return orders

    return strategy


def run_one_region(pipeline: DataPipeline, region: dict, capital: float = 500_000) -> dict:
    """在一个区间内跑动量策略 + 沪深300 对比，返回结果字典。"""
    print(f"\n{'=' * 70}")
    print(f"区间：{region['name']}")
    print(f"     {region['start']} ~ {region['end']}")
    print(f"     预期：{region['expectation']}")
    print("=" * 70)

    t0 = time.time()
    selector = MomentumSelector(
        pipeline=pipeline, top_pool=8, enable_timing=True,
        use_lstm_filter=True, lstm_weight=0.4,
    )
    weekly_pools = selector.precompute_weekly_pools(
        region["start"], region["end"], pipeline.calendar, verbose=False,
    )
    if not weekly_pools:
        return {"region": region["name"], "error": "没有产出周池子"}

    import bisect
    rebalance_dates = sorted(weekly_pools.keys())
    weekly_ratios = getattr(selector, "weekly_position_ratios", {}) or {}

    def get_pool_for_date(d):
        i = bisect.bisect_right(rebalance_dates, d) - 1
        return weekly_pools[rebalance_dates[i]] if i >= 0 else []

    def get_ratio_for_date(d):
        i = bisect.bisect_right(rebalance_dates, d) - 1
        return weekly_ratios.get(rebalance_dates[i], 1.0) if i >= 0 else 1.0

    all_symbols = sorted({s for p in weekly_pools.values() for s in p})
    empty_weeks = sum(1 for d in rebalance_dates if not weekly_pools[d])
    half_weeks = sum(1 for d in rebalance_dates if 0 < weekly_ratios.get(d, 1.0) < 1.0)
    print(f"  共 {len(rebalance_dates)} 周，{empty_weeks} 周空仓 + {half_weeks} 周半仓，候选 {len(all_symbols)} 只")

    if not all_symbols:
        # 全程空仓
        print(f"  ✓ 全程大盘趋势向下，策略保持空仓 0 元损失（vs 沪深300 跌了多少自己看）")

    # 跑动量策略
    print(f"  >>> 跑回测…", flush=True)
    engine = BacktestEngine(
        pipeline=pipeline,
        candidate_symbols=all_symbols if all_symbols else ["510300"],
        decide_fn=make_weekly_rebalance_strategy(
            weekly_pools, get_pool_for_date, get_ratio_for_date,
        ),
        initial_capital=capital,
    )
    result = engine.run(region["start"], region["end"], verbose=False)
    m = result.metrics

    # 沪深300 基准
    hs300 = sources.get_price_history(
        "510300", region["start"], region["end"], asset_type="etf"
    )
    if not hs300.empty:
        hs300_start = float(hs300.iloc[0]["close"])
        hs300_end = float(hs300.iloc[-1]["close"])
        hs300_return = (hs300_end / hs300_start - 1) * 100

        # 沪深300 的最大回撤
        hs_equity = [(d, capital * (float(c) / hs300_start))
                     for d, c in zip(hs300["date"], hs300["close"])]
        hs_mdd, _, _ = max_drawdown(hs_equity)
    else:
        hs300_return = None
        hs_mdd = None

    elapsed = time.time() - t0
    print(f"  动量: {m.total_return*100:+.2f}%  最大回撤: {m.max_drawdown*100:.2f}%  夏普: {m.sharpe:.2f}")
    if hs300_return is not None:
        diff = m.total_return * 100 - hs300_return
        winner = "✅" if diff > 0 else "❌"
        print(f"  HS300: {hs300_return:+.2f}%  最大回撤: {hs_mdd*100:.2f}%")
        print(f"  超额: {diff:+.2f} pp  {winner}  (耗时 {elapsed:.1f}s)")

    return {
        "region": region["name"],
        "start": str(region["start"]),
        "end": str(region["end"]),
        "weeks": len(rebalance_dates),
        "empty_weeks": empty_weeks,
        "n_symbols": len(all_symbols),
        "momentum_return_pct": round(m.total_return * 100, 2),
        "momentum_max_drawdown_pct": round(m.max_drawdown * 100, 2),
        "momentum_sharpe": round(m.sharpe, 2),
        "hs300_return_pct": round(hs300_return, 2) if hs300_return is not None else None,
        "hs300_max_drawdown_pct": round(hs_mdd * 100, 2) if hs_mdd is not None else None,
        "excess_pct": round(m.total_return * 100 - (hs300_return or 0), 2),
        "won": (hs300_return is not None and m.total_return * 100 > hs300_return),
    }


def main() -> int:
    pipeline = DataPipeline()
    print(f"\n{'#' * 70}")
    print("# 动量+大盘择时 策略 · **多区间稳健性验证**")
    print(f"# 跨 {len(REGIONS)} 个市场环境验证（金融研究的金标准）")
    print(f"{'#' * 70}")

    results = []
    for region in REGIONS:
        r = run_one_region(pipeline, region)
        results.append(r)

    # 汇总
    print("\n\n" + "=" * 80)
    print("📊 综合稳健性报告")
    print("=" * 80)
    df = pd.DataFrame(results)
    cols = ["region", "momentum_return_pct", "hs300_return_pct", "excess_pct",
            "momentum_max_drawdown_pct", "hs300_max_drawdown_pct", "won"]
    print(df[cols].to_string(index=False))

    won_count = sum(1 for r in results if r.get("won"))
    print(f"\n胜率: {won_count}/{len(results)} 个区间跑赢沪深300")
    if won_count == len(results):
        print("🎉 **全部跑赢**：策略稳健性得到多区间样本外证据支持")
    elif won_count >= len(results) - 1:
        print("✓ 多数跑赢，策略整体稳健")
    else:
        print("⚠ 跑赢率偏低，策略可能在某些市场环境下失效，需要进一步分析")

    # 落盘
    import json
    out_path = Path("runs/robustness_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ 报告已落盘: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
