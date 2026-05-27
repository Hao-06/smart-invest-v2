"""动量策略参数扫描 —— 在多区间寻找**真正稳健**的参数组合。

为避免过拟合到单一区间，本脚本：
1. 定义 4 种代表性参数组合（基准 / 敏感 / 宽松 / 激进）
2. 在 3 个不同市场环境区间分别测试
3. 输出每种组合的「**平均超额** + 胜率 + 最大单次跑输**」综合评分
4. 推荐最稳健的参数（**不是单次最高的，是综合最稳的**）

设计原则：
- 不追求「**任何单次最高**」 —— 那是过拟合
- 追求「**最差情况下不死，平均情况下能赢**」—— 真稳健
"""
from __future__ import annotations

import bisect
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from aifund.backtest import BacktestEngine
from aifund.data import DataPipeline, sources
from aifund.stockpool import MomentumSelector

# 3 个测试区间（与 robustness_test.py 一致）
REGIONS = [
    {"name": "🐻 2024 Q1", "start": date(2024, 2, 1), "end": date(2024, 4, 30)},
    {"name": "🐂 9.24",   "start": date(2024, 9, 1), "end": date(2024, 11, 30)},
    {"name": "🐂 2025 H1","start": date(2025, 6, 1), "end": date(2025, 7, 31)},
]

# 4 种参数组合（**保守 → 激进**）
PARAM_SETS = [
    {
        "name": "基准 (当前默认)",
        "trend_short_ma": 5, "trend_long_ma": 20,
        "max_recent_gain_pct": 15.0,
    },
    {
        "name": "敏感均线 (3MA vs 10MA)",
        "trend_short_ma": 3, "trend_long_ma": 10,
        "max_recent_gain_pct": 15.0,
    },
    {
        "name": "去除反转过滤 (允许追强)",
        "trend_short_ma": 5, "trend_long_ma": 20,
        "max_recent_gain_pct": 999.0,  # 实际等于关掉
    },
    {
        "name": "激进 (敏感+去过滤)",
        "trend_short_ma": 3, "trend_long_ma": 10,
        "max_recent_gain_pct": 999.0,
    },
]


def make_weekly_strategy(weekly_pools, get_pool_for_date):
    """周一调仓策略（与其他脚本一致）。"""
    def strategy(snapshot, portfolio):
        current = get_pool_for_date(snapshot.as_of)
        if not current:
            orders = []
            for sym, pos in list(portfolio.positions.items()):
                if pos.shares > 0:
                    sellable = pos.sellable_shares(snapshot.as_of)
                    if sellable > 0:
                        orders.append({"symbol": sym, "side": "SELL",
                                       "shares": (sellable // 100) * 100,
                                       "name": pos.name, "reason": "大盘弱"})
            return orders
        if snapshot.as_of not in weekly_pools:
            return []
        orders = []
        for sym, pos in list(portfolio.positions.items()):
            if pos.shares > 0 and sym not in current:
                sellable = pos.sellable_shares(snapshot.as_of)
                if sellable > 0:
                    orders.append({"symbol": sym, "side": "SELL",
                                   "shares": (sellable // 100) * 100,
                                   "name": pos.name, "reason": "调出池"})
        existing = {s for s in current if portfolio.get_position(s) and portfolio.get_position(s).shares > 0}
        new_targets = [s for s in current if s not in existing]
        if new_targets:
            cash_per = portfolio.cash * 0.95 / len(new_targets)
            for sym in new_targets:
                sd = snapshot.get(sym)
                if sd is None or not sd.last_close:
                    continue
                shares = int(cash_per / (sd.last_close * 100)) * 100
                if shares >= 100:
                    orders.append({"symbol": sym, "side": "BUY", "shares": shares,
                                   "name": sd.name, "reason": "动量入池"})
        return orders
    return strategy


def test_one(pipeline: DataPipeline, params: dict, region: dict) -> dict:
    """对一组参数 × 一个区间跑一次回测，返回结果。"""
    selector = MomentumSelector(
        pipeline=pipeline,
        top_pool=8,
        trend_short_ma=params["trend_short_ma"],
        trend_long_ma=params["trend_long_ma"],
        max_recent_gain_pct=params["max_recent_gain_pct"],
        enable_timing=True,
    )
    weekly_pools = selector.precompute_weekly_pools(
        region["start"], region["end"], pipeline.calendar, verbose=False,
    )
    if not weekly_pools:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "won": False}

    rebalance_dates = sorted(weekly_pools.keys())
    def get_pool(d):
        i = bisect.bisect_right(rebalance_dates, d) - 1
        return weekly_pools[rebalance_dates[i]] if i >= 0 else []

    all_symbols = sorted({s for p in weekly_pools.values() for s in p})
    if not all_symbols:
        # 全空仓，权益保持初始
        hs300 = sources.get_price_history("510300", region["start"], region["end"], asset_type="etf")
        hs_ret = ((float(hs300.iloc[-1]["close"]) / float(hs300.iloc[0]["close"]) - 1) * 100
                  if not hs300.empty else 0.0)
        return {"return_pct": 0.0, "hs300_pct": hs_ret, "excess_pct": -hs_ret, "won": -hs_ret > 0}

    engine = BacktestEngine(
        pipeline=pipeline,
        candidate_symbols=all_symbols,
        decide_fn=make_weekly_strategy(weekly_pools, get_pool),
        initial_capital=500_000,
    )
    result = engine.run(region["start"], region["end"], verbose=False)
    m = result.metrics

    hs300 = sources.get_price_history("510300", region["start"], region["end"], asset_type="etf")
    hs300_ret = ((float(hs300.iloc[-1]["close"]) / float(hs300.iloc[0]["close"]) - 1) * 100
                 if not hs300.empty else None)
    excess = m.total_return * 100 - hs300_ret if hs300_ret is not None else None
    return {
        "return_pct": round(m.total_return * 100, 2),
        "max_drawdown_pct": round(m.max_drawdown * 100, 2),
        "hs300_pct": round(hs300_ret, 2) if hs300_ret is not None else None,
        "excess_pct": round(excess, 2) if excess is not None else None,
        "won": (excess is not None and excess > 0),
    }


def main() -> int:
    pipeline = DataPipeline()
    print("=" * 80)
    print("动量策略 · 参数扫描")
    print(f"  {len(PARAM_SETS)} 种参数 × {len(REGIONS)} 个区间 = {len(PARAM_SETS) * len(REGIONS)} 次测试")
    print("=" * 80)

    rows: list[dict] = []
    for params in PARAM_SETS:
        print(f"\n--- 参数: {params['name']} (短MA={params['trend_short_ma']}, "
              f"长MA={params['trend_long_ma']}, 过滤阈值={params['max_recent_gain_pct']:.0f}%) ---")
        for region in REGIONS:
            t0 = time.time()
            r = test_one(pipeline, params, region)
            elapsed = time.time() - t0
            r["params"] = params["name"]
            r["region"] = region["name"]
            rows.append(r)
            won_mark = "✅" if r["won"] else "❌"
            print(f"  {region['name']:<12}  动量 {r['return_pct']:+6.2f}%  HS300 {r.get('hs300_pct', 0):+6.2f}%  "
                  f"超额 {r['excess_pct']:+6.2f}pp  {won_mark}  ({elapsed:.0f}s)")

    # 汇总分析
    print("\n" + "=" * 80)
    print("📊 参数对比")
    print("=" * 80)
    df = pd.DataFrame(rows)

    summary = df.groupby("params").agg(
        avg_excess=("excess_pct", "mean"),
        win_rate=("won", "mean"),
        worst_excess=("excess_pct", "min"),
        best_excess=("excess_pct", "max"),
    ).round(2)
    summary["综合评分"] = (summary["avg_excess"] + summary["worst_excess"]).round(2)
    summary = summary.sort_values("综合评分", ascending=False)
    print(summary.to_string())

    print(f"\n🏆 推荐参数组合（综合评分 = 平均超额 + 最差超额，越高越稳健）:")
    print(f"   {summary.index[0]}")

    import json
    out_path = Path("runs/parameter_sweep_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ 报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
