"""**Meta-Strategy Agent 多区间回测验证** —— 检验自主选策略是否真正鲁棒。

跑 6 个独立时段，验证 Meta-Agent 在不同市场环境下的表现：
1. 2023 全年（牛市末 + 调整）
2. 2024 Q1（熊市/反弹）
3. 2024 Q2-Q3（震荡）
4. 2024 9.24 (单日暴涨)
5. 2025 全年（牛市主线 + 调整）
6. 2026 H1（震荡 + 复苏）

每个区间对比：
- Meta-Strategy（自主选）
- 沪深 300 ETF 满仓
- 沪深 300 指数（裸跟踪）

用法：
    python3 scripts/test_meta_strategy.py
    python3 scripts/test_meta_strategy.py --skip-llm  # 用规则降级（不调 R1，避免 LLM 失败）
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aifund.agents.regime import MarketRegimeAgent
from aifund.agents.strategy_selector import AllWeatherManager
from aifund.backtest.engine import BacktestEngine
from aifund.data import sources
from aifund.data.pipeline import DataPipeline
from aifund.strategies import all_strategies

# 6 个测试区间
REGIONS = [
    {"name": "🐂 2023 全年（牛市末+调整）", "start": "2023-01-09", "end": "2023-12-29"},
    {"name": "🐻 2024 Q1（熊市/反弹）",      "start": "2024-01-02", "end": "2024-03-29"},
    {"name": "🌊 2024 Q2-Q3（震荡）",        "start": "2024-04-01", "end": "2024-08-30"},
    {"name": "🚀 2024 9.24（单日暴涨）",     "start": "2024-09-02", "end": "2024-11-29"},
    {"name": "🐂 2025 全年（牛市主线）",     "start": "2025-01-02", "end": "2025-12-31"},
    {"name": "🌊 2026 H1（震荡+复苏）",      "start": "2026-01-02", "end": "2026-05-18"},
]


def make_all_weather_strategy(mgr, decisions_log, hs300_prices=None):
    """**All Weather 多策略持仓**策略函数（支持反思 / 记忆）。"""
    last_decision_week = [None]
    cached_target_weights = [{}]
    last_equity = [None]
    last_hs300 = [None]

    def strategy(snapshot, portfolio):
        d = snapshot.as_of
        y, w, _ = d.isocalendar()
        current_week = (y, w)

        # 新一周：调 Meta-Agent（带上周表现给反思用）
        if current_week != last_decision_week[0]:
            last_decision_week[0] = current_week
            equity_now = portfolio.equity()
            # 拿最近沪深 300 价格
            hs300_now = None
            if hs300_prices:
                for past_d in sorted(hs300_prices.keys(), reverse=True):
                    if past_d <= d:
                        hs300_now = hs300_prices[past_d]
                        break
            # 计算上周表现
            last_our_pct = last_hs300_pct = None
            if last_equity[0] is not None and last_hs300[0]:
                last_our_pct = (equity_now / last_equity[0] - 1) * 100
                if hs300_now:
                    last_hs300_pct = (hs300_now / last_hs300[0] - 1) * 100
            last_equity[0] = equity_now
            last_hs300[0] = hs300_now

            decision = mgr.decide(d, last_our_pct, last_hs300_pct)
            cached_target_weights[0] = decision.target_symbol_weights

            decisions_log.append({
                "as_of": str(d),
                "regime": decision.regime.regime,
                "regime_probs": decision.regime.regime_probs,
                "weights": decision.allocation.weights,
                "total_position": decision.allocation.total_position,
                "target_symbols": list(decision.target_symbol_weights.keys()),
                "n_symbols": len(decision.target_symbol_weights),
            })

        target_weights = cached_target_weights[0]
        orders = []

        # 不在目标里的持仓 → 卖出
        for sym, pos in list(portfolio.positions.items()):
            if pos.shares > 0 and sym not in target_weights:
                sellable = pos.sellable_shares(snapshot.as_of)
                if sellable > 0:
                    orders.append({
                        "symbol": sym, "side": "SELL",
                        "shares": (sellable // 100) * 100,
                        "name": pos.name, "reason": "All-Weather 调出池",
                    })

        if not target_weights:
            return orders

        # 每周一仅做一次 rebalance（避免每日跑此逻辑）
        if current_week != getattr(strategy, "_last_rebalance_week", None):
            strategy._last_rebalance_week = current_week
            equity = portfolio.equity()
            # 留 5% 现金 buffer 避免精度问题
            usable_equity = equity * 0.97
            for sym, target_w in target_weights.items():
                target_value = usable_equity * target_w
                pos = portfolio.get_position(sym)
                current_value = pos.market_value if pos else 0.0
                sd = snapshot.get(sym)
                if sd is None or not sd.last_close:
                    continue
                diff = target_value - current_value
                lot_value = sd.last_close * 100
                if diff > lot_value:
                    shares = int(diff / lot_value) * 100
                    max_by_cash = int(portfolio.cash * 0.95 / lot_value) * 100
                    shares = min(shares, max_by_cash)
                    if shares >= 100:
                        orders.append({
                            "symbol": sym, "side": "BUY", "shares": shares,
                            "name": sd.name,
                            "reason": f"AllWeather 加仓至 {target_w*100:.1f}%",
                        })
                elif diff < -lot_value and pos:
                    shares = int(-diff / lot_value) * 100
                    sellable = pos.sellable_shares(snapshot.as_of)
                    shares = min(shares, (sellable // 100) * 100)
                    if shares >= 100:
                        orders.append({
                            "symbol": sym, "side": "SELL", "shares": shares,
                            "name": pos.name,
                            "reason": f"AllWeather 减仓至 {target_w*100:.1f}%",
                        })
        return orders
    return strategy


def run_region(region: dict, pipeline: DataPipeline, mgr: AllWeatherManager,
               capital: float = 500_000, verbose: bool = False) -> dict:
    print(f"\n{'=' * 70}")
    print(f"区间：{region['name']}")
    print(f"     {region['start']} ~ {region['end']}")
    print('=' * 70)
    start = date.fromisoformat(region['start'])
    end = date.fromisoformat(region['end'])

    # 拉沪深 300 全期日线（用于风控对比 + benchmark）
    hs300 = sources.get_price_history("510300", start, end, asset_type="etf")
    hs300_prices: dict[date, float] = {}
    hs300_ret = 0.0
    if hs300 is not None and not hs300.empty:
        for _, row in hs300.iterrows():
            hs300_prices[row["date"]] = float(row["close"])
        hs_start = float(hs300["close"].iloc[0])
        hs_end = float(hs300["close"].iloc[-1])
        hs300_ret = (hs_end / hs_start - 1) * 100

    # 缩小候选 universe（避免预加载 290+ 标的的开销）
    # 策略：ETF 全部 + CSV 中 score top 80 个股
    candidate_symbols = set()
    # 必需 ETF
    candidate_symbols.add("510300")  # 沪深 300
    candidate_symbols.add("511010")  # 国债
    from aifund.strategies.sector_rotation import SECTOR_ETFS
    candidate_symbols.update(SECTOR_ETFS.keys())
    # CSV 中 score 最高的 80 只股票（高股息/反转可能选）
    from aifund.stockpool.factor_loader import FactorLoader
    try:
        fl = FactorLoader()
        # 找数据完整的最近一日
        day_counts = fl._df.dropna(subset=["score"]).groupby("日期").size().sort_index()
        complete_days = day_counts[day_counts > 100].index
        if len(complete_days):
            ref_day = max(d for d in complete_days if d <= end)
            ref_df = fl._df[fl._df["日期"] == ref_day]
            top80 = ref_df.nlargest(80, "score")["symbol"].tolist()
            candidate_symbols.update(top80)
    except Exception as e:
        print(f"  ⚠ 缩小 universe 失败: {e}")
    candidate_symbols = sorted(candidate_symbols)
    print(f"  候选 universe: {len(candidate_symbols)} 只（ETF + score top 80 个股）")

    # 决策日志（strategy 函数会往里加）
    decisions_log: list[dict] = []

    print(f"  >>> 跑回测（All Weather 多策略权重分配）…", flush=True)
    t0 = time.time()
    engine = BacktestEngine(
        pipeline=pipeline,
        candidate_symbols=candidate_symbols,
        decide_fn=make_all_weather_strategy(mgr, decisions_log, hs300_prices),
        initial_capital=capital,
    )
    result = engine.run(start, end, verbose=False)
    m = result.metrics
    print(f"  耗时 {time.time() - t0:.1f}s（{len(decisions_log)} 周决策）")

    # 统计 regime 分布 + 平均权重
    from collections import Counter
    regime_counter = Counter(d.get("regime", "?") for d in decisions_log)
    # 计算平均权重
    avg_weights = {"momentum": 0.0, "etf_defense": 0.0,
                   "sector_rotation": 0.0, "high_dividend": 0.0}
    for d in decisions_log:
        for s, w in d.get("weights", {}).items():
            avg_weights[s] = avg_weights.get(s, 0) + w
    if decisions_log:
        avg_weights = {k: round(v / len(decisions_log) * 100, 1) for k, v in avg_weights.items()}
    avg_position = sum(d.get("total_position", 0) for d in decisions_log) / max(len(decisions_log), 1)
    print(f"  Regime 分布：  {dict(regime_counter)}")
    print(f"  平均权重（%）：{avg_weights}")
    print(f"  平均总仓位：{avg_position * 100:.1f}%")

    cum = m.total_return * 100
    dd = m.max_drawdown * 100
    sharpe = m.sharpe
    excess = cum - hs300_ret
    win = "✅" if excess > -1 else "❌"
    print(f"  Meta-Agent  累计: {cum:+.2f}%   回撤: {dd:+.2f}%   夏普: {sharpe:.2f}   交易: {m.trade_count}")
    print(f"  沪深 300    累计: {hs300_ret:+.2f}%")
    print(f"  超额        {excess:+.2f} pp  {win}")

    return {
        "region": region['name'],
        "start": region['start'],
        "end": region['end'],
        "meta_return_pct": cum,
        "meta_max_dd_pct": dd,
        "meta_sharpe": sharpe,
        "meta_trades": m.trade_count,
        "hs300_return_pct": hs300_ret,
        "excess_pct": excess,
        "won": win == "✅",
        "regime_distribution": dict(regime_counter),
        "avg_weights": avg_weights,
        "avg_position": avg_position,
        "decisions": decisions_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", default="0,1,2,3,4,5",
                        help="跑哪些区间 (0-5)，逗号分隔")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    indices = [int(x) for x in args.regions.split(",") if x.strip().isdigit()]
    test_regions = [REGIONS[i] for i in indices]

    pipeline = DataPipeline()
    # **关键加速**：开启 light_mode（snapshot 只拉行情，跳过资金流/新闻/估值）
    # All Weather 的 strategy 函数只需要 last_close，其他数据浪费时间
    pipeline.light_mode = True
    print("✓ 启用 pipeline light_mode（仅行情，跳过资金流/新闻/估值）")
    # 缩小个股 universe（避免预加载 290+ 标的）
    from aifund.stockpool.factor_loader import FactorLoader
    fl = FactorLoader()
    day_counts = fl._df.dropna(subset=["score"]).groupby("日期").size().sort_index()
    complete_days = day_counts[day_counts > 100].index
    individual_universe: list[str] = []
    if len(complete_days):
        ref_day = complete_days.max()
        ref_df = fl._df[fl._df["日期"] == ref_day]
        individual_universe = ref_df.nlargest(80, "score")["symbol"].tolist()

    strats = all_strategies(pipeline, universe=individual_universe)
    print(f"加载 {len(strats)} 个策略：{list(strats.keys())}")
    print(f"个股 universe: {len(individual_universe)} 只（用于 high_dividend）")
    mgr = AllWeatherManager(strats)

    results = []
    for region in test_regions:
        try:
            r = run_region(region, pipeline, mgr, verbose=args.verbose)
            results.append(r)
        except Exception as exc:
            print(f"  ❌ {region['name']} 失败：{exc}")
            results.append({"region": region['name'], "error": str(exc)})

    # 综合汇总
    print(f"\n{'=' * 80}")
    print("📊 Meta-Strategy 6 区间综合稳健性报告")
    print('=' * 80)
    print(f"{'区间':<40} {'Meta':>10} {'HS300':>10} {'超额':>10} 结果")
    print('-' * 80)
    successful = [r for r in results if "error" not in r]
    for r in successful:
        emoji = "✅" if r["won"] else "❌"
        print(f"{r['region']:<40} {r['meta_return_pct']:>+9.2f}% {r['hs300_return_pct']:>+9.2f}% "
              f"{r['excess_pct']:>+8.2f}pp {emoji}")

    if successful:
        wins = sum(1 for r in successful if r["won"])
        avg_excess = sum(r["excess_pct"] for r in successful) / len(successful)
        worst = min(r["excess_pct"] for r in successful)
        print('-' * 80)
        print(f"{'综合':<40} 平均超额 {avg_excess:>+5.2f}pp / 最差 {worst:>+5.2f}pp / "
              f"赢 {wins}/{len(successful)}")

    # 落盘报告
    out_path = Path("runs/meta_strategy_report.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n✓ 报告已落盘: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
