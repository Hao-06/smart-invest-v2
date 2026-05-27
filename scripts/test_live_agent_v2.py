"""**LiveTradingAgentV2 · 纯推荐版回测** —— 验证「Agent 只给推荐 / 平台管执行」架构。

回测模拟（假设平台行为）：
1. 每天调用 agent.recommend(as_of) → 收到 JSON 推荐
2. 平台**等权买入**所有推荐的票（**T+5 自动平仓** A 股短期策略常用规则）
3. 期末按持仓估值 + 现金计算总资产

注：这是**对平台行为的简化假设**，仅供回测演示。
真实复赛平台行为以官方技术文档为准。

用法：
    python3 scripts/test_live_agent_v2.py --start 2026-01-02 --end 2026-05-18
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aifund.agents.live_agent_v2 import LiveTradingAgentV2
from aifund.data import sources
from aifund.data.pipeline import DataPipeline
from aifund.strategies import all_strategies


def preload_prices(symbols, start, end):
    """并发拉取 OHLC（{symbol: {date: {open, close}}}）"""
    import threading
    price_cache: dict = {}
    lock = threading.Lock()

    def _fetch(sym):
        try:
            asset_type = "etf" if (len(sym) == 6 and sym[:2] in ("51", "15", "58", "56")) else "stock"
            df = sources.get_price_history(sym, start - timedelta(days=30), end + timedelta(days=10),
                                            asset_type=asset_type)
            if df is None or df.empty:
                return
            d = {}
            for _, row in df.iterrows():
                d[row["date"]] = {"open": float(row["open"]), "close": float(row["close"])}
            with lock:
                price_cache[sym] = d
        except Exception:
            pass

    threads = [threading.Thread(target=_fetch, args=(s,), daemon=True) for s in symbols]
    for th in threads:
        th.start()
    deadline = time.time() + 180
    for th in threads:
        th.join(timeout=max(0.1, deadline - time.time()))
    return price_cache


def get_price(sym, dt, price_cache, field="close"):
    bars = price_cache.get(sym)
    if not bars:
        return None
    if dt in bars:
        return bars[dt].get(field)
    # 找最近 ≤ dt 的
    for past in sorted(bars.keys(), reverse=True):
        if past <= dt:
            return bars[past].get(field)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument("--capital", type=float, default=500_000)
    parser.add_argument("--hold-days", type=int, default=5,
                        help="假设平台 T+N 自动平仓的 N 天数")
    parser.add_argument("--name", default="2026 H1 V2 纯推荐")
    parser.add_argument("--out", default="runs/live_agent_v2_report.json",
                        help="结果 JSON 输出路径（多区间验证时各区间用不同文件名）")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"=== 🎯 LiveTradingAgentV2 · 纯推荐版回测 · {args.name} ===")
    print(f"区间：{start} ~ {end}")
    print(f"初始资金：{args.capital:,.0f}")
    print(f"平台模拟规则：每天等权买入 V2 推荐 + T+{args.hold_days} 自动平仓")
    print()

    pipeline = DataPipeline()
    pipeline.light_mode = True

    # 候选 universe
    candidate_symbols = {"510300", "511010"}
    from aifund.strategies.sector_rotation import SECTOR_ETFS
    candidate_symbols.update(SECTOR_ETFS.keys())
    from aifund.stockpool.factor_loader import FactorLoader
    fl = FactorLoader()
    # PIT 正确：用回测起点前最近一个「完整交易日」的因子快照选 universe ——
    # 杜绝「用未来因子分数挑历史股票」的前视偏差（look-ahead bias），
    # 同时跳过 CSV 边界稀疏日（避免得到残缺的 1 只票 universe）。
    pit_universe = fl.pit_top_n(start, n=80)
    individual_universe = pit_universe["symbol"].tolist()
    if individual_universe:
        pit_day = pit_universe["日期"].iloc[0]
        print(f"个股 universe：{len(individual_universe)} 只（PIT 因子快照 @ {pit_day}）")
    else:
        lo, _ = fl.date_range
        print(f"个股 universe：空（{start} 早于因子数据起点 {lo} → 本区间仅 ETF 策略生效）")
    candidate_symbols.update(individual_universe)
    print(f"候选 universe（含 ETF）：{len(candidate_symbols)} 只")

    strats = all_strategies(pipeline, universe=individual_universe)
    print(f"加载 {len(strats)} 策略：{list(strats.keys())}")

    # 回测固定 offline 模式：regime/allocator 走规则判断 ——
    # 确定性、可复现，且杜绝「LLM 知道未来」的信息泄漏。
    agent = LiveTradingAgentV2(pipeline=pipeline, strategies=strats, offline=True)
    print("Agent 模式：offline（规则判断，确定性回测）")

    print(f"\n>>> 预加载行情…", flush=True)
    t0 = time.time()
    price_cache = preload_prices(sorted(candidate_symbols), start, end)
    print(f"  ✓ {len(price_cache)}/{len(candidate_symbols)} 只 (耗时 {time.time() - t0:.1f}s)")

    # 沪深 300 基准
    hs300 = sources.get_price_history("510300", start, end, asset_type="etf")
    hs300_return = ((float(hs300["close"].iloc[-1]) / float(hs300["close"].iloc[0]) - 1) * 100
                    if not hs300.empty else 0)
    trading_days = sorted(hs300["date"].tolist())
    print(f"  交易日数：{len(trading_days)}")

    # 模拟平台
    cash = args.capital
    # 持仓：{symbol: [(buy_date, shares, buy_price)]} — 用列表跟踪 FIFO 平仓
    holdings: dict[str, list] = {}
    daily_equity = []
    n_recommends = 0
    n_no_action = 0
    n_buys = 0
    n_sells = 0

    print(f"\n>>> 开始每日模拟…", flush=True)
    for i, today in enumerate(trading_days):
        # 1. T+hold_days 自动平仓
        for sym in list(holdings.keys()):
            remaining = []
            for buy_date, shares, buy_price in holdings[sym]:
                hold_days = (today - buy_date).days
                if hold_days >= args.hold_days:
                    # 平仓
                    sell_price = get_price(sym, today, price_cache, "open")
                    if sell_price is None:
                        remaining.append((buy_date, shares, buy_price))
                        continue
                    amount = shares * sell_price
                    fee = max(amount * 0.00025, 5.0)
                    tax = amount * 0.0005
                    cash += amount - fee - tax
                    n_sells += 1
                else:
                    remaining.append((buy_date, shares, buy_price))
            if remaining:
                holdings[sym] = remaining
            else:
                del holdings[sym]

        # 2. V2 给出今日推荐
        try:
            recs, decision = agent.recommend(today)
        except Exception as exc:
            print(f"  {today}: Agent 失败 ({exc})", flush=True)
            # 估值跳过
            equity = cash + sum(
                shares * (get_price(sym, today, price_cache) or buy_price)
                for sym, lots in holdings.items()
                for buy_date, shares, buy_price in lots
            )
            daily_equity.append((today, equity))
            continue

        # 3. 模拟平台买入：按 volume 推荐买（如果现金够）
        if recs:
            n_recommends += 1
            for r in recs:
                sym = r["symbol"]
                vol = int(r["volume"])
                if vol <= 0:
                    continue
                buy_price = get_price(sym, today, price_cache, "open")
                if buy_price is None or buy_price <= 0:
                    continue
                cost = vol * buy_price
                fee = max(cost * 0.00025, 5.0)
                total = cost + fee
                if cash < total:
                    # 按可用现金截断
                    max_vol = (int((cash - 5) / (buy_price * 100))) * 100
                    if max_vol < 100:
                        continue
                    vol = max_vol
                    cost = vol * buy_price
                    fee = max(cost * 0.00025, 5.0)
                    total = cost + fee
                cash -= total
                holdings.setdefault(sym, []).append((today, vol, buy_price))
                n_buys += 1
        else:
            n_no_action += 1

        # 4. 按收盘价估值
        equity = cash + sum(
            sum(shares * (get_price(sym, today, price_cache) or buy_price)
                for buy_date, shares, buy_price in lots)
            for sym, lots in holdings.items()
        )
        daily_equity.append((today, equity))

        if (i + 1) % 20 == 0:
            n_pos = sum(len(lots) for lots in holdings.values())
            print(f"  [{i+1}/{len(trading_days)}] {today} equity={equity:,.0f} "
                  f"cash={cash:,.0f} 持仓批次={n_pos}", flush=True)

    # 期末汇总
    final = daily_equity[-1][1] if daily_equity else args.capital
    total_ret = (final / args.capital - 1) * 100
    excess = total_ret - hs300_return

    peak = args.capital
    max_dd = 0.0
    for _, eq in daily_equity:
        peak = max(peak, eq)
        dd = (eq / peak - 1) * 100
        max_dd = min(max_dd, dd)

    print(f"\n{'='*70}")
    print(f"  📊 LiveAgentV2 · 纯推荐版结果")
    print(f"{'='*70}")
    print(f"  推荐日数：{n_recommends} / 无操作日：{n_no_action}")
    print(f"  成交：买入 {n_buys} 笔 / 卖出 {n_sells} 笔 (T+{args.hold_days} 平仓)")
    print(f"  📈 V2 累计：{total_ret:+.2f}%   回撤：{max_dd:+.2f}%")
    print(f"  📊 沪深300：{hs300_return:+.2f}%")
    print(f"  🎯 超额：{excess:+.2f} pp  {'✅' if excess > -1 else '❌'}")
    print(f"  💰 ¥{args.capital:,.0f} → ¥{final:,.0f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 构造每日净值序列（V2 + 沪深300 同步基准）—— 供可视化看板画净值曲线
    hs300_by_date = {row["date"]: float(row["close"]) for _, row in hs300.iterrows()} if not hs300.empty else {}
    hs300_first = float(hs300["close"].iloc[0]) if not hs300.empty else 1.0
    equity_series = []
    for d, eq in daily_equity:
        hs_close = hs300_by_date.get(d)
        equity_series.append({
            "date": str(d),
            "v2_equity": eq,
            "v2_pct": (eq / args.capital - 1) * 100,
            "hs300_close": hs_close,
            "hs300_pct": ((hs_close / hs300_first - 1) * 100) if hs_close else None,
        })

    out_path.write_text(json.dumps({
        "region": args.name, "start": str(start), "end": str(end),
        "capital": args.capital, "final": final,
        "total_return_pct": total_ret, "max_drawdown_pct": max_dd,
        "hs300_return_pct": hs300_return, "excess_pct": excess,
        "n_recommend_days": n_recommends, "n_no_action": n_no_action,
        "n_buys": n_buys, "n_sells": n_sells, "hold_days": args.hold_days,
        "daily_equity": equity_series,
        "decisions": [
            {"as_of": str(d.as_of), "n_recs": len(d.recommendations),
             "recs": d.recommendations, "regime": d.regime, "weights": d.weights}
            for d in agent.decision_log
        ],
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n✓ 报告：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
