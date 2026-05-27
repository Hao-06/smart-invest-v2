"""**LiveTradingAgent 比赛模拟回测** —— 真正按驼灵大赛规则的回测。

工作流（每个交易日重复）：
1. 平台调用 `agent.recommend(as_of, portfolio)` → 收到 JSON 买入建议
2. 模拟平台按建议成交（**T 日开盘价**，**100 股整数倍**，扣手续费）
3. 更新持仓 + 现金
4. 期末按可用资金总额排名

注意（按官方规则）：
- 只支持 BUY（卖出由平台机制处理 / 复赛后对接）
- 100 股整数倍
- 推荐标的须当日可交易
- 每天调用一次

用法：
    python3 scripts/test_live_agent.py --start 2026-01-02 --end 2026-05-18
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

from aifund.agents.live_agent import LiveTradingAgent, PortfolioSnapshot
from aifund.data import sources
from aifund.data.pipeline import DataPipeline
from aifund.strategies import all_strategies


def get_open_price(symbol: str, as_of: date,
                   price_cache: dict[str, dict[date, dict]]) -> float | None:
    """获取标的某日开盘价（假设平台按 T 日开盘价成交）。"""
    if symbol in price_cache and as_of in price_cache[symbol]:
        bar = price_cache[symbol][as_of]
        return bar.get("open") if isinstance(bar, dict) else None
    return None


def is_tradable(symbol: str, as_of: date,
                price_cache: dict[str, dict[date, dict]]) -> bool:
    """简单判断标的当日是否可交易（有行情数据）。"""
    if symbol not in price_cache:
        return False
    return as_of in price_cache[symbol]


def get_close_price(symbol: str, as_of: date,
                    price_cache: dict[str, dict[date, dict]]) -> float | None:
    """获取某日收盘价（用于持仓估值）。"""
    if symbol not in price_cache:
        return None
    # 找 ≤ as_of 的最近收盘价
    if as_of in price_cache[symbol]:
        bar = price_cache[symbol][as_of]
        return bar.get("close") if isinstance(bar, dict) else None
    for past in sorted(price_cache[symbol].keys(), reverse=True):
        if past <= as_of:
            bar = price_cache[symbol][past]
            return bar.get("close") if isinstance(bar, dict) else None
    return None


def simulate_trade(symbol: str, volume: int, open_price: float,
                   portfolio: dict, commission_rate: float = 0.00025,
                   min_commission: float = 5.0,
                   stamp_tax_rate: float = 0.0005) -> tuple[bool, str]:
    """模拟交易：volume > 0 买入 / volume < 0 卖出。

    A 股规则：
    - 100 股整数倍
    - 买入扣现金 + 佣金；卖出加现金 - 佣金 - 印花税（千一）

    Returns:
        (success, message)
    """
    if open_price is None or open_price <= 0:
        return False, f"{symbol}: 无开盘价"
    if volume == 0 or abs(volume) % 100 != 0:
        return False, f"{symbol}: volume {volume} 不是 100 整数倍"

    is_buy = volume > 0
    abs_vol = abs(volume)
    amount = abs_vol * open_price

    if is_buy:
        # 买入
        fee = max(amount * commission_rate, min_commission)
        total_cost = amount + fee
        if portfolio["cash"] < total_cost:
            max_vol = (int((portfolio["cash"] - min_commission) / (open_price * 100))) * 100
            if max_vol < 100:
                return False, f"{symbol}: 买入现金不足"
            abs_vol = max_vol
            amount = abs_vol * open_price
            fee = max(amount * commission_rate, min_commission)
            total_cost = amount + fee
        portfolio["cash"] -= total_cost
        portfolio["holdings"][symbol] = portfolio["holdings"].get(symbol, 0) + abs_vol
        return True, f"{symbol}: BUY {abs_vol}@{open_price:.2f}"
    else:
        # 卖出
        current = portfolio["holdings"].get(symbol, 0)
        if current < abs_vol:
            abs_vol = (current // 100) * 100
            if abs_vol < 100:
                return False, f"{symbol}: 持仓不足卖出"
        amount = abs_vol * open_price
        fee = max(amount * commission_rate, min_commission)
        tax = amount * stamp_tax_rate
        net_in = amount - fee - tax
        portfolio["cash"] += net_in
        portfolio["holdings"][symbol] -= abs_vol
        if portfolio["holdings"][symbol] == 0:
            del portfolio["holdings"][symbol]
        return True, f"{symbol}: SELL {abs_vol}@{open_price:.2f} 入账{net_in:.2f}"


# 向后兼容别名
simulate_buy = simulate_trade


def mark_to_market(portfolio: dict, as_of: date,
                   price_cache: dict[str, dict[date, float]]) -> float:
    """计算组合当前总市值 = 现金 + 持仓市值"""
    equity = portfolio["cash"]
    for sym, shares in portfolio["holdings"].items():
        if shares <= 0:
            continue
        price = get_close_price(sym, as_of, price_cache)
        if price is None:
            continue
        equity += shares * price
    return equity


def preload_prices(symbols: list[str], start: date, end: date,
                   verbose: bool = True) -> dict[str, dict[date, float]]:
    """并发拉取所有候选标的的行情（含 OHLC）。返回 {symbol: {date: close}}。"""
    import threading

    price_cache: dict[str, dict[date, float]] = {}
    lock = threading.Lock()

    def _fetch(sym):
        try:
            asset_type = "etf" if (len(sym) == 6 and sym[:2] in ("51", "15", "58", "56")) else "stock"
            df = sources.get_price_history(sym, start - timedelta(days=30), end + timedelta(days=5),
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

    threads = []
    for sym in symbols:
        th = threading.Thread(target=_fetch, args=(sym,), daemon=True)
        th.start()
        threads.append(th)
    deadline = time.time() + 180
    for th in threads:
        remaining = max(0.1, deadline - time.time())
        th.join(timeout=remaining)
    if verbose:
        print(f"  ✓ 预加载 {len(price_cache)}/{len(symbols)} 只标的行情", flush=True)
    return price_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument("--capital", type=float, default=500_000)
    parser.add_argument("--name", default="2026 H1 真样本外")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"=== 🎯 LiveTradingAgent · 比赛模拟回测 · {args.name} ===")
    print(f"区间：{start} ~ {end}")
    print(f"初始资金：{args.capital:,.0f}")
    print(f"输出：每日 JSON 买入建议（符合驼灵大赛规则）")
    print()

    pipeline = DataPipeline()
    pipeline.light_mode = True

    # 候选 universe（ETF + CSV top 80）
    candidate_symbols = {"510300", "511010"}
    from aifund.strategies.sector_rotation import SECTOR_ETFS
    candidate_symbols.update(SECTOR_ETFS.keys())
    from aifund.stockpool.factor_loader import FactorLoader
    fl = FactorLoader()
    day_counts = fl._df.dropna(subset=["score"]).groupby("日期").size().sort_index()
    complete_days = day_counts[day_counts > 100].index
    individual_universe = []
    if len(complete_days):
        ref_day = complete_days.max()
        ref_df = fl._df[fl._df["日期"] == ref_day]
        individual_universe = ref_df.nlargest(80, "score")["symbol"].tolist()
        candidate_symbols.update(individual_universe)
    print(f"候选 universe：{len(candidate_symbols)} 只（ETF + score top 80 个股）")

    # 创建 Agent
    strats = all_strategies(pipeline, universe=individual_universe)
    print(f"加载 {len(strats)} 策略：{list(strats.keys())}")
    agent = LiveTradingAgent(pipeline=pipeline, strategies=strats)

    # 预加载行情
    print(f"\n>>> 预加载行情（{len(candidate_symbols)} 只 + 沪深 300 基准）…", flush=True)
    t0 = time.time()
    all_syms = sorted(candidate_symbols)
    price_cache = preload_prices(all_syms, start, end)
    print(f"  耗时 {time.time() - t0:.1f}s")

    # 拿沪深 300 基准
    hs300 = sources.get_price_history("510300", start, end, asset_type="etf")
    hs300_start_close = float(hs300["close"].iloc[0]) if not hs300.empty else 1.0
    hs300_end_close = float(hs300["close"].iloc[-1]) if not hs300.empty else 1.0
    hs300_return = (hs300_end_close / hs300_start_close - 1) * 100

    # 交易日列表（沪深 300 ETF 数据日期）
    if hs300.empty:
        print("✗ 无沪深 300 数据")
        return 1
    trading_days = sorted(hs300["date"].tolist())
    print(f"\n交易日数：{len(trading_days)}")

    # 初始化组合
    portfolio = {"cash": args.capital, "holdings": {}}
    daily_equity: list[tuple[date, float]] = []
    n_recommends = 0
    n_no_action = 0
    n_buys = 0
    n_sells = 0

    print(f"\n>>> 开始每日模拟…", flush=True)
    for i, today in enumerate(trading_days):
        # 1. 计算当前组合状态（用昨日收盘价估值，模拟开盘前快照）
        prev_day = trading_days[i - 1] if i > 0 else today
        equity = mark_to_market(portfolio, prev_day, price_cache)
        snapshot = PortfolioSnapshot(
            as_of=today, cash=portfolio["cash"],
            holdings=dict(portfolio["holdings"]), equity=equity,
        )

        # 2. Agent 给出今日推荐
        try:
            recs, decision = agent.recommend(today, snapshot)
        except Exception as exc:
            print(f"  {today}: Agent 失败 ({type(exc).__name__}: {exc})", flush=True)
            daily_equity.append((today, equity))
            continue

        # 3. 模拟买入
        if recs:
            n_recommends += 1
            for r in recs:
                sym = r["symbol"]
                vol = int(r["volume"])
                open_price = None
                if sym in price_cache:
                    bars = price_cache[sym].get(today)
                    if bars:
                        open_price = bars.get("open", bars.get("close"))
                if open_price is None or not is_tradable(sym, today, price_cache):
                    continue
                ok, msg = simulate_trade(sym, vol, open_price, portfolio)
                if ok:
                    if vol > 0:
                        n_buys += 1
                    else:
                        n_sells += 1
        else:
            n_no_action += 1

        # 4. 用今日收盘价计算净值
        equity_eod = mark_to_market(portfolio, today, price_cache)
        daily_equity.append((today, equity_eod))

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(trading_days)}] {today} equity={equity_eod:,.0f} "
                  f"cash={portfolio['cash']:,.0f} 持仓={len(portfolio['holdings'])}",
                  flush=True)

    # 期末汇总
    final_equity = daily_equity[-1][1] if daily_equity else args.capital
    total_return = (final_equity / args.capital - 1) * 100
    excess = total_return - hs300_return

    # 最大回撤
    peak = args.capital
    max_dd = 0.0
    for _, eq in daily_equity:
        peak = max(peak, eq)
        dd = (eq / peak - 1) * 100
        if dd < max_dd:
            max_dd = dd

    print(f"\n{'='*70}")
    print(f"  📊 LiveTradingAgent 比赛模拟结果")
    print(f"{'='*70}")
    print(f"  交易日数：{len(trading_days)}")
    print(f"  推荐日数：{n_recommends} / 无操作日：{n_no_action}")
    print(f"  实际成交：买入 {n_buys} 笔 / 卖出 {n_sells} 笔")
    print(f"")
    print(f"  📈 Agent 累计收益：{total_return:+.2f}%   最大回撤：{max_dd:+.2f}%")
    print(f"  📊 沪深 300 同期：{hs300_return:+.2f}%")
    print(f"  🎯 超额收益：{excess:+.2f} pp  {'✅' if excess > -1 else '❌'}")
    print(f"")
    print(f"  💰 初始资金 ¥{args.capital:,.0f} → 期末资金 ¥{final_equity:,.0f}")

    # 落盘报告
    out_path = Path("runs/live_agent_report.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "region": args.name,
        "start": str(start), "end": str(end),
        "initial_capital": args.capital,
        "final_equity": final_equity,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "hs300_return_pct": hs300_return,
        "excess_pct": excess,
        "n_trading_days": len(trading_days),
        "n_recommend_days": n_recommends,
        "n_no_action_days": n_no_action,
        "n_buys": n_buys,
        "daily_equity": [{"date": str(d), "equity": e} for d, e in daily_equity],
        "decisions": [
            {
                "as_of": str(d.as_of),
                "type": d.decision_type,
                "n_recs": len(d.recommendations),
                "recs": d.recommendations,
                "regime": d.regime,
                "weights": d.weights,
                "health_action": d.health_action,
            }
            for d in agent.decision_log
        ],
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n✓ 决策日志已落盘：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
