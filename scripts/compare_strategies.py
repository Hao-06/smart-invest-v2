"""策略对比回测：多 Agent 团队 vs 买入持有 baseline vs 沪深300 基准。

为设计书的「回测结果」一章生成硬数据：在同一区间下对比三条曲线 ——
- A) 多 Agent 团队（咱们的核心方案）
- B) 等权买入持有候选标的（被动基准 1）
- C) 沪深300 指数（行业标准基准）

输出：
- 控制台打印三方对比表
- ``runs/comparison_<start>_<end>/`` 目录下：
  - ``metrics.json``     —— 全部绩效指标
  - ``equity_curves.csv``—— 三策略每日净值
  - ``comparison.png``   —— 三曲线对比图（若装了 matplotlib）

用法：
    python scripts/compare_strategies.py --start 2026-04-15 --end 2026-04-30 \\
        --symbols 600519,601318 --capital 500000

⚠️ 多 Agent 策略每个交易日 ≈ 6 次 LLM 调用（5 分析师 + 基金经理），
   60 日回测约 3-6 元 RMB。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# 让本脚本无需安装即可运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from aifund.agents import AgentTeam
from aifund.backtest import BacktestEngine, Portfolio
from aifund.backtest.metrics import (
    PerformanceMetrics, annualized_return, max_drawdown,
    sharpe_ratio, total_return, volatility, calmar_ratio,
)
from aifund.data import DataPipeline, sources
from aifund.data.models import MarketSnapshot
from config.settings import settings


# ---------------------------------------------------------------------------
# Baseline 策略：等权买入持有
# ---------------------------------------------------------------------------


def make_buy_and_hold_strategy(target_total_position_pct: float = 0.95):
    """构造一个等权买入持有策略。

    第一日按候选标的的可成交价等权分配资金（留 ``1-target`` 比例现金），
    后续日全部空操作。简单、稳定，作为对照基准。
    """

    def strategy(snapshot: MarketSnapshot, portfolio: Portfolio) -> list[dict]:
        # 已有任何持仓 → 不再下单
        if any(p.shares > 0 for p in portfolio.positions.values()):
            return []
        candidates = [s for s in snapshot.tradable_symbols if snapshot.get(s) and snapshot.get(s).last_close]
        if not candidates:
            return []
        per_stock_cap = portfolio.cash * target_total_position_pct / len(candidates)
        orders: list[dict] = []
        for sym in candidates:
            sd = snapshot.get(sym)
            if sd is None or not sd.last_close:
                continue
            # 100 股整数倍下取整
            shares = int(per_stock_cap // (sd.last_close * 100)) * 100
            if shares >= 100:
                orders.append({
                    "symbol": sym, "side": "BUY", "shares": shares,
                    "name": sd.name, "reason": "Baseline 等权建仓",
                })
        return orders

    return strategy


# ---------------------------------------------------------------------------
# 沪深300 基准（业内标准 benchmark）
# ---------------------------------------------------------------------------


def build_hs300_equity_curve(
    start_date: str | date, end_date: str | date, initial_capital: float,
) -> tuple[list[tuple[date, float]], PerformanceMetrics | None]:
    """以沪深300指数（或 ETF 510300 代理）作为纯持有 benchmark，返回与 BacktestResult 兼容的净值曲线 + 指标。

    优先用指数行情接口；失败时回退到沪深300 ETF（510300，多源容灾覆盖）。
    """
    hs300 = sources.get_index_history("000300", start_date, end_date)
    if hs300 is None or hs300.empty:
        # 指数源不可用 → 用沪深300 ETF 510300 作为代理（ETF 有腾讯备用源）
        print("  [基准] 指数接口不可用，回退到沪深300 ETF 510300 作为代理")
        hs300 = sources.get_price_history("510300", start_date, end_date, asset_type="etf")
    if hs300 is None or hs300.empty:
        return [], None

    hs300 = hs300.sort_values("date").reset_index(drop=True)
    base_close = float(hs300.iloc[0]["close"])
    if base_close <= 0:
        return [], None

    equity_curve: list[tuple[date, float]] = []
    for _, row in hs300.iterrows():
        d = row["date"]
        try:
            close = float(row["close"])
        except (TypeError, ValueError):
            continue
        if close <= 0:
            continue
        equity = initial_capital * (close / base_close)
        equity_curve.append((d, equity))

    if not equity_curve:
        return equity_curve, None

    # 直接基于净值曲线计算指标（沪深300 无交易笔数，胜率/换手率为 0）
    mdd, peak, trough = max_drawdown(equity_curve)
    days = (equity_curve[-1][0] - equity_curve[0][0]).days
    metrics = PerformanceMetrics(
        start_date=equity_curve[0][0],
        end_date=equity_curve[-1][0],
        days=days,
        initial_capital=initial_capital,
        final_equity=equity_curve[-1][1],
        total_return=total_return(equity_curve, initial_capital),
        annualized_return=annualized_return(equity_curve, initial_capital),
        max_drawdown=mdd,
        drawdown_peak=peak,
        drawdown_trough=trough,
        sharpe=sharpe_ratio(equity_curve),
        calmar=calmar_ratio(equity_curve, initial_capital),
        volatility=volatility(equity_curve),
        win_rate=0.0,
        profit_factor=0.0,
        win_count=0,
        loss_count=0,
        trade_count=0,
        turnover=0.0,
    )
    return equity_curve, metrics


# ---------------------------------------------------------------------------
# 运行对比
# ---------------------------------------------------------------------------


def metrics_to_row(name: str, m: PerformanceMetrics) -> dict[str, object]:
    return {
        "策略": name,
        "累计收益率": f"{m.total_return * 100:+.2f}%",
        "年化收益率": f"{m.annualized_return * 100:+.2f}%",
        "最大回撤": f"{m.max_drawdown * 100:.2f}%",
        "夏普比率": round(m.sharpe, 3),
        "Calmar": round(m.calmar, 3),
        "年化波动率": f"{m.volatility * 100:.2f}%",
        "胜率": f"{m.win_rate * 100:.1f}%",
        "交易笔数": m.trade_count,
        "换手率": round(m.turnover, 3),
        "期末权益": round(m.final_equity, 2),
    }


def metrics_to_serializable(m: PerformanceMetrics) -> dict[str, object]:
    """把 dataclass 转为可 JSON 序列化的纯字典。"""
    return {
        "start_date": str(m.start_date) if m.start_date else None,
        "end_date": str(m.end_date) if m.end_date else None,
        "days": m.days,
        "initial_capital": m.initial_capital,
        "final_equity": m.final_equity,
        "total_return": m.total_return,
        "annualized_return": m.annualized_return,
        "max_drawdown": m.max_drawdown,
        "drawdown_peak": str(m.drawdown_peak) if m.drawdown_peak else None,
        "drawdown_trough": str(m.drawdown_trough) if m.drawdown_trough else None,
        "sharpe": m.sharpe,
        "calmar": m.calmar,
        "volatility": m.volatility,
        "win_rate": m.win_rate,
        "profit_factor": (
            m.profit_factor if m.profit_factor != float("inf") else None
        ),
        "win_count": m.win_count,
        "loss_count": m.loss_count,
        "trade_count": m.trade_count,
        "turnover": m.turnover,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="策略对比回测：多 Agent vs 买入持有")
    parser.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--symbols", help="候选标的，逗号分隔（不与 --auto-pool 同用）")
    parser.add_argument("--auto-pool", action="store_true",
                        help="启用自主选股（每周一从沪深300 选 top-K）")
    parser.add_argument("--selector", choices=("auto", "momentum"), default="momentum",
                        help="选股器：auto = 多因子综合打分；momentum = 动量+大盘择时（推荐）")
    parser.add_argument("--auto-csv", help="多因子 CSV 路径，默认 ~/量化金融/多因子策略优化-re/5.2…")
    parser.add_argument("--top1", type=int, default=20, help="自动选股第 1 层粗筛保留数")
    parser.add_argument("--top2", type=int, default=8, help="自动选股第 2 层最终保留数")
    parser.add_argument("--no-timing", action="store_true",
                        help="禁用大盘择时（momentum 选股器才有效）")
    parser.add_argument("--capital", type=float, default=settings.backtest.initial_capital)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--skip-agent", action="store_true",
                        help="跳过多 Agent 策略，只跑 baseline（用于无 key 时快速验证）")
    args = parser.parse_args()

    if args.auto_pool and args.symbols:
        print("⚠ --auto-pool 与 --symbols 互斥，使用 --auto-pool 模式（忽略 --symbols）")
    if not args.auto_pool and not args.symbols:
        print("✗ 必须指定 --symbols 或 --auto-pool 之一")
        return 2

    symbols = (
        [s.strip() for s in args.symbols.replace("，", ",").split(",") if s.strip()]
        if args.symbols else []
    )
    if not args.auto_pool and not symbols:
        print("✗ --symbols 不能为空")
        return 2

    out_dir = settings.paths.runs / f"comparison_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"对比区间：{args.start} ~ {args.end}")
    print(f"模式：{'🤖 自主选股漏斗' if args.auto_pool else '👤 手工指定标的'}")
    if not args.auto_pool:
        print(f"候选标的：{symbols}")
    print(f"初始资金：{args.capital:,.0f} 元")
    print(f"输出目录：{out_dir}")

    pipeline = DataPipeline(lookback_days=args.lookback)
    results: dict[str, object] = {}

    # ------ 自主选股漏斗（如启用）------
    auto_ctx: dict | None = None
    if args.auto_pool:
        from aifund.stockpool import AutoSelector, MomentumSelector
        import bisect
        if args.selector == "momentum":
            selector = MomentumSelector(
                pipeline=pipeline,
                factor_csv_path=args.auto_csv,
                top_pool=args.top2,
                enable_timing=not args.no_timing,
            )
            csv_start, csv_end = selector.loader.date_range if selector.loader else (None, None)
            print(f"\n[自动选股] 模式: 🚀 动量+大盘择时")
            print(f"[自动选股] CSV 数据覆盖: {csv_start} ~ {csv_end}")
            print(f"[自动选股] 周频持仓 {args.top2} 只 · 大盘择时: "
                  f"{'禁用' if args.no_timing else '启用（510300 ETF 5MA/20MA）'}")
        else:
            selector = AutoSelector(
                pipeline=pipeline,
                factor_csv_path=args.auto_csv,
                top1=args.top1,
                top2=args.top2,
            )
            csv_start, csv_end = selector.loader.date_range
            print(f"\n[自动选股] 模式: 多因子综合打分")
            print(f"[自动选股] CSV 数据覆盖: {csv_start} ~ {csv_end}")
            print(f"[自动选股] 第 1 层 → top {args.top1}，第 2 层 → top {args.top2}（每周一重选）")
        print(">>> 预计算周频候选池…")
        weekly_pools = selector.precompute_weekly_pools(
            args.start, args.end, pipeline.calendar, verbose=True
        )
        if not weekly_pools:
            print("✗ 选股漏斗未产出候选池，可能区间在 CSV 覆盖之外")
            return 4
        rebalance_dates = sorted(weekly_pools.keys())
        all_symbols = sorted({s for pool in weekly_pools.values() for s in pool})
        # Baseline 用首个非空池子（大盘择时让早期可能空仓）
        initial_pool = []
        for d in rebalance_dates:
            if weekly_pools[d]:
                initial_pool = weekly_pools[d]
                break
        empty_weeks = sum(1 for d in rebalance_dates if not weekly_pools[d])
        print(f"\n[自动选股] 共 {len(rebalance_dates)} 周，"
              f"{empty_weeks} 周空仓避险，去重候选 {len(all_symbols)} 只")
        print(f"[自动选股] Baseline 用首个非空周池子: {initial_pool}")

        def get_pool_for_date(d: date) -> list[str]:
            i = bisect.bisect_right(rebalance_dates, d) - 1
            return weekly_pools[rebalance_dates[i]] if i >= 0 else []

        auto_ctx = {
            "weekly_pools": weekly_pools,
            "rebalance_dates": rebalance_dates,
            "all_symbols": all_symbols,
            "initial_pool": initial_pool,
            "get_pool_for_date": get_pool_for_date,
        }
        # 同时落盘周池子，便于设计书引用
        import json as _json
        with (out_dir / "weekly_pools.json").open("w", encoding="utf-8") as _f:
            _json.dump(
                {str(d): syms for d, syms in weekly_pools.items()},
                _f, ensure_ascii=False, indent=2, default=str,
            )

    # ------ 策略 B: 买入持有 baseline（一定先跑，零 LLM 成本）------
    baseline_pool = auto_ctx["initial_pool"] if auto_ctx else symbols
    print(f"\n>>> 跑 Baseline 等权买入持有（{len(baseline_pool)} 只标的）…")
    engine_bh = BacktestEngine(
        pipeline=pipeline,
        candidate_symbols=baseline_pool,
        decide_fn=make_buy_and_hold_strategy(),
        initial_capital=args.capital,
    )
    result_bh = engine_bh.run(args.start, args.end, verbose=False)
    print(f"  期末权益 {result_bh.metrics.final_equity:,.2f} "
          f"({result_bh.metrics.total_return * 100:+.2f}%)，"
          f"夏普 {result_bh.metrics.sharpe:.2f}")

    # ------ 策略 A: 多 Agent ------
    if args.skip_agent:
        print("\n>>> 跳过多 Agent 策略（--skip-agent）")
        result_ag = None
    else:
        if not settings.llm.deepseek_api_key:
            print("\n✗ 未配置 DEEPSEEK_API_KEY；如需跑多 Agent 策略请先填 .env，"
                  "或加 --skip-agent 仅跑 baseline。")
            return 3
        agent_pool = auto_ctx["all_symbols"] if auto_ctx else symbols
        print(f"\n>>> 跑多 Agent 策略（{len(agent_pool)} 只标的合集，每周自动重选 top-{args.top2}）…")
        team = AgentTeam(log_dir=out_dir / "decisions")
        if auto_ctx:
            # 包装 decide_fn：每日先过滤到当前周的池子
            def auto_pool_decide(snapshot, portfolio):
                current = auto_ctx["get_pool_for_date"](snapshot.as_of)
                if not current:
                    return []
                return team.decide(
                    snapshot, portfolio, symbols=current, save_log=True
                ).orders
            decide_fn = auto_pool_decide
        else:
            decide_fn = team.as_decide_fn(save_log=True)

        engine_ag = BacktestEngine(
            pipeline=pipeline,
            candidate_symbols=agent_pool,
            decide_fn=decide_fn,
            initial_capital=args.capital,
        )
        result_ag = engine_ag.run(args.start, args.end, verbose=True)
        print(f"  期末权益 {result_ag.metrics.final_equity:,.2f} "
              f"({result_ag.metrics.total_return * 100:+.2f}%)，"
              f"夏普 {result_ag.metrics.sharpe:.2f}")

    # ------ 沪深300 基准 ------
    print("\n>>> 拉取沪深300 指数基准（行业标准 benchmark）…")
    hs300_curve, hs300_metrics = build_hs300_equity_curve(
        args.start, args.end, args.capital
    )
    if hs300_metrics is not None:
        print(f"  期末权益 {hs300_metrics.final_equity:,.2f} "
              f"({hs300_metrics.total_return * 100:+.2f}%)，"
              f"夏普 {hs300_metrics.sharpe:.2f}")
    else:
        print("  ⚠ 沪深300 数据拉取失败，跳过")

    # ------ 输出对比表 ------
    print("\n" + "=" * 70)
    print("策略对比")
    print("=" * 70)
    rows = []
    if result_ag:
        rows.append(metrics_to_row("多 Agent 团队", result_ag.metrics))
    rows.append(metrics_to_row("Baseline 买入持有", result_bh.metrics))
    if hs300_metrics is not None:
        rows.append(metrics_to_row("沪深300 基准", hs300_metrics))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # ------ 持久化 ------
    metrics_payload = {
        "params": {
            "start": args.start, "end": args.end,
            "symbols": symbols if not args.auto_pool else None,
            "capital": args.capital,
            "auto_pool": args.auto_pool,
            "top1": args.top1 if args.auto_pool else None,
            "top2": args.top2 if args.auto_pool else None,
            "baseline_pool": baseline_pool,
            "agent_pool_size": len(auto_ctx["all_symbols"]) if auto_ctx else None,
            "weekly_rebalances": len(auto_ctx["rebalance_dates"]) if auto_ctx else None,
        },
        "baseline_buy_and_hold": metrics_to_serializable(result_bh.metrics),
        "multi_agent": metrics_to_serializable(result_ag.metrics) if result_ag else None,
        "hs300_benchmark": metrics_to_serializable(hs300_metrics) if hs300_metrics else None,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # ------ 三策略权益曲线 CSV ------
    eq_data: dict[str, list] = {}
    for label, res in (("baseline", result_bh), ("multi_agent", result_ag)):
        if res is None:
            continue
        for d, v in res.portfolio.equity_curve:
            eq_data.setdefault(label, []).append({"date": d, "equity": v})
    if hs300_curve:
        for d, v in hs300_curve:
            eq_data.setdefault("hs300", []).append({"date": d, "equity": v})
    if eq_data:
        merged = None
        for label, items in eq_data.items():
            df_lbl = pd.DataFrame(items).rename(columns={"equity": f"equity_{label}"})
            merged = df_lbl if merged is None else merged.merge(df_lbl, on="date", how="outer")
        if merged is not None:
            merged = merged.sort_values("date")
            merged.to_csv(out_dir / "equity_curves.csv", index=False)
            print(f"\n✓ 权益曲线已保存到 {out_dir / 'equity_curves.csv'}")

    # ------ 可选：matplotlib 双曲线对比图 ------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 中文字体兜底
        plt.rcParams["font.sans-serif"] = [
            "Arial Unicode MS", "Hiragino Sans GB", "PingFang SC",
            "SimHei", "Microsoft YaHei", "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(11, 5.5))
        plot_series = [
            ("多 Agent 团队", result_ag.portfolio.equity_curve if result_ag else None,
             "#d4af37", 2.6, "-"),
            ("Baseline 买入持有", result_bh.portfolio.equity_curve, "#5ac8fa", 2.0, "-"),
            ("沪深300 基准", hs300_curve, "#ff3b30", 1.8, "--"),
        ]
        for label, curve, color, lw, ls in plot_series:
            if not curve:
                continue
            xs = [d for d, _ in curve]
            ys = [v for _, v in curve]
            ax.plot(xs, ys, label=label, color=color, linewidth=lw, linestyle=ls)
        ax.axhline(args.capital, color="gray", linestyle=":",
                   alpha=0.5, label=f"初始资金 ¥{args.capital:,.0f}")
        ax.set_title(f"策略对比 · {args.start} ~ {args.end}", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期")
        ax.set_ylabel("组合权益 (元)")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        png_path = out_dir / "comparison.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"✓ 对比图已保存到 {png_path}")
    except ImportError:
        print("ℹ matplotlib 未安装，已跳过 PNG 输出")
    except Exception as exc:  # noqa: BLE001
        print(f"ℹ 绘图失败：{type(exc).__name__}: {exc}")

    print(f"\n✓ 全部结果已保存到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
