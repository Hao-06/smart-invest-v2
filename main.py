"""智投未来 · 多智能体 A股投研系统 —— 命令行入口。

主入口（LiveTradingAgentV2）：
    python main.py check
        环境与配置自检。

    python main.py recommend --date 2026-05-25
        V2 当日推荐：regime 概率 → 4 策略权重 → top 8 推荐 JSON。
        默认 offline 模式（无需 API key）；加 --use-llm 启用 R1 推理（实盘场景）。

    python main.py validate --start 2024-01-02 --end 2026-05-18
        V2 连续回测验证（offline 确定性）。

兼容入口（旧架构 AgentTeam，已被 V2 替代，保留以避免破坏既有脚本）：
    python main.py decide --date ... --symbols ...
    python main.py backtest --start ... --end ... --symbols ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.settings import settings


def cmd_check(_args: argparse.Namespace) -> int:
    """自检：确认配置与依赖就绪。"""
    print("智投未来 · 系统自检")
    print("-" * 40)
    print(f"项目根目录   : {settings.paths.root}")
    print(f"数据缓存目录 : {settings.paths.data_cache}")
    print(f"LLM 提供方   : {settings.llm.provider}")
    print(f"V3 模型      : {settings.llm.deepseek_fast_model}    (role=fast, 旧架构用)")
    print(f"R1 模型      : {settings.llm.deepseek_deep_model}    (role=deep, V2 实盘用)")

    key = settings.llm.deepseek_api_key
    if key:
        print(f"DeepSeek Key : 已配置（{key[:6]}…，长度 {len(key)}）")
    else:
        print("DeepSeek Key : ✗ 未配置 —— 请复制 .env.example 为 .env 并填入密钥")

    print(f"初始资金     : {settings.backtest.initial_capital:,.0f} 元")
    print(f"单票上限     : {settings.backtest.max_position_per_stock:.0%}")

    missing: list[str] = []
    for mod in ("akshare", "pandas", "numpy", "openai", "streamlit"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"依赖缺失     : ✗ {', '.join(missing)} —— 请运行 pip install -r requirements.txt")
        return 1
    print("依赖检查     : ✓ 全部就绪")
    return 0


def _parse_symbols(text: str) -> list[str]:
    return [s.strip() for s in text.replace("，", ",").split(",") if s.strip()]


def cmd_decide(args: argparse.Namespace) -> int:
    """对指定交易日产出今日投资建议。"""
    # 延迟导入：cmd_check 不需要这些模块（也不需要 key）
    from aifund.agents import AgentTeam
    from aifund.backtest import Portfolio
    from aifund.data import DataPipeline
    from aifund.output import build_markdown_report, orders_to_competition_json

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        print("✗ --symbols 不能为空")
        return 2

    pipeline = DataPipeline(lookback_days=args.lookback)
    snapshot = pipeline.snapshot(symbols, args.date)
    if not snapshot.tradable_symbols:
        print(f"✗ {args.date} 候选标的均无可用行情")
        return 3

    team = AgentTeam()
    portfolio = Portfolio(initial_capital=args.capital)
    decision = team.decide(snapshot, portfolio, save_log=True)

    print(f"\n# 命题 JSON 输出（{args.mode} 模式）")
    print(orders_to_competition_json(decision.orders, mode=args.mode, indent=2))

    # 写一份 markdown 报告
    report = build_markdown_report(decision, snapshot, portfolio)
    report_path = settings.paths.runs / f"report_{args.date}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n# 决策报告：{report_path}", file=sys.stderr)
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """LiveTradingAgentV2 当日推荐 —— 输出符合官方规则的 JSON。"""
    import json as _json
    from datetime import date as _date

    from aifund.agents.live_agent_v2 import LiveTradingAgentV2
    from aifund.data.pipeline import DataPipeline
    from aifund.strategies import all_strategies
    from aifund.strategies.sector_rotation import SECTOR_ETFS
    from aifund.stockpool.factor_loader import FactorLoader

    today = _date.fromisoformat(args.date)
    pipeline = DataPipeline()
    pipeline.light_mode = True

    # PIT universe（用回测起点前最近完整因子日选股池，杜绝前视）
    fl = FactorLoader()
    pit_universe = fl.pit_top_n(today, n=args.universe_size)
    individual_universe = pit_universe["symbol"].tolist()
    if individual_universe:
        pit_day = pit_universe["日期"].iloc[0]
        print(f"# 个股 universe: {len(individual_universe)} 只（PIT 因子快照 @ {pit_day}）",
              file=sys.stderr)
    else:
        print(f"# 个股 universe: 空（{today} 早于因子数据覆盖，仅 ETF 策略生效）",
              file=sys.stderr)

    strats = all_strategies(pipeline, universe=individual_universe)
    agent = LiveTradingAgentV2(
        pipeline=pipeline, strategies=strats, offline=not args.use_llm,
    )
    recs, decision = agent.recommend(today)

    # 决策上下文（写 stderr，不污染 JSON 输出）
    print(f"# Agent 模式: {'LLM 推理（DeepSeek-R1）' if args.use_llm else 'offline 规则判断'}",
          file=sys.stderr)
    print(f"# Regime: {decision.regime.get('regime', 'unknown')} "
          f"(confidence {decision.regime.get('confidence', 0):.2f})", file=sys.stderr)
    print(f"# Weights: {decision.weights}", file=sys.stderr)
    print(f"# 推荐 {len(recs)} 只", file=sys.stderr)
    print(file=sys.stderr)

    # JSON 推荐（stdout，便于管道）
    print(_json.dumps(recs, ensure_ascii=False, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """LiveTradingAgentV2 区间回测 —— 委托给 scripts/test_live_agent_v2.py。"""
    import subprocess

    script = Path(__file__).parent / "scripts" / "test_live_agent_v2.py"
    out = args.out or f"runs/v2_validation_{args.start}_{args.end}.json"
    cmd = [
        sys.executable, str(script),
        "--start", args.start, "--end", args.end,
        "--name", args.name or f"validate {args.start}~{args.end}",
        "--out", out,
    ]
    return subprocess.call(cmd)


def cmd_backtest(args: argparse.Namespace) -> int:
    """[旧架构] 在区间内用多 Agent 团队驱动回测 —— V2 用户请用 `validate`。"""
    from aifund.agents import AgentTeam
    from aifund.backtest import BacktestEngine
    from aifund.data import DataPipeline

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        print("✗ --symbols 不能为空")
        return 2

    pipeline = DataPipeline(lookback_days=args.lookback)
    team = AgentTeam(log_dir=settings.paths.runs / f"backtest_{args.start}_{args.end}")
    engine = BacktestEngine(
        pipeline=pipeline,
        candidate_symbols=symbols,
        decide_fn=team.as_decide_fn(save_log=True),
        initial_capital=args.capital,
    )
    result = engine.run(args.start, args.end, verbose=True)
    result.print_report()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aifund",
        description="智投未来 · 多智能体 A股投研系统",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="环境与配置自检")
    p_check.set_defaults(func=cmd_check)

    # ────── V2 主入口 ──────
    p_rec = sub.add_parser("recommend", help="[V2] 当日推荐 JSON（默认 offline，无需 API key）")
    p_rec.add_argument("--date", required=True, help="决策日 YYYY-MM-DD")
    p_rec.add_argument("--universe-size", type=int, default=80,
                       help="PIT 选股池大小，默认 80")
    p_rec.add_argument("--use-llm", action="store_true",
                       help="启用 DeepSeek-R1 推理（实盘场景；需配置 DEEPSEEK_API_KEY）")
    p_rec.set_defaults(func=cmd_recommend)

    p_val = sub.add_parser("validate", help="[V2] 区间回测验证（offline 确定性）")
    p_val.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    p_val.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    p_val.add_argument("--name", default=None, help="区间标签（用于报告标题）")
    p_val.add_argument("--out", default=None, help="结果 JSON 输出路径")
    p_val.set_defaults(func=cmd_validate)

    # ────── 旧架构（保留兼容） ──────
    p_decide = sub.add_parser("decide", help="[旧] 6 分析师团队单日决策（已被 recommend 替代）")
    p_decide.add_argument("--date", required=True, help="决策日，格式 YYYY-MM-DD")
    p_decide.add_argument("--symbols", required=True,
                          help="候选标的，逗号分隔，如 600519,601318")
    p_decide.add_argument("--capital", type=float, default=settings.backtest.initial_capital,
                          help="组合初始资金，默认 50 万")
    p_decide.add_argument("--lookback", type=int, default=120,
                          help="行情回看交易日数，默认 120")
    p_decide.add_argument("--mode", choices=("strict", "extended"), default="strict",
                          help="JSON 模式：strict=仅买入；extended=负数表卖出")
    p_decide.set_defaults(func=cmd_decide)

    p_bt = sub.add_parser("backtest", help="多 Agent 驱动的区间回测")
    p_bt.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    p_bt.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    p_bt.add_argument("--symbols", required=True,
                      help="候选标的，逗号分隔")
    p_bt.add_argument("--capital", type=float, default=settings.backtest.initial_capital)
    p_bt.add_argument("--lookback", type=int, default=120)
    p_bt.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
