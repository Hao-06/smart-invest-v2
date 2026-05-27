"""系统健康检查 —— 验证所有模块、数据流、输出格式连通正常。

设计目标：在不消耗 LLM 调用的前提下，端到端覆盖项目的所有非 LLM 组件，
让 Hao 演示前或评委拿到代码后，**一条命令证明系统稳定**。

涵盖的检查（共 12 项）：
1. Python 版本与关键依赖完整性
2. 配置加载与密钥状态
3. 缓存目录可读写
4. 交易日历加载 + 边界查询
5. 行情数据多源拉取（含回退）
6. 资金流 / 新闻数据获取
7. 数据管道生成 point-in-time 快照
8. 技术指标计算
9. 回测引擎（买入持有 baseline，不需要 LLM）
10. JSON 输出与 markdown 报告格式化
11. Agent 模块导入与类型继承
12. Streamlit 看板模块导入

加 ``--with-llm`` 标志会额外做一次最小 LLM 调用验证（约 0.005 元 RMB）。

退出码：
    0 = 全部通过
    1 = 有失败项（脚本最后会列出失败列表）
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

# 项目根目录加入路径，让本脚本无需安装即可运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 检查框架
# ---------------------------------------------------------------------------


class CheckResult:
    """单项检查的结果。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed: bool = False
        self.message: str = ""
        self.elapsed_sec: float = 0.0
        self.traceback: str = ""

    def __str__(self) -> str:
        mark = "✓" if self.passed else "✗"
        return f"  [{mark}] {self.name:<32} {self.message}  ({self.elapsed_sec:.2f}s)"


def run_check(name: str, fn) -> CheckResult:
    """运行单项检查，捕获所有异常。"""
    res = CheckResult(name)
    t0 = time.perf_counter()
    try:
        ok, message = fn()
        res.passed = bool(ok)
        res.message = str(message)
    except Exception as exc:  # noqa: BLE001  健康检查就是要兜住所有异常
        res.passed = False
        res.message = f"{type(exc).__name__}: {exc}"
        res.traceback = traceback.format_exc()
    res.elapsed_sec = round(time.perf_counter() - t0, 2)
    return res


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    s = f"Python {v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return True, s
    return False, f"{s} 过低，推荐 3.11+"


def check_dependencies() -> tuple[bool, str]:
    """关键依赖能否 import。"""
    required = [
        "akshare", "pandas", "numpy", "pyarrow",
        "openai", "tenacity", "dotenv", "pydantic",
        "streamlit", "plotly",
    ]
    missing = []
    versions: dict[str, str] = {}
    for mod_name in required:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "?")
        except ImportError:
            missing.append(mod_name)
    if missing:
        return False, f"缺失：{missing}"
    return True, f"{len(required)} 项就绪"


def check_config() -> tuple[bool, str]:
    """配置加载 + key 状态。"""
    from config.settings import settings

    cfg = settings.llm
    if not cfg.deepseek_api_key:
        return False, "DEEPSEEK_API_KEY 未配置（如不调 LLM 可忽略）"
    return True, (
        f"key 已配置 ({cfg.deepseek_api_key[:6]}…)，"
        f"fast={cfg.deepseek_fast_model}，deep={cfg.deepseek_deep_model}"
    )


def check_cache_dir() -> tuple[bool, str]:
    """数据缓存目录可读写。"""
    from config.settings import settings

    test_path = settings.paths.data_cache / ".health_probe"
    try:
        test_path.write_text("ok")
        if test_path.read_text() != "ok":
            return False, "读写不一致"
        test_path.unlink()
    except Exception as exc:  # noqa: BLE001
        return False, f"读写失败：{exc}"
    return True, f"目录可读写：{settings.paths.data_cache}"


def check_trade_calendar() -> tuple[bool, str]:
    """交易日历可加载，关键查询正常。"""
    from aifund.data import TradeCalendar

    cal = TradeCalendar()
    if len(cal) < 100:
        return False, f"日历过短（{len(cal)} 天）"
    sample = date(2025, 6, 16)  # 一个已知工作日
    latest = cal.latest(sample)
    prev = cal.prev(sample, 5)
    if latest is None or prev is None:
        return False, "边界查询返回 None"
    return True, f"{len(cal)} 天，{cal.dates[0]} ~ {cal.dates[-1]}"


def check_price_data() -> tuple[bool, str]:
    """行情多源拉取（默认贵州茅台 600519 近 30 个交易日）。"""
    from aifund.data import sources

    end = date.today()
    start = end - timedelta(days=60)
    df = sources.get_price_history("600519", start, end)
    if df is None or df.empty:
        return False, "行情拉取返回空"
    if len(df) < 5:
        return False, f"只拉到 {len(df)} 根，怀疑数据源问题"
    src = df.attrs.get("source", "cache")
    return True, f"{len(df)} 根 K 线（来源：{src}）"


def check_fund_flow_and_news() -> tuple[bool, str]:
    """资金流 + 新闻拉取（这两个接口经常成败相关，一起测）。"""
    from aifund.data import sources

    issues = []
    flow = sources.get_fund_flow("600519")
    if flow is None or flow.empty:
        issues.append("资金流空")
    news = sources.get_news("600519", limit=5)
    if news is None or news.empty:
        issues.append("新闻空")
    if len(issues) == 2:
        return False, "; ".join(issues)
    return True, (
        f"资金流 {len(flow)} 行，新闻 {len(news)} 条"
        + (f"（部分失败：{issues}）" if issues else "")
    )


def check_data_pipeline() -> tuple[bool, str]:
    """数据管道生成时点正确的 MarketSnapshot。"""
    from aifund.data import DataPipeline, TradeCalendar

    cal = TradeCalendar()
    if not cal.dates:
        return False, "交易日历为空，跳过"
    as_of = cal.latest(date.today() - timedelta(days=2))
    if as_of is None:
        return False, "找不到合适的 as_of 日"

    pipe = DataPipeline(lookback_days=30)
    snap = pipe.snapshot(["600519"], as_of)
    sd = snap.get("600519")
    if sd is None or not sd.has_price:
        return False, f"快照中无 600519 行情（as_of={as_of}）"
    # 时点正确性检查
    if sd.price_history["date"].max() > as_of:
        return False, "前视偏差：最末行情日 > as_of"
    return True, f"as_of={as_of}，行情 {sd.bars} 根，新闻 {len(sd.news)}，资金流 {len(sd.fund_flow)}"


def check_indicators() -> tuple[bool, str]:
    """技术指标 summarize 在合成数据上正常输出。"""
    import numpy as np
    import pandas as pd
    from aifund import indicators

    n = 60
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
    df = pd.DataFrame({
        "date": [date(2026, 1, 1) + timedelta(days=i) for i in range(n)],
        "open": prices * 0.99, "high": prices * 1.02, "low": prices * 0.98, "close": prices,
        "volume": np.random.randint(10000, 50000, n).astype(float),
        "amount": np.nan, "amplitude": np.nan, "pct_change": np.nan,
        "change": np.nan, "turnover": np.nan,
    })
    s = indicators.summarize(df)
    if not s.get("available"):
        return False, f"summarize 返回不可用：{s.get('reason')}"
    required_keys = ("close", "ma5", "rsi14", "macd", "bollinger", "returns_pct")
    missing = [k for k in required_keys if k not in s]
    if missing:
        return False, f"缺关键字段：{missing}"
    return True, f"close={s['close']}，rsi14={s['rsi14']}，ma_trend={s['ma_trend']}"


def check_backtest_engine() -> tuple[bool, str]:
    """跑一个 3 天的买入持有回测（不需要 LLM）。"""
    from aifund.backtest import BacktestEngine, Portfolio
    from aifund.data import DataPipeline, TradeCalendar

    cal = TradeCalendar()
    end = cal.latest(date.today() - timedelta(days=3))
    if end is None:
        return False, "找不到合适的回测末日"
    start = cal.prev(end, 3) or end

    def buy_and_hold(snapshot, portfolio: Portfolio):
        if any(p.shares > 0 for p in portfolio.positions.values()):
            return []
        return [
            {"symbol": "600519", "side": "BUY", "shares": 100, "name": "贵州茅台"}
        ]

    pipe = DataPipeline(lookback_days=30)
    engine = BacktestEngine(
        pipeline=pipe, candidate_symbols=["600519"],
        decide_fn=buy_and_hold, initial_capital=500_000,
    )
    result = engine.run(start, end, verbose=False)
    if not result.portfolio.equity_curve:
        return False, "回测无净值曲线"
    trades = len(result.portfolio.trades)
    final = result.portfolio.equity_curve[-1][1]
    return True, f"{start} ~ {end}，{trades} 笔交易，期末权益 ¥{final:,.0f}"


def check_output_formatter() -> tuple[bool, str]:
    """命题 JSON + markdown 报告格式化。"""
    from aifund.agents.base import AgentOpinion
    from aifund.agents.manager import ManagerDecision
    from aifund.backtest import Order, Portfolio
    from aifund.data.models import MarketSnapshot, StockData
    from aifund.output import (
        build_markdown_report,
        orders_to_competition_json,
        orders_to_competition_payload,
    )

    # 1) 严格模式只输出 BUY
    orders = [
        Order("600519", "BUY", 100, "贵州茅台"),
        Order("601318", "SELL", 200, "中国平安"),
    ]
    strict = orders_to_competition_payload(orders, mode="strict")
    if len(strict) != 1 or strict[0]["volume"] != 100:
        return False, f"strict 模式输出错误：{strict}"
    # 2) extended 模式允许负 volume
    ext = orders_to_competition_payload(orders, mode="extended")
    if len(ext) != 2 or ext[1]["volume"] != -200:
        return False, f"extended 模式输出错误：{ext}"
    # 3) 100 股整数倍校验
    bad = orders_to_competition_payload([Order("600519", "BUY", 150, "茅台")])
    if bad:
        return False, "非 100 股倍数应被剔除"
    # 4) 空数组
    empty = orders_to_competition_json([], mode="strict")
    if empty != "[]":
        return False, f"空操作日应返回 []：{empty}"
    # 5) markdown 报告
    as_of = date(2026, 5, 13)
    sd = StockData(symbol="600519", name="贵州茅台", as_of=as_of)
    snap = MarketSnapshot(as_of=as_of, stocks={"600519": sd})
    op = AgentOpinion("技术面分析师", "600519", "看多", 0.5, 0.7, "测试理由")
    decision = ManagerDecision(
        as_of=as_of, orders=[orders[0]], rationale="测试",
        analyst_opinions={"600519": [op]},
    )
    md = build_markdown_report(decision, snap, Portfolio(initial_capital=500_000))
    if "投资决策报告" not in md or "技术面分析师" not in md:
        return False, "markdown 报告内容缺失"
    return True, f"strict/extended/empty/markdown 4 项均正常"


def check_agent_imports() -> tuple[bool, str]:
    """Agent 模块可导入，类继承关系正确（不实例化 → 不需要 LLM key）。"""
    from aifund.agents import (
        Agent, AgentOpinion, AgentTeam, FundManager,
        FundFlowAnalyst, NewsAnalyst, RiskAnalyst, TechnicalAnalyst, ValuationAnalyst,
    )

    analysts = (TechnicalAnalyst, FundFlowAnalyst, NewsAnalyst, RiskAnalyst, ValuationAnalyst)
    if not all(issubclass(c, Agent) for c in analysts + (FundManager,)):
        return False, "Agent 子类继承关系异常"
    return True, f"{len(analysts)} 个分析师 + 基金经理 + AgentTeam 导入正常"


def check_streamlit_import() -> tuple[bool, str]:
    """Streamlit 看板模块语法 + import 检查（不真正启动）。"""
    import ast

    dashboard_path = ROOT / "app" / "dashboard.py"
    if not dashboard_path.exists():
        return False, "app/dashboard.py 不存在"
    try:
        ast.parse(dashboard_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return False, f"dashboard.py 语法错误：{exc}"
    # 看板内部还要靠的两个三方库
    try:
        import streamlit  # noqa: F401
        import plotly  # noqa: F401
    except ImportError as exc:
        return False, f"看板依赖缺失：{exc}"
    return True, "语法正确，依赖齐备"


def check_llm_minimal_call() -> tuple[bool, str]:
    """最小 LLM 调用：只调一次 V3，输出固定 JSON 验证连通（约 0.005 元）。"""
    from aifund.llm import get_llm_client

    client = get_llm_client("fast")
    resp = client.chat(
        system="你是一个测试 Agent，严格只回复 JSON 对象。",
        user='请回复 {"status": "ok", "echo": "ping"}',
        json_mode=True,
        max_tokens=64,
    )
    data = resp.parse_json()
    if not isinstance(data, dict) or data.get("status") != "ok":
        return False, f"LLM 响应非预期：{data}"
    return True, (
        f"模型 {resp.model}，prompt {resp.prompt_tokens} → "
        f"completion {resp.completion_tokens} token"
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


CHECKS = [
    ("Python 版本", check_python_version),
    ("依赖完整性", check_dependencies),
    ("配置加载", check_config),
    ("缓存目录可读写", check_cache_dir),
    ("交易日历", check_trade_calendar),
    ("行情数据多源拉取", check_price_data),
    ("资金流 + 新闻", check_fund_flow_and_news),
    ("数据管道时点快照", check_data_pipeline),
    ("技术指标计算", check_indicators),
    ("回测引擎", check_backtest_engine),
    ("输出格式化", check_output_formatter),
    ("Agent 模块导入", check_agent_imports),
    ("Streamlit 看板", check_streamlit_import),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="智投未来 · 系统健康检查")
    parser.add_argument("--with-llm", action="store_true",
                        help="额外做一次最小 LLM 调用（约 0.005 元 RMB）")
    parser.add_argument("--verbose", action="store_true",
                        help="失败时打印完整 traceback")
    args = parser.parse_args()

    checks = list(CHECKS)
    if args.with_llm:
        checks.append(("LLM 最小调用（V3）", check_llm_minimal_call))

    print("=" * 70)
    print(f"{'智投未来 · 系统健康检查':^60}")
    print("=" * 70)

    results: list[CheckResult] = []
    for name, fn in checks:
        print(f"\n→ 正在检查：{name}", flush=True)
        res = run_check(name, fn)
        results.append(res)
        print(res)
        if not res.passed and args.verbose and res.traceback:
            print(res.traceback)

    # 总结
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    failed = [r for r in results if not r.passed]
    total_time = sum(r.elapsed_sec for r in results)

    print()
    print("=" * 70)
    if not failed:
        print(f"{'✅ 全部通过 (' + str(passed) + '/' + str(total) + ')':^60}")
    else:
        print(f"{'⚠ 通过 ' + str(passed) + '/' + str(total) + '，失败 ' + str(len(failed)) + ' 项':^60}")
    print(f"{'总耗时 ' + f'{total_time:.1f}' + ' s':^60}")
    print("=" * 70)

    if failed:
        print("\n失败项：")
        for r in failed:
            print(f"  ✗ {r.name}: {r.message}")
        print("\n（加 --verbose 查看完整 traceback）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
