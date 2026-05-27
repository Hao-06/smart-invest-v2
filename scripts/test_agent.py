"""Agent 端到端联通测试 —— Hao 配好 DEEPSEEK_API_KEY 后跑这个。

依次验证：
1) API key 配置生效，能成功调用 DeepSeek-V3（分析师层）
2) 技术面分析师能对单只标的输出一份 AgentOpinion（含 reasoning）
3) 完整投研团队（4 分析师 + 基金经理 R1）能在单日快照上产生一份决策

任一步失败会打印明确的错误提示。耗时大约 30–90 秒。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aifund.agents import AgentTeam, TechnicalAnalyst
from aifund.backtest import Portfolio
from aifund.data import DataPipeline
from config.settings import settings


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> int:
    if not settings.llm.deepseek_api_key:
        print("✗ 未检测到 DEEPSEEK_API_KEY")
        print("  请在项目根目录的 .env 文件里填入：")
        print("    DEEPSEEK_API_KEY=sk-...")
        return 1
    print(f"✓ DEEPSEEK_API_KEY 已配置（{settings.llm.deepseek_api_key[:6]}…）")
    print(f"  fast 模型: {settings.llm.deepseek_fast_model}")
    print(f"  deep 模型: {settings.llm.deepseek_deep_model}")

    # --- 准备数据 ---
    _section("Step 1 / 拉取测试数据：贵州茅台 600519 @ 2026-05-13")
    pipeline = DataPipeline(lookback_days=60)
    snapshot = pipeline.snapshot(["600519"], "2026-05-13")
    sd = snapshot.get("600519")
    if sd is None or not sd.has_price:
        print("✗ 数据拉取失败：行情为空")
        return 2
    print(f"✓ 行情根数 {sd.bars}，最新收盘 {sd.last_close}")
    print(f"  行业 {sd.industry or '未知'}，新闻 {len(sd.news)} 条，资金流 {len(sd.fund_flow)} 行")

    # --- 单分析师 ---
    _section("Step 2 / 调用技术面分析师（DeepSeek-V3）")
    tech = TechnicalAnalyst()
    op = tech.analyze(snapshot, "600519")
    print(f"  角色: {op.agent_name}")
    print(f"  方向: {op.action}  分值: {op.score:+.2f}  置信度: {op.confidence:.2f}")
    print(f"  推理: {op.reasoning}")
    if op.key_facts:
        print(f"  关键事实:")
        for kf in op.key_facts:
            print(f"    · {kf}")
    if op.risks:
        print(f"  风险:")
        for rk in op.risks:
            print(f"    · {rk}")
    print(f"  Tokens: prompt={op.metadata.get('tokens_prompt')}, "
          f"completion={op.metadata.get('tokens_completion')}, "
          f"耗时={op.metadata.get('elapsed_sec')}s")
    if op.metadata.get("error"):
        print("✗ 分析师调用失败，请检查 key / 网络")
        return 3

    # --- 全团队决策 ---
    _section("Step 3 / 完整投研团队决策（4 分析师 + 基金经理 R1）")
    team = AgentTeam()
    portfolio = Portfolio(initial_capital=500_000)
    decision = team.decide(snapshot, portfolio, save_log=True)

    for sym, ops in decision.analyst_opinions.items():
        print(f"\n[{sym}] 分析师矩阵:")
        for o in ops:
            print(f"  · {o.agent_name:<10} {o.action:<4} score={o.score:+.2f} "
                  f"conf={o.confidence:.2f}  — {o.reasoning[:60]}")

    print(f"\n基金经理总结：{decision.rationale}")
    print(f"订单数量：{len(decision.orders)}")
    for od in decision.orders:
        print(f"  · {od.side} {od.symbol} {od.name} {od.shares}股  ({od.reason})")
    if decision.rejected:
        print("放弃标的：")
        for rj in decision.rejected:
            print(f"  · {rj.get('symbol')}：{rj.get('reason')}")
    if decision.reasoning_content:
        print(f"\n基金经理思维链（R1 reasoning_content，前 500 字）：")
        print(decision.reasoning_content[:500])

    print(f"\n✓ 决策已保存到 {team.log_dir}/")
    print("\n全部联通测试通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
