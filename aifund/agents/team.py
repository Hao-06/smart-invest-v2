"""Agent 团队编排：把分析师并行调用 + 基金经理综合决策串起来。

对外暴露两种使用方式：
- ``AgentTeam.decide(snapshot, portfolio)`` → ``ManagerDecision``：拿完整产物（含每个分析师的意见 + 基金经理的推理链路），用于决策报告/看板呈现。
- ``AgentTeam.as_decide_fn()`` → ``Callable``：返回符合 ``BacktestEngine.decide_fn`` 协议的回调，直接喂给回测引擎。

并行：4 个分析师对同一只标的的调用相互独立，用 ``ThreadPoolExecutor`` 并发，
显著缩短单日决策耗时（4× 加速，I/O 密集场景近线性）。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from aifund.agents.analysts import build_analyst_team
from aifund.agents.base import Agent, AgentOpinion
from aifund.agents.manager import FundManager, ManagerDecision
from aifund.backtest.portfolio import Order, Portfolio
from aifund.data.models import MarketSnapshot
from config.settings import settings


class AgentTeam:
    """投研团队：4 位分析师 + 1 位基金经理。"""

    def __init__(
        self,
        analysts: list[Agent] | None = None,
        manager: FundManager | None = None,
        max_workers: int = 4,
        log_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            analysts: 自定义分析师列表；None 时使用标准 4 人团。
            manager: 自定义基金经理；None 时使用默认 R1 经理。
            max_workers: 分析师并发上限。
            log_dir: 决策日志落盘目录；None 时默认 ``runs/decisions``。
        """
        self.analysts: list[Agent] = analysts or build_analyst_team()
        self.manager: FundManager = manager or FundManager()
        self.max_workers: int = max_workers
        self.log_dir: Path = Path(log_dir) if log_dir else (settings.paths.runs / "decisions")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 单标的意见收集（并行）
    # ------------------------------------------------------------------
    def collect_opinions(self, snapshot: MarketSnapshot, symbol: str) -> list[AgentOpinion]:
        """对单只标的并发调用全部分析师。"""
        opinions: list[AgentOpinion | None] = [None] * len(self.analysts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(agent.analyze, snapshot, symbol): idx
                for idx, agent in enumerate(self.analysts)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    opinions[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001  最外层兜底
                    opinions[idx] = AgentOpinion(
                        agent_name=self.analysts[idx].name,
                        symbol=symbol,
                        reasoning=f"分析师异常：{type(exc).__name__}: {exc}",
                        confidence=0.0,
                        metadata={"error": True},
                    )
        return [op for op in opinions if op is not None]

    # ------------------------------------------------------------------
    # 全量决策
    # ------------------------------------------------------------------
    def decide(
        self,
        snapshot: MarketSnapshot,
        portfolio: Portfolio,
        symbols: list[str] | None = None,
        save_log: bool = True,
    ) -> ManagerDecision:
        """对候选标的执行：分析师并行 → 基金经理综合 → 输出订单。

        Args:
            snapshot: 决策基准日的市场快照。
            portfolio: 当前组合状态。
            symbols: 显式限定候选；None 时取 ``snapshot.tradable_symbols``。
            save_log: 是否把决策落盘为 JSON 供后续审计。
        """
        candidates = symbols or snapshot.tradable_symbols
        opinions: dict[str, list[AgentOpinion]] = {
            sym: self.collect_opinions(snapshot, sym) for sym in candidates
        }
        decision = self.manager.decide(snapshot, opinions, portfolio)
        if save_log:
            self._save_decision(decision)
        return decision

    # ------------------------------------------------------------------
    # 回测适配
    # ------------------------------------------------------------------
    def as_decide_fn(
        self, save_log: bool = True
    ) -> Callable[[MarketSnapshot, Portfolio], list[Order]]:
        """返回符合 ``BacktestEngine.decide_fn`` 协议的回调。"""
        def _fn(snapshot: MarketSnapshot, portfolio: Portfolio) -> list[Order]:
            return self.decide(snapshot, portfolio, save_log=save_log).orders
        return _fn

    # ------------------------------------------------------------------
    # 决策日志
    # ------------------------------------------------------------------
    def _save_decision(self, decision: ManagerDecision) -> None:
        if decision.as_of is None:
            return
        # 文件名：YYYY-MM-DD__<时间戳>.json，方便同日多次复盘对比
        ts = datetime.now().strftime("%H%M%S")
        path = self.log_dir / f"{decision.as_of}__{ts}.json"
        payload = decision.to_dict()
        # 把 R1 思维链单独保留（reasoning_content 在 metadata 之外，避免吞掉）
        if decision.reasoning_content:
            payload["reasoning_content"] = decision.reasoning_content
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass  # 日志写失败不阻塞主流程
