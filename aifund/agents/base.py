"""Agent 抽象基类与统一意见模型。

每个分析师 / 基金经理都派生自 ``Agent``。子类只需声明：
- ``name``：中文角色名（用于日志、决策报告）
- ``role``：LLM 角色 —— ``"fast"`` (V3, 分析师) / ``"deep"`` (R1, 基金经理)
- ``_system_prompt()`` / ``_build_user_prompt()``：两个 prompt 构造方法

基类负责调 LLM、解析 JSON、容错、记录 token 用量。
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aifund.data.models import MarketSnapshot, StockData
from aifund.llm import LLMClient, get_llm_client
from aifund.llm.client import Role


# ---------------------------------------------------------------------------
# 统一意见模型
# ---------------------------------------------------------------------------


@dataclass
class AgentOpinion:
    """单个 Agent 对单只标的的判断。

    所有分析师的输出都归一为此结构，便于基金经理 Agent 综合处理与决策报告呈现。
    """

    agent_name: str
    symbol: str
    action: str = "中性"  # "看多" / "看空" / "中性"
    score: float = 0.0  # [-1, 1] 数值化方向（-1 强空 → +1 强多）
    confidence: float = 0.5  # [0, 1] 置信度
    reasoning: str = ""  # 核心理由（限 1-3 句）
    key_facts: list[str] = field(default_factory=list)  # 关键事实点
    risks: list[str] = field(default_factory=list)  # 风险提示
    metadata: dict[str, Any] = field(default_factory=dict)  # 模型 / token / 用时

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "symbol": self.symbol,
            "action": self.action,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "key_facts": self.key_facts,
            "risks": self.risks,
        }


# ---------------------------------------------------------------------------
# Agent 基类
# ---------------------------------------------------------------------------


class Agent(ABC):
    """分析师 / 基金经理共用的 Agent 基类。"""

    #: 中文角色名（子类必须覆盖）
    name: str = "Agent"
    #: LLM 角色（fast = 分析师 / deep = 基金经理）
    role: Role = "fast"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm: LLMClient = llm or get_llm_client(role=self.role)

    # ------------------------------------------------------------------
    # Prompt 接口（子类必填）
    # ------------------------------------------------------------------
    @abstractmethod
    def _system_prompt(self) -> str:
        """Agent 角色定义与输出格式约束。"""

    @abstractmethod
    def _build_user_prompt(self, snapshot: MarketSnapshot, stock: StockData) -> str:
        """单只标的的数据上下文。"""

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def analyze(self, snapshot: MarketSnapshot, symbol: str) -> AgentOpinion:
        """对 ``symbol`` 输出一份意见。失败时返回 confidence=0 的中性意见。"""
        stock = snapshot.get(symbol)
        if stock is None or not stock.has_price:
            return self._empty_opinion(symbol, "无可用行情数据")

        system = self._system_prompt()
        user = self._build_user_prompt(snapshot, stock)

        t0 = time.perf_counter()
        try:
            response = self.llm.chat(system=system, user=user, json_mode=True)
        except Exception as exc:  # noqa: BLE001  LLM 失败兜底
            return self._empty_opinion(symbol, f"LLM 调用失败: {type(exc).__name__}: {exc}", error=True)
        elapsed = round(time.perf_counter() - t0, 2)

        try:
            data = response.parse_json()
        except ValueError as exc:
            return self._empty_opinion(symbol, f"LLM 响应非合法 JSON: {exc}", error=True)

        return self._make_opinion(symbol, data, response, elapsed)

    # ------------------------------------------------------------------
    # 解析与兜底
    # ------------------------------------------------------------------
    def _make_opinion(
        self,
        symbol: str,
        data: Any,
        response: Any,
        elapsed_sec: float,
    ) -> AgentOpinion:
        """把 LLM 解析后的 dict 装配成 AgentOpinion。"""
        if not isinstance(data, dict):
            return self._empty_opinion(symbol, f"LLM 输出非对象：{type(data).__name__}", error=True)

        action = self._normalize_action(data.get("action") or data.get("方向") or "中性")
        score = self._clamp(self._to_float(data.get("score", data.get("评分", 0.0))), -1.0, 1.0)
        confidence = self._clamp(self._to_float(data.get("confidence", data.get("置信度", 0.5))), 0.0, 1.0)

        return AgentOpinion(
            agent_name=self.name,
            symbol=symbol,
            action=action,
            score=score,
            confidence=confidence,
            reasoning=str(data.get("reasoning") or data.get("理由") or "").strip(),
            key_facts=self._as_str_list(data.get("key_facts") or data.get("关键事实")),
            risks=self._as_str_list(data.get("risks") or data.get("风险")),
            metadata={
                "model": getattr(response, "model", ""),
                "role": self.role,
                "tokens_prompt": getattr(response, "prompt_tokens", 0),
                "tokens_completion": getattr(response, "completion_tokens", 0),
                "elapsed_sec": elapsed_sec,
                "reasoning_content": getattr(response, "reasoning_content", "") or "",
            },
        )

    def _empty_opinion(self, symbol: str, reason: str, error: bool = False) -> AgentOpinion:
        return AgentOpinion(
            agent_name=self.name,
            symbol=symbol,
            action="中性",
            score=0.0,
            confidence=0.0,
            reasoning=reason,
            metadata={"error": error, "role": self.role},
        )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _to_float(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _normalize_action(action: Any) -> str:
        """把 LLM 给的方向词归一为「看多 / 看空 / 中性」。"""
        s = str(action).strip().lower()
        if any(k in s for k in ("看多", "买入", "做多", "bullish", "buy", "long", "positive")):
            return "看多"
        if any(k in s for k in ("看空", "卖出", "做空", "bearish", "sell", "short", "negative")):
            return "看空"
        return "中性"

    @staticmethod
    def _as_str_list(v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v)]

    @staticmethod
    def _safe_json(obj: Any, max_chars: int = 4000) -> str:
        """把上下文数据序列化为 JSON 字符串，截断超长内容。"""
        try:
            s = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            s = str(obj)
        return s if len(s) <= max_chars else s[:max_chars] + "\n...（已截断）"
