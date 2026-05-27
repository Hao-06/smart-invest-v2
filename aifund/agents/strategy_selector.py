"""**Strategy Selector Agent** —— 根据 Market Regime 选最佳策略。

工作流：
1. 接收 MarketRegimeAgent 的 RegimeOpinion（regime + 置信度 + 关键指标）
2. 用 DeepSeek-R1 综合判断：选择哪个策略最适合当前 regime
3. 输出 StrategyChoice（策略名 + 选择理由 + 思维链）

设计要点：
- 用 R1 思维链 → 给评委看完整推理
- 候选策略由 ``aifund.strategies.all_strategies(...)`` 提供，每个策略有 ``description`` 和 ``suitable_regimes``
- 不下单，仅选策略；下单由选中策略自己负责
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from aifund.agents.regime import RegimeOpinion
from aifund.llm import get_llm_client, LLMClient
from aifund.strategies.base import BaseStrategy


@dataclass
class StrategyChoice:
    """Strategy Selector 的输出。"""
    as_of: date
    strategy_name: str           # 选中的策略 ID
    confidence: float            # 选择信心度 0.0-1.0
    reasoning: str               # R1 思维链 + 选择理由
    regime_opinion: RegimeOpinion  # 引用的市场状态判断
    raw_response: dict[str, Any] = field(default_factory=dict)


class StrategySelectorAgent:
    """根据市场状态选最佳策略的 Agent。"""

    name = "策略选择 Agent"
    role = "deep"  # 用 R1

    def __init__(self, strategies: dict[str, BaseStrategy],
                 llm: LLMClient | None = None) -> None:
        self.strategies = strategies
        self.llm = llm or get_llm_client(role=self.role)

    def _system_prompt(self) -> str:
        return """你是一名资深的多策略基金经理。你的任务是根据当前市场状态（regime）从策略库中选择最适合的一个策略。

**选择原则**：
1. 优先选择 ``suitable_regimes`` 字段包含当前 regime 的策略
2. 多个候选策略都匹配时，选**最稳健**的（不一定是最激进的）
3. 高不确定性时（regime confidence < 0.5）优先选 etf_defense 兜底
4. 突破启动时（breakout）可选 momentum 或 sector_rotation，但要警惕假突破
5. **若 Performance Monitor 提示「之前策略连续跑输」，必须避开上次的策略，换一个候选**

**输出严格 JSON**：
{
  "strategy_name": "<策略 ID，必须是候选清单里的>",
  "confidence": 0.0~1.0,
  "reasoning": "为什么选这个策略的简明理由（100-200 字）"
}"""

    def _user_prompt(self, regime: RegimeOpinion,
                     monitor_feedback: str | None = None,
                     last_strategy: str | None = None) -> str:
        strategies_desc = "\n".join(
            f"- **{name}**: {s.description}（适用：{', '.join(s.suitable_regimes)}）"
            for name, s in self.strategies.items()
        )
        feedback_block = ""
        if monitor_feedback:
            feedback_block = f"""

**⚠️ Performance Monitor 反馈**（**重要**）：
{monitor_feedback}
上次使用的策略：**{last_strategy}** —— 表现不佳，请**避开它**选另一个策略。
"""
        return f"""**当前市场状态识别**（来自 Market Regime Agent）：
- 日期：{regime.as_of}
- regime: **{regime.regime}**（置信度 {regime.confidence:.2f}）
- 关键指标：
{json.dumps(regime.key_signals, ensure_ascii=False, indent=2)}
- 状态判断理由：{regime.reasoning[-500:] if len(regime.reasoning) > 500 else regime.reasoning}
{feedback_block}
**策略库**（请从中选一个）：
{strategies_desc}

请输出选择策略的 JSON。"""

    def select(self, regime: RegimeOpinion,
               monitor_feedback: str | None = None,
               last_strategy: str | None = None) -> StrategyChoice:
        """根据 regime 选策略。"""
        # 失败时回退到 etf_defense
        fallback = StrategyChoice(
            as_of=regime.as_of, strategy_name="etf_defense", confidence=0.3,
            reasoning="LLM 失败 / 异常情况，回退到 ETF 防御策略",
            regime_opinion=regime,
        )
        try:
            response = self.llm.chat(
                system=self._system_prompt(),
                user=self._user_prompt(regime, monitor_feedback, last_strategy),
                json_mode=True,
            )
            data = response.parse_json()
            choice = data.get("strategy_name", "etf_defense")
            if choice not in self.strategies:
                choice = "etf_defense"  # 不在候选清单 → 兜底
            # 如果 Monitor 触发了重选，但 LLM 居然又选了同一个策略 → 强制换 ETF 兜底
            if monitor_feedback and last_strategy and choice == last_strategy:
                choice = "etf_defense"
            return StrategyChoice(
                as_of=regime.as_of,
                strategy_name=choice,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=(response.reasoning_content or "")
                          + "\n\n[结论] " + data.get("reasoning", ""),
                regime_opinion=regime,
                raw_response=data,
            )
        except Exception as exc:
            fallback.reasoning = f"LLM 失败（{exc}）→ 回退到 ETF 防御"
            return fallback


class AllWeatherManager:
    """**All Weather 风格 Meta-Strategy Manager** —— 多策略权重分配 + 实时执行。

    每周一调用 `decide()`：
    1. Market Regime Agent 输出 regime 概率分布
    2. Weight Allocator Agent 输出 4 策略权重
    3. 每个策略各自 select() 给出持仓建议
    4. 合并为「**符号 → 目标市值权重**」字典

    返回 `AllWeatherDecision`：包含 regime + 权重 + 每策略持仓 + 合并后的目标权重。
    """

    #: 权重低于此阈值的策略跳过 select() 调用（节省时间）
    MIN_WEIGHT_TO_SELECT = 0.10
    #: 记忆窗口大小（保留最近 N 周决策 + 表现给 R1 反思用）
    MEMORY_SIZE = 4

    def __init__(self, strategies: dict[str, BaseStrategy],
                 regime_agent=None, allocator_agent=None) -> None:
        from aifund.agents.regime import MarketRegimeAgent
        from aifund.agents.weight_allocator import WeightAllocatorAgent
        self.strategies = strategies
        self.regime_agent = regime_agent or MarketRegimeAgent()
        self.allocator = allocator_agent or WeightAllocatorAgent(strategies)
        # **记忆机制**：维护最近 N 周的决策 + 实际表现
        self._memory: list[dict] = []

    def _detect_fresh_breakout(self, current_regime) -> bool:
        """识别「**长熊后反弹首周**」：最近 ≥ 2 周 trending_bear → 本周首次 bull/breakout"""
        if len(self._memory) < 2:
            return False
        last_two = self._memory[-2:]
        was_bear = all(m.get("regime") == "trending_bear" for m in last_two)
        now_bull = current_regime.regime in ("trending_bull", "breakout")
        return was_bear and now_bull

    def decide(self, as_of: date,
               last_our_pct: float | None = None,
               last_hs300_pct: float | None = None) -> "AllWeatherDecision":
        # 回填上次决策的实际表现（供反思）
        if last_our_pct is not None and last_hs300_pct is not None and self._memory:
            self._memory[-1]["actual_excess_pp"] = round(last_our_pct - last_hs300_pct, 2)

        regime = self.regime_agent.analyze(as_of)
        fresh_breakout = self._detect_fresh_breakout(regime)
        allocation = self.allocator.allocate(
            regime, memory=self._memory, fresh_breakout=fresh_breakout,
        )
        # 每个策略各自选股（跳过权重 < MIN_WEIGHT_TO_SELECT 的，避免无谓 LSTM 推理）
        strategy_outputs: dict[str, Any] = {}
        for strat_name, weight in allocation.weights.items():
            if weight >= self.MIN_WEIGHT_TO_SELECT and strat_name in self.strategies:
                strat = self.strategies[strat_name]
                strategy_outputs[strat_name] = strat.select(as_of)
        # 调整权重：把跳过的策略权重按比例分配给剩下的（保持总仓位不变）
        kept_weight_sum = sum(allocation.weights[s] for s in strategy_outputs)
        original_sum = sum(allocation.weights.values())
        if kept_weight_sum > 0 and original_sum > kept_weight_sum:
            scale = original_sum / kept_weight_sum
            for s in strategy_outputs:
                allocation.weights[s] *= scale
        # 合并到「标的 → 目标市值权重」
        target_symbol_weights = self._merge_to_symbol_weights(
            strategy_outputs, allocation.weights
        )

        # 记录到 memory（下次决策时反思用）
        self._memory.append({
            "as_of": str(as_of),
            "regime": regime.regime,
            "regime_probs": dict(regime.regime_probs),
            "weights": dict(allocation.weights),
            "fresh_breakout": fresh_breakout,
            "actual_excess_pp": None,  # 下次 decide 时填
        })
        if len(self._memory) > self.MEMORY_SIZE:
            self._memory = self._memory[-self.MEMORY_SIZE:]

        return AllWeatherDecision(
            as_of=as_of, regime=regime, allocation=allocation,
            strategy_outputs=strategy_outputs,
            target_symbol_weights=target_symbol_weights,
        )

    @staticmethod
    def _merge_to_symbol_weights(
        strategy_outputs: dict[str, Any],
        strategy_weights: dict[str, float],
    ) -> dict[str, float]:
        """把「每策略持仓 + 策略权重」合并为「**标的 → 目标市值权重**」字典。

        例：
        - momentum 策略权重 0.3，输出 4 只票 → 每只票从该策略获得 0.3/4 = 0.075 权重
        - etf_defense 策略权重 0.45，输出 [510300] → 510300 从该策略获得 0.45
        - 两个策略都买 510300 → 累加
        """
        symbol_weights: dict[str, float] = {}
        for strat_name, output in strategy_outputs.items():
            strat_weight = strategy_weights.get(strat_name, 0.0)
            if strat_weight <= 0 or not output.symbols:
                continue
            # 策略内部 position_ratio 影响其实际仓位（如 momentum 半仓时 0.5）
            effective_weight = strat_weight * output.position_ratio
            per_symbol_weight = effective_weight / len(output.symbols)
            for sym in output.symbols:
                symbol_weights[sym] = symbol_weights.get(sym, 0) + per_symbol_weight
        return symbol_weights


@dataclass
class AllWeatherDecision:
    """All Weather 完整决策链路。"""
    as_of: date
    regime: Any                                # RegimeOpinion
    allocation: Any                            # WeightAllocation
    strategy_outputs: dict[str, Any]           # {strategy: StrategyOutput}
    target_symbol_weights: dict[str, float]    # {symbol: target_weight}

    def summary(self) -> str:
        parts = [f"{name}={w*100:.0f}%"
                 for name, w in self.allocation.weights.items() if w > 0.01]
        return (f"[{self.as_of}] regime={self.regime.regime}({self.regime.confidence:.2f}) "
                f"→ {' / '.join(parts)} "
                f"(共 {len(self.target_symbol_weights)} 标的)")


class MetaStrategyManager:
    """**Meta-Strategy Manager** —— 整合 Regime + Selector + Performance Monitor + 策略执行。

    每周一调用一次：
    1. RegimeAgent.analyze(as_of) → 市场状态
    2. **PerformanceMonitor.check(as_of) → 是否需要强制重选**
    3. SelectorAgent.select(regime, monitor_feedback) → 选策略（必要时带反馈）
    4. strategy.select(as_of) → 实际持仓建议

    返回 ``MetaDecision`` 包含完整决策链路 + 思维链 + Monitor 反馈。
    """

    def __init__(self, strategies: dict[str, BaseStrategy],
                 regime_agent=None, selector_agent=None,
                 performance_monitor=None) -> None:
        from aifund.agents.regime import MarketRegimeAgent
        from aifund.agents.performance_monitor import PerformanceMonitor
        self.strategies = strategies
        self.regime_agent = regime_agent or MarketRegimeAgent()
        self.selector_agent = selector_agent or StrategySelectorAgent(strategies)
        self.monitor = performance_monitor or PerformanceMonitor()
        self._last_strategy: str | None = None

    def decide(self, as_of: date,
               last_our_pct: float | None = None,
               last_hs300_pct: float | None = None) -> "MetaDecision":
        """每周调仓决策。

        Args:
            as_of: 决策日。
            last_our_pct: 上次决策日至今我们的收益率（百分比）—— 喂给 Monitor。
            last_hs300_pct: 上次决策日至今沪深300 的收益率（百分比）。
        """
        # 1. 喂 Monitor 上周数据
        if last_our_pct is not None and last_hs300_pct is not None:
            self.monitor.record(as_of, last_our_pct, last_hs300_pct)

        # 2. Regime 分析
        regime = self.regime_agent.analyze(as_of)

        # 3. Monitor 检查
        check = self.monitor.check(as_of)
        monitor_feedback = check.reason if check.should_reevaluate else None

        # 4. Selector 选策略（带 Monitor 反馈）
        choice = self.selector_agent.select(
            regime,
            monitor_feedback=monitor_feedback,
            last_strategy=self._last_strategy if check.should_reevaluate else None,
        )
        self._last_strategy = choice.strategy_name

        # 5. 策略执行
        strategy = self.strategies[choice.strategy_name]
        output = strategy.select(as_of)
        return MetaDecision(
            as_of=as_of,
            regime=regime,
            choice=choice,
            strategy_output=output,
            performance_check=check,
        )


@dataclass
class MetaDecision:
    """Meta-Strategy 完整决策链路（用于回测 / 落盘 / Streamlit 展示）。"""
    as_of: date
    regime: RegimeOpinion
    choice: StrategyChoice
    strategy_output: Any  # StrategyOutput
    performance_check: Any | None = None  # PerformanceCheck（可选）

    @property
    def symbols(self) -> list[str]:
        return self.strategy_output.symbols

    @property
    def position_ratio(self) -> float:
        return self.strategy_output.position_ratio

    def summary(self) -> str:
        monitor_tag = ""
        if self.performance_check and self.performance_check.should_reevaluate:
            monitor_tag = " ⚠️Monitor 触发重选"
        return (
            f"[{self.as_of}] regime={self.regime.regime}({self.regime.confidence:.2f}) "
            f"→ 选 {self.choice.strategy_name}({self.choice.confidence:.2f}){monitor_tag} "
            f"→ 持仓 {self.strategy_output.symbols[:3]}... "
            f"ratio={self.strategy_output.position_ratio:.2f}"
        )
