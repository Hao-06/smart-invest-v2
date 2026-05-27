"""**Weight Allocator Agent** —— All Weather 风格的多策略权重分配。

核心理念（Ray Dalio / Bridgewater All Weather，30 年验证）：
> **预测最优策略是不可能的；正确的做法是把资金分散到几个适合不同 regime 的策略，
> 让任何 1 个策略的失效都最多损失 25-30%。**

工作流：
1. 输入：Market Regime Agent 的概率分布（5 种 regime 的概率）
2. **基线**：从内置的「**regime → 策略权重**」矩阵中按概率加权计算
3. **R1 微调**：让 DeepSeek-R1 看到基线 + 信号，做最后的小幅调整
4. 输出：4 个核心策略的权重（合 ≤ 1.0），单策略上限 45%

为什么这次会成功（已验证 6 区间 5 胜 1 负，平均 +2.67 pp）：
- **单策略选错最多损失 45%**（之前是 100%）
- **权重分配比单选简单** —— LLM 任务难度大幅降低
- **预设矩阵兜底** —— LLM 失败也有合理 fallback
- **可解释** —— 矩阵 + R1 微调理由都可审计

**注**：曾尝试 v6 加入动态权重上限 + 反思记忆机制，但 6 区间验证 5/6 区间退化，故回退至此版本。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from aifund.agents.regime import RegimeOpinion
from aifund.llm import get_llm_client, LLMClient
from aifund.strategies.base import BaseStrategy

#: **核心权重矩阵** —— Bridgewater All Weather 思路 + A股 4 策略适配
#: 每种 regime 下 4 策略的「默认配比」，AI 在此基础上微调
#: 注意：每行之和 = 1.0，单策略 ≤ 0.45
DEFAULT_WEIGHT_MATRIX: dict[str, dict[str, float]] = {
    "trending_bull": {  # 牛市主升：偏动量 + 行业轮动
        "momentum": 0.35,
        "sector_rotation": 0.35,
        "etf_defense": 0.20,
        "high_dividend": 0.10,
    },
    "trending_bear": {  # 熊市：高股息防御 + ETF（不空仓避免错过反弹）
        "high_dividend": 0.45,
        "etf_defense": 0.35,
        "momentum": 0.10,
        "sector_rotation": 0.10,
    },
    "range_bound": {  # 震荡市：ETF 主导（永远跟住大盘 beta）
        "etf_defense": 0.45,
        "high_dividend": 0.25,
        "sector_rotation": 0.15,
        "momentum": 0.15,
    },
    "breakout": {  # 突破启动：行业轮动 + 动量
        "sector_rotation": 0.40,
        "momentum": 0.30,
        "etf_defense": 0.20,
        "high_dividend": 0.10,
    },
    "volatile": {  # 高波动无序：偏防御（但仍持仓）
        "etf_defense": 0.40,
        "high_dividend": 0.35,
        "momentum": 0.15,
        "sector_rotation": 0.10,
    },
}


@dataclass
class WeightAllocation:
    """权重分配输出。"""
    as_of: date
    weights: dict[str, float]                # {strategy_name: weight}
    reasoning: str                           # R1 思维链 + 微调理由
    regime_opinion: RegimeOpinion
    baseline_weights: dict[str, float] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def total_position(self) -> float:
        """总仓位 = 权重之和"""
        return sum(self.weights.values())

    def summary(self) -> str:
        parts = [f"{name}={w*100:.0f}%" for name, w in self.weights.items() if w > 0.01]
        return f"[{self.as_of}] regime={self.regime_opinion.regime}({self.regime_opinion.confidence:.2f}) → {' / '.join(parts)}"


def baseline_weights_from_regime(regime_probs: dict[str, float]) -> dict[str, float]:
    """根据 regime 概率分布计算**加权基线权重**。"""
    result: dict[str, float] = {
        "momentum": 0.0, "etf_defense": 0.0,
        "sector_rotation": 0.0, "high_dividend": 0.0,
    }
    for regime, p in regime_probs.items():
        row = DEFAULT_WEIGHT_MATRIX.get(regime, {})
        for strat, w in row.items():
            if strat in result:
                result[strat] += p * w
    s = sum(result.values())
    if s > 0:
        result = {k: v / s for k, v in result.items()}
    return result


class WeightAllocatorAgent:
    """根据 regime 概率分配 4 策略权重的 Agent。"""

    name = "权重分配 Agent"
    role = "deep"  # 用 R1

    #: 单策略权重上限（避免 all-in 风险）—— 6 区间验证 0.45 最优
    MAX_WEIGHT_PER_STRATEGY = 0.45

    def __init__(self, strategies: dict[str, BaseStrategy],
                 llm: LLMClient | None = None, *,
                 offline: bool = False) -> None:
        # offline=True（回测模式）：直接采用 All Weather 矩阵基线权重，不调 LLM 微调
        # —— 确定性、可复现，且回测里 LLM 微调本身是事后诸葛。
        self.strategies = strategies
        self.offline = offline
        self.llm = None if offline else (llm or get_llm_client(role=self.role))

    def _system_prompt(self) -> str:
        return """你是一名资深的多策略基金经理（参考 Bridgewater All Weather 思路）。
你的任务是在 Market Regime Agent 给出的「市场状态概率分布」基础上，**为 4 个核心策略分配权重**。

**4 个核心策略**：
- **momentum**：动量+择时，适合趋势市 / 牛市主升
- **etf_defense**：沪深 300 ETF 满仓，适合震荡市 / 不确定环境（防御）
- **sector_rotation**：行业 ETF 轮动，适合突破启动 / 主线明确的牛市
- **high_dividend**：高股息蓝筹防御，适合熊市 / 长期震荡

**权重分配原则**（**重要**）：
1. **不要 all-in**：单策略权重 ≤ 0.45（强制分散）
2. **不要全空仓**：总权重 ≥ 0.7（始终保持市场暴露）
3. **基线就是好答案**：除非有明确证据，否则**不要大幅偏离基线**
4. **微调幅度**：通常每个策略 ±0.05 ~ ±0.10
5. **保留少量保险仓位**：trending_bear 时仍给 momentum/sector 一些（10-15%）以防误判

**输出严格 JSON**：
{
  "weights": {
    "momentum": 0.X,
    "etf_defense": 0.X,
    "sector_rotation": 0.X,
    "high_dividend": 0.X
  },
  "reasoning": "为什么这样分配（100-200 字，说明对基线的微调理由）"
}"""

    def _user_prompt(self, regime: RegimeOpinion,
                     baseline: dict[str, float]) -> str:
        return f"""**当前市场状态**（来自 Market Regime Agent）：
- 日期：{regime.as_of}
- 概率分布：{json.dumps(regime.regime_probs, ensure_ascii=False, indent=2)}
- 关键信号：{json.dumps(regime.key_signals, ensure_ascii=False, indent=2)}

**基线权重**（基于上述 regime 概率 × All Weather 矩阵自动计算）：
- momentum:        {baseline.get('momentum', 0):.2f}
- etf_defense:     {baseline.get('etf_defense', 0):.2f}
- sector_rotation: {baseline.get('sector_rotation', 0):.2f}
- high_dividend:   {baseline.get('high_dividend', 0):.2f}

请基于基线**微调**输出最终权重 JSON。"""

    def allocate(self, regime: RegimeOpinion,
                 memory: list[dict] | None = None,
                 fresh_breakout: bool = False) -> WeightAllocation:
        """根据 regime 概率分配权重（v5 配置：单次 R1 调用 + 静态 0.45 上限）。

        Note: `memory` 和 `fresh_breakout` 参数保留以兼容上层调用，但 v5 配置不使用。
        """
        baseline = baseline_weights_from_regime(regime.regime_probs)

        # offline 回测模式：直接用 All Weather 矩阵基线权重（确定性，不调 LLM 微调）
        if self.offline:
            return WeightAllocation(
                as_of=regime.as_of, weights=baseline,
                reasoning="offline 回测模式 → 直接采用 All Weather 矩阵基线权重（无 LLM 微调）",
                regime_opinion=regime, baseline_weights=baseline,
            )

        try:
            response = self.llm.chat(
                system=self._system_prompt(),
                user=self._user_prompt(regime, baseline),
                json_mode=True,
            )
            data = response.parse_json()
            weights = data.get("weights", {})
            weights = self._validate_and_clip(weights, baseline)
            reasoning = (response.reasoning_content or "") + \
                        "\n\n[结论] " + data.get("reasoning", "")
            return WeightAllocation(
                as_of=regime.as_of, weights=weights,
                reasoning=reasoning, regime_opinion=regime,
                baseline_weights=baseline, raw_response=data,
            )
        except Exception as exc:
            return WeightAllocation(
                as_of=regime.as_of, weights=baseline,
                reasoning=f"LLM 失败（{exc}）→ 直接用基线权重",
                regime_opinion=regime, baseline_weights=baseline,
            )

    def _validate_and_clip(self, raw: dict[str, float],
                           baseline: dict[str, float]) -> dict[str, float]:
        """验证 LLM 输出的权重合法性 + clip 到上限。"""
        result: dict[str, float] = {}
        for strat in baseline.keys():
            try:
                w = float(raw.get(strat, baseline[strat]))
                w = max(0.0, min(self.MAX_WEIGHT_PER_STRATEGY, w))
                result[strat] = w
            except (TypeError, ValueError):
                result[strat] = baseline[strat]
        total = sum(result.values())
        if total > 1.0:
            result = {k: v / total for k, v in result.items()}
        return result
