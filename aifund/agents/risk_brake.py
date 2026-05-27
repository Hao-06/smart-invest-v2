"""**组合级风险刹车** —— 比赛实战的「保命机制」。

设计哲学（驼灵大赛实战导向）：
- 比赛排名按「**期末可用资金**」算。开局亏 -15% 几乎无法翻盘
- 即使 Meta-Agent 选错策略 / 黑天鹅事件 / 突发暴跌，**风控刹车也能兜底**

两条独立的刹车规则：

1. **绝对回撤刹车**（救命用）
   - 组合从历史高点回撤 > ``max_drawdown_pct`` (默认 8%) → **强制切 ETF 满仓**
   - 理由：限制最大损失，保住排名

2. **相对跑输刹车**（避免硬扛）
   - 组合相对沪深 300 跑输 > ``max_relative_underperformance_pct`` (默认 5%) → **强制 Meta-Agent 重选**
   - 理由：当前策略明显失效，必须重新评估

风控刹车是「**外层包装**」—— 仍保留 Meta-Agent 的所有智能决策，只在出现风险信号时强制干预。

设计模式：状态机（StateMachine）
- ``NORMAL``：Meta-Agent 自由决策
- ``ABSOLUTE_BRAKE``：绝对回撤触发，**ETF 兜底 + 冷却期**（默认 3 周）
- ``RELATIVE_BRAKE``：相对跑输触发，Meta-Agent 重选（带 monitor_feedback）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

BrakeState = Literal["NORMAL", "ABSOLUTE_BRAKE", "RELATIVE_BRAKE"]


@dataclass
class RiskBrakeOutput:
    """风险刹车检查结果。"""
    as_of: date
    state: BrakeState
    force_etf: bool          # 是否强制 ETF 满仓
    force_reselect: bool     # 是否强制 Meta-Agent 重选策略
    reasoning: str           # 触发原因
    current_drawdown_pct: float
    current_underperformance_pct: float


class RiskBrake:
    """跟踪组合 + 沪深300 收益曲线，在风险信号触发时强制干预。"""

    def __init__(
        self,
        max_drawdown_pct: float = 8.0,
        max_relative_underperformance_pct: float = 5.0,
        absolute_brake_cooldown_weeks: int = 3,
    ) -> None:
        self.max_drawdown_pct = max_drawdown_pct
        self.max_relative_underperformance_pct = max_relative_underperformance_pct
        self.absolute_brake_cooldown_weeks = absolute_brake_cooldown_weeks

        # 状态
        self.state: BrakeState = "NORMAL"
        self._equity_peak: float | None = None        # 组合历史最高净值
        self._hs300_initial: float | None = None      # 沪深300 初始值
        self._equity_initial: float | None = None     # 组合初始净值
        self._cooldown_remaining: int = 0             # 绝对刹车冷却周数

    def update(self, as_of: date, equity: float, hs300_price: float) -> RiskBrakeOutput:
        """每周调仓时调用，更新状态并返回是否触发刹车。"""
        # 第一次调用：初始化
        if self._equity_initial is None:
            self._equity_initial = equity
            self._hs300_initial = hs300_price
            self._equity_peak = equity
            return RiskBrakeOutput(
                as_of=as_of, state="NORMAL",
                force_etf=False, force_reselect=False,
                reasoning="风控初始化，记录起点",
                current_drawdown_pct=0.0,
                current_underperformance_pct=0.0,
            )

        # 更新历史高点
        if equity > self._equity_peak:
            self._equity_peak = equity

        # 计算关键指标
        drawdown_pct = (equity / self._equity_peak - 1) * 100  # 负数
        our_total_pct = (equity / self._equity_initial - 1) * 100
        hs300_total_pct = (hs300_price / self._hs300_initial - 1) * 100
        underperformance_pct = our_total_pct - hs300_total_pct  # 负数 = 跑输

        # 冷却期内仍维持绝对刹车
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return RiskBrakeOutput(
                as_of=as_of, state="ABSOLUTE_BRAKE",
                force_etf=True, force_reselect=False,
                reasoning=f"绝对刹车冷却中（剩 {self._cooldown_remaining} 周）→ ETF 兜底",
                current_drawdown_pct=drawdown_pct,
                current_underperformance_pct=underperformance_pct,
            )

        # 检查 1：绝对回撤刹车（最高优先级）
        if drawdown_pct <= -self.max_drawdown_pct:
            self.state = "ABSOLUTE_BRAKE"
            self._cooldown_remaining = self.absolute_brake_cooldown_weeks
            return RiskBrakeOutput(
                as_of=as_of, state="ABSOLUTE_BRAKE",
                force_etf=True, force_reselect=False,
                reasoning=f"⚠️ 绝对回撤 {drawdown_pct:.2f}% ≤ -{self.max_drawdown_pct}% 阈值 "
                          f"→ 强制 ETF 满仓 + 冷却 {self.absolute_brake_cooldown_weeks} 周",
                current_drawdown_pct=drawdown_pct,
                current_underperformance_pct=underperformance_pct,
            )

        # 检查 2：相对跑输刹车
        if underperformance_pct <= -self.max_relative_underperformance_pct:
            self.state = "RELATIVE_BRAKE"
            return RiskBrakeOutput(
                as_of=as_of, state="RELATIVE_BRAKE",
                force_etf=False, force_reselect=True,
                reasoning=f"⚠️ 相对沪深300 跑输 {underperformance_pct:.2f}pp ≤ "
                          f"-{self.max_relative_underperformance_pct}pp 阈值 → 强制 Meta-Agent 重选",
                current_drawdown_pct=drawdown_pct,
                current_underperformance_pct=underperformance_pct,
            )

        # 正常态
        self.state = "NORMAL"
        return RiskBrakeOutput(
            as_of=as_of, state="NORMAL",
            force_etf=False, force_reselect=False,
            reasoning=f"风控正常（回撤 {drawdown_pct:+.2f}%，相对 {underperformance_pct:+.2f}pp）",
            current_drawdown_pct=drawdown_pct,
            current_underperformance_pct=underperformance_pct,
        )
