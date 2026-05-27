"""**Performance Monitor Agent** —— 让 Meta-Strategy 系统能够自我修正。

设计哲学：
- Strategy Selector 选完策略后，并不能保证后续表现一定好
- 当前策略**连续跑输**沪深 300 太多时，应该触发**强制重新评估**
- 重新评估时，把「**为什么失败**」反馈给 R1，让它考虑别的策略

工作流：
1. 每周一调仓时：
   a. 先用 Performance Monitor 检查最近 N 周表现
   b. 如果触发「重选信号」→ 让 Strategy Selector 强制重新选（带「失败原因」反馈）
   c. 否则继续用上次的策略
2. 决策落盘：包含 Monitor 输出 + 选择是否切换

触发规则（可调）：
- 连续 ``window_weeks`` 周（默认 2）跑输沪深 300
- 累计跑输 > ``threshold_pp`` pp（默认 5pp）
- → 触发强制重新评估

防 over-reaction：
- 触发后强制等 ``cooldown_weeks`` 周（默认 1）才能再次触发
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class PerformanceCheck:
    """Performance Monitor 的输出。"""
    as_of: date
    should_reevaluate: bool       # 是否需要强制重新评估
    reason: str                   # 触发/不触发的原因
    recent_excess_pp: float       # 最近窗口累计超额（百分点）
    weeks_underperforming: int    # 连续跑输周数
    history: list[dict[str, Any]] = field(default_factory=list)


class PerformanceMonitor:
    """跟踪 Meta-Strategy 表现 + 失败时触发自我修正。"""

    def __init__(
        self,
        window_weeks: int = 2,       # 滚动窗口长度
        threshold_pp: float = 5.0,   # 累计跑输 > 5pp 触发
        cooldown_weeks: int = 1,     # 触发后冷却期
    ) -> None:
        self.window_weeks = window_weeks
        self.threshold_pp = threshold_pp
        self.cooldown_weeks = cooldown_weeks
        # 内部状态：每周记录 (as_of, our_return_pct, hs300_return_pct)
        self._weekly_log: list[tuple[date, float, float]] = []
        self._last_trigger_week: int = -100  # 上次触发的周序号

    def record(self, as_of: date, our_return_pct: float, hs300_return_pct: float) -> None:
        """每周调仓时记录一次（用 T-1 收盘为基准）。"""
        self._weekly_log.append((as_of, our_return_pct, hs300_return_pct))

    def check(self, as_of: date) -> PerformanceCheck:
        """检查最近 ``window_weeks`` 周是否触发重选信号。"""
        # 窗口数据不足
        if len(self._weekly_log) < self.window_weeks:
            return PerformanceCheck(
                as_of=as_of, should_reevaluate=False,
                reason=f"周数据不足（{len(self._weekly_log)}/{self.window_weeks}），不触发",
                recent_excess_pp=0.0,
                weeks_underperforming=0,
                history=list(self._weekly_log),
            )

        # 取最近 window_weeks 周
        recent = self._weekly_log[-self.window_weeks:]
        weekly_excess = [our - hs for _, our, hs in recent]
        # 累计超额（按复合方式 = ∏(1+x) - 1，简化为加和）
        cum_excess = sum(weekly_excess)
        # 连续跑输周数（从末尾倒数，连续 < 0 的周数）
        weeks_under = 0
        for ex in reversed(weekly_excess):
            if ex < 0:
                weeks_under += 1
            else:
                break

        # 冷却期检查
        current_week = len(self._weekly_log) - 1
        in_cooldown = (current_week - self._last_trigger_week) < self.cooldown_weeks

        # 触发条件：连续 window 周跑输 + 累计 > threshold + 不在冷却
        triggered = (
            weeks_under >= self.window_weeks
            and cum_excess <= -self.threshold_pp
            and not in_cooldown
        )

        if triggered:
            self._last_trigger_week = current_week
            reason = (
                f"连续 {weeks_under} 周跑输大盘，累计超额 {cum_excess:.2f}pp "
                f"≤ -{self.threshold_pp}pp 阈值 → 触发强制重选"
            )
        elif in_cooldown:
            reason = f"距上次触发仅 {current_week - self._last_trigger_week} 周（冷却 {self.cooldown_weeks} 周），不触发"
        else:
            reason = (
                f"最近 {self.window_weeks} 周累计超额 {cum_excess:+.2f}pp / "
                f"连续跑输 {weeks_under} 周 → 未达触发阈值"
            )

        return PerformanceCheck(
            as_of=as_of,
            should_reevaluate=triggered,
            reason=reason,
            recent_excess_pp=cum_excess,
            weeks_underperforming=weeks_under,
            history=[
                {"as_of": str(d), "our_pct": round(o, 2), "hs300_pct": round(h, 2),
                 "excess_pct": round(o - h, 2)}
                for d, o, h in recent
            ],
        )
