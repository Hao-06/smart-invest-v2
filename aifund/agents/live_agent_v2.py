"""**LiveTradingAgentV2** —— 重新理解需求后的「**纯推荐型**」实战智能体。

**关键洞察**（来自 Hao 重新解读官方规则）：
- 智能体**只给「**今天看好的股票 + 建议买入股数**」**
- **不需要**维护 portfolio / cash / holdings / SELL
- 平台**自主**执行 + 管理资金 + 控制卖出（具体规则复赛后对接技术文档）

设计哲学：
- **Agent = 纯策略大脑**：每天分析市场 → 推荐 N 只票
- 输出严格符合官方 JSON 格式
- 当日无操作 → 返回 `[]`
- 不依赖外部 portfolio 状态（**完全无状态 + 可重入**）

工作流（每天 9:00 前调用）：
1. Daily Regime 分析（R1）
2. Weight Allocator 决定策略偏好（R1）
3. 各策略 select() 给出选股
4. 合并 → 取综合得分 top N
5. 输出 JSON 推荐清单

注：
- volume 按「**每只建议价值 ≈ 5 万 / 价格 → 100 股整数倍**」自动算
- 单日推荐数量上限 8（避免分散到无法买入）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from aifund.agents.daily_health_check import DailyHealthCheckAgent, HS300_ETF
from aifund.agents.regime import MarketRegimeAgent
from aifund.agents.weight_allocator import WeightAllocatorAgent
from aifund.data import sources
from aifund.data.pipeline import DataPipeline
from aifund.strategies import BaseStrategy
from aifund.strategies.sector_rotation import SECTOR_ETFS as _SECTOR_ETFS

#: 标的名称缓存。
#: ETF 名称全部内置 —— 个股信息接口对 ETF 代码必然失败且每次耗时约 45s，
#: 绝不能在回测里逐日重复调用。
_NAME_CACHE: dict[str, str] = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "511010": "5年期国债ETF",
    "159949": "创业板50ETF",
    "588000": "科创50ETF",
    **{code: f"{name}ETF" for code, name in _SECTOR_ETFS.items()},
}


@dataclass
class DailyDecision:
    """每日决策完整记录（可审计 + 落盘）。"""
    as_of: date
    recommendations: list[dict[str, Any]]
    decision_type: str                       # "health_emergency" / "meta_agent" / "no_action"
    reasoning: str
    regime: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    health_action: str | None = None
    chosen_strategies: list[str] = field(default_factory=list)


class LiveTradingAgentV2:
    """**纯推荐型**实战智能体（无状态，每天独立调用）。"""

    name = "纯推荐型智能体"

    # 配置参数
    MAX_RECOMMENDATIONS = 8                  # 单日最多推荐几只
    TARGET_VALUE_PER_STOCK = 50_000          # 每只推荐价值约 5 万元
    MIN_VOLUME = 100                         # A 股最小单元
    MIN_WEIGHT_TO_SELECT = 0.10              # 策略权重低于此跳过

    def __init__(
        self,
        pipeline: DataPipeline,
        strategies: dict[str, BaseStrategy],
        regime_agent: MarketRegimeAgent | None = None,
        allocator: WeightAllocatorAgent | None = None,
        health_check: DailyHealthCheckAgent | None = None,
        *,
        offline: bool = False,
    ) -> None:
        # offline=True：回测模式 —— regime/allocator 全程走规则判断，不调 LLM。
        # 历史回测调 LLM 既有未来信息泄漏（模型知道 as_of 之后的事）又不可复现；
        # LLM 推理仅用于实盘（复赛）。
        self.pipeline = pipeline
        self.offline = offline
        self.strategies = strategies
        self.regime_agent = regime_agent or MarketRegimeAgent(offline=offline)
        self.allocator = allocator or WeightAllocatorAgent(strategies, offline=offline)
        # 健康检查的 portfolio 跟踪由外部维护（如果需要）
        self.health_check = health_check or DailyHealthCheckAgent()
        self.decision_log: list[DailyDecision] = []

    def recommend(self, as_of: date) -> tuple[list[dict[str, Any]], DailyDecision]:
        """每日推荐入口。

        Args:
            as_of: 今天日期。

        Returns:
            (recommendations, decision):
                recommendations: 符合官方 JSON 的列表
                decision: 完整决策记录（落盘可审计）
        """
        # ============ 步骤 1: 市场健康检查（不依赖 portfolio）============
        # 检查大盘是否暴跌（市场层面）
        market_signal = self._check_market_signal(as_of)
        if market_signal == "crash":
            # 大盘崩盘 → 推荐 ETF 防御
            recs = self._etf_defense_recommendation(as_of)
            decision = DailyDecision(
                as_of=as_of, recommendations=recs,
                decision_type="health_emergency",
                reasoning="🚨 大盘单日跌幅 > 3% → 推荐 ETF 防御",
                health_action="market_crash",
            )
            self.decision_log.append(decision)
            return recs, decision

        # ============ 步骤 2: Meta-Agent 主决策（每天跑）============
        regime = self.regime_agent.analyze(as_of)
        allocation = self.allocator.allocate(regime)

        # ============ 步骤 3: 各策略选股（按权重排序）============
        symbol_scores: dict[str, float] = {}  # {symbol: weighted_score}
        chosen_strategies = []
        for strat_name, weight in allocation.weights.items():
            if weight < self.MIN_WEIGHT_TO_SELECT or strat_name not in self.strategies:
                continue
            output = self.strategies[strat_name].select(as_of)
            if not output.symbols:
                continue
            chosen_strategies.append(strat_name)
            effective_weight = weight * output.position_ratio
            # 策略内部排序的隐含分数：第 1 名 1.0、第 2 名 0.9、...
            for i, sym in enumerate(output.symbols):
                rank_score = 1.0 - i * 0.05
                symbol_scores[sym] = symbol_scores.get(sym, 0) + effective_weight * rank_score

        # ============ 步骤 4: 取综合得分 top N，输出推荐 ============
        if not symbol_scores:
            decision = DailyDecision(
                as_of=as_of, recommendations=[],
                decision_type="no_action",
                reasoning="无候选股票",
                regime={"regime": regime.regime, "probs": regime.regime_probs,
                        "confidence": regime.confidence},
                weights=allocation.weights,
            )
            self.decision_log.append(decision)
            return [], decision

        sorted_symbols = sorted(symbol_scores.items(), key=lambda x: -x[1])
        top_picks = sorted_symbols[: self.MAX_RECOMMENDATIONS]

        recs = []
        for sym, score in top_picks:
            asset_type = "etf" if self._is_etf(sym) else "stock"
            price = self._get_latest_price(sym, as_of, asset_type=asset_type)
            if price is None or price <= 0:
                continue
            # 计算 volume：每只约 5 万元 → 100 股整数倍
            volume = (int(self.TARGET_VALUE_PER_STOCK / price) // 100) * 100
            if volume < self.MIN_VOLUME:
                continue
            recs.append({
                "symbol": sym,
                "symbol_name": self._get_symbol_name(sym),
                "volume": volume,
            })

        decision = DailyDecision(
            as_of=as_of, recommendations=recs,
            decision_type="meta_agent" if recs else "no_action",
            reasoning=f"Regime {regime.regime}({regime.confidence:.2f}) → "
                      f"Weights {allocation.weights}\n[R1 推理]\n{allocation.reasoning[:300]}",
            regime={"regime": regime.regime, "probs": regime.regime_probs,
                    "confidence": regime.confidence},
            weights=allocation.weights,
            chosen_strategies=chosen_strategies,
        )
        self.decision_log.append(decision)
        return recs, decision

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _check_market_signal(self, as_of: date) -> str:
        """检查市场是否暴跌（不依赖 portfolio）。返回 'normal' / 'crash'。"""
        try:
            df = sources.get_price_history(
                HS300_ETF, as_of - timedelta(days=10), as_of, asset_type="etf",
            )
            if df is None or df.empty or len(df) < 2:
                return "normal"
            df = df.sort_values("date").reset_index(drop=True)
            today_close = float(df["close"].iloc[-1])
            yesterday_close = float(df["close"].iloc[-2])
            change_pct = (today_close / yesterday_close - 1) * 100
            if change_pct <= -3.0:
                return "crash"
        except Exception:
            pass
        return "normal"

    def _etf_defense_recommendation(self, as_of: date) -> list[dict[str, Any]]:
        """大盘崩盘时推荐 ETF 防御。"""
        price = self._get_latest_price(HS300_ETF, as_of, asset_type="etf")
        if price is None or price <= 0:
            return []
        volume = (int(self.TARGET_VALUE_PER_STOCK / price) // 100) * 100
        if volume < self.MIN_VOLUME:
            return []
        return [{
            "symbol": HS300_ETF,
            "symbol_name": "沪深300ETF",
            "volume": volume,
        }]

    def _get_latest_price(self, symbol: str, as_of: date,
                          asset_type: str = "stock") -> float | None:
        try:
            df = sources.get_price_history(
                symbol, as_of - timedelta(days=10), as_of, asset_type=asset_type,
            )
            if df is None or df.empty:
                return None
            return float(df.sort_values("date")["close"].iloc[-1])
        except Exception:
            return None

    def _get_symbol_name(self, symbol: str) -> str:
        if symbol in _NAME_CACHE:
            return _NAME_CACHE[symbol]
        # ETF 不在个股信息接口里 —— 直接用代码兜底，绝不发起注定失败的网络请求
        if self._is_etf(symbol):
            _NAME_CACHE[symbol] = symbol
            return symbol
        try:
            info = sources.get_stock_info(symbol)
            name = str(info.get("股票简称") or info.get("name") or "").strip()
        except Exception:
            name = ""
        # 无论成功失败都写缓存：失败也缓存代码本身，避免每天重试昂贵的失败请求
        _NAME_CACHE[symbol] = name or symbol
        return _NAME_CACHE[symbol]

    @staticmethod
    def _is_etf(symbol: str) -> bool:
        if not symbol or len(symbol) != 6:
            return False
        return symbol[:2] in ("51", "15", "58", "56")
