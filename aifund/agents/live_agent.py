"""**LiveTradingAgent** —— 真正的「**日频实战智能体**」，符合驼灵大赛官方规则。

设计哲学：
- **每天**调用一次 .recommend(as_of, current_portfolio) → 输出 JSON 买入建议
- **完全自主**：不需要人工干预
- **应急刹车 + Meta-Agent 智能调权** 双层架构
- 输出格式严格符合官方：`[{"symbol": "...", "symbol_name": "...", "volume": 100}]`

工作流（每日 9:00 调用）：
1. **DailyHealthCheck**：检测大盘崩盘 / 组合大跌 / 持仓异常
   - 触发 → 推荐买 ETF 防御
   - 正常 → 进入主流程
2. **Meta-Agent 主流程**（每天都跑）：
   - Market Regime Agent (R1) → 输出 regime 概率分布
   - Weight Allocator (R1) → 决定 4 策略权重
   - 各策略 select() → 给出目标持仓股池
3. **对比当前持仓**：
   - 已持有的股票 → 跳过（避免重复买入）
   - 缺的股票 → 计算买入股数（按当前可用现金 + 100 股整数倍）
4. **输出 JSON 列表**

注意（按官方规则）：
- volume 必须是 100 股的整数倍
- 推荐标的须当日可交易
- 当日无操作 → 返回 `[]`
- 卖出机制（不支持/平台管/复赛后调）按官方文档定义
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from aifund.agents.daily_health_check import DailyHealthCheckAgent, HS300_ETF
from aifund.agents.regime import MarketRegimeAgent
from aifund.agents.weight_allocator import WeightAllocatorAgent
from aifund.data import sources
from aifund.data.pipeline import DataPipeline
from aifund.strategies import BaseStrategy

#: 标的名称缓存（symbol → name），减少重复查询
_NAME_CACHE: dict[str, str] = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "511010": "5年期国债ETF",
}


@dataclass
class PortfolioSnapshot:
    """**当前组合状态**（由比赛平台或回测引擎提供）。"""
    as_of: date
    cash: float                              # 可用现金
    holdings: dict[str, int]                 # {symbol: shares}
    equity: float                            # 总资产 = cash + 持仓市值
    holdings_market_value: dict[str, float] = field(default_factory=dict)  # 每只票市值


@dataclass
class DailyDecision:
    """每日决策完整记录（落盘用，可审计）。"""
    as_of: date
    recommendations: list[dict[str, Any]]    # 输出的 JSON 列表
    decision_type: str                       # "health_emergency" / "meta_agent" / "no_action"
    reasoning: str                           # 决策理由（含 R1 思维链）
    regime: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    health_action: str | None = None
    portfolio_snapshot: dict[str, Any] = field(default_factory=dict)


class LiveTradingAgent:
    """**日频实战智能体** —— 驼灵大赛核心入口。"""

    name = "实战智能体"

    def __init__(
        self,
        pipeline: DataPipeline,
        strategies: dict[str, BaseStrategy],
        regime_agent: MarketRegimeAgent | None = None,
        allocator: WeightAllocatorAgent | None = None,
        health_check: DailyHealthCheckAgent | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.strategies = strategies
        self.regime_agent = regime_agent or MarketRegimeAgent()
        self.allocator = allocator or WeightAllocatorAgent(strategies)
        self.health_check = health_check or DailyHealthCheckAgent()
        # 持久化记忆（跨日）
        self.decision_log: list[DailyDecision] = []

    #: 单次推荐买入的目标占总资产的最大比例（防止单笔砸光现金）
    MAX_BUY_RATIO_PER_DAY = 0.30
    #: 单只票目标仓位最大值
    MAX_POSITION_PER_SYMBOL = 0.15
    #: 权重低于此阈值的策略跳过 select（与 AllWeatherManager 一致）
    MIN_WEIGHT_TO_SELECT = 0.10
    #: 卖出阈值：当前持仓股不在目标股池里 + 持有超过 N 天时卖出
    SELL_AFTER_DAYS_NOT_IN_TARGET = 5
    #: 是否启用 SELL（按官方规则未明确，假设 volume 负数 = 卖出）
    ENABLE_SELL = True

    def recommend(
        self,
        as_of: date,
        portfolio: PortfolioSnapshot,
    ) -> tuple[list[dict[str, Any]], DailyDecision]:
        """每日决策入口。

        Args:
            as_of: 今天日期。
            portfolio: 当前组合状态（平台/回测引擎提供）。

        Returns:
            (recommendations, decision):
                recommendations: 直接输出给平台的 JSON 列表
                decision: 完整决策记录（含 R1 思维链 + 信号 + 推理过程）
        """
        # ============ 步骤 1: 日频健康检查 ============
        health = self.health_check.check(
            as_of=as_of,
            current_equity=portfolio.equity,
            current_positions=list(portfolio.holdings.keys()),
        )
        if health.action != "no_action":
            # 应急 → 推荐买 ETF
            recs = self._emergency_recommendation(as_of, portfolio, health)
            decision = DailyDecision(
                as_of=as_of, recommendations=recs,
                decision_type="health_emergency",
                reasoning=health.reasoning,
                health_action=health.action,
                portfolio_snapshot={"cash": portfolio.cash, "equity": portfolio.equity,
                                    "n_holdings": len(portfolio.holdings)},
            )
            self.decision_log.append(decision)
            return recs, decision

        # ============ 步骤 2: Meta-Agent 主决策 ============
        regime = self.regime_agent.analyze(as_of)
        allocation = self.allocator.allocate(regime)

        # ============ 步骤 3: 各策略给出目标持仓 ============
        target_holdings: dict[str, float] = {}  # {symbol: target_weight}
        for strat_name, weight in allocation.weights.items():
            if weight < self.MIN_WEIGHT_TO_SELECT or strat_name not in self.strategies:
                continue
            output = self.strategies[strat_name].select(as_of)
            if not output.symbols:
                continue
            effective_weight = weight * output.position_ratio
            per_symbol = effective_weight / len(output.symbols)
            for sym in output.symbols:
                target_holdings[sym] = target_holdings.get(sym, 0) + per_symbol

        # 限制单只票上限
        target_holdings = {s: min(w, self.MAX_POSITION_PER_SYMBOL)
                           for s, w in target_holdings.items()}

        # ============ 步骤 4a: 卖出 — 当前持仓不在目标池 + 持有超过 N 天 ============
        sell_recs = self._compute_sell_recommendations(as_of, portfolio, target_holdings)

        # ============ 步骤 4b: 买入 — 目标池缺的股票 ============
        # 假设卖出后现金增加，预估增加额（仅用于本次买入预算估算）
        estimated_extra_cash = sum(
            portfolio.holdings_market_value.get(r["symbol"], 0)
            for r in sell_recs
        ) if sell_recs else 0
        buy_recs = self._compute_buy_recommendations(
            as_of, portfolio, target_holdings,
            extra_cash_estimate=estimated_extra_cash,
        )

        # 合并：卖出在前，买入在后（符合实战 T+1 现金回流逻辑）
        recs = sell_recs + buy_recs

        decision = DailyDecision(
            as_of=as_of, recommendations=recs,
            decision_type="meta_agent" if recs else "no_action",
            reasoning=f"Regime {regime.regime}({regime.confidence:.2f}) → "
                      f"Weights {allocation.weights}\n[R1 推理]\n{allocation.reasoning[:500]}",
            regime={"regime": regime.regime, "probs": regime.regime_probs,
                    "confidence": regime.confidence},
            weights=allocation.weights,
            portfolio_snapshot={"cash": portfolio.cash, "equity": portfolio.equity,
                                "n_holdings": len(portfolio.holdings)},
        )
        self.decision_log.append(decision)
        return recs, decision

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------
    def _emergency_recommendation(
        self,
        as_of: date,
        portfolio: PortfolioSnapshot,
        health,
    ) -> list[dict[str, Any]]:
        """应急情况下推荐买入 ETF。"""
        if portfolio.cash < 1000 or HS300_ETF in portfolio.holdings:
            return []  # 没现金 / 已持有 → 不动
        price = self._get_latest_price(HS300_ETF, as_of, asset_type="etf")
        if price is None or price <= 0:
            return []
        # 应急时把 50% 现金买入 ETF
        target_value = portfolio.cash * 0.5
        volume = (int(target_value / price) // 100) * 100
        if volume < 100:
            return []
        return [{
            "symbol": HS300_ETF,
            "symbol_name": "沪深300ETF",
            "volume": volume,
        }]

    def _compute_sell_recommendations(
        self,
        as_of: date,
        portfolio: PortfolioSnapshot,
        target_holdings: dict[str, float],
    ) -> list[dict[str, Any]]:
        """**卖出推荐** —— 当前持仓不在目标股池里时卖出。

        规则：
        - 当前持仓股不在今日 target_holdings 里
        - 已持有超过 5 天（避免 T+1 限制 / 频繁换仓）
        - 卖出建议用 **负 volume** 表示

        注：官方规则没明确 SELL 格式，这里**假设 volume 负数 = 卖出**。
        实际比赛中如不支持，平台会忽略，相当于自动持有。
        """
        if not self.ENABLE_SELL:
            return []

        recs = []
        target_symbols = set(target_holdings.keys())
        for sym, shares in portfolio.holdings.items():
            if shares <= 0:
                continue
            # 在目标池里 → 不卖
            if sym in target_symbols:
                continue
            # 卖出股数：全部卖出（100 股整数倍）
            sell_volume = (shares // 100) * 100
            if sell_volume < 100:
                continue
            recs.append({
                "symbol": sym,
                "symbol_name": self._get_symbol_name(sym),
                "volume": -sell_volume,   # 负数 = 卖出
            })
        return recs

    def _compute_buy_recommendations(
        self,
        as_of: date,
        portfolio: PortfolioSnapshot,
        target_holdings: dict[str, float],
        extra_cash_estimate: float = 0.0,
    ) -> list[dict[str, Any]]:
        """根据目标持仓 vs 当前持仓，计算今日买入建议。

        Args:
            extra_cash_estimate: 同日卖出预计释放的现金（用于扩大买入预算）。
        """
        # 限制今日新建仓总预算（含同日卖出释放的现金）
        effective_cash = portfolio.cash + extra_cash_estimate * 0.95  # 保守估计
        budget = min(
            effective_cash * 0.95,
            portfolio.equity * self.MAX_BUY_RATIO_PER_DAY,
        )

        recs = []
        sorted_targets = sorted(target_holdings.items(), key=lambda x: -x[1])
        for sym, target_w in sorted_targets:
            if budget < 1000:
                break
            if sym in portfolio.holdings and portfolio.holdings[sym] > 0:
                continue
            asset_type = "etf" if self._is_etf(sym) else "stock"
            price = self._get_latest_price(sym, as_of, asset_type=asset_type)
            if price is None or price <= 0:
                continue
            target_value = portfolio.equity * target_w
            actual_budget = min(target_value, budget)
            volume = (int(actual_budget / price) // 100) * 100
            if volume < 100:
                continue
            budget -= volume * price
            recs.append({
                "symbol": sym,
                "symbol_name": self._get_symbol_name(sym),
                "volume": volume,
            })
        return recs

    def _get_latest_price(self, symbol: str, as_of: date,
                           asset_type: str = "stock") -> float | None:
        """获取标的当日（或最近）收盘价。"""
        try:
            df = sources.get_price_history(
                symbol, as_of - timedelta(days=10), as_of, asset_type=asset_type,
            )
            if df is None or df.empty:
                return None
            df = df.sort_values("date").reset_index(drop=True)
            return float(df["close"].iloc[-1])
        except Exception:
            return None

    def _get_symbol_name(self, symbol: str) -> str:
        if symbol in _NAME_CACHE:
            return _NAME_CACHE[symbol]
        try:
            info = sources.get_stock_info(symbol)
            name = str(info.get("股票简称") or info.get("name") or "").strip()
            if name:
                _NAME_CACHE[symbol] = name
                return name
        except Exception:
            pass
        return symbol

    @staticmethod
    def _is_etf(symbol: str) -> bool:
        """简单判断：5XXXXX / 15XXXX 是 ETF 代码"""
        if not symbol or len(symbol) != 6:
            return False
        prefix = symbol[:2]
        return prefix in ("51", "15", "58", "56")
