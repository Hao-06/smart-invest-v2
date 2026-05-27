"""基金经理 Agent（综合决策层）。

接收四位分析师的意见 + 当前组合状态，输出今日的具体交易订单。

使用 DeepSeek-R1（``role="deep"``）—— 推理模型在「权衡多方意见、做风险调整」
这类核心决策上明显胜过通用模型；其 ``reasoning_content`` 思维链也被保留下来
进入决策报告，对评委直接展示「Agent 是怎么想清楚的」。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from aifund.agents.base import Agent, AgentOpinion
from aifund.backtest.portfolio import Order, Portfolio, Side
from aifund.data.models import MarketSnapshot, StockData
from aifund.llm import LLMClient, get_llm_client
from aifund.llm.client import Role
from config.settings import settings


# ---------------------------------------------------------------------------
# 决策结果
# ---------------------------------------------------------------------------


@dataclass
class ManagerDecision:
    """基金经理一次决策的完整产物（含订单 + 推理链路，供报告呈现）。"""

    as_of: date | None
    orders: list[Order] = field(default_factory=list)
    rationale: str = ""  # 基金经理总体决策逻辑
    rejected: list[dict[str, str]] = field(default_factory=list)  # 主动放弃的标的及原因
    analyst_opinions: dict[str, list[AgentOpinion]] = field(default_factory=dict)
    reasoning_content: str = ""  # R1 的思维链（仅 deep 模型有）
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": str(self.as_of) if self.as_of else None,
            "orders": [
                {"symbol": o.symbol, "name": o.name, "side": o.side,
                 "shares": o.shares, "reason": o.reason}
                for o in self.orders
            ],
            "rationale": self.rationale,
            "rejected": self.rejected,
            "analyst_opinions": {
                sym: [op.to_dict() for op in ops]
                for sym, ops in self.analyst_opinions.items()
            },
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# 基金经理 Agent
# ---------------------------------------------------------------------------


class FundManager(Agent):
    """综合分析师意见 + 组合状态，输出今日交易订单。"""

    name = "基金经理"
    role: Role = "deep"  # 用 DeepSeek-R1 推理

    def __init__(self, llm: LLMClient | None = None) -> None:
        super().__init__(llm)
        # 沿用全局风控参数，但允许临时调整
        cfg = settings.backtest
        self.max_position_per_stock: float = cfg.max_position_per_stock
        self.max_total_position: float = cfg.max_total_position
        self.lot_size: int = cfg.lot_size

    # ------------------------------------------------------------------
    # Agent 基类接口（仅用于单标的快查，主流程用下方 decide()）
    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:  # pragma: no cover - 主流程不走这条
        return "基金经理由 decide() 驱动，不通过 analyze() 单标的调用。"

    def _build_user_prompt(self, snapshot: MarketSnapshot, stock: StockData) -> str:  # pragma: no cover
        return ""

    # ------------------------------------------------------------------
    # 主流程：综合决策
    # ------------------------------------------------------------------
    def decide(
        self,
        snapshot: MarketSnapshot,
        opinions: dict[str, list[AgentOpinion]],
        portfolio: Portfolio,
    ) -> ManagerDecision:
        """综合所有分析师意见 + 当前组合状态，输出今日订单。

        Args:
            snapshot: 决策基准日的市场快照。
            opinions: {symbol: [分析师意见, ...]}，由调度层并行收集而成。
            portfolio: 当前组合状态（现金、持仓、T+1 约束）。
        """
        decision = ManagerDecision(as_of=snapshot.as_of, analyst_opinions=opinions)

        # 没有任何意见 → 空操作
        if not opinions:
            decision.rationale = "无候选标的意见，不交易。"
            return decision

        system = self._build_system_prompt()
        user = self._build_decision_prompt(snapshot, opinions, portfolio)

        t0 = time.perf_counter()
        try:
            response = self.llm.chat(system=system, user=user, json_mode=True)
        except Exception as exc:  # noqa: BLE001
            decision.rationale = f"LLM 调用失败：{type(exc).__name__}: {exc}"
            decision.metadata["error"] = True
            return decision
        elapsed = round(time.perf_counter() - t0, 2)

        try:
            data = response.parse_json()
        except ValueError as exc:
            decision.rationale = f"LLM 响应非合法 JSON：{exc}"
            decision.metadata["error"] = True
            return decision
        if not isinstance(data, dict):
            decision.rationale = f"LLM 输出非对象：{type(data).__name__}"
            decision.metadata["error"] = True
            return decision

        decision.rationale = str(data.get("reasoning") or data.get("rationale") or "").strip()
        decision.rejected = self._parse_rejected(data.get("rejected"))
        decision.reasoning_content = response.reasoning_content or ""
        decision.metadata = {
            "model": response.model,
            "tokens_prompt": response.prompt_tokens,
            "tokens_completion": response.completion_tokens,
            "elapsed_sec": elapsed,
        }

        raw_orders = data.get("orders") or []
        if not isinstance(raw_orders, list):
            raw_orders = []
        decision.orders = self._parse_and_validate_orders(raw_orders, snapshot, portfolio)
        return decision

    # ------------------------------------------------------------------
    # Prompt 构造
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        max_pos = int(self.max_position_per_stock * 100)
        max_tot = int(self.max_total_position * 100)
        return f"""你是一名稳健的 AI 基金经理，下设 6 位独立分析师：技术面 / 资金面 /
消息面 / 风控 / 估值面（均为 LLM）+ 事件预测 Agent（基于双向 LSTM 的量化模型）。
你的方法论参考中银证券**多策略复合**、银河证券**行业差异化打分**、华泰金工
**多 Agent 协同 × 失效自我感知**等卖方实战经验。

**核心投研框架**：
- **自上而下 + 自下而上**（最重要 ⚠️）：每天**先看 `market_signal` 字段判断大盘势能**，
  再看个股。**逆势硬扛是基金经理最大的禁忌**：
  - `market_signal = "弱势下跌"`（候选标的 5 日中位 < -3%）→ **建仓减半甚至不建新仓**，
    优先减持已有持仓的弱势标的
  - `market_signal = "震荡偏弱"`（< -1%）→ 仅强一致信号建小仓
  - `market_signal = "横盘震荡"`（-1% ~ +1%）→ 按默认仓位策略
  - `market_signal = "震荡偏强"`（+1% ~ +3%）→ 正常建仓
  - `market_signal = "强势上涨"`（> +3%）→ 可适度加大单笔仓位
- **多策略复合而非单点押注**（中银 S1-S7 实证）：单一信号容易失效，6 维独立判断
  同向时信号才真正强。任何单一维度的看多 / 看空都不应导致激进操作。
- **score³ 非线性加权**（银河实证）：高一致性的强信号应当获得**显著更大**的权重 ——
  3 个维度共振 + 强 confidence 远胜 6 个维度纠缠的弱信号之和。
- **风险预算意识**：所有的「机会」要折算成「风险调整后的预期收益」。风控警示时，
  即使其他维度看多，也应当**降级**为小仓位试探或不动。
- **失效自我感知**（华泰 AI 量化失效应对）：当 6 个 Agent 之间存在**严重分歧**，
  或多个 Agent 的 confidence 都偏低时，说明当前市场环境难判断，**应优先选择不动**。
- **A 股 T+1 与制度约束**：买入次日才能卖出；订单股数必须 100 整数倍；不卖空。

**决策原则（按优先级）**：

1. **现金不是默认安全选项！**（最重要 ⚠️）：
   - 持有过多现金 = **错过被动的市场上涨收益**，长期看是真亏损
   - **目标总仓位 65-85%**（即现金占比 15-35%）
   - 现金 > 50% 时**必须**主动寻找建仓机会，不允许「保守观望」做决策默认值
   - 仅在 `market_signal = "弱势下跌"` 或风控明确警示 ≥ 2 个标的时，才允许现金占比 > 50%

2. **激进的分级仓位策略**（重要变更：阈值大幅放宽）：
   - **重仓建仓**：≥ 3 个 Agent 看多 → 建 **22-28% 总资产**（接近单票上限 30%）
   - **中仓建仓**：2 个 Agent 看多 + 风控不强烈看空 → 建 **12-18% 总资产**
   - **小仓试探**：1 个 Agent 强看多（conf > 0.6）+ 无强反对 → 建 **5-10% 总资产**
   - **不动**：≥ 3 个 Agent 明确看空 **或** 风控强烈警示（confidence > 0.8）

3. **市场环境放大器**：
   - `市场强势上涨` / `震荡偏强` → **每档仓位 + 5%**，且现金 < 30% 才合规
   - `市场弱势下跌` → 每档仓位 − 5%，避险优先

4. **风控的角色**（精细化）：
   - 风控看空 confidence 0.5-0.7：仓位**打 7 折**，仍建仓
   - 风控看空 confidence 0.7-0.85：仓位**打 4 折**，建最小仓试探
   - 风控看空 confidence > 0.85：**完全不买**（仅此一条硬否决）

5. **事件预测 Agent 的特殊地位**（LSTM 量化）：
   - 涨停概率 > 50% → 仓位 +5%（强买入信号）
   - 涨停概率 30-50% → 中性信号
   - 涨停概率 < 20% → 仓位 -5%

6. **持仓管理（T+1）**：已持有标的若 ≥ 3 个维度转空 → 卖出 50-100%
   （`sellable_shares_today` 字段告诉你今日可卖几股，0 = 昨天才买的）

7. **多标的并行机会**：一天可以同时下 3-5 笔订单，不必担心「交易过多」 ——
   只要每笔订单都有清晰的多维信号支撑。**单天 0 订单是失败**，除非所有标的都满足
   「不动」条件。

8. **仓位上限**：单笔建仓不超过现金的 50%；单标的市值不超过总资产 30%。

**A 股硬约束**：
- 买入 / 卖出股数必须为 **100 的整数倍**（不满足的订单会被引擎拒绝）。
- 单票市值不超过总资产的 **{max_pos}%**。
- 全部仓位不超过总资产的 **{max_tot}%**（留 ≥{100 - max_tot}% 现金）。
- 卖出股数不得超过当日 ``sellable_shares_today``（T+1 制度）。

**输出格式**（严格只输出 JSON，不要 markdown 围栏与解释文字）：
{{
  "orders": [
    {{"symbol": "600519", "side": "BUY" 或 "SELL", "shares": 100 的倍数整数, "reason": "简短理由（含触发的核心维度共振）"}}
  ],
  "reasoning": "本次综合决策的核心逻辑：哪几个维度共振 / 哪些维度否决 / 仓位为何如此（≤150 字）",
  "rejected": [
    {{"symbol": "...", "reason": "为何放弃该标的（最常见：信号分歧 / 风控警示 / 估值过高）"}}
  ]
}}

无操作时 ``orders`` 返回空数组 `[]`，``reasoning`` 仍要写明「为何按兵不动」（哪些维度
分歧 / 哪些 confidence 偏低）。
"""

    def _build_decision_prompt(
        self,
        snapshot: MarketSnapshot,
        opinions: dict[str, list[AgentOpinion]],
        portfolio: Portfolio,
    ) -> str:
        # ------ 组合状态 ------
        snap = portfolio.snapshot()
        cash_pct = (portfolio.cash / portfolio.equity()) if portfolio.equity() > 0 else 1.0
        position_summary: list[dict[str, Any]] = []
        for pos_d in snap["positions"]:
            sym = pos_d["symbol"]
            pos = portfolio.get_position(sym)
            sellable = pos.sellable_shares(snapshot.as_of) if pos else 0
            position_summary.append({
                **pos_d,
                "sellable_shares_today": sellable,
            })

        portfolio_block = {
            "as_of": str(snapshot.as_of),
            "cash": round(portfolio.cash, 2),
            "cash_pct": round(cash_pct * 100, 2),
            "equity": round(portfolio.equity(), 2),
            "initial_capital": portfolio.initial_capital,
            "positions": position_summary,
            "constraints": {
                "max_position_per_stock_pct": int(self.max_position_per_stock * 100),
                "max_total_position_pct": int(self.max_total_position * 100),
                "lot_size": self.lot_size,
            },
        }

        # ------ 市场环境感知（自上而下视角）------
        # 用候选标的最近 5 / 20 日收益的中位数作为「大盘势能」代理
        market_block = self._summarize_market_environment(snapshot, opinions)

        # ------ 分析师意见（按标的分组）------
        candidates_block: list[dict[str, Any]] = []
        for sym, ops in opinions.items():
            stock = snapshot.get(sym)
            entry: dict[str, Any] = {
                "symbol": sym,
                "name": stock.name if stock else sym,
                "industry": (stock.industry if stock else "") or "未知",
                "last_close": stock.last_close if stock else None,
                "opinions": [op.to_dict() for op in ops],
            }
            candidates_block.append(entry)

        return (
            "## 市场环境（自上而下视角，**决策前必读**）\n"
            f"{json.dumps(market_block, ensure_ascii=False, indent=2, default=str)}\n\n"
            "## 当前组合状态\n"
            f"{json.dumps(portfolio_block, ensure_ascii=False, indent=2, default=str)}\n\n"
            "## 候选标的与分析师意见\n"
            f"{json.dumps(candidates_block, ensure_ascii=False, indent=2, default=str)}\n\n"
            "请**先看市场环境**，再综合分析师意见和当前组合，给出今日交易决策。"
        )

    @staticmethod
    def _summarize_market_environment(
        snapshot: MarketSnapshot,
        opinions: dict[str, list[AgentOpinion]],
    ) -> dict[str, Any]:
        """从候选池的近期表现推导「市场环境信号」—— 自上而下的大盘势能代理。

        没有真实的沪深300 实时数据时，**候选标的的中位涨跌幅**是个简单但有效的市场温度计：
        - 多标的同向大跌 → 大盘弱势 / 风险事件 → 应保守
        - 多标的同向上涨 → 大盘强势 → 可积极
        """
        ret_5d_list: list[float] = []
        ret_20d_list: list[float] = []

        for sym in opinions:
            stock = snapshot.get(sym)
            if stock is None or not stock.has_price or len(stock.price_history) < 25:
                continue
            close = stock.price_history["close"].astype(float)
            try:
                ret_5d_list.append(float(close.iloc[-1] / close.iloc[-6] - 1) * 100)
                ret_20d_list.append(float(close.iloc[-1] / close.iloc[-21] - 1) * 100)
            except (IndexError, ZeroDivisionError):
                continue

        def _median(xs: list[float]) -> float | None:
            if not xs:
                return None
            xs = sorted(xs)
            n = len(xs)
            return round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 2)

        median_5d = _median(ret_5d_list)
        median_20d = _median(ret_20d_list)

        # 综合 5 / 20 日中位涨跌得到环境信号（数值化 + 标签化）
        if median_5d is None:
            signal = "数据不足"
            stance = "正常"
        elif median_5d < -3.0:
            signal = "弱势下跌"
            stance = "防御（建仓减半，宁可空仓）"
        elif median_5d < -1.0:
            signal = "震荡偏弱"
            stance = "谨慎（仅强信号建小仓）"
        elif median_5d < 1.0:
            signal = "横盘震荡"
            stance = "中性（按 prompt 默认仓位策略）"
        elif median_5d < 3.0:
            signal = "震荡偏强"
            stance = "积极（可正常建仓）"
        else:
            signal = "强势上涨"
            stance = "进取（可适度加大单笔仓位）"

        # 也聚合所有分析师的看多 / 看空票数（跨标的、跨 Agent）
        bull = bear = neutral = 0
        for ops in opinions.values():
            for op in ops:
                if op.action == "看多":
                    bull += 1
                elif op.action == "看空":
                    bear += 1
                else:
                    neutral += 1
        total = bull + bear + neutral
        bull_pct = round(bull / total * 100, 1) if total else 0.0
        bear_pct = round(bear / total * 100, 1) if total else 0.0

        return {
            "candidates_median_5d_return_pct": median_5d,
            "candidates_median_20d_return_pct": median_20d,
            "market_signal": signal,
            "recommended_stance": stance,
            "analyst_vote_summary": {
                "bullish_pct": bull_pct,
                "bearish_pct": bear_pct,
                "total_opinions": total,
            },
        }

    # ------------------------------------------------------------------
    # 订单校验
    # ------------------------------------------------------------------
    def _parse_and_validate_orders(
        self,
        raw_orders: list[Any],
        snapshot: MarketSnapshot,
        portfolio: Portfolio,
    ) -> list[Order]:
        """把 LLM 输出的订单数组归一并做合规初筛（引擎层还会再校验一次）。"""
        orders: list[Order] = []
        for item in raw_orders:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or item.get("代码") or "").strip()
            if not symbol or len(symbol) < 6:
                continue
            symbol = "".join(ch for ch in symbol if ch.isdigit()).zfill(6)[-6:]

            side_raw = str(item.get("side") or item.get("方向") or "").upper()
            shares_raw = item.get("shares") or item.get("volume") or 0
            try:
                shares = int(shares_raw)
            except (TypeError, ValueError):
                continue

            # 兼容「正负 volume」语义
            if side_raw not in ("BUY", "SELL"):
                if shares > 0:
                    side_raw = "BUY"
                elif shares < 0:
                    side_raw = "SELL"
                else:
                    continue
            shares = abs(shares)
            if shares <= 0 or shares % self.lot_size != 0:
                continue

            stock = snapshot.get(symbol)
            name = item.get("name") or item.get("symbol_name") or (stock.name if stock else "")
            reason = str(item.get("reason") or item.get("理由") or "")

            orders.append(Order(
                symbol=symbol, side=side_raw,  # type: ignore[arg-type]
                shares=shares, name=str(name), reason=reason,
            ))
        return orders

    @staticmethod
    def _parse_rejected(raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                sym = str(item.get("symbol") or item.get("代码") or "")
                reason = str(item.get("reason") or item.get("理由") or "")
                if sym:
                    out.append({"symbol": sym, "reason": reason})
        return out
