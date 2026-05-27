"""分析师 Agent（4 个并行子角色）。

- ``TechnicalAnalyst``：技术面（K线 / 均线 / MACD / RSI / 量能 / 形态学）
- ``FundFlowAnalyst``：资金面（主力净流入 / 北向资金）
- ``NewsAnalyst``：消息面（个股近期新闻情绪 + 关键事件）
- ``RiskAnalyst``：风控面（波动率 / 回撤 / 流动性 / 行业 / 估值风险）
- ``ValuationAnalyst``：估值面（PE/PB/PS + 历史分位 + 股息率，基本面价值视角）

五个分析师并行调用 DeepSeek-V3（``role="fast"``），每个对单只标的输出
统一格式的 ``AgentOpinion``，再交给基金经理 Agent 综合权衡。
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from aifund import indicators
from aifund.agents.base import Agent
from aifund.data.models import MarketSnapshot, StockData

# 所有分析师共用的 JSON 输出格式约束 —— 抽出来避免重复
_JSON_FORMAT = """\
**输出格式**（严格只输出 JSON，无 markdown 围栏、无解释文字）：
{
  "action": "看多" 或 "看空" 或 "中性",
  "score": 数字，[-1, 1]，越正越看多，
  "confidence": 数字，[0, 1]，置信度，
  "reasoning": 一两句话核心判断（不超过 80 字），
  "key_facts": ["关键事实 1", ...] 最多 5 条,
  "risks": ["风险点 1", ...] 最多 3 条
}"""


# ---------------------------------------------------------------------------
# 数据摘要工具（把 DataFrame 压成对 LLM 友好的紧凑字典）
# ---------------------------------------------------------------------------


def _round_num(v: Any, ndigits: int = 2) -> float | None:
    try:
        f = float(v)
        if np.isnan(f):
            return None
        return round(f, ndigits)
    except (TypeError, ValueError):
        return None


def summarize_fund_flow(fund_flow: pd.DataFrame) -> dict[str, Any]:
    """把资金流 DataFrame 压成最近 5/10 日净流入与趋势的紧凑字典。"""
    if fund_flow is None or fund_flow.empty:
        return {"available": False, "reason": "无资金流数据"}
    df = fund_flow.tail(20).copy()

    def sum_tail(col: str, n: int) -> float | None:
        if col not in df.columns:
            return None
        s = df[col].tail(n).dropna()
        if s.empty:
            return None
        return round(float(s.sum()) / 10000.0, 2)  # 元 → 万元

    last_row = df.iloc[-1] if not df.empty else None

    def latest(col: str) -> float | None:
        if last_row is None or col not in df.columns:
            return None
        return _round_num(last_row[col])

    return {
        "available": True,
        "as_of": str(last_row["date"]) if last_row is not None else None,
        "bars": len(df),
        "today": {
            "main_net_wan": _round_num((last_row["main_net"] or 0) / 10000.0)
            if last_row is not None and "main_net" in df.columns else None,
            "main_net_pct": latest("main_net_pct"),
            "pct_change": latest("pct_change"),
        },
        "cumulative_main_net_wan": {
            "last_5d": sum_tail("main_net", 5),
            "last_10d": sum_tail("main_net", 10),
            "last_20d": sum_tail("main_net", 20),
        },
        "cumulative_xlarge_net_wan": {
            "last_5d": sum_tail("xlarge_net", 5),
            "last_10d": sum_tail("xlarge_net", 10),
        },
        "small_money_5d_wan": sum_tail("small_net", 5),
    }


def summarize_hsgt(hsgt: pd.DataFrame) -> dict[str, Any]:
    """把北向资金 DataFrame 压成最近态度。"""
    if hsgt is None or hsgt.empty:
        return {"available": False}
    df = hsgt.tail(5).copy()

    def latest(col: str) -> float | None:
        return _round_num(df.iloc[-1][col]) if col in df.columns else None

    cum_5d = None
    if "net_buy" in df.columns:
        s = df["net_buy"].dropna()
        if not s.empty:
            cum_5d = round(float(s.sum()) / 10000.0, 2)  # 元 → 万元

    return {
        "available": True,
        "today_net_buy_wan": _round_num((latest("net_buy") or 0) / 10000.0),
        "cumulative_net_buy_5d_wan": cum_5d,
        "today_inflow_wan": _round_num((latest("net_inflow") or 0) / 10000.0),
    }


def summarize_news(news: pd.DataFrame, max_items: int = 8) -> dict[str, Any]:
    """把新闻 DataFrame 压成关键标题 + 内容摘要，控制 token 用量。"""
    if news is None or news.empty:
        return {"available": False, "items": []}
    df = news.head(max_items)
    items = []
    for _, row in df.iterrows():
        content = str(row.get("content") or "")
        items.append({
            "publish_time": str(row.get("publish_time") or ""),
            "title": str(row.get("title") or "").strip(),
            "summary": content[:200] + ("…" if len(content) > 200 else ""),
            "source": str(row.get("source") or ""),
        })
    return {"available": True, "count": len(items), "items": items}


def summarize_risk_facts(stock: StockData) -> dict[str, Any]:
    """汇总该标的的内在风险特征（波动率、回撤、流动性、市值、行业）。"""
    summary = indicators.summarize(stock.price_history)
    if not summary.get("available"):
        return {"available": False, "reason": "行情不足以评估风险"}

    valuation_risk = None
    if stock.valuation:
        pct = stock.valuation.get("percentile_in_1y", {}) or {}
        pe_pct = pct.get("pe_ttm")
        pb_pct = pct.get("pb")
        # 任一估值分位 > 0.8 → 历史高位（估值风险偏高）
        if (pe_pct is not None and pe_pct > 0.8) or (pb_pct is not None and pb_pct > 0.8):
            valuation_risk = "历史高位（≥80% 分位），估值压力偏大"
        elif (pe_pct is not None and pe_pct < 0.3) or (pb_pct is not None and pb_pct < 0.3):
            valuation_risk = "历史低位（≤30% 分位），估值偏便宜"

    return {
        "available": True,
        "as_of": summary.get("as_of"),
        "volatility_annualized_pct": summary.get("volatility_annualized_pct"),
        "max_drawdown_60d_pct": summary.get("max_drawdown_60d_pct"),
        "atr14": summary.get("atr14"),
        "price_position_60d": summary.get("price_position_60d"),
        "return_skew_5d": summary.get("return_skew_5d"),
        "industry": stock.industry or "未知",
        "float_market_cap_wan": (
            round(stock.float_market_cap / 10000.0, 2) if stock.float_market_cap else None
        ),
        "listed_date": str(stock.info.get("上市时间", "")),
        "valuation_risk_note": valuation_risk,  # 估值面风险提示（None = 估值适中）
    }


def summarize_valuation_facts(stock: StockData) -> dict[str, Any]:
    """汇总该标的的估值面特征：PE/PB/PS + 在近一年的历史分位 + 市值。"""
    val = stock.valuation or {}
    if not val:
        return {"available": False, "reason": "无估值数据"}
    return {
        "available": True,
        "as_of": val.get("as_of"),
        "industry": stock.industry or "未知",
        "latest": val.get("latest", {}),
        "percentile_in_1y": val.get("percentile_in_1y", {}),
        "note": (
            "percentile 0 = 历史最低估值（便宜）；1 = 历史最高（贵）；"
            "0.3-0.7 区间为合理区间。"
        ),
    }


# ---------------------------------------------------------------------------
# 1) 技术面分析师
# ---------------------------------------------------------------------------


class TechnicalAnalyst(Agent):
    name = "技术面分析师"
    role = "fast"

    def _system_prompt(self) -> str:
        return f"""你是一名资深的 A 股技术面分析师，专注于日线级别的趋势、动量与形态判断。
你的方法论来自卖方金工研究的实战经验（华泰金工 / 中银证券量化系列）。

**核心投研框架**：
- **多频段融合视角**（华泰 AI 量价端到端思想）：日 / 周 / 月不同尺度的信号要相互印证，
  短期信号若与中期趋势相悖则置信度打折。
- **困境反转 vs 短期反转**（中银 S4 经验）：2-3 年级别的反转效应真实存在，但 1 周内
  的反转往往是噪声。看到「跌幅过深」时先判断是中期反转机会还是短期破位继续。

**判断维度（按重要性排序）**：
1. **趋势主线**：均线排列（多头/空头/纠缠）、MACD 状态。趋势永远是技术分析的第一原则。
2. **多周期共振**：1d / 5d / 20d 收益率方向是否一致 —— 共振时信号强，背离时不轻易下注。
3. **位置与极值**：60 日分位（>0.85 高位警惕 / <0.15 低位关注）、布林带位置、RSI 超买超卖。
4. **量能配合**：放量上涨（量比>1.5）= 资金推升；缩量上涨（量比<0.7）= 抛压不足但需警惕动能不足。
5. **形态学语义**（`candle` 字段已经把形态归一为语义标签）：
   - 高开放量启动、长上影抛压、长下影支撑、十字星胶着 —— 直接采信 `pattern` 标签
   - `gap_pct > 1%` 高开 + `body_ratio > 0.5` 强实体 + `close_position > 0.5` → 强势启动
6. **偏度与极值** `return_skew_5d`：正偏 = 偶尔大涨拉高均值（典型上涨）；负偏 = 偶尔大跌（破位）

**判断窗口**：未来 1-5 个交易日。
**纪律**：不做基本面、不做估值判断；不引用未提供的数据；多周期背离时主动下调 confidence。

{_JSON_FORMAT}"""

    def _build_user_prompt(self, snapshot: MarketSnapshot, stock: StockData) -> str:
        ind = indicators.summarize(stock.price_history)
        return (
            f"标的：{stock.symbol} {stock.name}（行业：{stock.industry or '未知'}）\n"
            f"决策基准日：{snapshot.as_of}\n"
            f"技术指标：\n{self._safe_json(ind)}\n\n"
            "请给出技术面判断。"
        )


# ---------------------------------------------------------------------------
# 2) 资金面分析师
# ---------------------------------------------------------------------------


class FundFlowAnalyst(Agent):
    name = "资金面分析师"
    role = "fast"

    def _system_prompt(self) -> str:
        return f"""你是一名 A 股资金面分析师，关注机构资金与外资动向。
你的方法论参考中银证券量化行业轮动系列 S5「资金流策略」与华泰逐笔成交深度学习模型。

**核心投研框架**：
- **聪明钱原则**：超大单 + 大单代表机构与游资，小单代表散户。聪明钱与散户**反向**时
  通常是机构的判断更靠谱。
- **资金 × 价格的"博弈"维度**（华泰逐笔成交思路）：仅看资金净额不够，还要看资金动作
  与价格走势是否一致 —— **价升资入** = 健康趋势；**价升资出** = 拉高出货警惕；
  **价跌资入** = 主力低位吸筹（潜在机会）；**价跌资出** = 共识看空。
- **多窗口验证**：近 5 日 / 10 日累计净流入提供短中期视角，比单日噪声大的数据可靠。

**判断维度**：
1. **主力资金趋势**：近 5 / 10 日累计净流入（正 = 机构入场；负 = 出逃）+ 净流入占比
2. **聪明钱 vs 散户**：超大单 / 大单累计 vs 小单累计的方向与背离
3. **价资关系**：把资金动作与近期价格变化对照，识别 4 种博弈模式（见上）
4. **北向资金风向**：当日 + 近 5 日累计 —— **全市场情绪温度计**，与个股共振时信号增强

**单位**：金额单位为「万元」。判断窗口：1-5 个交易日。
**纪律**：不引用未提供的数据；价资背离时主动降低 confidence 并在 reasoning 里点出背离类型。

{_JSON_FORMAT}"""

    def _build_user_prompt(self, snapshot: MarketSnapshot, stock: StockData) -> str:
        fund = summarize_fund_flow(stock.fund_flow)
        hsgt = summarize_hsgt(snapshot.hsgt_flow)
        return (
            f"标的：{stock.symbol} {stock.name}（行业：{stock.industry or '未知'}）\n"
            f"决策基准日：{snapshot.as_of}\n"
            f"个股资金流摘要：\n{self._safe_json(fund)}\n\n"
            f"全市场北向资金摘要：\n{self._safe_json(hsgt)}\n\n"
            "请综合个股与外资动向给出资金面判断。"
        )


# ---------------------------------------------------------------------------
# 3) 消息面分析师
# ---------------------------------------------------------------------------


class NewsAnalyst(Agent):
    name = "消息面分析师"
    role = "fast"

    def _system_prompt(self) -> str:
        return f"""你是一名 A 股消息面分析师，从近期新闻中提炼对短期股价的影响。
你的方法论受华泰金工 LLM-FADT「大模型增强文本选股」与「LLM 驱动的因子语义进化」启发。

**核心投研框架**（不只做情感分类，要做"博观"式语义解读）：
- **事件分类与影响半径**：业绩 / 分红 / 增发 / 收购 / 减持 / 政策 / 监管 / 诉讼 / 人事 / 行业景气
  → 每类事件的「方向 × 时效」不同：业绩超预期 = 强多 / 短期；监管处罚 = 强空 / 中期。
- **新旧信息辨别**：同一利好若已重复见报多次，市场往往已充分计价 —— 「重复定价」会
  让看似好消息的实际影响接近 0。判断时关注「**该信息是否首次进入市场视野**」。
- **管理层意图与催化剂**：公告里的承诺（回购、增持、扩产）+ 关键人物表态 ≈ 短期催化剂。
- **信息可信度分层**：官方公告 > 权威媒体（证券时报/中国证券报） > 财经媒体 > 自媒体 > 传闻。
  低可信度信息要打折，传闻通常不构成 action 依据。
- **行业景气联动**：个股新闻要放在所属行业的景气度背景里 —— 行业上行期的利好放大，
  行业下行期的同类利好往往不及预期。

**判断维度**：
1. **整体情绪倾向**：偏多 / 偏空 / 中性，配合主要事件类型
2. **关键催化剂**：识别 1-2 个最具影响力的事件并说明其影响半径
3. **市场预期差**：信息是否已被市场充分计价（重复见报次数、股价是否已反应）
4. **可信度**：信息源等级 → 调整 confidence

**判断窗口**：1-5 个交易日。
**纪律**：缺新闻时坦诚说明并把 confidence 降到 ≤0.3；不要把猜测当事实。

{_JSON_FORMAT}"""

    def _build_user_prompt(self, snapshot: MarketSnapshot, stock: StockData) -> str:
        news = summarize_news(stock.news)
        return (
            f"标的：{stock.symbol} {stock.name}（行业：{stock.industry or '未知'}）\n"
            f"决策基准日：{snapshot.as_of}\n"
            f"近期新闻（已按时间倒序，最多 8 条）：\n{self._safe_json(news)}\n\n"
            "请基于新闻给出消息面判断；缺新闻时坦诚说明并降低 confidence。"
        )


# ---------------------------------------------------------------------------
# 4) 风控分析师
# ---------------------------------------------------------------------------


class RiskAnalyst(Agent):
    """从「标的内在风险」角度评估，不涉及组合层仓位（那是基金经理的事）。"""

    name = "风控分析师"
    role = "fast"

    def _system_prompt(self) -> str:
        return f"""你是一名风控分析师，从标的**内在风险**的角度给出评估
（不考虑当前组合中的仓位 —— 那是基金经理的工作）。你借鉴中银证券**高估值预警系统**
与华泰金工**AI 模型泛化与失效应对**的方法论。

**核心投研框架**：
- **多维风险拼图**：单一指标不足以判断风险。波动率 + 回撤 + 流动性 + 行业属性 + 估值
  分位要交叉验证 —— 任两个维度同时报警时，风险才真正算「高」。
- **高估值预警**（中银金工实践）：当估值（PE/PB）处于**滚动 6 年 95% 分位以上**，触发
  「估值泡沫」预警，此类标的的风险溢价显著放大。本 Agent 已收到 `valuation_risk_note` 字段。
- **拥挤交易识别**：若价格位置在 60 日 95% 分位 + 波动率显著上升 + 量比异常放大，
  通常是**情绪过热**而非真正趋势，需警惕回撤。
- **市场结构感知**：A 股小市值（流通市值 < 30 亿）存在流动性折价 + 退市风险 + 庄家
  操纵风险，对应 confidence 应主动下调。

**判断维度**：
1. **波动性**：年化波动率（>40% 高 / 20-40% 中 / <20% 低）
2. **回撤特征**：近 60 日最大回撤（< -25% 风险偏高 / -10%~-25% 正常 / > -10% 强势）
3. **价格位置**：60 日分位（接近 1 警惕高位回落 / 接近 0 关注是否破位继续）
4. **流动性 / 市值**：流通市值 < 50 亿小盘股流动性差；< 30 亿应避免重仓
5. **行业 + 估值风险**：所属行业政策 / 周期位置 + `valuation_risk_note` 是否报警
6. **形态偏度**：`return_skew_5d` 极端负偏 → 警惕尾部风险

**输出含义**（语义反转！注意）：
- ``action`` = "看多" 表示「**风险可控、敢配置**」（不是预测涨）
- ``action`` = "看空" 表示「**风险偏高、应规避或轻仓**」（不是预测跌）
- ``score`` 反映风险友好度（+1 安全 / -1 危险）

{_JSON_FORMAT}"""

    def _build_user_prompt(self, snapshot: MarketSnapshot, stock: StockData) -> str:
        risk = summarize_risk_facts(stock)
        return (
            f"标的：{stock.symbol} {stock.name}\n"
            f"决策基准日：{snapshot.as_of}\n"
            f"风险特征摘要：\n{self._safe_json(risk)}\n\n"
            "请给出该标的的内在风险评估。"
        )


# ---------------------------------------------------------------------------
# 5) 估值面分析师（基本面 / 价值投资视角）
# ---------------------------------------------------------------------------


class ValuationAnalyst(Agent):
    """从估值面（PE/PB/PS + 历史分位）角度评估。

    与 ``RiskAnalyst`` 互补：风控看的是「会不会大跌」，估值看的是「贵不贵 / 性价比」。
    """

    name = "估值面分析师"
    role = "fast"

    def _system_prompt(self) -> str:
        return f"""你是一名 A 股估值面分析师，从基本面与历史估值分位的角度评估标的。
你的方法论来自中国银河证券**国企/科技/消费基本面因子选股**系列与中银证券**高估值预警系统**。

**核心投研框架**（行业差异化 + 历史分位双维度）：
- **行业差异化原则**（银河证券实证）：不同行业天然估值中枢差异巨大 ——
  银行/地产/煤炭 PE 中枢 5-10；消费/科技 PE 中枢 30-60；不能用同一套阈值套所有行业。
  **核心是看「该标的 vs 其行业历史区间」的相对位置**。
- **历史分位 > 绝对估值**：单看 PE = 20 没意义；看 PE 在近 1 年 / 5 年的分位才有意义。
  > 80% 分位 = 历史贵；< 20% 分位 = 历史便宜。
- **价值陷阱警惕**：低估值不等于机会。若估值历史低位但基本面持续恶化（盈利下滑 / 行业
  萎缩 / 监管打压），属于「**价值陷阱**」—— 看似便宜实则越买越亏。这是 Hao 之前
  多因子策略在 2024-2025 跑输沪深300 67% 的关键教训。
- **均值回归 vs 趋势延续**：估值历史极值（>95% 或 <5% 分位）通常会向均值回归，
  但需要确认是「估值修复」还是「业绩证伪」。

**判断维度**：
1. **PE_TTM / PE_静**（市盈率）：行业内相对位置 + 历史分位双判断
2. **PB**（市净率）：< 1 破净（资产折价但需查原因）；> 5 高估
3. **PS_TTM**（市销率）：成长股核心指标
4. **PEG**（PE/盈利增速）：< 1 低估 / 1-2 合理 / > 2 高估（结合盈利增速判断）
5. **PCF**（市现率）：现金流估值，越低越被低估
6. **历史分位**（核心权重最高！）：
   - `pe_ttm_percentile > 0.8` → 历史**高位**，估值杀风险大
   - 0.3-0.7 → 合理区间
   - `pe_ttm_percentile < 0.3` → 历史**低位**，但需排查价值陷阱
7. **流通市值**（`float_mv_wan`）：< 50 亿小盘股流动性溢价/风险；> 1000 亿大盘股价值稳定
8. **行业属性**（输入字段 `industry`）：用作横向对标参考

**典型逻辑模板**：
- 高分位 + 业绩支撑弱 / 行业景气下行 → **强看空**（估值杀风险）
- 高分位 + 业绩持续高增长 → **中性偏空**（贵但有支撑）
- 低分位 + 基本面稳定 + 行业未恶化 → **强看多**（均值回归机会）
- 低分位 + 基本面持续恶化 → **看空**（价值陷阱）
- 分位中性 → 估值面不构成主要驱动，confidence 偏低

**纪律**：缺估值数据时坦诚说明并把 confidence 降到 ≤0.3；不要瞎猜数字。

{_JSON_FORMAT}"""

    def _build_user_prompt(self, snapshot: MarketSnapshot, stock: StockData) -> str:
        val = summarize_valuation_facts(stock)
        return (
            f"标的：{stock.symbol} {stock.name}（行业：{stock.industry or '未知'}）\n"
            f"决策基准日：{snapshot.as_of}\n"
            f"估值面摘要：\n{self._safe_json(val)}\n\n"
            "请基于估值面给出判断；缺数据时坦诚说明并降低 confidence。"
        )


# ---------------------------------------------------------------------------
# 6) 事件预测 Agent（LSTM 量化模型，不调 LLM）
# ---------------------------------------------------------------------------


class EventPredictionAgent(Agent):
    """基于 LSTM 模型的事件预测 Agent —— 输出**次日涨停概率**。

    与其他 5 位 LLM 分析师不同，本 Agent 不调用大模型，直接用本地训练好的
    双向 LSTM 推理。这让多 Agent 团队形成「**LLM × 量化双轨决策**」的差异化 ——
    LLM 擅长综合上下文与定性判断，LSTM 擅长定量预测短期事件型机会。

    返回的 AgentOpinion：
    - ``action``：``"看多"``（概率 > 0.6）/ ``"看空"``（< 0.4）/ ``"中性"``
    - ``score`` ∈ [-1, 1] 由概率映射
    - ``confidence``：直接取概率（看多时）或 1-概率（看空时）
    - ``key_facts``：含具体涨停概率数值
    """

    name = "事件预测Agent"
    role = "fast"  # 占位，本 Agent 不实际使用 LLM

    def __init__(self) -> None:
        # 不调 super().__init__() —— 我们不需要 LLM 客户端
        # 但为了类型一致性，仍设 role 属性
        from aifund.ml import LimitUpPredictor

        self.predictor = LimitUpPredictor()

    def _system_prompt(self) -> str:  # pragma: no cover - 本 Agent 不走 LLM
        return ""

    def _build_user_prompt(self, snapshot: MarketSnapshot, stock: StockData) -> str:  # pragma: no cover
        return ""

    def analyze(self, snapshot: MarketSnapshot, symbol: str) -> "AgentOpinion":
        """完全覆盖基类实现：不调 LLM，直接用 LSTM 推理。"""
        from aifund.agents.base import AgentOpinion

        stock = snapshot.get(symbol)
        if stock is None or not stock.has_price:
            return AgentOpinion(
                agent_name=self.name, symbol=symbol,
                reasoning="无可用行情，跳过预测", confidence=0.0,
                metadata={"role": self.role, "error": False},
            )

        result = self.predictor.predict(stock.price_history)
        if not result.get("available"):
            return AgentOpinion(
                agent_name=self.name, symbol=symbol, action="中性",
                reasoning=f"LSTM 暂不可用：{result.get('reason', '未知原因')}",
                confidence=0.0,
                metadata={"role": self.role, "error": False},
            )

        prob = float(result["limit_up_probability"])

        if prob > 0.6:
            action = "看多"
            score = min(1.0, (prob - 0.5) * 2)
            confidence = min(0.95, prob)
            reasoning = (
                f"LSTM 模型预测次日涨停概率 {prob * 100:.1f}%，"
                f"显著高于市场平均（~3-5%），短线机会信号强烈。"
            )
        elif prob < 0.4:
            action = "看空"
            score = max(-1.0, -(0.5 - prob) * 2)
            confidence = min(0.95, 1 - prob)
            reasoning = (
                f"LSTM 模型预测次日涨停概率仅 {prob * 100:.1f}%，"
                f"短线无突破信号；当前形态偏弱。"
            )
        else:
            action = "中性"
            score = (prob - 0.5) * 2
            confidence = 0.4
            reasoning = (
                f"LSTM 预测次日涨停概率 {prob * 100:.1f}%，"
                f"处于中性区间，无明确短线方向。"
            )

        return AgentOpinion(
            agent_name=self.name, symbol=symbol,
            action=action, score=round(score, 3),
            confidence=round(confidence, 3),
            reasoning=reasoning,
            key_facts=[
                f"次日涨停概率（LSTM）= {prob * 100:.2f}%",
                "特征：20 日窗口 × 16 维（K 线形态 + 均线偏离 + 波动率 + 动量 + 偏度）",
                "模型：双向 LSTM（128 隐藏 × 2 层）+ 全连接分类头",
            ],
            risks=[
                "LSTM 训练数据截至 2025-05；行情结构剧变时模型可能失效",
                "短线信号有效期通常 1-2 个交易日，不宜外推",
            ],
            metadata={
                "model": result.get("model", "LSTM-LimitUp-v1"),
                "probability": prob,
                "role": self.role,
            },
        )


# ---------------------------------------------------------------------------
# 工厂：一次性创建标准分析师团
# ---------------------------------------------------------------------------


def build_analyst_team(include_event_predictor: bool = True) -> list[Agent]:
    """返回标准的分析师团。

    - 默认 **6 位**：5 位 LLM 分析师 + 1 位 LSTM 量化 Agent
    - ``include_event_predictor=False`` 时退化为 5 位 LLM 分析师团

    形成完整的「技术 / 资金 / 消息 / 风控 / 估值 / 量化事件」六维独立判断矩阵。
    """
    team: list[Agent] = [
        TechnicalAnalyst(),
        FundFlowAnalyst(),
        NewsAnalyst(),
        RiskAnalyst(),
        ValuationAnalyst(),
    ]
    if include_event_predictor:
        team.append(EventPredictionAgent())
    return team
