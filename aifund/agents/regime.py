"""**Market Regime Agent** —— 实时识别市场状态（趋势 / 震荡 / 熊市 / 暴涨）。

这是 Meta-Strategy 系统的「**眼睛**」—— 先看清楚当前是什么市场，再决定用什么策略。

输入：
- 沪深 300 最近 60 天行情（用于计算 ADX / 趋势 / 波动率）
- 行业 ETF 涨跌幅分布（用于判断板块分化度）
- 大盘资金流（北向 / 主力净流入）—— 可选

输出：
- regime: 5 种标签 之一
  * trending_bull   —— 趋势上行（牛市主升）
  * trending_bear   —— 趋势下行（熊市）
  * range_bound     —— 震荡市（无明确方向）
  * breakout        —— 突破启动（如 9.24 单日暴涨）
  * volatile        —— 高波动无序（极端情况）
- confidence: 0.0 ~ 1.0
- reasoning: R1 思维链推理过程
- key_signals: 决策依据的指标值

设计要点：
- 用 **DeepSeek-R1**（强推理 + 思维链）—— 给评委看 AI 是怎么诊断市场的
- 先用规则算出关键指标（ADX / 涨幅 / 波动率），再让 R1 综合判断
- 失败时降级到「unknown」regime（让 Selector Agent 选最稳健策略）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from aifund.data import sources
from aifund.llm import get_llm_client, LLMClient
from aifund.stockpool.momentum_selector import _calculate_adx
from aifund.strategies.sector_rotation import SECTOR_ETFS

HS300_ETF = "510300"

#: 5 种 regime 标签 + 描述（给 LLM 看）
REGIME_DEFINITIONS = {
    "trending_bull": "趋势上行：大盘均线多头排列，连续创新高，资金持续流入",
    "trending_bear": "趋势下行：大盘均线空头排列，连续下跌，资金持续流出",
    "range_bound": "震荡市：大盘在 ±5% 区间反复，无明确方向，无主线",
    "breakout": "突破启动：单日或几日暴涨 / 暴跌（如 9.24 类）启动新趋势",
    "volatile": "高波动无序：日内振幅极大，资金恐慌或狂热",
}


@dataclass
class RegimeOpinion:
    """市场状态识别结果 —— 输出**概率分布**而非单一标签（科学性体现）。

    `regime_probs` 是 5 种 regime 的概率分布，sum=1.0。
    `regime` (primary_regime) 是概率最高的那个；`confidence` 是其概率。
    """
    as_of: date
    regime: str                            # primary regime（最高概率）
    confidence: float                      # primary regime 的概率
    regime_probs: dict[str, float] = field(default_factory=dict)  # 全部 5 种概率
    reasoning: str = ""
    key_signals: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class MarketRegimeAgent:
    """识别市场状态的 Agent。"""

    name = "市场状态识别 Agent"
    role = "deep"  # 用 R1

    def __init__(self, llm: LLMClient | None = None, *,
                 offline: bool = False) -> None:
        # offline=True（回测模式）：全程走规则判断，不构造也不调用 LLM。
        # 历史回测里调 LLM 既是「事后诸葛」（模型知道 as_of 之后发生的事），
        # 又不可复现 —— 回测必须确定性。LLM 推理仅用于实盘（复赛）。
        self.offline = offline
        self.llm: LLMClient | None = (
            None if offline else (llm or get_llm_client(role=self.role))
        )

    # ------------------------------------------------------------------
    # 信号计算（规则部分，给 R1 当 input）
    # ------------------------------------------------------------------
    def _compute_signals(self, as_of: date) -> dict[str, Any]:
        """计算市场状态相关的关键指标。"""
        signals: dict[str, Any] = {"as_of": str(as_of)}

        # 1. 沪深 300 趋势 + 强度
        try:
            df = sources.get_price_history(
                HS300_ETF, as_of - timedelta(days=90), as_of, asset_type="etf",
            )
            if df is None or len(df) < 30:
                signals["error"] = "沪深300 数据不足"
                return signals
            df = df.sort_values("date").reset_index(drop=True)
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            # 趋势：5MA vs 10MA vs 20MA
            ma3 = float(close.tail(3).mean())
            ma10 = float(close.tail(10).mean())
            ma20 = float(close.tail(20).mean())
            ma60 = float(close.tail(60).mean()) if len(close) >= 60 else ma20

            # ADX 趋势强度
            adx = _calculate_adx(high, low, close, period=14)

            # 涨幅
            ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 6 else 0
            ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
            ret_60d = float(close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) > 61 else 0

            # 波动率（年化）
            log_returns = np.log(close / close.shift(1)).dropna().tail(20)
            volatility = float(log_returns.std() * np.sqrt(252)) * 100 if len(log_returns) > 5 else 0

            # 最大单日涨跌幅（看是否暴涨/暴跌）
            daily_returns = (close.pct_change().dropna().tail(20) * 100)
            max_daily = float(daily_returns.max()) if len(daily_returns) else 0
            min_daily = float(daily_returns.min()) if len(daily_returns) else 0

            signals.update({
                "hs300_close": round(float(close.iloc[-1]), 4),
                "ma3_over_ma10": round(ma3 / ma10, 4),
                "ma10_over_ma20": round(ma10 / ma20, 4),
                "ma20_over_ma60": round(ma20 / ma60, 4),
                "adx": round(adx, 2),
                "ret_5d_pct": round(ret_5d, 2),
                "ret_20d_pct": round(ret_20d, 2),
                "ret_60d_pct": round(ret_60d, 2),
                "annualized_volatility_pct": round(volatility, 2),
                "max_daily_pct_recent20": round(max_daily, 2),
                "min_daily_pct_recent20": round(min_daily, 2),
            })
        except Exception as exc:
            signals["hs300_error"] = f"{type(exc).__name__}: {exc}"

        # 2. 行业 ETF 涨跌幅分化度
        try:
            sector_rets = []
            for etf_code in list(SECTOR_ETFS)[:10]:  # 取 10 个代表性 ETF
                try:
                    edf = sources.get_price_history(
                        etf_code, as_of - timedelta(days=35), as_of, asset_type="etf",
                    )
                    if edf is None or len(edf) < 21:
                        continue
                    eclose = edf.sort_values("date")["close"].astype(float)
                    r = float(eclose.iloc[-1] / eclose.iloc[-21] - 1) * 100
                    sector_rets.append(r)
                except Exception:
                    continue
            if len(sector_rets) >= 5:
                signals["sector_dispersion_pct"] = round(float(np.std(sector_rets)), 2)
                signals["sector_max_pct"] = round(max(sector_rets), 2)
                signals["sector_min_pct"] = round(min(sector_rets), 2)
                signals["sector_count_positive"] = sum(1 for r in sector_rets if r > 0)
                signals["sector_count_total"] = len(sector_rets)
        except Exception as exc:
            signals["sector_error"] = f"{type(exc).__name__}: {exc}"

        # 3. **北向资金净流入**（聪明钱信号 —— 通常领先大盘 1-2 周）
        try:
            hsgt = sources.get_hsgt_flow()
            if hsgt is not None and not hsgt.empty and "date" in hsgt.columns:
                # 截到 as_of 之前的数据（防前视偏差）
                mask = hsgt["date"] <= as_of
                sub = hsgt.loc[mask].sort_values("date").tail(30)
                if len(sub) >= 5:
                    flow_col = "net_inflow" if "net_inflow" in sub.columns else "net_buy"
                    if flow_col in sub.columns:
                        recent_5d = float(sub[flow_col].tail(5).sum())
                        recent_20d = float(sub[flow_col].tail(20).sum())
                        # 单位转换（接口给的是「亿元」，做容错）
                        signals["north_flow_5d_yi"] = round(recent_5d, 1)
                        signals["north_flow_20d_yi"] = round(recent_20d, 1)
                        # 流入信号强度
                        if recent_5d > 100:
                            signals["north_flow_signal"] = "强流入(看多)"
                        elif recent_5d > 30:
                            signals["north_flow_signal"] = "温和流入"
                        elif recent_5d < -100:
                            signals["north_flow_signal"] = "强流出(看空)"
                        elif recent_5d < -30:
                            signals["north_flow_signal"] = "温和流出"
                        else:
                            signals["north_flow_signal"] = "中性"
        except Exception as exc:
            signals["north_flow_error"] = f"{type(exc).__name__}: {exc}"

        return signals

    # ------------------------------------------------------------------
    # LLM 推理（R1 思维链）
    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        return """你是一名资深的 A股市场状态识别专家。基于给定的技术指标和市场数据，**输出 5 种市场状态的概率分布**（科学性体现：承认不确定性）。

5 种市场状态定义：
- **trending_bull**：趋势上行（牛市主升） —— 均线多头排列，ADX > 25，连续创新高
- **trending_bear**：趋势下行（熊市） —— 均线空头排列，ADX > 25 但方向向下
- **range_bound**：震荡市 —— 均线粘合，ADX < 20，无明确方向
- **breakout**：突破启动 —— 单日或几日暴涨/暴跌（5 日涨幅 > 5% 或单日 > 3%）
- **volatile**：高波动无序 —— 年化波动率 > 30%，单日振幅极大

输出严格 JSON：
{
  "regime_probs": {
    "trending_bull": 0.X,
    "trending_bear": 0.X,
    "range_bound": 0.X,
    "breakout": 0.X,
    "volatile": 0.X
  },
  "reasoning": "判断的简明依据（100-200 字）"
}

**重要原则**：
1. 概率合必须 = 1.0
2. **不要给出 0.95+ 的「绝对确定」概率** —— 市场永远有不确定性
3. 典型分布：明确趋势市 primary 概率 0.5-0.7；不明确时 0.3-0.4
4. ADX > 25 + 均线明确方向：高 trending_bull 或 trending_bear
5. ADX < 20 + 板块分化大：高 range_bound + volatile
6. 5 日涨跌幅 > 5%：高 breakout
7. 多个信号矛盾时：分布更均匀（每个 0.15-0.25）"""

    def _user_prompt(self, signals: dict[str, Any]) -> str:
        return f"""请分析当前 A股市场状态。

**沪深 300 关键指标**（{signals.get('as_of', 'N/A')}）：
- 短期/中期/长期均线比值：3MA/10MA={signals.get('ma3_over_ma10', 'N/A')}，10MA/20MA={signals.get('ma10_over_ma20', 'N/A')}，20MA/60MA={signals.get('ma20_over_ma60', 'N/A')}
- ADX 趋势强度：{signals.get('adx', 'N/A')}
- 涨幅：5日 {signals.get('ret_5d_pct', 'N/A')}% / 20日 {signals.get('ret_20d_pct', 'N/A')}% / 60日 {signals.get('ret_60d_pct', 'N/A')}%
- 年化波动率：{signals.get('annualized_volatility_pct', 'N/A')}%
- 近 20 日单日最大涨幅：{signals.get('max_daily_pct_recent20', 'N/A')}%
- 近 20 日单日最大跌幅：{signals.get('min_daily_pct_recent20', 'N/A')}%

**行业 ETF 分布**（10 个主流 ETF 过去 20 日涨跌幅）：
- 标准差：{signals.get('sector_dispersion_pct', 'N/A')}%
- 最强 +{signals.get('sector_max_pct', 'N/A')}% / 最弱 {signals.get('sector_min_pct', 'N/A')}%
- 上涨家数：{signals.get('sector_count_positive', 'N/A')} / {signals.get('sector_count_total', 'N/A')}

**北向资金净流入**（聪明钱信号，通常领先大盘 1-2 周）：
- 近 5 日累计：{signals.get('north_flow_5d_yi', 'N/A')} 亿元
- 近 20 日累计：{signals.get('north_flow_20d_yi', 'N/A')} 亿元
- 信号：**{signals.get('north_flow_signal', 'N/A')}**

请综合判断 regime。**特别关注北向资金信号** —— 强流入是看多重要佐证，强流出是看空预警。
请输出 regime 判断的 JSON。"""

    def analyze(self, as_of: date) -> RegimeOpinion:
        """识别 as_of 日的市场状态。"""
        signals = self._compute_signals(as_of)
        # 如果信号本身就失败了，直接返回 unknown
        if "error" in signals or "hs300_error" in signals:
            return RegimeOpinion(
                as_of=as_of, regime="range_bound", confidence=0.3,
                reasoning=f"数据获取失败 ({signals.get('error') or signals.get('hs300_error')})，降级到震荡市保守判断",
                key_signals=signals,
            )

        # offline 回测模式：跳过 LLM，直接用规则判断（确定性、可复现、无未来信息泄漏）
        if self.offline:
            return self._rule_based_fallback(
                as_of, signals, RuntimeError("offline 回测模式"))

        try:
            response = self.llm.chat(
                system=self._system_prompt(),
                user=self._user_prompt(signals),
                json_mode=True,
            )
            data = response.parse_json()
            probs = data.get("regime_probs", {})
            # 归一化（保险）
            valid_regimes = ["trending_bull", "trending_bear", "range_bound",
                             "breakout", "volatile"]
            probs = {r: float(probs.get(r, 0.0)) for r in valid_regimes}
            s = sum(probs.values())
            if s > 0:
                probs = {r: v / s for r, v in probs.items()}
            else:
                probs = {r: 0.2 for r in valid_regimes}
            primary = max(probs, key=probs.get)
            return RegimeOpinion(
                as_of=as_of,
                regime=primary,
                confidence=probs[primary],
                regime_probs=probs,
                reasoning=(response.reasoning_content or "")
                          + "\n\n[结论] " + data.get("reasoning", ""),
                key_signals=signals,
                raw_response=data,
            )
        except Exception as exc:
            # LLM 失败：用规则降级判断
            return self._rule_based_fallback(as_of, signals, exc)

    def _rule_based_fallback(self, as_of: date, signals: dict[str, Any],
                              exc: Exception) -> RegimeOpinion:
        """LLM 失败时用规则判断（保底逻辑）—— 同样输出概率分布。"""
        adx = signals.get("adx", 20)
        ret_5d = signals.get("ret_5d_pct", 0)
        ma_ratio = signals.get("ma10_over_ma20", 1.0)
        vol = signals.get("annualized_volatility_pct", 20)

        # 规则映射：每个 regime 都分配一些概率（避免极端值）
        probs = {
            "trending_bull": 0.1, "trending_bear": 0.1,
            "range_bound": 0.4,  # 默认偏震荡
            "breakout": 0.1, "volatile": 0.3,
        }
        if abs(ret_5d) > 5:
            probs = {"breakout": 0.55, "trending_bull": 0.15,
                     "trending_bear": 0.15, "range_bound": 0.1, "volatile": 0.05}
        elif vol > 30:
            probs = {"volatile": 0.5, "range_bound": 0.25, "trending_bear": 0.15,
                     "trending_bull": 0.05, "breakout": 0.05}
        elif adx > 25 and ma_ratio > 1.005:
            probs = {"trending_bull": 0.55, "breakout": 0.2, "range_bound": 0.15,
                     "trending_bear": 0.05, "volatile": 0.05}
        elif adx > 25 and ma_ratio < 0.995:
            probs = {"trending_bear": 0.55, "range_bound": 0.2, "volatile": 0.15,
                     "trending_bull": 0.05, "breakout": 0.05}
        primary = max(probs, key=probs.get)
        return RegimeOpinion(
            as_of=as_of, regime=primary, confidence=probs[primary],
            regime_probs=probs,
            reasoning=f"规则判断（{exc}）：ADX={adx}, 5日涨幅={ret_5d}%, "
                      f"均线比值={ma_ratio}, 波动率={vol}% → 概率分布",
            key_signals=signals,
        )
