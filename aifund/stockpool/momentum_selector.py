"""**动量 + 大盘择时** 选股器（学术验证 50 年的 alpha 因子）。

放弃单纯多因子估值打分（在牛市跑输大盘的根本原因），改用经典动量策略：

- **第 1 层 · 横截面动量**（Jegadeesh & Titman 1993）：选近 20 日涨幅 top-N
- **第 2 层 · 行业分散 + 反转过滤**：同行业最多 2 只，剔除短期涨过头（>15% / 5 日）
- **第 3 层 · 大盘择时**（Trend Following）：沪深300 5MA > 20MA 时满仓，否则空仓

设计理由：
- 横截面动量是金融实证里最稳健的 alpha 之一，**学术界与业界 50 年验证**
- 牛市里动量延续效应最强；熊市里大盘择时避免硬扛回撤
- 不用 LLM、不用 CSV 因子，**完全实时数据驱动** —— 可在任意时间段跑

参考文献：
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers.
- Faber, M. T. (2007). A quantitative approach to tactical asset allocation.
"""
from __future__ import annotations

import bisect
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """计算最新一日的 ADX（Average Directional Index，**趋势强度指标**）。

    Wilder 1978 经典定义：
    - ADX > 25 → 趋势市（无论上涨下跌方向都明确）
    - ADX 20-25 → 弱趋势（过渡区）
    - ADX < 20 → 震荡市（无方向，应避免趋势追随）

    返回最末日 ADX 值；数据不足时返回 25.0（中性，不阻止任何决策）。
    """
    if len(close) < period * 2 + 1:
        return 25.0
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    # 方向移动：+DM / -DM
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder 平滑（EMA with alpha=1/period）
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan))
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else 25.0

from aifund.data import sources
from aifund.data.calendar import TradeCalendar
from aifund.data.pipeline import DataPipeline
from aifund.stockpool.factor_loader import FactorLoader

#: 沪深300 ETF 代码（择时用）
HS300_ETF = "510300"


class MomentumSelector:
    """动量 + 大盘择时 选股器。"""

    def __init__(
        self,
        pipeline: DataPipeline,
        factor_csv_path: Path | str | None = None,
        universe: list[str] | None = None,
        momentum_window: int = 20,
        max_per_industry: int = 2,
        max_recent_gain_pct: float = 15.0,
        top_pool: int = 8,
        trend_short_ma: int = 3,   # 经多区间扫描验证 3/10 综合最优
        trend_long_ma: int = 10,
        enable_timing: bool = True,
        use_realtime_momentum: bool = False,
        universe_top_n: int | None = None,
        use_lstm_filter: bool = False,
        lstm_weight: float = 0.4,
    ) -> None:
        """
        Args:
            pipeline: 数据管道（共享缓存）。
            factor_csv_path: 可选；用于从 CSV 提取沪深300 成分股 + 行业信息。
            universe: 显式指定标的池；若 None 则从 CSV 提取沪深300 成分股。
            momentum_window: 动量窗口（默认 20 个交易日）。
            max_per_industry: 同行业最多保留股数。
            max_recent_gain_pct: 近 5 日涨幅超过此值的剔除（防止追高）。
            top_pool: 第 2 层最终输出股数。
            trend_short_ma: 大盘择时短期均线（默认 5）。
            trend_long_ma: 大盘择时长期均线（默认 20）。
            enable_timing: 是否开启大盘择时。
            use_realtime_momentum: True 时不读 CSV 的 Factor_MOM_60，而是从实时行情
                自己算近 ``momentum_window`` 日动量。**用于 CSV 覆盖范围之外的时段**
                （如 2026 年），可跑「真正样本外」的 forward test。
            universe_top_n: 限制 universe 大小（按 CSV 最后一日 score 降序取前 N）。
                配合 ``use_realtime_momentum=True`` 用，避免拉全 293 只标的的行情。
        """
        self.pipeline = pipeline
        self.momentum_window = momentum_window
        self.max_per_industry = max_per_industry
        self.max_recent_gain_pct = max_recent_gain_pct
        self.top_pool = top_pool
        self.trend_short_ma = trend_short_ma
        self.trend_long_ma = trend_long_ma
        self.enable_timing = enable_timing
        self.use_realtime_momentum = use_realtime_momentum
        self.use_lstm_filter = use_lstm_filter
        self.lstm_weight = float(lstm_weight)
        self._lstm_predictor = None  # 懒加载

        # 加载多因子 CSV（包含 Factor_MOM_60 历史动量因子）
        self.loader: FactorLoader | None = None
        # 用 `is not None` 而非真值判断：显式传入空列表 [] 表示「该区间无可选个股」，
        # 必须原样保留为空 universe —— 不能掉进下面那段非 PIT 的 CSV 回退。
        if universe is not None:
            self.universe: list[str] = list(universe)
        else:
            self.loader = FactorLoader(factor_csv_path)
            df = self.loader._df[["symbol", "简称"]].drop_duplicates("symbol")
            self.universe = df["symbol"].tolist()
            # 缩小 universe（按 score 取前 N）。
            # ⚠️ 这里用 `complete_days.max()`（CSV 最新一日）——**仅适用于「实盘当下」场景**
            # （universe 未显式指定时）。回测必须由调用方传入 PIT universe，绝不能走这条路，
            # 否则会用未来因子分数挑历史股票（look-ahead bias）。
            if universe_top_n and universe_top_n < len(self.universe):
                day_counts = (
                    self.loader._df.dropna(subset=["score"])
                    .groupby("日期").size().sort_index()
                )
                complete_days = day_counts[day_counts > 100].index
                if len(complete_days):
                    ref_day = complete_days.max()
                    ref_df = self.loader._df[self.loader._df["日期"] == ref_day]
                    top = ref_df.nlargest(universe_top_n, "score")["symbol"].tolist()
                    self.universe = top

    # ------------------------------------------------------------------
    # 大盘择时（双层决策：方向 hysteresis + 强度 ADX）
    # ------------------------------------------------------------------
    def market_position_target(self, as_of: date) -> float:
        """返回当前应使用的**仓位比例**：1.0 满仓 / 0.5 半仓 / 0.0 空仓。

        **双层决策架构**（解决「震荡市被甩耳光 vs 突变市反应慢」+ 「震荡市追涨杀跌」的两难）：

        第一层：**方向判断**（不对称 hysteresis）
        - 短 MA > 长 MA × 1.001 → **趋势上**
        - 短 MA < 长 MA × 0.995 → **趋势下**
        - 中间胶着区 → 沿用上次方向

        第二层：**强度判断**（ADX Average Directional Index）
        - ADX > 25 → 强趋势 → 满仓
        - ADX < 20 → 震荡市 → **半仓（保留 50% 现金等趋势启动）**
        - 20-25 过渡区 → 沿用上次仓位

        综合决策：
        - 趋势下 → 0.0 空仓（无论 ADX）
        - 趋势上 + 强趋势 → 1.0 满仓
        - 趋势上 + 震荡 → 0.5 半仓
        - 趋势上 + 过渡区 → 沿用上次

        数据不足或失败时默认 1.0（不阻止建仓）。
        """
        if not self.enable_timing:
            self.last_position_ratio = 1.0
            return 1.0
        try:
            df = sources.get_price_history(
                HS300_ETF, as_of - timedelta(days=90), as_of, asset_type="etf"
            )
            if df is None or len(df) < self.trend_long_ma + 5:
                self.last_position_ratio = 1.0
                return 1.0
            close = df["close"].astype(float)
            high = df["high"].astype(float) if "high" in df.columns else close
            low = df["low"].astype(float) if "low" in df.columns else close
            short_ma = float(close.tail(self.trend_short_ma).mean())
            long_ma = float(close.tail(self.trend_long_ma).mean())

            # ── 第一层：方向（不对称 hysteresis）
            ratio = short_ma / long_ma if long_ma > 0 else 1.0
            ENTER_LONG = 1.001  # 0.1% 就建仓（快速捕捉突变）
            EXIT_LONG = 0.995   # 必须 -0.5% 才空仓（避免震荡甩出）
            prev_trend_up = getattr(self, "_last_trend_up", True)
            if ratio > ENTER_LONG:
                self._last_trend_up = True
            elif ratio < EXIT_LONG:
                self._last_trend_up = False
            trend_up = getattr(self, "_last_trend_up", True)

            # 趋势下 → 空仓（ADX 不参与决策）+ 累计空仓周数
            if not trend_up:
                self._bear_streak = getattr(self, "_bear_streak", 0) + 1
                self.last_position_ratio = 0.0
                return 0.0

            # ── 关键修复 v7：仅当**连续空仓 ≥ 2 周后**翻转上行才强制满仓
            #   这区分「反弹真翻转」（长熊后启动，9.24/Q1 类）和「震荡假翻转」（频繁反复，2026 类）
            #   - 长熊后翻转 → 强制满仓捕捉「新鲜动量」
            #   - 震荡翻转 → 不奖励，按 ADX 走（震荡市自然半仓）
            bear_streak = getattr(self, "_bear_streak", 0)
            just_flipped_up = trend_up and not prev_trend_up
            self._bear_streak = 0   # 翻转上后归零计数
            if just_flipped_up and bear_streak >= 2:
                self.last_position_ratio = 1.0
                return 1.0

            # ── 第二层：强度（ADX）—— 仅在 trend 稳定后才用 ADX 调节仓位
            adx_value = _calculate_adx(high, low, close, period=14)
            ADX_TREND = 25.0    # ADX > 25 = 强趋势
            ADX_RANGE = 20.0    # ADX < 20 = 震荡
            if adx_value >= ADX_TREND:
                ratio_out = 1.0   # 强趋势 → 满仓
            elif adx_value < ADX_RANGE:
                ratio_out = 0.5   # 震荡市 → 半仓
            else:
                # 过渡区：沿用上次仓位（init 时默认 1.0）
                ratio_out = getattr(self, "last_position_ratio", 1.0)
                if ratio_out == 0.0:
                    ratio_out = 0.5
            self.last_position_ratio = ratio_out
            return ratio_out
        except Exception:
            self.last_position_ratio = 1.0
            return 1.0

    def market_trend_is_up(self, as_of: date) -> bool:
        """向后兼容：仓位 > 0 即视为「up」（外部 select 是否进场用这个判定）。"""
        return self.market_position_target(as_of) > 0.0

    # ------------------------------------------------------------------
    # 第 1 层：横截面动量（CSV 预算 or 实时计算 两种模式）
    # ------------------------------------------------------------------
    def momentum_filter(self, as_of: date, top: int = 20) -> list[dict]:
        """选 60 日动量 top-N 股票。

        两种模式：
        - **CSV 模式**（默认）：从 Hao 多因子 CSV 里读 ``Factor_MOM_60``，秒级
        - **实时模式** (``use_realtime_momentum=True``)：从实时行情自己算 60 日涨幅，
          支持 CSV 覆盖范围之外的时段（如 2026 年）

        Returns:
            含 ``symbol``、``momentum_pct``、``recent_5d_pct`` 的字典列表，
            按 momentum_pct 降序。
        """
        if self.use_realtime_momentum:
            return self._momentum_filter_realtime(as_of, top)
        return self._momentum_filter_csv(as_of, top)

    def _momentum_filter_csv(self, as_of: date, top: int = 20) -> list[dict]:
        """从 CSV 的 Factor_MOM_60 字段读取（秒级）。"""
        if not self.loader:
            return []
        factor_df = self.loader.factors_by_date(as_of)
        if factor_df.empty:
            return []
        sub = factor_df[["symbol", "Factor_MOM_60", "Factor_REV_5"]].dropna()
        sub = sub.rename(columns={
            "Factor_MOM_60": "momentum_pct",
            "Factor_REV_5": "recent_5d_pct",
        })
        sub["momentum_pct"] = sub["momentum_pct"] * 100
        sub["recent_5d_pct"] = sub["recent_5d_pct"] * 100
        sub = sub.sort_values("momentum_pct", ascending=False)
        return sub.head(top).to_dict("records")

    def _momentum_filter_realtime(self, as_of: date, top: int = 20) -> list[dict]:
        """从实时行情**并行**计算 60 日动量 + 5 日涨幅。

        早期串行循环 80 只 ×  100-200ms = 12s+；改为 ThreadPoolExecutor 并行后
        实测降到 1-2s，看板「当日推荐」体感大幅提升。
        """
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock

        rows: list[dict] = []
        lock = Lock()

        def _fetch(symbol: str) -> None:
            try:
                # 拉 130 个日历日（≈ 85+ 交易日），保证有足够数据算 60 日动量
                df = sources.get_price_history(
                    symbol, as_of - timedelta(days=130), as_of
                )
            except Exception:
                return
            if df is None or len(df) < 65:
                return
            df = df.sort_values("date").reset_index(drop=True)
            close = df["close"].astype(float)
            try:
                mom_pct = float(close.iloc[-1] / close.iloc[-61] - 1) * 100
                recent_5d = float(close.iloc[-1] / close.iloc[-6] - 1) * 100
            except (IndexError, ZeroDivisionError, ValueError):
                return
            row = {"symbol": symbol, "momentum_pct": mom_pct, "recent_5d_pct": recent_5d}
            with lock:
                rows.append(row)

        # 16 线程并发；I/O 密集（parquet/网络），不会撞 GIL
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(_fetch, self.universe))

        if not rows:
            return []
        df_m = pd.DataFrame(rows).sort_values("momentum_pct", ascending=False)
        return df_m.head(top).to_dict("records")

    # ------------------------------------------------------------------
    # 第 2 层（可选）：LSTM 涨停概率重排序
    # ------------------------------------------------------------------
    def _get_lstm_predictor(self):
        """懒加载 LSTM 预测器（首次使用时加载，避免 import 成本）。"""
        if self._lstm_predictor is not None:
            return self._lstm_predictor
        try:
            from aifund.ml.lstm_predictor import LimitUpPredictor
            self._lstm_predictor = LimitUpPredictor()
            return self._lstm_predictor
        except Exception as e:
            print(f"[momentum] LSTM 预测器加载失败：{e}", flush=True)
            return None

    def lstm_filter(self, candidates: list[dict], as_of: date) -> list[dict]:
        """用 LSTM 涨停概率对候选**加权重排**。

        综合分 = (1 - w) × 标准化动量分 + w × LSTM 涨停概率分

        - 标准化动量分：把 momentum_pct 映射到 [0, 1]（min-max）
        - LSTM 概率分：直接使用 [0, 1]
        - 权重 w = ``self.lstm_weight``（默认 0.4）

        无法计算 LSTM 的票（数据不足 / 早期 / 停牌）保留原排序权重 0。
        在候选列表上**就地**加入 ``lstm_prob`` / ``combined_score`` 字段。
        """
        predictor = self._get_lstm_predictor()
        if predictor is None or not candidates:
            return candidates

        # 1. 拉行情，跑 LSTM 推理
        for c in candidates:
            try:
                df = sources.get_price_history(
                    c["symbol"], as_of - timedelta(days=80), as_of
                )
                result = predictor.predict(df)
                c["lstm_prob"] = (
                    float(result["limit_up_probability"])
                    if result.get("available") else 0.0
                )
            except Exception:
                c["lstm_prob"] = 0.0

        # 2. 标准化动量分到 [0, 1]
        moms = [c["momentum_pct"] for c in candidates]
        m_min, m_max = min(moms), max(moms)
        m_range = max(m_max - m_min, 1e-6)
        # 3. 综合打分
        w = self.lstm_weight
        for c in candidates:
            mom_norm = (c["momentum_pct"] - m_min) / m_range  # [0, 1]
            c["combined_score"] = (1 - w) * mom_norm + w * c["lstm_prob"]
        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 第 3 层：反转过滤 + 选 top-K
    # ------------------------------------------------------------------
    def reversal_filter(self, candidates: list[dict]) -> list[str]:
        """剔除近 5 日涨幅过高的（防追高）+ 取 top-K。"""
        survivors = [
            c for c in candidates if c["recent_5d_pct"] <= self.max_recent_gain_pct
        ]
        return [c["symbol"] for c in survivors[: self.top_pool]]

    # ------------------------------------------------------------------
    # 一站式
    # ------------------------------------------------------------------
    def select(self, as_of: date, verbose: bool = False) -> list[str]:
        """对 as_of 日运行：动量 → (可选) LSTM 涵盖 → 反转过滤 → 大盘择时。"""
        if self.enable_timing and not self.market_trend_is_up(as_of):
            if verbose:
                print(f"[momentum] {as_of} 大盘趋势向下，空仓避险")
            return []

        momentum = self.momentum_filter(as_of, top=self.top_pool * 2 + 4)
        if not momentum:
            return []

        # 可选：LSTM 涨停概率加权重排
        if self.use_lstm_filter:
            momentum = self.lstm_filter(momentum, as_of)

        survivors = self.reversal_filter(momentum)

        if verbose:
            print(f"[momentum] {as_of} 选出 {len(survivors)} 只:")
            for c in momentum[: len(survivors)]:
                if c["symbol"] in survivors:
                    extra = (f"  LSTM={c.get('lstm_prob', 0)*100:.1f}%"
                             if self.use_lstm_filter else "")
                    print(f"  {c['symbol']}  20日动量={c['momentum_pct']:+.2f}%  "
                          f"近5日={c['recent_5d_pct']:+.2f}%{extra}")
        return survivors

    # ------------------------------------------------------------------
    # 周频预计算（供回测使用）
    # ------------------------------------------------------------------
    def precompute_weekly_pools(
        self,
        start_date: date,
        end_date: date,
        calendar: TradeCalendar | None = None,
        verbose: bool = True,
    ) -> dict[date, list[str]]:
        """预计算 [start, end] 区间内每周首个交易日的候选池。

        同时把每周的**目标仓位比例**（满仓 1.0 / 半仓 0.5 / 空仓 0.0）缓存到
        ``self.weekly_position_ratios``（``dict[date, float]``），供 strategy 函数
        在调仓时控制 cash deployment。
        """
        cal = calendar or self.pipeline.calendar
        all_dates = cal.range(start_date, end_date)
        if not all_dates:
            self.weekly_position_ratios = {}
            return {}

        pools: dict[date, list[str]] = {}
        ratios: dict[date, float] = {}
        last_week_key: tuple[int, int] | None = None
        for d in all_dates:
            year, week, _ = d.isocalendar()
            week_key = (year, week)
            if week_key != last_week_key:
                # 先取仓位比例（也会更新内部 hysteresis 状态）
                pos_ratio = self.market_position_target(d) if self.enable_timing else 1.0
                ratios[d] = pos_ratio
                if verbose:
                    ratio_note = (
                        "🟢 满仓" if pos_ratio >= 0.99
                        else ("🟡 半仓震荡" if pos_ratio > 0 else "⚫ 空仓避险")
                    )
                    print(f"[momentum] 预计算 {d} ({ratio_note} ratio={pos_ratio:.2f})…", flush=True)
                pools[d] = self.select(d, verbose=False) if pos_ratio > 0 else []
                if verbose:
                    print(f"  → {pools[d]}", flush=True)
                last_week_key = week_key
        self.weekly_position_ratios = ratios
        return pools
