"""技术指标计算。

基于规范化的日线行情 DataFrame（列见 `data.sources.PRICE_COLUMNS`）计算常用
技术指标。所有计算函数为纯函数，不修改入参。

`summarize()` 把最新一期的指标汇总成对 LLM 友好的字典 —— 这是技术面分析师
Agent 的标准输入：既给出原始数值，也给出「多头排列 / MACD 金叉」这类已归纳的
信号词，降低 LLM 误读数字的风险。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 基础指标（纯函数）
# ---------------------------------------------------------------------------


def sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均。"""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均。"""
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指标 RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD：返回 (DIF, DEA, 柱状值)。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = (dif - dea) * 2
    return dif, dea, hist


def bollinger(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """布林带：返回 (上轨, 中轨, 下轨)。"""
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    return mid + n_std * std, mid, mid - n_std * std


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真实波幅 ATR。入参需含 high/low/close 列。"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def annualized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """年化波动率（按 252 个交易日）。"""
    returns = close.pct_change()
    return returns.rolling(window, min_periods=window).std() * np.sqrt(252)


def rolling_max_drawdown(close: pd.Series, window: int = 60) -> float:
    """近 window 日的最大回撤（负值）。"""
    seg = close.tail(window)
    if seg.empty:
        return float("nan")
    running_max = seg.cummax()
    drawdown = seg / running_max - 1.0
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# 形态学特征（参考 LSTM 涨停板预测项目的特征工程）
# ---------------------------------------------------------------------------


def candle_pattern(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """K 线形态学特征：跳空 / 上下影线 / 实体占比 / 收盘位置 / 日内振幅。

    这些是事件驱动型短线交易里识别启动信号的关键特征：
    - ``gap`` > 1% 高开 + 大实体 = 强势启动
    - ``upper_shadow`` 长 + 小实体 = 上方抛压
    - ``close_position`` 接近 1 = 当日强势收盘
    """
    df = df.copy()
    # 「前一日收盘」用 close.shift(1) 近似（无 preclose 列时的兜底）
    prev_close = df["close"].shift(1)
    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    full_range = (df["high"] - df["low"]).replace(0, np.nan)

    df["gap"] = (df["open"] - prev_close) / prev_close
    df["intraday_range"] = (df["high"] - df["low"]) / df["open"]
    df["close_position"] = (df["close"] - df["open"]) / full_range
    df["upper_shadow"] = (df["high"] - body_high) / df["open"]
    df["lower_shadow"] = (body_low - df["low"]) / df["open"]
    df["body_ratio"] = (df["close"] - df["open"]).abs() / full_range
    return df


def return_skewness(close: pd.Series, window: int = 5) -> pd.Series:
    """近 window 日收益率偏度。

    正偏 = 偶尔大涨拉高均值（典型上涨形态）；
    负偏 = 偶尔大跌（典型下跌或破位形态）。
    """
    returns = close.pct_change()
    return returns.rolling(window, min_periods=window).skew()


# ---------------------------------------------------------------------------
# 指标汇总
# ---------------------------------------------------------------------------


def compute(price_df: pd.DataFrame) -> pd.DataFrame:
    """在行情 DataFrame 上追加指标列，返回新 DataFrame。"""
    df = price_df.copy()
    close = df["close"]
    df["ma5"] = sma(close, 5)
    df["ma10"] = sma(close, 10)
    df["ma20"] = sma(close, 20)
    df["ma60"] = sma(close, 60)
    df["rsi14"] = rsi(close, 14)
    df["macd_dif"], df["macd_dea"], df["macd_hist"] = macd(close)
    df["boll_upper"], df["boll_mid"], df["boll_lower"] = bollinger(close)
    df["atr14"] = atr(df, 14)
    df["volatility20"] = annualized_volatility(close, 20)
    df["vol_ma5"] = sma(df["volume"], 5)
    # 形态学特征（参考 LSTM 涨停板预测项目）
    df = candle_pattern(df)
    df["return_skew_5d"] = return_skewness(close, 5)
    return df


def _round(value: object, ndigits: int = 4) -> float | None:
    """安全地四舍五入，NaN / None 转为 None。"""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return round(f, ndigits)


def _ma_trend(ma5: float, ma10: float, ma20: float) -> str:
    """根据短中期均线相对位置判断排列形态。"""
    if any(v is None or np.isnan(v) for v in (ma5, ma10, ma20)):
        return "数据不足"
    if ma5 > ma10 > ma20:
        return "多头排列"
    if ma5 < ma10 < ma20:
        return "空头排列"
    return "均线纠缠"


def summarize(price_df: pd.DataFrame, min_bars: int = 20) -> dict[str, object]:
    """把最新一期技术指标汇总为对 LLM 友好的字典。

    Returns:
        含 available 标志的字典。数据不足时 available=False。
    """
    if price_df is None or len(price_df) < min_bars:
        return {"available": False, "reason": f"行情不足 {min_bars} 根", "bars": 0 if price_df is None else len(price_df)}

    df = compute(price_df)
    last = df.iloc[-1]
    close = float(last["close"])

    # 收益率
    ret = {}
    for label, n in (("1d", 1), ("5d", 5), ("20d", 20)):
        if len(df) > n:
            prev = float(df.iloc[-1 - n]["close"])
            ret[label] = round((close / prev - 1) * 100, 2) if prev else None
        else:
            ret[label] = None

    # MACD 状态
    dif, dea, hist = last["macd_dif"], last["macd_dea"], last["macd_hist"]
    prev_hist = df.iloc[-2]["macd_hist"] if len(df) > 1 else np.nan
    if not np.isnan(hist) and not np.isnan(prev_hist):
        if prev_hist <= 0 < hist:
            macd_state = "金叉（柱状由负转正）"
        elif prev_hist >= 0 > hist:
            macd_state = "死叉（柱状由正转负）"
        elif hist > 0:
            macd_state = "多头区间"
        else:
            macd_state = "空头区间"
    else:
        macd_state = "数据不足"

    # 布林带位置
    bu, bl = last["boll_upper"], last["boll_lower"]
    if not np.isnan(bu) and not np.isnan(bl) and bu > bl:
        boll_pos = round((close - bl) / (bu - bl), 2)  # 0=下轨, 1=上轨
    else:
        boll_pos = None

    # 价格在近 60 日区间的分位
    window60 = df["close"].tail(60)
    lo, hi = float(window60.min()), float(window60.max())
    price_pos_60 = round((close - lo) / (hi - lo), 2) if hi > lo else None

    # 量能比：当日量 / 5 日均量
    vol, vol_ma5 = last["volume"], last["vol_ma5"]
    if not np.isnan(vol) and not np.isnan(vol_ma5) and vol_ma5 > 0:
        volume_ratio = round(float(vol) / float(vol_ma5), 2)
    else:
        volume_ratio = None  # 备用行情源可能缺成交量

    ma5, ma10, ma20 = _round(last["ma5"]), _round(last["ma10"]), _round(last["ma20"])

    # 形态学（百分比形式，便于 LLM 理解）
    candle = {
        "gap_pct": _round(last.get("gap", np.nan) * 100, 2)
            if not np.isnan(last.get("gap", np.nan)) else None,
        "intraday_range_pct": _round(last.get("intraday_range", np.nan) * 100, 2),
        "close_position": _round(last.get("close_position", np.nan), 2),
        "upper_shadow_pct": _round(last.get("upper_shadow", np.nan) * 100, 2),
        "lower_shadow_pct": _round(last.get("lower_shadow", np.nan) * 100, 2),
        "body_ratio": _round(last.get("body_ratio", np.nan), 2),
        "pattern": _candle_pattern_label(last),
    }

    as_of = last["date"]
    return {
        "available": True,
        "as_of": as_of.isoformat() if isinstance(as_of, date) else str(as_of),
        "bars": len(df),
        "close": round(close, 4),
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": _round(last["ma60"]),
        "ma_trend": _ma_trend(ma5, ma10, ma20),
        "rsi14": _round(last["rsi14"], 2),
        "macd": {
            "dif": _round(dif),
            "dea": _round(dea),
            "hist": _round(hist),
            "state": macd_state,
        },
        "bollinger": {
            "upper": _round(bu),
            "mid": _round(last["boll_mid"]),
            "lower": _round(bl),
            "position": boll_pos,  # 0=贴下轨, 1=贴上轨
        },
        "returns_pct": ret,
        "return_skew_5d": _round(last.get("return_skew_5d", np.nan)),
        "volatility_annualized_pct": _round(
            None if np.isnan(last["volatility20"]) else last["volatility20"] * 100, 2
        ),
        "atr14": _round(last["atr14"]),
        "max_drawdown_60d_pct": _round(rolling_max_drawdown(df["close"], 60) * 100, 2),
        "price_position_60d": price_pos_60,  # 0=60日最低, 1=60日最高
        "volume_ratio": volume_ratio,  # 当日量/5日均量；None=数据源未提供成交量
        "candle": candle,  # 形态学：跳空 / 影线 / 实体占比 / 形态标签
    }


def _candle_pattern_label(row: pd.Series) -> str:
    """根据形态学指标输出语义标签（启动 / 滞涨 / 长上影 / 长下影 / 十字星 等）。"""
    try:
        gap = float(row.get("gap", np.nan))
        body_ratio = float(row.get("body_ratio", np.nan))
        upper = float(row.get("upper_shadow", np.nan))
        lower = float(row.get("lower_shadow", np.nan))
        close_pos = float(row.get("close_position", np.nan))
    except (TypeError, ValueError):
        return "数据不足"
    if any(np.isnan(v) for v in (body_ratio, upper, lower)):
        return "数据不足"
    # 十字星：实体极小
    if body_ratio < 0.15:
        return "十字星（多空胶着）"
    # 长上影：上影线 / 振幅 > 50%（强压力）
    full = max(upper + lower + body_ratio, 1e-6)
    if upper / full > 0.45:
        return "长上影（高位抛压）"
    if lower / full > 0.45:
        return "长下影（下方有支撑）"
    # 强势启动：高开 + 实体大 + 收盘接近高位
    if not np.isnan(gap) and gap > 0.01 and body_ratio > 0.5 and close_pos > 0.5:
        return "高开放量（启动信号）"
    if close_pos > 0.5:
        return "收盘强势"
    if close_pos < -0.5:
        return "收盘弱势"
    return "普通中性"
