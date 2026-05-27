"""LSTM 涨停板预测器。

基于 Hao 之前训练好的双向 LSTM 模型（``lstm_limit_up_best.pt``），输入近 20 个
交易日的 16 维技术形态特征，输出**次日涨停的概率**。

这是给「事件预测 Agent」用的核心推理模块 —— 让 multi-agent 系统不仅有 LLM 推理，
还有定量机器学习预测，形成「LLM × 量化」双轨决策的差异化。

设计要点：
- **特征工程**：完全复刻训练时的 16 个特征，确保推理一致性
- **归一化**：复用训练时 z-score 的 mean/std（保存在 .npz）
- **延迟加载**：模型 + 权重在首次 predict 调用时加载，import 本模块零成本
- **优雅降级**：数据不足、含 NaN、模型加载失败时返回 ``available=False``
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 与训练时严格一致
LOOKBACK = 20
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
FEATURE_COLS = [
    "daily_return", "intraday_range", "close_position", "gap",
    "upper_shadow", "lower_shadow", "body_ratio",
    "ma5_bias", "ma10_bias", "ma20_bias",
    "vol_5d", "vol_10d", "price_position_20",
    "momentum_5d", "momentum_10d", "return_skew_5d",
]

_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
_MODEL_PATH = _CHECKPOINT_DIR / "lstm_limit_up_best.pt"
_NORM_PATH = _CHECKPOINT_DIR / "lstm_norm_params.npz"


# ---------------------------------------------------------------------------
# 模型定义（必须与训练时完全一致，否则权重加载会出错）
# ---------------------------------------------------------------------------


class _LSTMLimitUpPredictor(nn.Module):
    """双向 LSTM + 全连接分类头。结构与训练代码 1:1 复刻。"""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 训练时输入形状为 (batch, features, seq_len)，需要 permute 为 (batch, seq, features)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last).squeeze(-1)


# ---------------------------------------------------------------------------
# 特征工程（与训练严格对齐）
# ---------------------------------------------------------------------------


def engineer_lstm_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """从日线 OHLC 计算 16 维 LSTM 特征。

    与训练代码（lstm_limit_up.py）的 ``load_and_feature_engineer`` 完全一致。
    入参需含 ``open / high / low / close`` 列；``preclose`` 若缺则按 ``close.shift(1)`` 补。
    """
    df = price_df.copy()
    eps = 1e-8
    if "preclose" not in df.columns:
        df["preclose"] = df["close"].shift(1)

    # 价格类
    df["daily_return"] = (df["close"] - df["preclose"]) / df["preclose"]
    df["intraday_range"] = (df["high"] - df["low"]) / df["open"]
    df["close_position"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + eps)
    df["gap"] = (df["open"] - df["preclose"]) / df["preclose"]

    # K 线形态
    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    df["upper_shadow"] = (df["high"] - body_high) / df["open"]
    df["lower_shadow"] = (body_low - df["low"]) / df["open"]
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + eps)

    # 均线偏离
    for w in (5, 10, 20):
        ma = df["close"].rolling(w, min_periods=w).mean()
        df[f"ma{w}_bias"] = (df["close"] - ma) / ma

    # 波动率
    for w in (5, 10):
        df[f"vol_{w}d"] = df["daily_return"].rolling(w, min_periods=w).std()

    # 20 日位置
    low_20 = df["low"].rolling(20, min_periods=20).min()
    high_20 = df["high"].rolling(20, min_periods=20).max()
    df["price_position_20"] = (df["close"] - low_20) / (high_20 - low_20 + eps)

    # 动量
    for w in (5, 10):
        df[f"momentum_{w}d"] = df["close"] / df["close"].shift(w) - 1

    # 收益率偏度
    df["return_skew_5d"] = df["daily_return"].rolling(5, min_periods=5).skew()

    return df


# ---------------------------------------------------------------------------
# 推理器（延迟加载 + 优雅降级）
# ---------------------------------------------------------------------------


class LimitUpPredictor:
    """LSTM 涨停板预测器。

    使用方式：
        predictor = LimitUpPredictor()
        result = predictor.predict(price_df)  # 首次调用时加载模型
        if result["available"]:
            prob = result["limit_up_probability"]
    """

    def __init__(self) -> None:
        self.model: _LSTMLimitUpPredictor | None = None
        self.norm_mean: np.ndarray | None = None
        self.norm_std: np.ndarray | None = None
        self._load_error: str | None = None
        self._loaded: bool = False

    def _lazy_load(self) -> bool:
        """首次调用时加载模型与归一化参数。失败时记录原因并返回 False。"""
        if self._loaded:
            return self.model is not None
        self._loaded = True

        if not _MODEL_PATH.exists() or not _NORM_PATH.exists():
            self._load_error = f"模型文件缺失：{_MODEL_PATH.name} / {_NORM_PATH.name}"
            return False

        try:
            params = np.load(str(_NORM_PATH))
            self.norm_mean = params["mean"].astype(np.float32)
            self.norm_std = params["std"].astype(np.float32) + 1e-8
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"归一化参数加载失败：{type(exc).__name__}: {exc}"
            return False

        try:
            model = _LSTMLimitUpPredictor(input_size=len(FEATURE_COLS))
            state = torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            # PyTorch 推理模式（关闭 dropout/batchnorm 的训练行为）
            model.train(False)
            self.model = model
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"模型权重加载失败：{type(exc).__name__}: {exc}"
            return False
        return True

    def predict(self, price_df: pd.DataFrame) -> dict[str, Any]:
        """根据日线行情 DataFrame 预测**次日涨停概率**。

        返回字典：
        - ``available``: bool —— 是否成功预测
        - ``limit_up_probability``: float ∈ [0,1]（available=True 时）
        - ``reason``: str —— available=False 时的失败原因
        """
        if price_df is None or len(price_df) < LOOKBACK + 25:
            return {
                "available": False,
                "reason": f"行情根数不足，需 ≥ {LOOKBACK + 25} 根",
            }
        if not self._lazy_load():
            return {"available": False, "reason": self._load_error or "未知错误"}

        # 计算特征
        feats = engineer_lstm_features(price_df)
        recent = feats.tail(LOOKBACK).reset_index(drop=True)

        # 完整性校验
        if len(recent) < LOOKBACK:
            return {"available": False, "reason": "特征窗口长度不足"}
        try:
            x = recent[FEATURE_COLS].to_numpy(dtype=np.float32)
        except KeyError as exc:
            return {"available": False, "reason": f"特征缺失：{exc}"}
        if np.isnan(x).any() or np.isinf(x).any():
            return {"available": False, "reason": "特征含 NaN/Inf（行情早期或停牌段）"}

        # 归一化
        assert self.norm_mean is not None and self.norm_std is not None
        x = (x - self.norm_mean) / self.norm_std

        # 推理：形状要求 (batch, features, seq_len) = (1, 16, 20)
        assert self.model is not None
        x_tensor = torch.from_numpy(x.T[None, :, :])
        with torch.no_grad():
            logits = self.model(x_tensor)
            prob = float(torch.sigmoid(logits).item())

        return {
            "available": True,
            "limit_up_probability": prob,
            "model": "LSTM-LimitUp-v1",
            "as_of": str(price_df["date"].iloc[-1]) if "date" in price_df.columns else "",
        }
