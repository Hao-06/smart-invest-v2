"""数据快照缓存（Parquet）。

竞赛命题明确要求结果「可审计、可复现」。数据层抓到的所有行情数据都落盘为本地
快照，后续回测与复盘直接读快照 —— 保证同一份数据可重复实验、不受行情源后续
变动影响，也避免重复请求触发限流。

数据层统一以 pandas.DataFrame 流转，缓存即以 Parquet 列式格式存储：类型保真、
体积小、跨工具可读。缓存键 = 命名空间 + 参数哈希。
- 历史数据（已收盘交易日的行情）视为永不过期；
- 当日实时类数据可在读取时传入较短 TTL。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import settings


class SnapshotCache:
    """基于 Parquet 文件的 DataFrame 快照缓存。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.paths.data_cache
        self.root.mkdir(parents=True, exist_ok=True)

    # -- 键与路径 ----------------------------------------------------------
    @staticmethod
    def _key(namespace: str, params: dict[str, Any]) -> str:
        raw = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"{namespace}__{digest}"

    def _path(self, namespace: str, params: dict[str, Any]) -> Path:
        return self.root / f"{self._key(namespace, params)}.parquet"

    # -- 读写 --------------------------------------------------------------
    def get(
        self,
        namespace: str,
        params: dict[str, Any],
        ttl: float | None = None,
    ) -> pd.DataFrame | None:
        """读取快照。

        Args:
            namespace: 数据类别（如 "price_history"）。
            params: 请求参数，参与键计算。
            ttl: 有效期（秒）；None 表示永不过期。
        Returns:
            命中且未过期则返回 DataFrame，否则 None。
        """
        path = self._path(namespace, params)
        if not path.exists():
            return None
        if ttl is not None and (time.time() - path.stat().st_mtime) > ttl:
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            # 缓存损坏时静默失效，让上层重新抓取
            return None

    def put(
        self,
        namespace: str,
        params: dict[str, Any],
        df: pd.DataFrame | None,
    ) -> None:
        """写入快照。df 为 None 或空时不写入。"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return
        path = self._path(namespace, params)
        tmp = path.with_suffix(".tmp")
        try:
            df.to_parquet(tmp, index=False)
            tmp.replace(path)  # 原子替换，避免半截文件
        except Exception:
            tmp.unlink(missing_ok=True)

    def clear(self, namespace: str | None = None) -> int:
        """清空缓存。指定 namespace 时只清该类别。返回删除的文件数。"""
        pattern = f"{namespace}__*.parquet" if namespace else "*.parquet"
        count = 0
        for p in self.root.glob(pattern):
            p.unlink(missing_ok=True)
            count += 1
        return count


# 全局单例
cache = SnapshotCache()
