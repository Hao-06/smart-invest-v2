"""数据层数据模型。

数据管道的统一输出。Agent 与回测引擎只依赖这些模型，不直接接触 AkShare。

所有模型都是「时点正确」（point-in-time）的：构造时已截断到 `as_of` 交易日，
不含任何未来数据 —— 这是回测不产生前视偏差（look-ahead bias）的关键保证，
也是命题「结果可审计、可复现」要求的基础。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class StockData:
    """单只标的截至某交易日的完整数据视图。"""

    symbol: str  # 6 位代码
    name: str  # 证券简称
    as_of: date  # 数据截止日（含当日）
    asset_type: str = "stock"  # stock / etf
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)  # 日线行情
    fund_flow: pd.DataFrame = field(default_factory=pd.DataFrame)  # 个股资金流历史
    news: pd.DataFrame = field(default_factory=pd.DataFrame)  # 近期新闻
    info: dict[str, object] = field(default_factory=dict)  # 基本信息：行业/市值/上市时间等
    valuation: dict[str, object] = field(default_factory=dict)  # 估值面：PE/PB/PS + 历史分位
    #: 各数据项的来源记录，用于审计（如 {"price_history": "tencent"}）
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def industry(self) -> str:
        """所属行业，缺失时返回空串。"""
        return str(self.info.get("行业", "") or "")

    @property
    def float_market_cap(self) -> float | None:
        """流通市值（元），缺失时返回 None。"""
        val = self.info.get("流通市值")
        try:
            return float(val) if val not in (None, "", "-") else None
        except (TypeError, ValueError):
            return None

    @property
    def has_price(self) -> bool:
        return not self.price_history.empty

    @property
    def last_close(self) -> float | None:
        """最新收盘价。"""
        if self.price_history.empty:
            return None
        return float(self.price_history.iloc[-1]["close"])

    @property
    def bars(self) -> int:
        """可用的日线根数。"""
        return len(self.price_history)

    def recent_prices(self, n: int) -> pd.DataFrame:
        """最近 n 个交易日的行情。"""
        return self.price_history.tail(n).reset_index(drop=True)


@dataclass
class MarketSnapshot:
    """某交易日的全市场决策快照：一次决策所需的全部输入。"""

    as_of: date
    stocks: dict[str, StockData] = field(default_factory=dict)
    hsgt_flow: pd.DataFrame = field(default_factory=pd.DataFrame)  # 北向资金历史
    index_history: pd.DataFrame = field(default_factory=pd.DataFrame)  # 大盘指数

    def get(self, symbol: str) -> StockData | None:
        return self.stocks.get(symbol)

    def add(self, stock: StockData) -> None:
        self.stocks[stock.symbol] = stock

    @property
    def symbols(self) -> list[str]:
        return list(self.stocks)

    @property
    def tradable_symbols(self) -> list[str]:
        """有有效行情、可参与决策的标的。"""
        return [s for s, d in self.stocks.items() if d.has_price]
