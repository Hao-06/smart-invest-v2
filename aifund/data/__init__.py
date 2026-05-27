"""A股多源数据管道。

多源容灾的数据抓取 + 快照缓存 + 时点正确的市场快照。
导入本包即自动绕过本机代理（见 `_net`）。

对外主要接口：
    DataPipeline     —— 统一数据入口，产出 MarketSnapshot
    TradeCalendar    —— 交易日历工具
    MarketSnapshot / StockData —— 数据模型
    sources          —— 底层多源数据函数（一般无需直接调用）
"""
from aifund.data import _net  # noqa: F401  导入即绕过代理
from aifund.data import sources
from aifund.data.calendar import TradeCalendar, as_date
from aifund.data.cache import cache
from aifund.data.models import MarketSnapshot, StockData
from aifund.data.pipeline import DataPipeline

__all__ = [
    "DataPipeline",
    "TradeCalendar",
    "as_date",
    "MarketSnapshot",
    "StockData",
    "sources",
    "cache",
]
