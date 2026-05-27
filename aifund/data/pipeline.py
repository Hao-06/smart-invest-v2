"""统一数据管道。

对外提供「时点正确」（point-in-time）的市场快照：给定标的与某个交易日 `as_of`，
返回该日**收盘后可获得**的全部数据，所有序列都已严格截断到 `as_of`，不含未来
信息。这是回测不产生前视偏差、决策可复现的根本保证。

Agent 与回测引擎只与本模块打交道，不直接调用 `sources`。
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from aifund.data import sources
from aifund.data.calendar import TradeCalendar, as_date
from aifund.data.models import MarketSnapshot, StockData


class DataPipeline:
    """市场数据的统一入口。"""

    def __init__(self, lookback_days: int = 250) -> None:
        """
        Args:
            lookback_days: 每只标的回看的交易日根数（行情窗口长度）。
        """
        self.lookback_days = lookback_days
        self._calendar: TradeCalendar | None = None

    @property
    def calendar(self) -> TradeCalendar:
        """交易日历（懒加载）。"""
        if self._calendar is None:
            self._calendar = TradeCalendar()
        return self._calendar

    # ------------------------------------------------------------------
    # 单只标的
    # ------------------------------------------------------------------
    def get_stock_data(
        self,
        symbol: str,
        as_of: str | date,
        asset_type: str = "stock",
        name: str | None = None,
    ) -> StockData:
        """构造单只标的截至 `as_of` 的时点数据视图。"""
        as_of = as_date(as_of)
        symbol = sources.normalize_symbol(symbol)
        # 日历天窗口预留余量，覆盖足够的交易日（含停牌/节假日）
        start = as_of - timedelta(days=int(self.lookback_days * 1.7) + 30)

        # -- 行情 --
        price = sources.get_price_history(symbol, start, as_of, asset_type=asset_type)
        price_src = price.attrs.get("source", "cache") if not price.empty else "none"
        if not price.empty:
            price = price[price["date"] <= as_of].tail(self.lookback_days)
            price = price.reset_index(drop=True)

        # -- 资金流（仅个股）--
        fund_flow = pd.DataFrame()
        flow_src = "n/a"
        if asset_type == "stock":
            fund_flow = sources.get_fund_flow(symbol)
            flow_src = fund_flow.attrs.get("source", "cache") if not fund_flow.empty else "none"
            if not fund_flow.empty:
                fund_flow = fund_flow[fund_flow["date"] <= as_of].reset_index(drop=True)

        # -- 新闻（仅个股）--
        news = pd.DataFrame()
        news_src = "n/a"
        if asset_type == "stock":
            news = sources.get_news(symbol)
            news_src = news.attrs.get("source", "cache") if not news.empty else "none"
            if not news.empty and "publish_time" in news.columns:
                # 按 as_of 截断，回测时不泄露未来新闻
                mask = news["publish_time"].dt.date <= as_of
                news = news[mask].reset_index(drop=True)

        # -- 基本信息（仅个股）：名称 / 行业 / 流通市值等 --
        info: dict[str, object] = {}
        valuation: dict[str, object] = {}
        if asset_type == "stock":
            info = sources.get_stock_info(symbol)
            valuation = sources.get_valuation_metrics(symbol)

        if name is None:
            # 优先用个股信息里的简称；失败时回退为代码本身。
            # 不再调用 sources.get_stock_name —— 那会重复触发同一个失败接口。
            cand = str(info.get("股票简称") or "").strip() if info else ""
            name = cand or symbol

        return StockData(
            symbol=symbol,
            name=name,
            as_of=as_of,
            asset_type=asset_type,
            price_history=price,
            fund_flow=fund_flow,
            news=news,
            info=info,
            valuation=valuation,
            provenance={
                "price_history": price_src,
                "fund_flow": flow_src,
                "news": news_src,
                "valuation": "lg" if valuation else "none",
            },
        )

    # ------------------------------------------------------------------
    # 全市场快照
    # ------------------------------------------------------------------
    #: 全局 light 模式开关 —— 设为 True 时 snapshot **只拉行情**，跳过资金流/新闻/估值/北向
    #: 用于 All Weather / 纯量化回测 —— 这些场景 strategy 函数只需要 last_close
    light_mode: bool = False

    def snapshot(
        self,
        symbols: list[str],
        as_of: str | date,
        asset_types: dict[str, str] | None = None,
    ) -> MarketSnapshot:
        """构造一次决策所需的全市场快照。

        Args:
            symbols: 候选标的代码列表。
            as_of: 决策基准交易日。
            asset_types: 可选，{代码: "stock"|"etf"}，未指定默认 "stock"。
        """
        as_of = as_date(as_of)
        asset_types = asset_types or {}
        snap = MarketSnapshot(as_of=as_of)

        for sym in symbols:
            sym_n = sources.normalize_symbol(sym)
            atype = asset_types.get(sym, asset_types.get(sym_n, "stock"))
            if self.light_mode:
                # 只拉行情，秒级
                snap.add(self._light_stock_data(sym_n, as_of, atype))
            else:
                snap.add(self.get_stock_data(sym_n, as_of, asset_type=atype))

        # -- 北向资金（仅在非 light 模式拉）--
        if not self.light_mode:
            hsgt = sources.get_hsgt_flow()
            if not hsgt.empty and "date" in hsgt.columns:
                hsgt = hsgt[hsgt["date"] <= as_of].reset_index(drop=True)
            snap.hsgt_flow = hsgt

        return snap

    def _light_stock_data(self, symbol: str, as_of: date, asset_type: str = "stock"):
        """轻量版 stock data：**仅行情**，跳过资金流/新闻/估值。"""
        import pandas as pd
        from aifund.data.models import StockData

        start = as_of - timedelta(days=int(self.lookback_days * 1.7) + 30)
        price = sources.get_price_history(symbol, start, as_of, asset_type=asset_type)
        if not price.empty:
            price = price[price["date"] <= as_of].tail(self.lookback_days).reset_index(drop=True)
        return StockData(
            symbol=symbol, name=symbol, as_of=as_of, asset_type=asset_type,
            price_history=price,
            fund_flow=pd.DataFrame(), news=pd.DataFrame(),
            info={}, valuation={},
            provenance={"price_history": "light", "fund_flow": "skipped",
                        "news": "skipped", "valuation": "skipped"},
        )
