"""回测引擎。

把数据管道、决策函数与组合模拟器串起来，按交易日推进时间，记录净值与成交。

**时间约定（严格防前视偏差）**：
- 在交易日 T 的「开盘前」，用截至 T-1 收盘后的 ``MarketSnapshot`` 调用决策函数；
- 订单按 T 日**开盘价**成交（最贴近真实可执行价）；
- 净值在 T 日**收盘价**估值并记入 ``equity_curve``。

决策函数签名：
    ``decide_fn(snapshot: MarketSnapshot, portfolio: Portfolio) -> list[Order | dict]``

支持的订单格式（adapter 自动归一化）：
- ``Order`` 对象
- ``{"symbol", "side", "shares", "name"?, "reason"?}``
- ``{"symbol", "volume"}`` —— 正数 = 买入，负数 = 卖出，单位「股」（须 100 的倍数）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from aifund.backtest.metrics import PerformanceMetrics, compute_performance
from aifund.backtest.portfolio import Order, Portfolio, Side, Trade
from aifund.data import sources
from aifund.data.calendar import as_date
from aifund.data.models import MarketSnapshot
from aifund.data.pipeline import DataPipeline

DecideFn = Callable[[MarketSnapshot, Portfolio], list]


# ---------------------------------------------------------------------------
# 订单归一化
# ---------------------------------------------------------------------------


def _coerce_order(item: object) -> Order | None:
    """把决策函数返回的各种形态统一成 Order；无法识别返回 None。"""
    if isinstance(item, Order):
        return item
    if not isinstance(item, dict):
        return None

    symbol = str(item.get("symbol") or item.get("代码") or "").strip()
    if not symbol:
        return None
    name = str(item.get("symbol_name") or item.get("name") or item.get("名称") or "")
    reason = str(item.get("reason") or item.get("理由") or "")

    if "side" in item and "shares" in item:
        side = str(item["side"]).upper()
        shares = int(item["shares"])
        if side not in ("BUY", "SELL") or shares <= 0:
            return None
        return Order(symbol=symbol, side=side, shares=shares, name=name, reason=reason)

    if "volume" in item:
        vol = int(item["volume"])
        if vol == 0:
            return None
        side: Side = "BUY" if vol > 0 else "SELL"
        return Order(symbol=symbol, side=side, shares=abs(vol), name=name, reason=reason)

    return None


# ---------------------------------------------------------------------------
# 回测结果
# ---------------------------------------------------------------------------


@dataclass
class DailyLog:
    """单日审计日志。"""

    date: date
    decision_as_of: date | None
    equity: float
    cash: float
    orders_requested: int
    orders_filled: int
    rejections: list[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    start_date: date
    end_date: date
    initial_capital: float
    portfolio: Portfolio
    metrics: PerformanceMetrics
    daily_logs: list[DailyLog]

    def print_report(self) -> None:
        """打印回测报告（绩效摘要 + 末日持仓）。"""
        print("=" * 60)
        print(f"回测区间 {self.start_date} ~ {self.end_date}")
        print("-" * 60)
        for k, v in self.metrics.to_dict().items():
            print(f"  {k:<12} {v}")
        print("-" * 60)
        print("期末持仓：")
        snap = self.portfolio.snapshot()
        if snap["positions"]:
            for pos in snap["positions"]:  # type: ignore[union-attr]
                print(
                    f"  {pos['symbol']} {pos['name']:<10} {pos['shares']:>6} 股 "
                    f"@ {pos['last_price']:>9.4f}  市值 {pos['market_value']:>12,.2f}  "
                    f"权重 {pos['weight']:>5.2f}%  浮盈 {pos['unrealized_pnl']:>+10.2f}"
                )
        else:
            print("  （空仓）")
        print(f"  现金 {snap['cash']:>14,.2f}    期末权益 {snap['equity']:,.2f}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# 回测引擎
# ---------------------------------------------------------------------------


class BacktestEngine:
    """日频回测引擎。"""

    def __init__(
        self,
        pipeline: DataPipeline,
        candidate_symbols: list[str],
        decide_fn: DecideFn,
        initial_capital: float | None = None,
        asset_types: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            pipeline: 已构造的 DataPipeline 实例（共享缓存与日历）。
            candidate_symbols: 候选标的代码池（决策函数从中选择交易对象）。
            decide_fn: 决策函数，按本模块文档的签名实现。
            initial_capital: 初始资金，None 时取 settings 默认（50 万）。
            asset_types: 可选，{代码: "stock"|"etf"}，未指定默认 "stock"。
        """
        self.pipeline = pipeline
        self.candidate_symbols = [sources.normalize_symbol(s) for s in candidate_symbols]
        self.decide_fn = decide_fn
        self.asset_types = asset_types or {}
        self.portfolio = Portfolio(initial_capital=initial_capital)

    # ------------------------------------------------------------------
    # 行情预加载
    # ------------------------------------------------------------------
    def _preload_prices(self, start: date, end: date) -> dict[str, pd.DataFrame]:
        """一次性预拉取所有候选标的在回测区间的行情，以日期为索引。

        **并发 + 硬性总超时**：用 daemon Thread + 共享 dict 并发拉取。
        - 每只标的内部已有 30s 硬超时（aifund.data.sources）
        - 整体加 ``_PRELOAD_TOTAL_TIMEOUT`` 秒兜底（默认 180s）
        - 超时后还没下载完的标的**直接跳过**，回测仍能跑（数据缺失的票不能被买）
        """
        import threading

        _PRELOAD_TOTAL_TIMEOUT = 180.0  # 整体最多等 3 分钟

        prices: dict[str, pd.DataFrame] = {}
        prices_lock = threading.Lock()

        def _fetch_one(sym):
            try:
                atype = self.asset_types.get(sym, "stock")
                df = sources.get_price_history(sym, start, end, asset_type=atype)
                if df is not None and not df.empty:
                    df = df.copy()
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    with prices_lock:
                        prices[sym] = df.set_index("date")
            except Exception:
                pass

        threads = []
        for sym in self.candidate_symbols:
            th = threading.Thread(target=_fetch_one, args=(sym,), daemon=True)
            th.start()
            threads.append(th)

        # 用一个全局 deadline 等所有线程；超时不等了
        import time
        deadline = time.time() + _PRELOAD_TOTAL_TIMEOUT
        for th in threads:
            remaining = max(0.1, deadline - time.time())
            th.join(timeout=remaining)
        # 不强制 kill 没结束的线程，它们是 daemon，主程序退出时自然死

        with prices_lock:
            return dict(prices)

    @staticmethod
    def _today_bar(prices: dict[str, pd.DataFrame], symbol: str, day: date) -> dict | None:
        """取某标的某日的 OHLC；不存在（停牌/无数据）返回 None。"""
        df = prices.get(symbol)
        if df is None or day not in df.index:
            return None
        row = df.loc[day]
        try:
            open_, close = float(row["open"]), float(row["close"])
        except (TypeError, ValueError):
            return None
        if not (open_ > 0 and close > 0):
            return None
        return {
            "open": open_,
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": close,
        }

    # ------------------------------------------------------------------
    # 单日执行
    # ------------------------------------------------------------------
    def _execute_order(
        self,
        order: Order,
        day: date,
        prices: dict[str, pd.DataFrame],
        log: DailyLog,
    ) -> bool:
        """按 day 当日开盘价模拟成交一笔订单。返回是否成交。"""
        bar = self._today_bar(prices, order.symbol, day)
        if bar is None:
            log.rejections.append(f"{order.symbol} 当日无行情（停牌/无数据），跳过")
            return False
        # 100 股整数倍合规：非法订单直接拒绝
        if order.shares <= 0 or order.shares % self.portfolio.lot_size != 0:
            log.rejections.append(
                f"{order.symbol} {order.side} {order.shares}股 不是 {self.portfolio.lot_size} 倍数"
            )
            return False
        try:
            if order.side == "BUY":
                self.portfolio.buy(order.symbol, order.shares, bar["open"], day, name=order.name)
            else:
                self.portfolio.sell(order.symbol, order.shares, bar["open"], day)
            return True
        except ValueError as exc:
            log.rejections.append(f"{order.symbol} {order.side} {order.shares}股 失败：{exc}")
            return False

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(
        self,
        start_date: str | date,
        end_date: str | date,
        verbose: bool = True,
    ) -> BacktestResult:
        """在 [start_date, end_date] 区间运行回测。

        Args:
            start_date / end_date: 回测起止（自然日，自动对齐到交易日）。
            verbose: 是否逐日打印进度。

        Returns:
            BacktestResult，含组合终态、绩效指标与逐日日志。
        """
        start_date, end_date = as_date(start_date), as_date(end_date)
        cal = self.pipeline.calendar
        dates = cal.range(start_date, end_date)
        if not dates:
            raise ValueError(f"区间 [{start_date}, {end_date}] 内无交易日")

        # 预拉取行情（多取几日以确保第一日有「前一交易日」数据）
        first = cal.prev(dates[0]) or dates[0]
        prices = self._preload_prices(first, dates[-1])

        logs: list[DailyLog] = []
        if verbose:
            print(
                f"[backtest] 区间 {dates[0]} ~ {dates[-1]}（{len(dates)} 交易日），"
                f"候选 {len(self.candidate_symbols)} 只，初始资金 {self.portfolio.initial_capital:,.0f}"
            )

        for idx, today in enumerate(dates):
            prev_day = cal.prev(today)
            log = DailyLog(
                date=today,
                decision_as_of=prev_day,
                equity=0.0,
                cash=0.0,
                orders_requested=0,
                orders_filled=0,
            )

            if prev_day is None:
                # 第一日无历史，仅做估值，不交易
                close_prices = {s: bar["close"] for s in self.candidate_symbols
                                if (bar := self._today_bar(prices, s, today))}
                log.equity = self.portfolio.mark_to_market(close_prices, today)
                log.cash = self.portfolio.cash
                logs.append(log)
                continue

            # 1) 构造决策快照（截至 prev_day）
            snapshot = self.pipeline.snapshot(
                self.candidate_symbols, prev_day, asset_types=self.asset_types
            )

            # 2) 调用决策函数（异常视为空操作，记录日志）
            try:
                raw_orders = self.decide_fn(snapshot, self.portfolio) or []
            except Exception as exc:  # noqa: BLE001
                log.rejections.append(f"决策函数异常：{type(exc).__name__}: {exc}")
                raw_orders = []

            orders = [o for o in (_coerce_order(it) for it in raw_orders) if o is not None]
            log.orders_requested = len(orders)

            # 3) T 日开盘价成交（卖出优先，先释放现金再买入）
            for order in sorted(orders, key=lambda o: 0 if o.side == "SELL" else 1):
                if self._execute_order(order, today, prices, log):
                    log.orders_filled += 1

            # 4) T 日收盘价估值
            close_prices = {s: bar["close"] for s in self.candidate_symbols
                            if (bar := self._today_bar(prices, s, today))}
            log.equity = self.portfolio.mark_to_market(close_prices, today)
            log.cash = self.portfolio.cash

            logs.append(log)
            if verbose:
                ret_pct = (log.equity / self.portfolio.initial_capital - 1) * 100
                tag = "" if not log.rejections else f"  ⚠ {len(log.rejections)} 项被拒"
                print(
                    f"  [{today}] 决策基于 {prev_day} | 订单 {log.orders_filled}/{log.orders_requested}"
                    f" | 现金 {log.cash:>12,.0f} | 权益 {log.equity:>12,.0f}"
                    f" ({ret_pct:+.2f}%){tag}"
                )

        metrics = compute_performance(
            self.portfolio.equity_curve, self.portfolio.trades, self.portfolio.initial_capital
        )
        return BacktestResult(
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=self.portfolio.initial_capital,
            portfolio=self.portfolio,
            metrics=metrics,
            daily_logs=logs,
        )
