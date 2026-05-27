"""账户与组合状态。

严格模拟 A 股交易约束 —— 这是回测可信度的基础：
- **T+1 制度**：当日买入次日才能卖出。按 lot 记录买入日期，FIFO 卖出时
  排除掉 ``date >= as_of`` 的批次。
- **100 股整数倍**：每笔买/卖股数必须是 ``settings.backtest.lot_size`` 倍数。
- **手续费**：佣金双边按金额 × ``commission_rate``，单笔最低 ``min_commission`` 元；
  印花税仅卖出单边，按金额 × ``stamp_tax_rate``。
- **现金约束**：买入需现金充足（含手续费）。
- **已实现盈亏**：卖出时按 FIFO 削减 lot，记录到 ``Trade.realized_pnl``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from config.settings import settings

Side = Literal["BUY", "SELL"]


# ---------------------------------------------------------------------------
# 订单 & 成交记录
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """决策函数输出的交易订单。"""

    symbol: str
    side: Side
    shares: int  # 正数；方向由 side 决定
    name: str = ""
    reason: str = ""  # 可选，用于审计与决策报告


@dataclass
class Trade:
    """已成交记录。"""

    date: date
    symbol: str
    name: str
    side: Side
    shares: int
    price: float
    amount: float  # shares * price
    fee: float
    tax: float
    cash_after: float
    realized_pnl: float = 0.0  # 仅卖出时记录


# ---------------------------------------------------------------------------
# 持仓
# ---------------------------------------------------------------------------


@dataclass
class LotPurchase:
    """单次买入批次，用于 T+1 判定与 FIFO 成本核算。"""

    date: date
    shares: int
    cost_per_share: float  # 纯成交价；手续费不摊入成本，单独从现金扣减


@dataclass
class Position:
    symbol: str
    name: str = ""
    lots: list[LotPurchase] = field(default_factory=list)
    last_price: float = 0.0  # mark-to-market 价

    @property
    def shares(self) -> int:
        return sum(lot.shares for lot in self.lots)

    @property
    def avg_cost(self) -> float:
        total = self.shares
        if total == 0:
            return 0.0
        return sum(lot.shares * lot.cost_per_share for lot in self.lots) / total

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.avg_cost) * self.shares if self.shares else 0.0

    def sellable_shares(self, as_of: date) -> int:
        """T+1 制度下，as_of 当日实际可卖股数（严格 lot.date < as_of）。"""
        return sum(lot.shares for lot in self.lots if lot.date < as_of)

    def add_lot(self, dt: date, shares: int, cost_per_share: float) -> None:
        self.lots.append(LotPurchase(dt, shares, cost_per_share))

    def reduce_shares(self, shares: int, as_of: date) -> float:
        """按 FIFO 削减可卖批次，返回卖出部分的成本基础（用于已实现盈亏）。

        如果可卖股数不足，会抛 ``ValueError`` —— 调用前应先 ``sellable_shares`` 校验。
        """
        remaining = shares
        cost_basis = 0.0
        sellable = sorted([l for l in self.lots if l.date < as_of], key=lambda l: l.date)
        locked = [l for l in self.lots if l.date >= as_of]
        kept: list[LotPurchase] = []
        for lot in sellable:
            if remaining <= 0:
                kept.append(lot)
                continue
            if lot.shares <= remaining:
                cost_basis += lot.shares * lot.cost_per_share
                remaining -= lot.shares
                # 整批卖完，不保留
            else:
                cost_basis += remaining * lot.cost_per_share
                kept.append(LotPurchase(lot.date, lot.shares - remaining, lot.cost_per_share))
                remaining = 0
        if remaining > 0:
            raise ValueError(
                f"{self.symbol} T+1 可卖 {shares - remaining} 股，请求卖出 {shares}（超出）"
            )
        self.lots = kept + locked
        return cost_basis


# ---------------------------------------------------------------------------
# 组合
# ---------------------------------------------------------------------------


class Portfolio:
    """投资组合：现金 + 多空持仓 + 净值曲线 + 成交记录。"""

    def __init__(
        self,
        initial_capital: float | None = None,
        commission_rate: float | None = None,
        stamp_tax_rate: float | None = None,
        min_commission: float | None = None,
        lot_size: int | None = None,
    ) -> None:
        cfg = settings.backtest
        self.initial_capital: float = float(initial_capital or cfg.initial_capital)
        self.cash: float = self.initial_capital
        self.commission_rate: float = commission_rate or cfg.commission_rate
        self.stamp_tax_rate: float = stamp_tax_rate or cfg.stamp_tax_rate
        self.min_commission: float = min_commission or cfg.min_commission
        self.lot_size: int = lot_size or cfg.lot_size

        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[date, float]] = []

    # ---------------- 查询 ----------------
    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    def equity(self) -> float:
        """总资产 = 现金 + 持仓市值。"""
        return self.cash + self.market_value()

    def position_weight(self, symbol: str) -> float:
        """单票市值占总资产比例。"""
        eq = self.equity()
        if eq <= 0:
            return 0.0
        pos = self.positions.get(symbol)
        return (pos.market_value / eq) if pos else 0.0

    # ---------------- 费用 ----------------
    def _calc_fees(self, amount: float, side: Side) -> tuple[float, float]:
        """返回 (佣金, 印花税)。"""
        fee = max(amount * self.commission_rate, self.min_commission)
        tax = amount * self.stamp_tax_rate if side == "SELL" else 0.0
        return round(fee, 4), round(tax, 4)

    # ---------------- 交易 ----------------
    def buy(self, symbol: str, shares: int, price: float, dt: date, name: str = "") -> Trade:
        if shares <= 0 or shares % self.lot_size != 0:
            raise ValueError(f"买入股数 {shares} 必须为 {self.lot_size} 的正整数倍")
        if price <= 0:
            raise ValueError(f"价格 {price} 无效")
        amount = shares * price
        fee, tax = self._calc_fees(amount, "BUY")
        total_cost = amount + fee + tax
        if total_cost > self.cash + 1e-6:
            raise ValueError(
                f"现金不足：买 {symbol} {shares}股 @{price:.2f} 需 {total_cost:.2f}，现 {self.cash:.2f}"
            )
        self.cash -= total_cost
        pos = self.positions.setdefault(symbol, Position(symbol=symbol, name=name))
        if name and not pos.name:
            pos.name = name
        pos.add_lot(dt, shares, price)
        pos.last_price = price  # 用最新成交价更新估值参考
        trade = Trade(dt, symbol, pos.name, "BUY", shares, price, amount, fee, tax, self.cash, 0.0)
        self.trades.append(trade)
        return trade

    def sell(self, symbol: str, shares: int, price: float, dt: date) -> Trade:
        pos = self.positions.get(symbol)
        if not pos or pos.shares == 0:
            raise ValueError(f"无 {symbol} 持仓，无法卖出")
        if shares <= 0 or shares % self.lot_size != 0:
            raise ValueError(f"卖出股数 {shares} 必须为 {self.lot_size} 的正整数倍")
        if price <= 0:
            raise ValueError(f"价格 {price} 无效")
        if shares > pos.sellable_shares(dt):
            raise ValueError(
                f"{symbol} T+1 可卖 {pos.sellable_shares(dt)} 股，请求卖出 {shares} 超出"
            )
        amount = shares * price
        fee, tax = self._calc_fees(amount, "SELL")
        cost_basis = pos.reduce_shares(shares, dt)
        proceeds = amount - fee - tax
        realized = proceeds - cost_basis
        self.cash += proceeds
        pos.last_price = price
        trade = Trade(dt, symbol, pos.name, "SELL", shares, price, amount, fee, tax, self.cash, realized)
        self.trades.append(trade)
        return trade

    # ---------------- 估值 ----------------
    def mark_to_market(self, prices: dict[str, float], dt: date) -> float:
        """以给定价格更新持仓最新价；未提供价格的标的保留上次值。

        记录当日净值到 equity_curve，并返回当日总资产。
        """
        for symbol, price in prices.items():
            pos = self.positions.get(symbol)
            if pos is not None and price and price > 0:
                pos.last_price = float(price)
        eq = self.equity()
        self.equity_curve.append((dt, eq))
        return eq

    # ---------------- 序列化 ----------------
    def snapshot(self) -> dict[str, object]:
        """当前组合状态的字典快照，便于打印与序列化。"""
        return {
            "cash": round(self.cash, 2),
            "market_value": round(self.market_value(), 2),
            "equity": round(self.equity(), 2),
            "return_pct": round((self.equity() / self.initial_capital - 1) * 100, 4),
            "positions": [
                {
                    "symbol": p.symbol,
                    "name": p.name,
                    "shares": p.shares,
                    "avg_cost": round(p.avg_cost, 4),
                    "last_price": round(p.last_price, 4),
                    "market_value": round(p.market_value, 2),
                    "weight": round(self.position_weight(p.symbol) * 100, 2),
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                }
                for p in self.positions.values()
                if p.shares > 0
            ],
        }
