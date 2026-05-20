"""Minimal limit order book for v1 of the world simulator.

The MM publishes a single (bid, ask, size) quote per tick. Anon agents
issue market orders against it. The book records every fill into
`trades` and tracks last_price for downstream bar aggregation.

This is deliberately simple — no order ID matching, no partial-fill
queueing across ticks, no level-2 depth. v3 will swap this out for
Nautilus's real matching engine when the strategy itself becomes a
participant agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Trade:
    side: str
    quantity: float
    price: float


@dataclass
class SimpleOrderBook:
    """Single-level book driven by the MM agent's quote."""

    best_bid: float | None = None
    best_ask: float | None = None
    quote_size: float = 0.0
    last_price: float | None = None
    trades: list[_Trade] = field(default_factory=list)

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    def set_quote(self, bid: float, ask: float, size: float) -> None:
        if ask <= bid:
            raise ValueError("ask must be > bid")
        if size <= 0:
            raise ValueError("quote size must be > 0")
        self.best_bid = bid
        self.best_ask = ask
        self.quote_size = size

    def market_order(self, side: str, quantity: float) -> _Trade | None:
        """Execute a market order against the current quote.

        Returns a Trade or None if no quote is live or quantity is non-positive.
        Quantity is clamped to the MM's quoted size (no walking the book).
        """
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        if quantity <= 0:
            return None
        if side == "BUY":
            if self.best_ask is None:
                return None
            price = self.best_ask
        else:
            if self.best_bid is None:
                return None
            price = self.best_bid

        fill_qty = min(quantity, self.quote_size)
        trade = _Trade(side=side, quantity=fill_qty, price=price)
        self.trades.append(trade)
        self.last_price = price
        return trade
