"""Tests for qts.world.order_book."""

from __future__ import annotations


def test_book_initial_state() -> None:  # T-WOB-1
    from qts.world.order_book import SimpleOrderBook

    book = SimpleOrderBook()
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.last_price is None


def test_set_quote_updates_best() -> None:  # T-WOB-2
    from qts.world.order_book import SimpleOrderBook

    book = SimpleOrderBook()
    book.set_quote(bid=29900.0, ask=30100.0, size=1.0)
    assert book.best_bid == 29900.0
    assert book.best_ask == 30100.0
    assert book.mid == 30000.0


def test_market_buy_lifts_offer() -> None:  # T-WOB-3
    from qts.world.order_book import SimpleOrderBook

    book = SimpleOrderBook()
    book.set_quote(bid=29900.0, ask=30100.0, size=1.0)
    fill = book.market_order(side="BUY", quantity=0.5)

    assert fill is not None
    assert fill.side == "BUY"
    assert fill.quantity == 0.5
    assert fill.price == 30100.0
    assert book.last_price == 30100.0


def test_market_sell_hits_bid() -> None:  # T-WOB-4
    from qts.world.order_book import SimpleOrderBook

    book = SimpleOrderBook()
    book.set_quote(bid=29900.0, ask=30100.0, size=1.0)
    fill = book.market_order(side="SELL", quantity=0.3)

    assert fill is not None
    assert fill.side == "SELL"
    assert fill.price == 29900.0


def test_market_order_rejected_when_no_quote() -> None:  # T-WOB-5
    from qts.world.order_book import SimpleOrderBook

    book = SimpleOrderBook()
    fill = book.market_order(side="BUY", quantity=1.0)
    assert fill is None


def test_market_order_size_clamped_to_quote_size() -> None:  # T-WOB-6
    from qts.world.order_book import SimpleOrderBook

    book = SimpleOrderBook()
    book.set_quote(bid=29900.0, ask=30100.0, size=0.1)
    fill = book.market_order(side="BUY", quantity=0.5)

    # MM only quoted 0.1 size — fill is partial up to quote
    assert fill is not None
    assert fill.quantity == 0.1


def test_trades_log_records_each_fill() -> None:  # T-WOB-7
    from qts.world.order_book import SimpleOrderBook

    book = SimpleOrderBook()
    book.set_quote(bid=29900.0, ask=30100.0, size=1.0)
    book.market_order(side="BUY", quantity=0.1)
    book.market_order(side="SELL", quantity=0.2)

    assert len(book.trades) == 2
    assert book.trades[0].side == "BUY"
    assert book.trades[1].side == "SELL"
