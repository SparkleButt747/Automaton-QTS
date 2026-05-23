"""T-PATHA-COMENTION-*: FNSPID co-mention edge construction."""

from __future__ import annotations

from datetime import date

from qts.propagation.equity.comention import (
    Article,
    extract_tickers,
)


def test_extract_tickers_matches_symbols_and_aliases() -> None:  # T-PATHA-COMENTION-1
    alias_map = {"aapl": "AAPL", "apple": "AAPL", "nvda": "NVDA", "nvidia": "NVDA"}
    text = "Apple's new chip pressures NVDA; analysts cite AAPL supply gains."
    found = extract_tickers(text, alias_map)
    assert found == {"AAPL", "NVDA"}


def test_extract_tickers_word_boundary_no_false_substring() -> None:  # T-PATHA-COMENTION-2
    alias_map = {"cat": "CAT"}
    assert extract_tickers("the category grew", alias_map) == set()
    assert extract_tickers("CAT raised guidance", alias_map) == {"CAT"}


def test_article_tickers_dedup_and_sorted() -> None:  # T-PATHA-COMENTION-3
    art = Article(date=date(2021, 5, 1), tickers=("NVDA", "AAPL", "AAPL"), text="x")
    assert art.tickers == ("AAPL", "NVDA")
