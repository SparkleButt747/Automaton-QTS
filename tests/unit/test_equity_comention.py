"""T-PATHA-COMENTION-*: FNSPID co-mention edge construction."""

from __future__ import annotations

from datetime import date

from qts.propagation.equity.comention import (
    Article,
    build_comention_edges,
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


def test_build_edges_canonical_pairs_and_pit_cutoff() -> None:  # T-PATHA-COMENTION-4
    arts = [
        Article(date=date(2020, 1, 1), tickers=("AAPL", "NVDA"), text="x"),
        Article(date=date(2020, 6, 1), tickers=("AAPL", "NVDA"), text="y"),
        Article(date=date(2025, 1, 1), tickers=("AAPL", "NVDA"), text="future"),  # after as_of
    ]
    edges = build_comention_edges(arts, as_of=date(2021, 1, 1), half_life_days=365.0)
    assert len(edges) == 1
    e = edges[0]
    assert (e.a, e.b) == ("AAPL", "NVDA")  # canonical a < b
    assert e.count == 2  # the 2025 article is excluded (look-ahead)
    assert e.last_seen == date(2020, 6, 1)
    assert e.weight > 0.0


def test_build_edges_recency_weight_decays() -> None:  # T-PATHA-COMENTION-5
    recent = [Article(date=date(2020, 12, 31), tickers=("AAPL", "NVDA"), text="r")]
    old = [Article(date=date(2018, 1, 1), tickers=("AAPL", "NVDA"), text="o")]
    as_of = date(2021, 1, 1)
    w_recent = build_comention_edges(recent, as_of=as_of)[0].weight
    w_old = build_comention_edges(old, as_of=as_of)[0].weight
    assert w_recent > w_old  # newer co-mentions weigh more
