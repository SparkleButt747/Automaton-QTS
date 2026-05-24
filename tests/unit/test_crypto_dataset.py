"""T-CRYPTO-DATA-*: build_crypto_contagion_dataset end-to-end on stub adapter + stub LLM."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from qts.propagation.crypto.dataset import build_crypto_contagion_dataset

from qts.models.base import Bar
from qts.propagation.crypto.events import ContagionEvent
from qts.propagation.crypto.universe import load_crypto_universe


class _StubAdapter:
    async def get_historical_bars(
        self, symbol: str, start: str, end: str, interval: str = "1m"
    ) -> list[Bar]:
        base = datetime(2023, 1, 1, tzinfo=UTC)
        rng = np.random.default_rng(abs(hash(symbol)) % 1000)
        closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 400))
        return [
            Bar(
                timestamp=base + timedelta(hours=h),
                symbol=symbol,
                open=float(closes[h]),
                high=float(closes[h]),
                low=float(closes[h]),
                close=float(closes[h]),
                volume=1.0,
                bar_count=1,
            )
            for h in range(400)
        ]


class _StubLLM:
    async def query(self, *a: object, **k: object) -> str:
        raise AssertionError("must use query_json")

    async def query_json(self, *a: object, **k: object) -> dict:
        return {"relation": "competitor", "direction": "negative", "confidence": 0.7}


def _uni(tmp_path: Path):
    p = tmp_path / "uni.yaml"
    p.write_text(
        "tokens:\n"
        "  BTC: {cluster: Major}\n"
        "  SOL: {cluster: L1}\n"
        "  SRM: {cluster: ExchangeEco}\n"
        "  FTT: {cluster: ExchangeToken, is_exchange_token: true}\n"
    )
    return load_crypto_universe(p)


def _seed(tmp_path: Path) -> Path:
    p = tmp_path / "seed.yaml"
    p.write_text(
        "links:\n  - {source: FTT, peer: SOL, relation: entity_exposure, direction: negative, confidence: 0.95}\n"
    )
    return p


def test_build_dataset_shapes_and_seed(tmp_path: Path) -> None:  # T-CRYPTO-DATA-1
    uni = _uni(tmp_path)
    events = [
        ContagionEvent("FTT", datetime(2023, 1, 13, 12, tzinfo=UTC), "insolvency", 8e9),
    ]
    ds = asyncio.run(
        build_crypto_contagion_dataset(
            uni,
            events,
            bar_adapter=_StubAdapter(),
            structural_seed_path=_seed(tmp_path),
            llm=_StubLLM(),
            cache_dir=tmp_path / "cache",
            horizon=3,
            est_window=200,
            start="2023-01-01",
            end="2023-01-20",
        )
    )
    assert ds.adj_type.shape == (len(uni.tokens), len(uni.tokens))
    assert len(ds.samples) == 1
    s = ds.samples[0]
    assert s.features.shape == (len(uni.tokens), ds.feature_dim)
    assert s.reactions.shape == (len(uni.tokens),)
    assert s.named_idx == uni.index_of("FTT")
    assert s.merit == s.reactions[s.named_idx]  # source reaction == do() seed
    from qts.propagation.crypto.links import CRYPTO_RELATIONS

    assert ds.adj_type[uni.index_of("FTT"), uni.index_of("SOL")] == CRYPTO_RELATIONS.index(
        "entity_exposure"
    )


def test_event_outside_panel_is_skipped(tmp_path: Path) -> None:  # T-CRYPTO-DATA-2
    uni = _uni(tmp_path)
    events = [ContagionEvent("FTT", datetime(2030, 1, 1, tzinfo=UTC), "insolvency", 8e9)]
    ds = asyncio.run(
        build_crypto_contagion_dataset(
            uni,
            events,
            bar_adapter=_StubAdapter(),
            structural_seed_path=_seed(tmp_path),
            llm=_StubLLM(),
            cache_dir=tmp_path / "cache",
            horizon=3,
            est_window=200,
            start="2023-01-01",
            end="2023-01-20",
        )
    )
    assert ds.samples == []
