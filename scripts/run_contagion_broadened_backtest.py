"""Broadened-trigger contagion backtest GATE (Phase 1). Detect idiosyncratic-drop events
over history -> existing dataset/fit/Null A/Null B/costed backtest -> GO/NO-GO verdict on
whether the broadened variant is worth taking live. Reuses run_crypto_contagion_v0.py's
flow with detector-sourced events, the detected event frequency, and the 72h horizon.

Usage: python scripts/run_contagion_broadened_backtest.py [universe.yaml] [structural.yaml]"""

import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qts.data.market.binance_adapter import BinanceBarAdapter
from qts.oversight.llm_client import LlamaCppClient
from qts.propagation.crypto.broadened import broadened_verdict, events_per_year
from qts.propagation.crypto.dataset import build_crypto_contagion_dataset
from qts.propagation.crypto.detect import IdiosyncraticDropDetector
from qts.propagation.crypto.gate import (
    contagion_backtest,
    evaluate_crypto_gate,
    event_study_linked_vs_unlinked,
    fit_crypto_propagation,
)
from qts.propagation.crypto.prices import fetch_price_panel
from qts.propagation.crypto.universe import load_crypto_universe
from qts.propagation.equity.gate import train_holdout_adjacency

logging.basicConfig(level=logging.INFO)
UNI = Path(sys.argv[1] if len(sys.argv) > 1 else "config/universe/crypto_contagion_live.yaml")
SEED = Path(sys.argv[2] if len(sys.argv) > 2 else "config/links/crypto_structural_live.yaml")
START, END = "2021-01-01", "2024-01-01"
HORIZON, EST_WINDOW = 72, 720  # 72h hold (feasibility-validated); 30-day (720h) beta estimation


async def main() -> None:
    uni = load_crypto_universe(UNI)
    adapter = BinanceBarAdapter()
    grid, closes = await fetch_price_panel(
        list(uni.tokens), bar_adapter=adapter, start=START, end=END, interval="1h"
    )
    if "BTC" not in closes:
        raise ValueError("BTC must be present in the universe (used as the market benchmark)")
    detector = IdiosyncraticDropDetector(
        threshold=0.15, window=24, est_window=EST_WINDOW, cooldown=72
    )
    events = detector.detect(closes, grid, closes["BTC"])
    print(f"detector fired {len(events)} events over {START}..{END}")
    if len(events) < 4:
        print("too few detected events — lower the threshold or widen the universe")
        return

    ds = await build_crypto_contagion_dataset(
        uni,
        events,
        bar_adapter=adapter,
        structural_seed_path=SEED,
        llm=LlamaCppClient(base_url="http://localhost:8080"),
        cache_dir=Path("data/crypto/link_cache_live"),
        horizon=HORIZON,
        est_window=EST_WINDOW,
        start=START,
        end=END,
    )
    print(f"{len(ds.samples)} usable events, {int((ds.adj_type >= 0).sum())} typed edges")
    if len(ds.samples) < 4:
        print("too few usable events after panel alignment")
        return

    adj_train, _ = train_holdout_adjacency(ds.adj_type, holdout_frac=0.3)
    model = fit_crypto_propagation(ds.samples, adj_type=adj_train, feature_dim=ds.feature_dim)
    named = np.array([s.named_idx for s in ds.samples])
    merit = np.array([s.merit for s in ds.samples])
    # pred uses sample[0] features: gate checks structural fit, not per-event accuracy (mirrors v0)
    pred = model.predict_np(ds.samples[0].features, ds.adj_type, named, merit)

    es = event_study_linked_vs_unlinked(ds.samples, adj_type=ds.adj_type)
    gate = evaluate_crypto_gate(pred, ds.samples, adj_type=ds.adj_type)
    epy = events_per_year(
        events,
        datetime.fromisoformat(START).replace(tzinfo=UTC),
        datetime.fromisoformat(END).replace(tzinfo=UTC),
    )
    bt = contagion_backtest(
        pred,
        ds,
        token_names=uni.tokens,
        top_k=3,
        cost_bps=7.5,
        horizon=HORIZON,
        events_per_year=epy,
    )
    print(
        f"\nNULL A: linked {es.mean_linked_car:+.4f} vs unlinked {es.mean_unlinked_car:+.4f} "
        f"| p={es.mann_whitney_p:.4f} {'SIG' if es.significant else 'n.s.'}"
    )
    print(
        f"NULL B: graph MSE {gate.graph_mse:.5f} vs pairwise {gate.pairwise_mse:.5f} "
        f"| {'BEATS' if gate.beats_pairwise else 'loses'}"
    )
    print(
        f"BACKTEST: MN mean {bt.market_neutral_mean:+.4f} Sharpe {bt.market_neutral_sharpe:+.2f} "
        f"| {bt.n_trades} trades | {epy:.0f} events/yr"
    )
    go = broadened_verdict(es, gate, bt)
    label = "GO — broadened variant has edge; proceed to Plan 2 (live)" if go else "NO-GO — stop"
    print(f"\nGATE: {label}")


if __name__ == "__main__":
    asyncio.run(main())
