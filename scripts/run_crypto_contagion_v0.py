"""Crypto contagion v0 — end-to-end verdict on real Binance + Qwen (spec §9.10, §12).
Build dataset -> fit operator (held-out links) -> Null A (event-study) + Null B (vs pairwise) ->
costed market-neutral + outright backtest -> print verdict. Usage:
  python scripts/run_crypto_contagion_v0.py [universe.yaml] [events.yaml] [structural.yaml]"""

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

from qts.data.market.binance_adapter import BinanceBarAdapter
from qts.oversight.llm_client import LlamaCppClient
from qts.propagation.crypto.dataset import build_crypto_contagion_dataset
from qts.propagation.crypto.events import load_contagion_events
from qts.propagation.crypto.gate import (
    contagion_backtest,
    evaluate_crypto_gate,
    event_study_linked_vs_unlinked,
    fit_crypto_propagation,
)
from qts.propagation.crypto.universe import load_crypto_universe
from qts.propagation.equity.gate import train_holdout_adjacency

logging.basicConfig(level=logging.INFO)
UNI = Path(sys.argv[1] if len(sys.argv) > 1 else "config/universe/crypto_contagion_v0.yaml")
EVENTS = Path(sys.argv[2] if len(sys.argv) > 2 else "config/events/contagion_v0.yaml")
SEED = Path(sys.argv[3] if len(sys.argv) > 3 else "config/links/crypto_structural_v0.yaml")
HORIZON, EST_WINDOW = 24, 720  # 1-day hold; 30-day (720h) beta estimation


async def main() -> None:
    uni = load_crypto_universe(UNI)
    events = load_contagion_events(EVENTS)
    ds = await build_crypto_contagion_dataset(
        uni,
        events,
        bar_adapter=BinanceBarAdapter(),
        structural_seed_path=SEED,
        llm=LlamaCppClient(base_url="http://localhost:8080"),
        cache_dir=Path("data/crypto/link_cache_v0"),
        horizon=HORIZON,
        est_window=EST_WINDOW,
        start="2021-01-01",
        end="2024-01-01",
    )
    print(f"{len(ds.samples)} usable events, {int((ds.adj_type >= 0).sum())} typed edges")
    if len(ds.samples) < 4:
        print("too few usable events (delisting?) — see spec §10 archive note")
        return
    adj_train, _adj_held = train_holdout_adjacency(ds.adj_type, holdout_frac=0.3)
    model = fit_crypto_propagation(ds.samples, adj_type=adj_train, feature_dim=ds.feature_dim)
    named = np.array([s.named_idx for s in ds.samples])
    merit = np.array([s.merit for s in ds.samples])
    pred = model.predict_np(ds.samples[0].features, ds.adj_type, named, merit)

    es = event_study_linked_vs_unlinked(ds.samples, adj_type=ds.adj_type)
    gate = evaluate_crypto_gate(pred, ds.samples, adj_type=ds.adj_type)
    bt = contagion_backtest(
        pred, ds, token_names=uni.tokens, top_k=3, cost_bps=7.5, horizon=HORIZON
    )
    print("\n=== NULL A (event-study: linked vs unlinked) ===")
    print(
        f"  linked mean CAR {es.mean_linked_car:+.4f} vs unlinked {es.mean_unlinked_car:+.4f} "
        f"| MW p={es.mann_whitney_p:.4f} {'SIGNIFICANT' if es.significant else 'n.s.'}"
    )
    print("=== NULL B (operator vs pairwise on linked peers) ===")
    print(
        f"  graph MSE {gate.graph_mse:.5f} vs pairwise {gate.pairwise_mse:.5f} "
        f"| graph hit {gate.graph_hit:.2%} vs {gate.pairwise_hit:.2%} "
        f"| {'BEATS' if gate.beats_pairwise else 'loses to'} pairwise"
    )
    print("=== COSTED BACKTEST (net 7.5bps/leg) ===")
    mn_sharpe = bt.market_neutral_sharpe
    print(f"  market-neutral: mean {bt.market_neutral_mean:+.4f}  Sharpe {mn_sharpe:+.2f}")
    print(f"  outright:       mean {bt.outright_mean:+.4f}  Sharpe {bt.outright_sharpe:+.2f}")
    print(f"  ({bt.n_trades} trades)")
    verdict = es.significant and gate.beats_pairwise and bt.market_neutral_sharpe > 1.0
    print(
        f"\nVERDICT: {'TRADEABLE SIGNAL' if verdict else 'no tradeable signal'} on this universe."
    )


if __name__ == "__main__":
    asyncio.run(main())
