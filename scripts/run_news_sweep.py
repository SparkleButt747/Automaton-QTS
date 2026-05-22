"""Run the news-reactive Optuna sweep against a live llama.cpp/Qwen server.

Generates a frozen bank of simulated FOMC episodes (randomised news->price
couplings), warms the classifier cache once via the LLM, runs the TPE study
maximising the lower-quantile excess-vs-buy-and-hold, then applies the best
params to the held-out real 2023-12-13 day as an n=1 validity check.

Requires a llama.cpp server (default http://localhost:8080) with the Qwen model
loaded. Cache-misses fall back to NEUTRAL signals if the server is down — so
confirm the server is up first (curl http://localhost:8080/v1/models).

Usage:
    # smoke run (fast, validates the real-LLM path end to end)
    .venv/bin/python -m scripts.run_news_sweep --episodes 6 --trials 8

    # full run (the real thing; ~20-30 min)
    .venv/bin/python -m scripts.run_news_sweep --episodes 50 --trials 150
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from qts.config import RiskLimits
from qts.data.real_episode import RealEpisode
from qts.optimisation.run_news_sweep import run_news_sweep
from qts.world.scenario import AnonAgentConfig, ScenarioConfig

logger = logging.getLogger(__name__)


def _base_scenario() -> ScenarioConfig:
    """A 6h BTCUSDT FOMC day: announcement at +3h leaves ample warmup + reaction bars."""
    return ScenarioConfig(
        name="fomc-sweep-base",
        symbol="BTCUSDT",
        start=datetime(2023, 12, 13, 16, 0, 0),
        end=datetime(2023, 12, 13, 22, 0, 0),
        tick=timedelta(minutes=1),
        fomc_announcement_at=datetime(2023, 12, 13, 19, 0, 0),
        fomc_expected_rate=5.25,
        starting_price=40_000.0,
        anon_agents=[
            AnonAgentConfig(agent_id="trend1", style="trend", aggressiveness=1.0),
            AnonAgentConfig(agent_id="sent1", style="sentiment", aggressiveness=1.0),
            AnonAgentConfig(agent_id="revert1", style="mean_revert", aggressiveness=1.0),
        ],
        mm_base_spread_bps=2.0,
        mm_vol_widen_k=1.0,
        powell_persona_id="powell",
    )


def _risk_limits() -> RiskLimits:
    return RiskLimits(
        max_daily_drawdown_pct=0.05,
        max_position_size_pct=0.20,
        max_open_positions=5,
        circuit_breaker_cooldown_seconds=300,
        sentiment_signal_max_scalar=3.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50, help="bank size (default 50)")
    parser.add_argument("--trials", type=int, default=150, help="Optuna trials (default 150)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--curated-root",
        type=Path,
        default=Path("data/real/fomc/2023-12-13"),
        help="held-out real FOMC day for the n=1 validity check",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    real_episode = None
    if (args.curated_root / "bars.csv").exists():
        real_episode = RealEpisode.from_disk(
            args.curated_root, symbol="BTCUSDT", source="fomc:2023-12-13"
        )
    else:
        logger.warning(
            "curated real day not found at %s — skipping validity check", args.curated_root
        )

    logger.info(
        "Running sweep: %d episodes x %d trials (seed=%d)", args.episodes, args.trials, args.seed
    )
    result = run_news_sweep(
        base_scenario=_base_scenario(),
        risk_limits=_risk_limits(),
        n_episodes=args.episodes,
        n_trials=args.trials,
        seed=args.seed,
        real_episode=real_episode,
    )

    print("\n" + "=" * 70)
    print("NEWS-REACTIVE OPTUNA SWEEP — RESULTS")
    print("=" * 70)
    print(f"trials: {result.n_completed} completed, {result.n_pruned} pruned")
    print(f"best train CVaR(excess-vs-hold @25th pct): {result.best_score:+.4f}")
    print("best params:")
    for k, v in result.best_params.items():
        print(f"    {k:>26}: {v:.4f}")
    if result.real_day is not None:
        rd = result.real_day
        verdict = "PASS" if rd.passed else "FAIL"
        print(f"\nheld-out real day 2023-12-13  [{verdict}]")
        print(f"    trades:       {rd.trades}")
        print(f"    strat return: {rd.strat_return:+.4f}")
        print(f"    hold return:  {rd.hold_return:+.4f}")
        print(f"    excess:       {rd.excess:+.4f}  (pass = trades>0 & excess>0 & strat>0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
