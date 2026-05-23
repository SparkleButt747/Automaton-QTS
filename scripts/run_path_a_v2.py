"""Path A v2 end-to-end feasibility gate: does relation-typed propagation beat correlation on real
unnamed-peer earnings reactions? Wires FNSPID co-mention links (LLM-filtered) + SUE merit
(yfinance) + node features + abnormal-return labels (Alpaca) -> EventSamples -> train -> gate
(spec §8).

Usage:
    .venv/bin/python -m scripts.run_path_a_v2 --universe config/universe/path_a_v2.yaml \\
        --fnspid PATH --start 2016-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import logging

from qts.propagation.equity.universe import load_universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--fnspid", required=True, help="FNSPID CSV slice")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    universe = load_universe(args.universe)
    logging.info(
        "universe: %d tickers, %d sectors", len(universe.tickers), len(universe.unique_sectors)
    )
    # End-to-end wiring (build_path_a_dataset) needs live FNSPID + yfinance + Alpaca + llama.cpp.
    # Steps: co-mention edges -> EconomicLinkClassifier -> LinkGraph -> build_typed_adjacency;
    #        compute_sue per ticker; node_feature_vector per node per event date;
    #        market_model_abnormal_return per node -> assemble_event_samples;
    #        split train/held-out links -> fit_typed_propagation -> evaluate_path_a_gate.
    raise SystemExit(
        "End-to-end run requires live data sources (FNSPID + yfinance + Alpaca + llama.cpp). "
        "Implement build_path_a_dataset wiring here, then run. See spec §4/§8."
    )


if __name__ == "__main__":
    main()
