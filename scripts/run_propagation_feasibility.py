"""Run the event-propagation graph feasibility gate at full scale and print the report.

Usage:
    .venv/bin/python -m scripts.run_propagation_feasibility --seed 0 --epochs 600
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from qts.propagation.sim import PropagationSimConfig, build_world, make_splits
from qts.propagation.train import evaluate_feasibility, fit_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--n-train", type=int, default=4000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    world = build_world(PropagationSimConfig(seed=args.seed))
    train, val, test, transfer = make_splits(
        world, np.random.default_rng(args.seed), n_train=args.n_train
    )
    model = fit_graph(world, train, val, epochs=args.epochs, seed=args.seed)
    report = evaluate_feasibility(world, model, test, transfer, seed=args.seed)

    print("\n" + "=" * 70)
    print("EVENT-PROPAGATION GRAPH — FEASIBILITY GATE")
    print("=" * 70)
    print(
        f"  test MSE   graph={report.test_mse_graph:.5f}  "
        f"corr={report.test_mse_corr:.5f}  noprop={report.test_mse_noprop:.5f}"
    )
    print(f"  subst MSE  graph={report.sub_mse_graph:.5f}  corr={report.sub_mse_corr:.5f}")
    print(
        f"  transfer   graph={report.transfer_sub_mse_graph:.5f}  "
        f"corr={report.transfer_sub_mse_corr:.5f}"
    )
    print(f"  prediction_pass={report.prediction_pass}  transfer_pass={report.transfer_pass}")
    print(f"  PASSED={report.passed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
