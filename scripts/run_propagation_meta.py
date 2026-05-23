"""Meta-train the propagation operator over a pool of worlds; print held-out transfer (design §15).

Episodic relation-resampling cracks the n-hop transfer wall the single-world fit_graph hits: one
shared operator trained over >= ~10 distinct worlds generalises its learned composition to unseen
worlds. This reproduces the headline result on demand.

Usage:
    .venv/bin/python -m scripts.run_propagation_meta --pool-size 2000 --steps 20000 --seed 0
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from qts.propagation.meta import evaluate_meta_transfer, train_meta
from qts.propagation.sim import PropagationSimConfig, build_world, generate_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-size", type=int, default=2000, help="distinct training worlds")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=50, help="held-out worlds to evaluate")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rng = np.random.default_rng(args.seed)
    world_seeds = [int(s) for s in rng.integers(0, 1_000_000, args.pool_size)]
    logging.info("meta-training over %d worlds for %d steps...", args.pool_size, args.steps)
    model = train_meta(world_seeds, steps=args.steps, seed=args.seed)

    sub_caps, term_caps, sub_wins, term_wins = [], [], [], []
    for i in range(args.n_test):
        world = build_world(PropagationSimConfig(seed=2_000_000 + i))  # disjoint from training pool
        eval_batch = generate_events(world, 1000, np.random.default_rng(99_000 + i))
        report = evaluate_meta_transfer(model, world, eval_batch, seed=0)
        sub_caps.append(report.sub_capture)
        term_caps.append(report.terminal_capture)
        sub_wins.append(report.sub_win)
        term_wins.append(report.terminal_win)

    n = args.n_test
    print("\n" + "=" * 70)
    print("META-PROPAGATION — HELD-OUT TRANSFER (episodic relation-resampling)")
    print("=" * 70)
    print(f"  pool={args.pool_size} worlds   steps={args.steps}   held-out worlds={n}")
    print(
        f"  SUB  (1-hop B): win {sum(sub_wins):>3}/{n}   "
        f"mean capture {100 * float(np.mean(sub_caps)):5.1f}%"
    )
    print(
        f"  TERM (2-hop C): win {sum(term_wins):>3}/{n}   "
        f"mean capture {100 * float(np.mean(term_caps)):5.1f}%"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
