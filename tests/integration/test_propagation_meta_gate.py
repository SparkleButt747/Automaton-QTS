"""T-PROP-META-GATE-1: episodic relation-resampling transfers to UNSEEN worlds (design §15).

This is the claim that cracks the T-PROP-GATE-3 wall: a single shared operator meta-trained over a
pool of distinct worlds generalises its learned composition to worlds it never saw — where the
single-world fit_graph fails. Asserts robust 1-hop transfer and positive 2-hop transfer on held-out
worlds (seeds disjoint from the training pool).
"""

from __future__ import annotations

import numpy as np

from qts.propagation.meta import evaluate_meta_transfer, train_meta
from qts.propagation.sim import PropagationSimConfig, build_world, generate_events


def test_meta_transfers_to_unseen_worlds() -> None:  # T-PROP-META-GATE-1
    # >= ~10 distinct worlds is the diversity floor (design §15); train one shared operator on 15.
    model = train_meta(range(15), steps=6000, batch_size=256, seed=0)

    sub_caps, term_caps, sub_wins = [], [], []
    for held_out_seed in range(900, 905):  # 5 worlds disjoint from the training pool [0, 15)
        world = build_world(PropagationSimConfig(seed=held_out_seed))
        eval_batch = generate_events(world, 1000, np.random.default_rng(50_000 + held_out_seed))
        report = evaluate_meta_transfer(model, world, eval_batch, n_history=10000, seed=0)
        sub_caps.append(report.sub_capture)
        term_caps.append(report.terminal_capture)
        sub_wins.append(report.sub_win)

    # 1-hop substitute transfers robustly to unseen worlds (the plateau is ~0.5; bar well below it)
    assert sum(sub_wins) >= 4  # graph beats correlational on >= 4/5 unseen worlds
    assert float(np.mean(sub_caps)) > 0.12
    # 2-hop terminal composition also transfers (noisier/shallower capture, so a softer bar)
    assert float(np.mean(term_caps)) > 0.04
