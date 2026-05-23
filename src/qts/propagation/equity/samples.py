"""EventSample — one earnings do()-intervention on the universe graph (Phase-3 input contract)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class EventSample:
    """A single event: named firm = the do() source (merit=SUE), targets = all nodes' reactions.

    ``features`` is the point-in-time node-feature matrix (n_nodes, feature_dim); ``reactions`` is
    the per-node abnormal return over the event window. ``named_idx`` indexes both, into the
    universe's sorted-ticker order.
    """

    named_idx: int
    merit: float
    event_date: date
    features: np.ndarray
    reactions: np.ndarray

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be 2-D (n_nodes, feature_dim)")
        if self.reactions.shape != (self.features.shape[0],):
            raise ValueError("reactions must be 1-D of length n_nodes")
        if not (0 <= self.named_idx < self.features.shape[0]):
            raise ValueError("named_idx out of range")

    @property
    def n_nodes(self) -> int:
        return int(self.features.shape[0])
