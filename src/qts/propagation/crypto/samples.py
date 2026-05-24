"""ContagionSample — one contagion do()-intervention on the token graph (Phase-3 input contract).
Mirrors equity EventSample (same field names so the Phase-3 train/eval harness reuses unchanged),
but carries a tz-aware ``event_ts`` (hour-precision) instead of a date."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class ContagionSample:
    named_idx: int  # the source token = do() intervention point
    merit: float  # the source's own realized BTC-adjusted move (the seed)
    event_ts: datetime
    features: np.ndarray  # (n_nodes, feature_dim)
    reactions: np.ndarray  # (n_nodes,) BTC-adjusted CAR per token

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
