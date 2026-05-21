"""Nautilus custom data type carrying a raw text event through the engine.

NOTE: this module deliberately does NOT use `from __future__ import annotations`.
The @customdataclass decorator introspects real type objects via __annotations__;
PEP 563 stringized annotations break it.
"""

from nautilus_trader.core.data import Data
from nautilus_trader.model.custom import customdataclass


@customdataclass
class NewsDataPoint(Data):
    """A text event as Nautilus custom data, interleaved with bars by the engine.

    Carries raw text (classification happens on receipt in the strategy, mirroring
    live trading). `ts_event` / `ts_init` are added by @customdataclass.
    """

    source: str = ""
    persona: str = ""
    text: str = ""
