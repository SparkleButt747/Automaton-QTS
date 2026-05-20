"""Legacy NautilusTrader adapter — deprecated in favour of qts.nautilus.

All classes and functions are re-exported from qts._legacy.nautilus_adapter.
"""

from qts._legacy.nautilus_adapter import (  # noqa: F401
    FILL_MODEL_IMMEDIATE,
    FILL_MODEL_REALISTIC,
    HAS_NAUTILUS,
    NautilusBacktestAdapter,
    nautilus_fill_to_qts,
    nautilus_position_to_qts,
    qts_bar_to_nautilus,
)
