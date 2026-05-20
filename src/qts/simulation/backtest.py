"""Legacy backtest engine — deprecated in favour of NautilusTrader.

All classes and functions are re-exported from qts._legacy.backtest.
"""

from qts._legacy.backtest import (  # noqa: F401
    BacktestEngine,
    BacktestResult,
    BacktestSettings,
    compute_backtest_statistics,
)
