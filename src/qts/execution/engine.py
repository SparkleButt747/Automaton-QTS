"""Legacy execution engine — deprecated in favour of NautilusTrader.

All classes and functions are re-exported from qts._legacy.execution_engine.
"""

from qts._legacy.execution_engine import (
    ExecutionEngine,
    ExecutionProtocol,
)

__all__ = ["ExecutionEngine", "ExecutionProtocol"]
