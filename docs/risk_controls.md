# Risk Controls Documentation

All hard limits enforced by QTS are defined in `config/risk_limits.json` and
loaded into the frozen `RiskLimits` Pydantic model.  **No limit may be
changed at runtime.**  Changes require an explicit file edit, a pre-commit
audit commit, and a human reviewer sign-off.

---

## Immutable Hard Limits

### `max_daily_drawdown_pct` — 2%

**Value:** `0.02`

**Rationale:**
A 2% daily loss cap limits maximum drawdown to a recoverable level for a
diversified crypto/equity portfolio.  At this level, ten consecutive
maximum-drawdown days would produce only a ~18% portfolio decline, well
within typical institutional risk tolerance.

**How the circuit breaker works:**

1. `RiskManager.check_daily_drawdown(daily_pnl, portfolio_value)` is called
   before every order.
2. `drawdown_pct = -daily_pnl / portfolio_value` (positive = loss).
3. If `drawdown_pct >= max_daily_drawdown_pct`:
   - `_trip_circuit_breaker()` records the current UTC timestamp.
   - `_halted = True`.
   - All subsequent calls to `approve_order()` raise `CircuitBreakerError`.
4. After `circuit_breaker_cooldown_seconds` the halt is automatically lifted
   on the next `is_halted()` call.
5. On a new trading session, `reset_daily_state()` must be called explicitly.

**AlertManager integration:**  
`AlertManager.check_drawdown(current_drawdown, limit=0.02)` emits:
- `WARNING` when drawdown reaches the `drawdown_threshold` (default 1%).
- `CRITICAL` when drawdown reaches or exceeds `max_daily_drawdown_pct`.

---

### `max_position_size_pct` — 5%

**Value:** `0.05`

**Rationale:**
Limits single-position risk to 5% of portfolio value, ensuring that even a
complete loss of one position cannot exceed a 5% total portfolio loss.  This
prevents over-concentration regardless of signal strength.

**How it is enforced:**

```python
fraction = proposed_size_usd / portfolio_value
allowed  = fraction <= max_position_size_pct
```

If `allowed` is False the order is silently rejected and a WARNING is logged.

---

### `max_open_positions` — 5

**Value:** `5`

**Rationale:**
Prevents over-diversification and excessive margin usage.  With 5 positions
at the 5% size cap, at most 25% of the portfolio is deployed at any time,
leaving 75% as cash/collateral buffer.

**How it is enforced:**

```python
allowed = len(current_positions) < max_open_positions
```

---

### `circuit_breaker_cooldown_seconds` — 3600

**Value:** `3600` (1 hour)

**Rationale:**
A 1-hour cooldown after a circuit breaker trip prevents the system from
immediately re-entering positions after a loss event, providing time for
human review and market conditions to normalise.

**Automatic lift:**  
After `cooldown_seconds` have elapsed, `is_halted()` automatically resets
`_halted = False` so no manual intervention is required for routine cooldowns.

---

### `sentiment_signal_max_scalar` — 2.0

**Value:** `2.0`

**Rationale:**
Caps the sentiment signal's maximum influence on the combined alpha at 2×
the base signal.  Without this cap, extreme sentiment readings (e.g.
viral news events) could dominate the alpha computation and override
technical signals.

**How it is enforced:**

In `SentimentFusion`, the final fused score is clipped to `[-1, 1]`.
The `sentiment_signal_max_scalar` limits the range of the raw fused score
before clipping, ensuring sentiment cannot inflate the combined alpha beyond
its intended weight.

---

## Who Can Change These Limits

**ONLY a human operator** may modify risk limits.  There is no programmatic
path to override or loosen them at runtime.

Changes must follow this process:

### How to Change a Risk Limit

1. **Prepare justification:** Document the proposed change, its rationale,
   expected impact on maximum drawdown, and approval authority.

2. **Edit the file:**
   ```bash
   # Edit the file
   vim config/risk_limits.json
   ```

3. **Commit with the `--force` flag** (bypasses the pre-commit immutability
   hook for `risk_limits.json`):
   ```bash
   git add config/risk_limits.json
   git commit --no-verify -m "risk: increase max_daily_drawdown_pct to 0.025 — approved by [NAME]"
   ```

4. **Audit trail:**  The commit message must reference the approver's name
   and a brief justification.  This forms the permanent audit record.

5. **Code review:**  The pull request must be reviewed and approved by at
   least one human with trading risk authority before merging.

6. **No hot-reload:**  The application must be restarted after any change
   to `risk_limits.json` because `RiskLimits` is loaded once at startup
   and the model is frozen (`model_config = {"frozen": True}`).

---

## Emergency Procedures

**Immediate halt without waiting for circuit breaker:**

```python
from qts.execution.risk import RiskManager
risk_manager._trip_circuit_breaker()
```

Or via the CLI:
```bash
qts halt --reason "manual halt: unexpected market conditions"
```

**Check current state:**
```bash
qts status
```

**Resume trading after manual halt:**
```python
risk_manager.reset_daily_state()
```
