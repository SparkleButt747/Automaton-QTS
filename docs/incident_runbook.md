# Incident Runbook

This runbook provides step-by-step procedures for operational incidents.
For each incident type the runbook covers: detection, containment,
investigation, resolution, and post-incident review.

---

## Incident 1 — Data Feed Goes Stale

**Symptoms:**
- `AlertManager` raises a CRITICAL alert: "Market data feed stale".
- `HealthChecker.check_market_feed()` returns `UNHEALTHY`.
- `DashboardState.data_timestamps["binance"]` shows age > threshold.
- No new bars are arriving for > 60 seconds.

**Containment:**
1. The execution engine will not emit new orders on stale snapshots (the
   `SignalPipeline.compute()` call returns `None` if the bar buffer is frozen).
2. No immediate manual halt is required unless positions are open and
   adverse price movement is occurring.

**Investigation:**
```bash
# Check the last bar timestamp
qts status --feed

# Inspect logs
tail -100 logs/qts.log | grep "market_feed\|bar_buffer\|binance"
```

**Resolution:**
1. Verify internet connectivity and Binance API status at
   `https://www.binancestatus.com/`.
2. Restart the market data adapter:
   ```bash
   qts restart --component market_feed
   ```
3. If the feed does not recover within 5 minutes, halt trading and
   investigate the Binance API key / IP whitelist.
4. Verify feed recovery: `qts status --feed` should show "FRESH".

**Post-incident:** Log the outage duration and any missed bars.  If bars
were missed during an open position, review the position for potential
adverse fills.

---

## Incident 2 — Circuit Breaker Triggered

**Symptoms:**
- `CircuitBreakerError` raised in execution logs.
- `AlertManager` raises CRITICAL drawdown alert.
- Dashboard shows daily drawdown >= `max_daily_drawdown_pct` (2%).
- All new order attempts are blocked.

**Containment:**
The circuit breaker is self-contained.  Trading is automatically halted for
`circuit_breaker_cooldown_seconds` (3600 s = 1 hour).  No immediate action
is required unless the drawdown is continuing.

**Investigation:**
```bash
# Check current drawdown and positions
qts status --risk

# Review today's trade log
qts trades --today --sort pnl

# Identify the triggering trade(s)
qts trades --today --loss --limit 5
```

**Resolution:**
1. Review the trade log and identify the failure mode.
2. If the drawdown event was a one-off (e.g., news spike), allow the
   cooldown to expire naturally.
3. If the drawdown was systematic, consider:
   - Reducing `entry_threshold` temporarily via the proposal approval process.
   - Reviewing the sentiment signal weight if `SENTIMENT_FADE` is the
     dominant failure mode.
4. After cooldown, confirm the circuit breaker has lifted:
   ```python
   risk_manager.is_halted()  # should return False
   ```
5. Trading resumes automatically after cooldown.

**Post-incident:** File a debrief summary.  Run `qts debrief` to trigger
the LLM analysis and capture proposals for human review.

---

## Incident 3 — LLM API Unavailable

**Symptoms:**
- `LLMClientProtocol.query()` raises `httpx.HTTPStatusError` or timeout.
- Regime classification and debrief tasks fail.
- Logs show: "LLM API call failed: ...".

**Containment:**
The LLM is on the **cold path** only.  A failed LLM call does **not**
block trading.  The execution engine continues using the last known
`vol_regime` and the stale regime override file.

**Investigation:**
```bash
# Check Anthropic API status
curl https://status.anthropic.com/

# Verify the API key is set
echo $ANTHROPIC_API_KEY | head -c 10
```

**Resolution:**
1. If it is a temporary outage, the system will retry automatically on the
   next scheduled debrief / regime classification.
2. If the API key has expired, rotate it:
   ```bash
   export ANTHROPIC_API_KEY=<new_key>
   qts restart --component llm_client
   ```
3. If the model ID has changed, update `ANTHROPIC_MODEL` in `.env`.

**Post-incident:** Backfill the missed debrief session manually:
```bash
qts debrief --date YYYY-MM-DD
```

---

## Incident 4 — Unexpected Large Loss

**Symptoms:**
- Single trade PnL > 2× average trade loss.
- Positions closed by `CIRCUIT_BREAKER` or `STOP_LOSS`.
- Human operator notices unusual dashboard activity.

**Immediate actions (first 5 minutes):**
1. **Halt trading immediately** (do not wait for circuit breaker):
   ```bash
   qts halt --reason "unexpected large loss: manual investigation"
   ```
2. Review the failing trade:
   ```bash
   qts trades --trade-id <id> --verbose
   ```
3. Review open positions and decide whether to close them manually:
   ```bash
   qts positions
   qts close-all --dry-run  # preview before executing
   ```

**Investigation:**
1. Identify the failure mode: was this a news spike, regime mismatch,
   or execution slip?
2. Check if the sentiment signal was misleading at entry.
3. Check the vol regime at trade entry.

**Resolution:**
1. If caused by a bug (e.g., incorrect signal calculation), patch and redeploy.
2. If caused by market conditions, adjust weights via the proposal process.
3. Resume trading only after root cause is identified:
   ```bash
   qts resume --force
   ```

**Post-incident:** Mandatory debrief and proposal review within 24 hours.

---

## Incident 5 — Database Connection Failure

**Symptoms:**
- `HealthChecker.check_database()` returns `UNHEALTHY`.
- Logs show: "Database connection failed: ...".
- Trade and signal writes fail and fall back to in-memory buffer.

**Containment:**
`TradeLogger` and `SignalLogger` both have in-memory fallback buffers.
Trading continues, but records are not persisted.  The in-memory buffer
will be lost on process restart.

**Immediate actions:**
1. Monitor the in-memory buffer size:
   ```bash
   qts status --db
   ```
2. Do **not** restart the process until the database is restored, to
   preserve the in-memory buffer.

**Investigation:**
```bash
# Check database connectivity directly
psql $DATABASE_URL -c "SELECT 1;"

# Check TimescaleDB service
systemctl status timescaledb  # or equivalent
docker ps | grep timescale
```

**Resolution:**
1. Restore database connectivity (restart the DB container, check disk
   space, check credentials).
2. Flush the in-memory buffer to the database:
   ```bash
   qts flush-buffer --to-db
   ```
3. Verify health:
   ```bash
   qts status --db  # should show HEALTHY
   ```

**Post-incident:** Review whether any trades were lost.  If the buffer
was flushed successfully, no data loss should have occurred.

---

## Incident 6 — Redis / Celery Worker Failure

**Symptoms:**
- `HealthChecker.check_redis()` returns `UNHEALTHY`.
- Celery tasks (sentiment update, GDELT polling) are queued but not processing.
- Logs show: "Redis ping failed: ...".

**Containment:**
Redis/Celery is on the **cold path** only (sentiment updates, scheduled
tasks).  The hot trading path is unaffected.  Sentiment scores will be
stale but this is handled gracefully by using the last known value.

**Investigation:**
```bash
# Check Redis
redis-cli -u $REDIS_URL ping  # should return PONG

# Check Celery workers
celery -A qts.tasks inspect active

# Check Redis service
docker ps | grep redis
systemctl status redis
```

**Resolution:**
1. Restart Redis:
   ```bash
   docker restart qts-redis
   # or
   systemctl restart redis
   ```
2. Restart Celery workers:
   ```bash
   celery -A qts.tasks worker --loglevel=info &
   ```
3. Verify tasks resume processing:
   ```bash
   celery -A qts.tasks inspect active
   ```
4. Check for missed sentiment updates and trigger a manual refresh:
   ```bash
   qts sentiment --refresh --symbol BTCUSDT
   ```

**Post-incident:** Review whether stale sentiment caused any adverse
trades during the outage period.
