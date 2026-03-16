#!/usr/bin/env bash
# ============================================================
# Docker Compose E2E Test for QTS
# Builds, starts, validates, and tears down the full stack.
# ============================================================
set -euo pipefail

COMPOSE_FILE="docker/docker-compose.yml"
COMPOSE="docker compose -f $COMPOSE_FILE"
MAX_WAIT=120  # seconds to wait for services to become healthy
POLL_INTERVAL=5

# Track results
PASSED=0
FAILED=0
RESULTS=()

pass() {
    PASSED=$((PASSED + 1))
    RESULTS+=("PASS: $1")
    echo "  ✓ $1"
}

fail() {
    FAILED=$((FAILED + 1))
    RESULTS+=("FAIL: $1 — $2")
    echo "  ✗ $1: $2"
}

cleanup() {
    echo ""
    echo "=== Tearing down stack ==="
    $COMPOSE down -v --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Building and starting QTS Docker stack ==="
$COMPOSE up -d --build 2>&1
echo ""

# ---- Wait for services to be healthy ----
echo "=== Waiting for services to become healthy (max ${MAX_WAIT}s) ==="
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if core services are healthy
    DB_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' qts-timescaledb 2>/dev/null || echo "missing")
    REDIS_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' qts-redis 2>/dev/null || echo "missing")
    APP_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' qts-app 2>/dev/null || echo "missing")

    echo "  [${ELAPSED}s] db=$DB_HEALTH redis=$REDIS_HEALTH app=$APP_HEALTH"

    if [ "$DB_HEALTH" = "healthy" ] && [ "$REDIS_HEALTH" = "healthy" ]; then
        # DB and Redis healthy is enough to start testing — app may take longer
        if [ "$APP_HEALTH" = "healthy" ] || [ $ELAPSED -ge 60 ]; then
            echo "  Core services ready."
            break
        fi
    fi

    sleep $POLL_INTERVAL
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "WARNING: Timed out waiting for services. Running tests anyway..."
fi

echo ""
echo "=== Running connectivity tests ==="

# Test 1: TimescaleDB
if docker exec qts-timescaledb pg_isready -U qts -d trading >/dev/null 2>&1; then
    pass "TimescaleDB connectivity"
else
    fail "TimescaleDB connectivity" "pg_isready failed"
fi

# Test 2: Redis
REDIS_RESULT=$(docker exec qts-redis redis-cli ping 2>/dev/null || echo "FAIL")
if [ "$REDIS_RESULT" = "PONG" ]; then
    pass "Redis connectivity"
else
    fail "Redis connectivity" "redis-cli ping returned: $REDIS_RESULT"
fi

# Test 3: App config loads
if docker exec qts-app python -c "from qts.config import get_settings; print('Config OK')" 2>/dev/null; then
    pass "App config loading"
else
    fail "App config loading" "Could not import/load settings"
fi

# Test 4: App -> DB connectivity
if docker exec qts-app python -c "
import asyncio
from qts.db.engine import ensure_tables
asyncio.run(ensure_tables())
print('DB OK')
" 2>/dev/null; then
    pass "App -> DB connectivity"
else
    fail "App -> DB connectivity" "ensure_tables() failed"
fi

# Test 5: Prometheus scraping
if curl -sf http://localhost:9091/api/v1/targets 2>/dev/null | grep -q 'qts'; then
    pass "Prometheus scraping QTS target"
else
    fail "Prometheus scraping QTS target" "Target 'qts' not found in /api/v1/targets"
fi

# Test 6: Grafana health
if curl -sf http://localhost:3000/api/health 2>/dev/null | grep -q 'ok'; then
    pass "Grafana health"
else
    fail "Grafana health" "Health endpoint did not return ok"
fi

# Test 7: Metrics endpoint
if docker exec qts-app python -c "
import urllib.request
resp = urllib.request.urlopen('http://localhost:8080/metrics')
body = resp.read().decode()
assert 'qts_' in body, 'No qts_ metrics found'
print('Metrics OK')
" 2>/dev/null; then
    pass "Prometheus metrics endpoint on :8080"
else
    fail "Prometheus metrics endpoint on :8080" "Could not reach /metrics or no qts_ metrics"
fi

echo ""
echo "============================================"
echo "  E2E Test Summary"
echo "============================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: $PASSED  Failed: $FAILED"
echo "============================================"

# Exit with failure if any tests failed
[ $FAILED -eq 0 ]
