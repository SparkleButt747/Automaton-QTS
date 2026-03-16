# ============================================================
# QTS Docker Image - Multi-stage build
# ============================================================

# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies into a prefix dir for easy copying
RUN pip install --upgrade pip && \
    pip install --prefix=/install . && \
    pip install --prefix=/install asyncpg && \
    pip install --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu


# --- Stage 2: Runtime image ---
FROM python:3.11-slim AS runtime

LABEL maintainer="QTS Team"
LABEL description="Quant Trading System - Multi-signal strategy engine"

# Create non-root user
RUN groupadd -r qts && useradd -r -g qts -d /app -s /sbin/nologin qts

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml ./

# Copy Alembic migrations if they exist
COPY alembic.ini ./
COPY migrations/ ./migrations/

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create data directory
RUN mkdir -p /app/data && chown -R qts:qts /app

# Switch to non-root user
USER qts

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from qts.config import get_settings; get_settings(); print('ok')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]

# Default command - start metrics server and block forever
CMD ["python", "-c", "from qts.monitoring.metrics import start_metrics_server; import os, signal; start_metrics_server(int(os.environ.get('QTS_METRICS_PORT', 8080))); signal.pause()"]
