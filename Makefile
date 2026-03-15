.PHONY: help install dev-install lint typecheck test fmt docker-up docker-down clean

PYTHON := python3
PIP := pip
SRC := src/qts
TESTS := tests

# Default target
help:
	@echo "QTS - Quant Trading System"
	@echo ""
	@echo "Available targets:"
	@echo "  install       Install production dependencies"
	@echo "  dev-install   Install all dependencies including dev tools"
	@echo "  fmt           Format code with black and ruff"
	@echo "  lint          Run ruff linter"
	@echo "  typecheck     Run mypy type checker"
	@echo "  test          Run test suite with coverage"
	@echo "  docker-up     Start Docker services (timescaledb, redis)"
	@echo "  docker-down   Stop Docker services"
	@echo "  clean         Remove build artifacts and caches"

install:
	$(PIP) install -e .

dev-install:
	$(PIP) install -e ".[dev]"
	pre-commit install

fmt:
	black $(SRC) $(TESTS) scripts
	ruff check --fix $(SRC) $(TESTS) scripts

lint:
	ruff check $(SRC) $(TESTS)
	black --check $(SRC) $(TESTS) scripts

typecheck:
	mypy $(SRC) --config-file pyproject.toml

test:
	pytest $(TESTS) -v --tb=short --cov=$(SRC) --cov-report=term-missing --cov-report=html

test-unit:
	pytest $(TESTS)/unit -v --tb=short

test-integration:
	pytest $(TESTS)/integration -v --tb=short

test-fast:
	pytest $(TESTS) -v --tb=short -x --no-cov

docker-up:
	docker-compose -f docker/docker-compose.yml up -d timescaledb redis
	@echo "Waiting for services to be ready..."
	@sleep 3
	@echo "Services started."

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
	@echo "Clean complete."

# Convenience aliases
check: lint typecheck
all: fmt lint typecheck test
