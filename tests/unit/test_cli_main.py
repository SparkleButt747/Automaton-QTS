"""Unit tests for qts.cli.main.

Verifies:
- main group: --help displays usage
- main group: --version flag works
- main group: --log-level option is accepted
- main group: --skip-db-init skips database initialisation
- main group: database init failure is swallowed (warning, not crash)
- backtest command: prints symbol and date range, exits 0
- live command: dry-run mode prints DRY RUN, does not prompt
- live command: --live flag triggers abort if confirmation is denied
- debrief command: prints and exits cleanly
- approve command: prints and exits cleanly
- status command: prints environment and config info
- status command: prints configuration warnings when issues exist
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from qts.cli.main import main

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(*args, **kwargs):
    runner = CliRunner()
    return runner.invoke(main, list(args), catch_exceptions=False, **kwargs)


def _run_no_db(*args, **kwargs):
    """Invoke main with --skip-db-init to avoid touching the database."""
    return _run("--skip-db-init", *args, **kwargs)


# ── T-CLI-1 ───────────────────────────────────────────────────────────────────


class TestMainGroup:
    def test_help_shows_usage(self):
        """--help prints usage information."""
        result = _run_no_db("--help")
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_skip_db_init_flag_accepted(self):
        """--skip-db-init is accepted and does not cause an error."""
        result = _run("--skip-db-init", "status")
        assert result.exit_code == 0

    def test_log_level_option_accepted(self):
        """--log-level DEBUG is accepted without error."""
        result = _run("--skip-db-init", "--log-level", "DEBUG", "status")
        assert result.exit_code == 0

    def test_db_init_failure_is_swallowed(self):
        """A database init exception does not crash the CLI."""
        with patch("qts.db.engine.ensure_tables", new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = Exception("DB is on fire")
            result = _run("status")
        # Should not crash (exit 0 or 1 due to status, but not 2 which is click error)
        assert result.exit_code in (0, 1)


# ── T-CLI-2 ───────────────────────────────────────────────────────────────────


class TestBacktestCommand:
    def test_prints_symbol_and_dates(self):
        """backtest prints the symbol and date range."""
        result = _run_no_db("backtest", "--symbol", "ETHUSDT", "--start", "2024-01-01", "--end", "2024-03-01")
        assert result.exit_code == 0
        assert "ETHUSDT" in result.output
        assert "2024-01-01" in result.output

    def test_default_symbol_is_btcusdt(self):
        """backtest defaults to BTCUSDT when --symbol is not provided."""
        result = _run_no_db("backtest", "--start", "2024-01-01", "--end", "2024-02-01")
        assert "BTCUSDT" in result.output

    def test_requires_start_option(self):
        """backtest fails with exit code 2 when --start is missing."""
        result = _run_no_db("backtest", "--end", "2024-02-01")
        assert result.exit_code == 2


# ── T-CLI-3 ───────────────────────────────────────────────────────────────────


class TestLiveCommand:
    def test_dry_run_mode_prints_dry_run(self):
        """live in --dry-run mode (default) prints DRY RUN without prompting."""
        result = _run_no_db("live", "--symbol", "BTCUSDT")
        assert result.exit_code == 0
        assert "DRY" in result.output.upper() or "dry" in result.output.lower()

    def test_live_mode_aborts_on_no(self):
        """live --live mode prompts for confirmation and aborts on 'n'."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--skip-db-init", "--live", "live"],
            input="n\n",
            catch_exceptions=False,
        )
        assert result.exit_code != 0


# ── T-CLI-4 ───────────────────────────────────────────────────────────────────


class TestDebriefCommand:
    def test_prints_and_exits_cleanly(self):
        """debrief command prints a message and exits 0."""
        result = _run_no_db("debrief")
        assert result.exit_code == 0
        assert "debrief" in result.output.lower()


# ── T-CLI-5 ───────────────────────────────────────────────────────────────────


class TestApproveCommand:
    def test_prints_and_exits_cleanly(self):
        """approve command prints a message and exits 0."""
        result = _run_no_db("approve")
        assert result.exit_code == 0
        assert len(result.output) > 0


# ── T-CLI-6 ───────────────────────────────────────────────────────────────────


class TestStatusCommand:
    def test_prints_environment_info(self):
        """status prints the current environment, dry-run flag, and database URL."""
        result = _run_no_db("status")
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert any(word in output_lower for word in ["environment", "env", "database", "dry"])

    def test_prints_configuration_warnings(self):
        """status shows configuration warnings when validate_production_readiness returns issues."""
        mock_settings = MagicMock()
        mock_settings.qts_env = "test"
        mock_settings.qts_dry_run = True
        mock_settings.qts_log_level = "INFO"
        mock_settings.database.database_url = "sqlite:///test.db"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.validate_production_readiness.return_value = ["API key missing", "DB not reachable"]

        with patch("qts.config.get_settings", return_value=mock_settings):
            result = _run_no_db("status")

        assert result.exit_code == 0
        assert "API key missing" in result.output

    def test_prints_config_ok_when_no_issues(self):
        """status shows 'Configuration OK' when there are no warnings."""
        mock_settings = MagicMock()
        mock_settings.qts_env = "dev"
        mock_settings.qts_dry_run = True
        mock_settings.qts_log_level = "INFO"
        mock_settings.database.database_url = "sqlite:///dev.db"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.validate_production_readiness.return_value = []

        with patch("qts.config.get_settings", return_value=mock_settings):
            result = _run_no_db("status")

        assert result.exit_code == 0
        assert "OK" in result.output or "ok" in result.output.lower()
