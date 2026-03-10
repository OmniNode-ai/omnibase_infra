# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for scripts/provision-cross-repo-tables.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "provision-cross-repo-tables.py"


def _load_script():
    """Load provision-cross-repo-tables.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "provision_cross_repo_tables", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
class TestScriptLoads:
    """Smoke test: script is importable and has required symbols."""

    def test_script_exists(self):
        assert SCRIPT_PATH.exists(), f"Script not found: {SCRIPT_PATH}"

    def test_cross_repo_tables_constant_defined(self):
        mod = _load_script()
        assert hasattr(mod, "CROSS_REPO_TABLES")
        assert isinstance(mod.CROSS_REPO_TABLES, dict)
        assert "idempotency_records" in mod.CROSS_REPO_TABLES

    def test_idempotency_records_has_create_table(self):
        mod = _load_script()
        stmts = mod.CROSS_REPO_TABLES["idempotency_records"]
        assert isinstance(stmts, list)
        assert any("CREATE TABLE IF NOT EXISTS" in s for s in stmts)

    def test_idempotency_records_has_indexes(self):
        mod = _load_script()
        stmts = mod.CROSS_REPO_TABLES["idempotency_records"]
        create_index_stmts = [s for s in stmts if "CREATE INDEX IF NOT EXISTS" in s]
        # Expect at least 3 indexes: processed_at, domain, correlation_id
        assert len(create_index_stmts) >= 3


@pytest.mark.unit
class TestDryRunMode:
    """Dry-run path: prints SQL without connecting to the database."""

    def test_dry_run_does_not_call_asyncpg(self, capsys):
        mod = _load_script()

        import asyncio

        asyncio.run(mod.provision("postgresql://fake/db", dry_run=True))

        captured = capsys.readouterr()
        assert "[DRY-RUN]" in captured.out
        assert "CREATE TABLE IF NOT EXISTS" in captured.out
        assert "idempotency_records" in captured.out


@pytest.mark.unit
class TestProvisionHappyPath:
    """Provision path: connects, executes DDL inside transaction, closes."""

    def test_provision_executes_all_statements(self):
        mod = _load_script()

        mock_conn = AsyncMock()
        mock_conn.transaction = MagicMock(return_value=AsyncMock())
        mock_conn.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
        mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)

        import asyncio

        with patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)):
            asyncio.run(
                mod.provision(
                    "postgresql://postgres:pw@localhost:5436/omniintelligence"
                )
            )

        # execute called once per statement
        expected_count = len(mod.CROSS_REPO_TABLES["idempotency_records"])
        assert mock_conn.execute.call_count == expected_count
        mock_conn.close.assert_called_once()


@pytest.mark.unit
class TestProvisionErrorHandling:
    """Error handling: SystemExit on connection failure."""

    def test_exits_on_connection_error(self):
        mod = _load_script()

        import asyncio

        with (
            patch(
                "asyncpg.connect",
                new=AsyncMock(side_effect=OSError("connection refused")),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            asyncio.run(
                mod.provision(
                    "postgresql://postgres:pw@localhost:5436/omniintelligence"
                )
            )

        assert exc_info.value.code == 1
