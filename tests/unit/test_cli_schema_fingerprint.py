# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the schema fingerprint CLI entry point (OMN-2087).

Tests cover the ``_main()``, ``_cli_stamp()``, and ``_cli_verify()`` functions
in ``util_schema_fingerprint.py``.  All database interactions are mocked --
these are pure unit tests with no live infrastructure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.runtime.model_schema_fingerprint_result import (
    ModelSchemaFingerprintResult,
)
from omnibase_infra.runtime.util_schema_fingerprint import (
    _STAMP_QUERY,
    _cli_stamp,
    _cli_verify,
    _main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_DSN = "postgresql://user:pass@localhost:5432/testdb"

_FAKE_FINGERPRINT_RESULT = ModelSchemaFingerprintResult(
    fingerprint="a" * 64,
    table_count=3,
    column_count=10,
    constraint_count=5,
    per_table_hashes=(("t1", "h1"), ("t2", "h2"), ("t3", "h3")),
)


def _make_mock_pool(
    *,
    execute: AsyncMock | None = None,
) -> MagicMock:
    """Build a mock asyncpg pool with an acquirable connection."""
    pool = MagicMock()
    conn = AsyncMock()
    if execute is not None:
        conn.execute = execute
    else:
        conn.execute = AsyncMock()

    acm = AsyncMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=acm)
    pool.close = AsyncMock()
    return pool


# ---------------------------------------------------------------------------
# TestMain -- argparse / env-var gate
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for _main() argument parsing and environment checks."""

    def test_no_args_exits_nonzero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Calling _main() with no subcommand prints help and exits 1."""
        with (
            patch("sys.argv", ["prog"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main()

        assert exc_info.value.code == 1

    def test_missing_env_var_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_main() exits 1 when OMNIBASE_INFRA_DB_URL is not set."""
        with (
            patch("sys.argv", ["prog", "stamp"]),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "OMNIBASE_INFRA_DB_URL" in captured.err

    def test_stamp_subcommand_calls_cli_stamp(self) -> None:
        """'stamp' subcommand dispatches to _cli_stamp with dry_run=False."""
        with (
            patch("sys.argv", ["prog", "stamp"]),
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": _FAKE_DSN}, clear=True),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint._cli_stamp",
                new_callable=AsyncMock,
            ) as mock_stamp,
        ):
            _main()

        mock_stamp.assert_awaited_once_with(_FAKE_DSN, dry_run=False)

    def test_stamp_dry_run_passes_flag(self) -> None:
        """'stamp --dry-run' dispatches to _cli_stamp with dry_run=True."""
        with (
            patch("sys.argv", ["prog", "stamp", "--dry-run"]),
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": _FAKE_DSN}, clear=True),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint._cli_stamp",
                new_callable=AsyncMock,
            ) as mock_stamp,
        ):
            _main()

        mock_stamp.assert_awaited_once_with(_FAKE_DSN, dry_run=True)

    def test_verify_subcommand_calls_cli_verify(self) -> None:
        """'verify' subcommand dispatches to _cli_verify."""
        with (
            patch("sys.argv", ["prog", "verify"]),
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": _FAKE_DSN}, clear=True),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint._cli_verify",
                new_callable=AsyncMock,
            ) as mock_verify,
        ):
            _main()

        mock_verify.assert_awaited_once_with(_FAKE_DSN)

    def test_fingerprint_mismatch_exits_2(self) -> None:
        """SchemaFingerprintMismatchError causes exit code 2."""
        from omnibase_infra.errors.error_schema_fingerprint import (
            SchemaFingerprintMismatchError,
        )

        with (
            patch("sys.argv", ["prog", "verify"]),
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": _FAKE_DSN}, clear=True),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint._cli_verify",
                new_callable=AsyncMock,
                side_effect=SchemaFingerprintMismatchError(
                    "mismatch",
                    expected_fingerprint="aaa",
                    actual_fingerprint="bbb",
                ),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main()

        assert exc_info.value.code == 2

    def test_fingerprint_missing_exits_2(self) -> None:
        """SchemaFingerprintMissingError causes exit code 2."""
        from omnibase_infra.errors.error_schema_fingerprint import (
            SchemaFingerprintMissingError,
        )

        with (
            patch("sys.argv", ["prog", "verify"]),
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": _FAKE_DSN}, clear=True),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint._cli_verify",
                new_callable=AsyncMock,
                side_effect=SchemaFingerprintMissingError(
                    "missing",
                    expected_owner="test",
                ),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main()

        assert exc_info.value.code == 2

    def test_generic_exception_exits_1(self) -> None:
        """Unexpected exceptions cause exit code 1."""
        with (
            patch("sys.argv", ["prog", "stamp"]),
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": _FAKE_DSN}, clear=True),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint._cli_stamp",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main()

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# TestCliStamp -- _cli_stamp() behaviour
# ---------------------------------------------------------------------------


class TestCliStamp:
    """Tests for _cli_stamp() async function."""

    @pytest.mark.asyncio
    async def test_dry_run_skips_update(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With dry_run=True the UPDATE query is NOT executed."""
        mock_pool = _make_mock_pool()

        with (
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.compute_schema_fingerprint",
                new_callable=AsyncMock,
                return_value=_FAKE_FINGERPRINT_RESULT,
            ),
        ):
            await _cli_stamp(_FAKE_DSN, dry_run=True)

        # The connection's execute should NOT have been called (no UPDATE)
        conn_mock = mock_pool.acquire.return_value.__aenter__.return_value
        conn_mock.execute.assert_not_awaited()

        # Pool should still be closed
        mock_pool.close.assert_awaited_once()

        captured = capsys.readouterr()
        assert "--dry-run" in captured.out
        assert "a" * 64 in captured.out

    @pytest.mark.asyncio
    async def test_stamp_executes_update(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With dry_run=False the UPDATE query IS executed with the fingerprint."""
        mock_pool = _make_mock_pool(execute=AsyncMock(return_value="UPDATE 1"))

        with (
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.compute_schema_fingerprint",
                new_callable=AsyncMock,
                return_value=_FAKE_FINGERPRINT_RESULT,
            ),
        ):
            await _cli_stamp(_FAKE_DSN, dry_run=False)

        conn_mock = mock_pool.acquire.return_value.__aenter__.return_value
        conn_mock.execute.assert_awaited_once_with(
            _STAMP_QUERY, _FAKE_FINGERPRINT_RESULT.fingerprint
        )

        mock_pool.close.assert_awaited_once()

        captured = capsys.readouterr()
        assert "updated" in captured.out.lower()

    @pytest.mark.asyncio
    async def test_stamp_fails_on_zero_rows_updated(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When conn.execute returns 'UPDATE 0', _cli_stamp exits with code 1."""
        mock_pool = _make_mock_pool(execute=AsyncMock(return_value="UPDATE 0"))

        with (
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.compute_schema_fingerprint",
                new_callable=AsyncMock,
                return_value=_FAKE_FINGERPRINT_RESULT,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await _cli_stamp(_FAKE_DSN, dry_run=False)

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "No rows updated" in captured.out

        mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stamp_prints_stats(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_cli_stamp prints fingerprint, table_count, column_count, constraint_count."""
        mock_pool = _make_mock_pool()

        with (
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.compute_schema_fingerprint",
                new_callable=AsyncMock,
                return_value=_FAKE_FINGERPRINT_RESULT,
            ),
        ):
            await _cli_stamp(_FAKE_DSN, dry_run=True)

        captured = capsys.readouterr()
        assert "table_count: 3" in captured.out
        assert "column_count: 10" in captured.out
        assert "constraint_count: 5" in captured.out

    @pytest.mark.asyncio
    async def test_pool_closed_on_error(self) -> None:
        """Pool is closed even when compute_schema_fingerprint raises."""
        mock_pool = _make_mock_pool()

        with (
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.compute_schema_fingerprint",
                new_callable=AsyncMock,
                side_effect=RuntimeError("db error"),
            ),
            pytest.raises(RuntimeError, match="db error"),
        ):
            await _cli_stamp(_FAKE_DSN, dry_run=False)

        mock_pool.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestCliVerify -- _cli_verify() behaviour
# ---------------------------------------------------------------------------


class TestCliVerify:
    """Tests for _cli_verify() async function."""

    @pytest.mark.asyncio
    async def test_verify_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Successful validation prints OK message."""
        mock_pool = _make_mock_pool()

        with (
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.validate_schema_fingerprint",
                new_callable=AsyncMock,
            ),
        ):
            await _cli_verify(_FAKE_DSN)

        mock_pool.close.assert_awaited_once()

        captured = capsys.readouterr()
        assert "OK" in captured.out

    @pytest.mark.asyncio
    async def test_verify_calls_validate(self) -> None:
        """_cli_verify passes pool and manifest to validate_schema_fingerprint."""
        mock_pool = _make_mock_pool()

        with (
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.validate_schema_fingerprint",
                new_callable=AsyncMock,
            ) as mock_validate,
        ):
            await _cli_verify(_FAKE_DSN)

        mock_validate.assert_awaited_once()
        call_args = mock_validate.call_args
        # _cli_verify passes (pool, manifest) as positional args
        assert call_args[0][0] is mock_pool

    @pytest.mark.asyncio
    async def test_verify_pool_closed_on_error(self) -> None:
        """Pool is closed even when validate_schema_fingerprint raises."""
        from omnibase_infra.errors.error_schema_fingerprint import (
            SchemaFingerprintMismatchError,
        )

        mock_pool = _make_mock_pool()

        with (
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "omnibase_infra.runtime.util_schema_fingerprint.validate_schema_fingerprint",
                new_callable=AsyncMock,
                side_effect=SchemaFingerprintMismatchError(
                    "drift",
                    expected_fingerprint="aaa",
                    actual_fingerprint="bbb",
                ),
            ),
            pytest.raises(SchemaFingerprintMismatchError),
        ):
            await _cli_verify(_FAKE_DSN)

        mock_pool.close.assert_awaited_once()
