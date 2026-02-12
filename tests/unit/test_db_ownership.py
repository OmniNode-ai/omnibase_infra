# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for DB ownership validation (OMN-2085)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors.error_db_ownership import (
    DbOwnershipMismatchError,
    DbOwnershipMissingError,
)
from omnibase_infra.runtime.util_db_ownership import validate_db_ownership


def _make_mock_pool(
    *, row: dict | None = None, side_effect: Exception | None = None
) -> MagicMock:
    """Create a mock asyncpg.Pool that returns the given row from fetchrow."""
    pool = MagicMock()
    conn = AsyncMock()

    if side_effect is not None:
        conn.fetchrow = AsyncMock(side_effect=side_effect)
    else:
        conn.fetchrow = AsyncMock(return_value=row)

    # asyncpg pool.acquire() returns an async context manager
    acm = AsyncMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=acm)

    return pool


class TestValidateDbOwnership:
    """Tests for validate_db_ownership()."""

    @pytest.mark.asyncio
    async def test_ownership_match(self) -> None:
        """Happy path: owner_service matches expected_owner."""
        pool = _make_mock_pool(row={"owner_service": "omnibase_infra"})
        # Should not raise
        await validate_db_ownership(
            pool=pool,
            expected_owner="omnibase_infra",
            correlation_id=uuid4(),
        )

    @pytest.mark.asyncio
    async def test_ownership_mismatch_raises(self) -> None:
        """Mismatch between expected and actual owner raises DbOwnershipMismatchError."""
        pool = _make_mock_pool(row={"owner_service": "omniclaude"})
        with pytest.raises(DbOwnershipMismatchError) as exc_info:
            await validate_db_ownership(
                pool=pool,
                expected_owner="omnibase_infra",
                correlation_id=uuid4(),
            )
        assert exc_info.value.expected_owner == "omnibase_infra"
        assert exc_info.value.actual_owner == "omniclaude"
        assert "omnibase_infra" in str(exc_info.value)
        assert "omniclaude" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_table_raises(self) -> None:
        """Query failure (e.g. table doesn't exist) raises DbOwnershipMissingError."""
        pool = _make_mock_pool(
            side_effect=Exception('relation "public.db_metadata" does not exist')
        )
        with pytest.raises(DbOwnershipMissingError) as exc_info:
            await validate_db_ownership(
                pool=pool,
                expected_owner="omnibase_infra",
                correlation_id=uuid4(),
            )
        assert exc_info.value.expected_owner == "omnibase_infra"
        assert (
            "run migrations" in str(exc_info.value).lower()
            or "hint" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_empty_table_raises(self) -> None:
        """Table exists but no rows raises DbOwnershipMissingError."""
        pool = _make_mock_pool(row=None)
        with pytest.raises(DbOwnershipMissingError) as exc_info:
            await validate_db_ownership(
                pool=pool,
                expected_owner="omnibase_infra",
                correlation_id=uuid4(),
            )
        assert exc_info.value.expected_owner == "omnibase_infra"
        assert "no rows" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_auto_generates_correlation_id(self) -> None:
        """correlation_id is auto-generated when not provided."""
        pool = _make_mock_pool(row={"owner_service": "omnibase_infra"})
        # Should not raise -- correlation_id defaults to None -> auto-generated
        await validate_db_ownership(
            pool=pool,
            expected_owner="omnibase_infra",
        )

    @pytest.mark.asyncio
    async def test_mismatch_error_has_hint(self) -> None:
        """Error message includes actionable hint for operators."""
        pool = _make_mock_pool(row={"owner_service": "wrong_service"})
        with pytest.raises(DbOwnershipMismatchError) as exc_info:
            await validate_db_ownership(
                pool=pool,
                expected_owner="omnibase_infra",
            )
        msg = str(exc_info.value)
        assert "hint" in msg.lower() or "OMNIBASE_INFRA_DB_URL" in msg


class TestDbOwnershipErrorTypes:
    """Tests for error type hierarchy and attributes."""

    def test_mismatch_is_runtime_host_error(self) -> None:
        """DbOwnershipMismatchError extends RuntimeHostError."""
        from omnibase_infra.errors.error_infra import RuntimeHostError

        err = DbOwnershipMismatchError(
            "test",
            expected_owner="a",
            actual_owner="b",
        )
        assert isinstance(err, RuntimeHostError)

    def test_missing_is_runtime_host_error(self) -> None:
        """DbOwnershipMissingError extends RuntimeHostError."""
        from omnibase_infra.errors.error_infra import RuntimeHostError

        err = DbOwnershipMissingError(
            "test",
            expected_owner="a",
        )
        assert isinstance(err, RuntimeHostError)

    def test_mismatch_attributes(self) -> None:
        """DbOwnershipMismatchError exposes expected_owner and actual_owner."""
        err = DbOwnershipMismatchError(
            "msg",
            expected_owner="omnibase_infra",
            actual_owner="omniclaude",
        )
        assert err.expected_owner == "omnibase_infra"
        assert err.actual_owner == "omniclaude"

    def test_missing_attributes(self) -> None:
        """DbOwnershipMissingError exposes expected_owner."""
        err = DbOwnershipMissingError(
            "msg",
            expected_owner="omnibase_infra",
        )
        assert err.expected_owner == "omnibase_infra"
