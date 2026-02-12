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
        """UndefinedTableError (table doesn't exist) raises DbOwnershipMissingError."""
        from asyncpg.exceptions import UndefinedTableError

        pool = _make_mock_pool(
            side_effect=UndefinedTableError(
                'relation "public.db_metadata" does not exist'
            )
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
    async def test_transient_connection_error_propagates(self) -> None:
        """Non-table errors (e.g. connection failures) propagate with original type."""
        pool = _make_mock_pool(side_effect=ConnectionRefusedError("connection refused"))
        with pytest.raises(ConnectionRefusedError):
            await validate_db_ownership(
                pool=pool,
                expected_owner="omnibase_infra",
                correlation_id=uuid4(),
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
    async def test_empty_expected_owner_raises_value_error(self) -> None:
        """Empty-string expected_owner raises ValueError before any DB query."""
        pool = _make_mock_pool(row={"owner_service": ""})
        with pytest.raises(ValueError, match="non-empty"):
            await validate_db_ownership(
                pool=pool,
                expected_owner="",
                correlation_id=uuid4(),
            )

    @pytest.mark.asyncio
    async def test_whitespace_expected_owner_raises_value_error(self) -> None:
        """Whitespace-only expected_owner raises ValueError before any DB query."""
        pool = _make_mock_pool(row={"owner_service": ""})
        with pytest.raises(ValueError, match="non-empty"):
            await validate_db_ownership(
                pool=pool,
                expected_owner="   ",
                correlation_id=uuid4(),
            )

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

    @pytest.mark.asyncio
    async def test_long_owner_service_is_truncated(self) -> None:
        """Very long owner_service from DB is truncated in error message."""
        long_owner = "x" * 200
        pool = _make_mock_pool(row={"owner_service": long_owner})
        with pytest.raises(DbOwnershipMismatchError) as exc_info:
            await validate_db_ownership(
                pool=pool,
                expected_owner="omnibase_infra",
                correlation_id=uuid4(),
            )
        # actual_owner should be truncated to 64 chars
        assert len(exc_info.value.actual_owner) == 64

    @pytest.mark.asyncio
    async def test_special_chars_in_owner_service(self) -> None:
        """Special characters in owner_service don't cause crashes."""
        special_owner = "evil'; DROP TABLE --\n\x00\t"
        pool = _make_mock_pool(row={"owner_service": special_owner})
        with pytest.raises(DbOwnershipMismatchError) as exc_info:
            await validate_db_ownership(
                pool=pool,
                expected_owner="omnibase_infra",
                correlation_id=uuid4(),
            )
        assert exc_info.value.expected_owner == "omnibase_infra"


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


class TestPluginOwnershipPropagation:
    """Tests that DB ownership errors propagate out of plugin.initialize().

    PluginRegistration.initialize() has a broad ``except Exception`` that
    converts failures into ``ModelDomainPluginResult.failed()``. DB ownership
    errors are P0 hard gates that MUST escape this handler so the kernel
    terminates. These tests confirm the re-raise path works.
    """

    @pytest.mark.asyncio
    async def test_mismatch_error_propagates_from_plugin(self) -> None:
        """DbOwnershipMismatchError escapes plugin.initialize() (not swallowed)."""
        from unittest.mock import patch

        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            PluginRegistration,
        )
        from omnibase_infra.runtime.protocol_domain_plugin import (
            ModelDomainPluginConfig,
        )

        plugin = PluginRegistration()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        # Simulate: pool creation succeeds, but ownership validation raises mismatch
        mismatch = DbOwnershipMismatchError(
            "wrong owner",
            expected_owner="omnibase_infra",
            actual_owner="omniclaude",
        )

        with (
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": "postgresql://x/y"}),
            patch(
                "omnibase_infra.nodes.node_registration_orchestrator.plugin.ModelPostgresPoolConfig.validate_dsn",
            ),
            patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool,
            patch(
                "omnibase_infra.nodes.node_registration_orchestrator.plugin.validate_db_ownership",
                new_callable=AsyncMock,
                side_effect=mismatch,
            ),
        ):
            mock_create_pool.return_value = MagicMock()

            with pytest.raises(DbOwnershipMismatchError) as exc_info:
                await plugin.initialize(config)

            assert exc_info.value.expected_owner == "omnibase_infra"
            assert exc_info.value.actual_owner == "omniclaude"

    @pytest.mark.asyncio
    async def test_missing_error_propagates_from_plugin(self) -> None:
        """DbOwnershipMissingError escapes plugin.initialize() (not swallowed)."""
        from unittest.mock import patch

        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            PluginRegistration,
        )
        from omnibase_infra.runtime.protocol_domain_plugin import (
            ModelDomainPluginConfig,
        )

        plugin = PluginRegistration()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        missing = DbOwnershipMissingError(
            "table not found",
            expected_owner="omnibase_infra",
        )

        with (
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": "postgresql://x/y"}),
            patch(
                "omnibase_infra.nodes.node_registration_orchestrator.plugin.ModelPostgresPoolConfig.validate_dsn",
            ),
            patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool,
            patch(
                "omnibase_infra.nodes.node_registration_orchestrator.plugin.validate_db_ownership",
                new_callable=AsyncMock,
                side_effect=missing,
            ),
        ):
            mock_create_pool.return_value = MagicMock()

            with pytest.raises(DbOwnershipMissingError) as exc_info:
                await plugin.initialize(config)

            assert exc_info.value.expected_owner == "omnibase_infra"
