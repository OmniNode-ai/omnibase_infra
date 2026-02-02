# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for PostgreSQL handlers in NodeContractPersistenceEffect.

Tests validate for each handler:
- Successful operation returns ModelBackendResult with success=True
- TimeoutError returns success=False with appropriate TIMEOUT error_code
- InfraConnectionError returns success=False with appropriate CONNECTION error_code
- InfraAuthenticationError returns success=False with appropriate AUTH error_code

Handlers tested:
- HandlerPostgresContractUpsert
- HandlerPostgresTopicUpdate
- HandlerPostgresMarkStale
- HandlerPostgresHeartbeat
- HandlerPostgresDeactivate
- HandlerPostgresCleanupTopics

Related Tickets:
    - OMN-1845: NodeContractPersistenceEffect implementation
    - OMN-1653: ContractRegistryReducer implementation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
)
from omnibase_infra.nodes.contract_registry_reducer.models import (
    ModelPayloadCleanupTopicReferences,
    ModelPayloadDeactivateContract,
    ModelPayloadMarkStale,
    ModelPayloadUpdateHeartbeat,
    ModelPayloadUpdateTopic,
    ModelPayloadUpsertContract,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_cleanup_topics import (
    HandlerPostgresCleanupTopics,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_contract_upsert import (
    HandlerPostgresContractUpsert,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_deactivate import (
    HandlerPostgresDeactivate,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_heartbeat import (
    HandlerPostgresHeartbeat,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_mark_stale import (
    HandlerPostgresMarkStale,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_topic_update import (
    HandlerPostgresTopicUpdate,
)

# Fixed test time for deterministic testing
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Mock Pool Fixtures
# =============================================================================


def create_mock_pool_with_acquire() -> MagicMock:
    """Create a mock asyncpg pool that uses acquire() context manager.

    Used by: HandlerPostgresContractUpsert, HandlerPostgresTopicUpdate,
    HandlerPostgresDeactivate, HandlerPostgresCleanupTopics
    """
    pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock()
    mock_conn.fetchval = AsyncMock()
    mock_conn.execute = AsyncMock()

    # Create an async context manager for acquire()
    mock_acquire = AsyncMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=mock_acquire)

    return pool


def create_mock_pool_direct_execute() -> MagicMock:
    """Create a mock asyncpg pool that uses execute() directly on pool.

    Used by: HandlerPostgresMarkStale, HandlerPostgresHeartbeat
    """
    pool = MagicMock()
    pool.execute = AsyncMock()
    return pool


# =============================================================================
# Payload Factories
# =============================================================================


def create_upsert_contract_payload() -> ModelPayloadUpsertContract:
    """Create a test payload for contract upsert."""
    return ModelPayloadUpsertContract(
        correlation_id=uuid4(),
        contract_id="test-node:1.0.0",
        node_name="test-node",
        version_major=1,
        version_minor=0,
        version_patch=0,
        contract_hash="abc123",
        contract_yaml={"name": "test-node", "version": "1.0.0"},
        is_active=True,
        registered_at=TEST_NOW,
        last_seen_at=TEST_NOW,
    )


def create_update_topic_payload() -> ModelPayloadUpdateTopic:
    """Create a test payload for topic update."""
    return ModelPayloadUpdateTopic(
        correlation_id=uuid4(),
        topic_suffix="{env}.onex.evt.test.topic.v1",
        direction="publish",
        contract_id="test-node:1.0.0",
        node_name="test-node",
        last_seen_at=TEST_NOW,
    )


def create_mark_stale_payload() -> ModelPayloadMarkStale:
    """Create a test payload for mark stale."""
    return ModelPayloadMarkStale(
        correlation_id=uuid4(),
        stale_cutoff=TEST_NOW,
        checked_at=TEST_NOW,
    )


def create_update_heartbeat_payload() -> ModelPayloadUpdateHeartbeat:
    """Create a test payload for heartbeat update."""
    return ModelPayloadUpdateHeartbeat(
        correlation_id=uuid4(),
        contract_id="test-node:1.0.0",
        node_name="test-node",
        last_seen_at=TEST_NOW,
    )


def create_deactivate_contract_payload() -> ModelPayloadDeactivateContract:
    """Create a test payload for contract deactivation."""
    return ModelPayloadDeactivateContract(
        correlation_id=uuid4(),
        contract_id="test-node:1.0.0",
        node_name="test-node",
        reason="shutdown",
        deactivated_at=TEST_NOW,
    )


def create_cleanup_topics_payload() -> ModelPayloadCleanupTopicReferences:
    """Create a test payload for topic cleanup."""
    return ModelPayloadCleanupTopicReferences(
        correlation_id=uuid4(),
        contract_id="test-node:1.0.0",
        node_name="test-node",
        cleaned_at=TEST_NOW,
    )


# =============================================================================
# HandlerPostgresContractUpsert Tests
# =============================================================================


class TestHandlerPostgresContractUpsertSuccess:
    """Test successful contract upsert operations."""

    @pytest.mark.asyncio
    async def test_contract_upsert_success(self) -> None:
        """Test successful contract upsert returns ModelBackendResult with success=True."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = {
            "contract_id": "test-node:1.0.0",
            "was_insert": True,
        }

        handler = HandlerPostgresContractUpsert(pool)
        payload = create_upsert_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0
        assert result.error is None
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_contract_upsert_update_case(self) -> None:
        """Test contract upsert for update (was_insert=False) returns success."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = {
            "contract_id": "test-node:1.0.0",
            "was_insert": False,
        }

        handler = HandlerPostgresContractUpsert(pool)
        payload = create_upsert_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_contract_upsert_no_result_returns_error(self) -> None:
        """Test contract upsert with no result returns error."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = None

        handler = HandlerPostgresContractUpsert(pool)
        payload = create_upsert_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_UPSERT_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id


class TestHandlerPostgresContractUpsertErrors:
    """Test contract upsert error handling."""

    @pytest.mark.asyncio
    async def test_contract_upsert_timeout_error(self) -> None:
        """Test TimeoutError returns success=False with TIMEOUT error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = TimeoutError("Operation timed out")

        handler = HandlerPostgresContractUpsert(pool)
        payload = create_upsert_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_TIMEOUT_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_contract_upsert_connection_error(self) -> None:
        """Test InfraConnectionError returns success=False with CONNECTION error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = InfraConnectionError("Connection refused")

        handler = HandlerPostgresContractUpsert(pool)
        payload = create_upsert_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_CONNECTION_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_contract_upsert_authentication_error(self) -> None:
        """Test InfraAuthenticationError returns success=False with AUTH error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = InfraAuthenticationError("Auth failed")

        handler = HandlerPostgresContractUpsert(pool)
        payload = create_upsert_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_AUTH_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_contract_upsert_generic_exception(self) -> None:
        """Test generic Exception returns success=False with UNKNOWN error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = RuntimeError("Unexpected error")

        handler = HandlerPostgresContractUpsert(pool)
        payload = create_upsert_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_UNKNOWN_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id


# =============================================================================
# HandlerPostgresTopicUpdate Tests
# =============================================================================


class TestHandlerPostgresTopicUpdateSuccess:
    """Test successful topic update operations."""

    @pytest.mark.asyncio
    async def test_topic_update_success(self) -> None:
        """Test successful topic update returns ModelBackendResult with success=True."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = {
            "topic_suffix": "onex.evt.test.topic.v1",
            "direction": "publish",
            "contract_ids": ["test-node:1.0.0"],
        }

        handler = HandlerPostgresTopicUpdate(pool)
        payload = create_update_topic_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_topic_update_no_result_returns_error(self) -> None:
        """Test topic update with no result returns error."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = None

        handler = HandlerPostgresTopicUpdate(pool)
        payload = create_update_topic_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_TOPIC_UPDATE_ERROR"
        assert result.backend_id == "postgres"


class TestHandlerPostgresTopicUpdateErrors:
    """Test topic update error handling."""

    @pytest.mark.asyncio
    async def test_topic_update_timeout_error(self) -> None:
        """Test TimeoutError returns success=False with TIMEOUT error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = TimeoutError("Operation timed out")

        handler = HandlerPostgresTopicUpdate(pool)
        payload = create_update_topic_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_TIMEOUT_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_topic_update_connection_error(self) -> None:
        """Test InfraConnectionError returns success=False with CONNECTION error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = InfraConnectionError("Connection refused")

        handler = HandlerPostgresTopicUpdate(pool)
        payload = create_update_topic_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_CONNECTION_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_topic_update_authentication_error(self) -> None:
        """Test InfraAuthenticationError returns success=False with AUTH error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = InfraAuthenticationError("Auth failed")

        handler = HandlerPostgresTopicUpdate(pool)
        payload = create_update_topic_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_AUTH_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_topic_update_generic_exception(self) -> None:
        """Test generic Exception returns success=False with UNKNOWN error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = RuntimeError("Unexpected error")

        handler = HandlerPostgresTopicUpdate(pool)
        payload = create_update_topic_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_UNKNOWN_ERROR"
        assert result.backend_id == "postgres"


# =============================================================================
# HandlerPostgresMarkStale Tests
# =============================================================================


class TestHandlerPostgresMarkStaleSuccess:
    """Test successful mark stale operations."""

    @pytest.mark.asyncio
    async def test_mark_stale_success(self) -> None:
        """Test successful mark stale returns ModelBackendResult with success=True."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.return_value = "UPDATE 5"

        handler = HandlerPostgresMarkStale(pool)
        payload = create_mark_stale_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_mark_stale_zero_rows_still_success(self) -> None:
        """Test mark stale with zero affected rows still returns success."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.return_value = "UPDATE 0"

        handler = HandlerPostgresMarkStale(pool)
        payload = create_mark_stale_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        assert result.backend_id == "postgres"


class TestHandlerPostgresMarkStaleErrors:
    """Test mark stale error handling."""

    @pytest.mark.asyncio
    async def test_mark_stale_timeout_error(self) -> None:
        """Test TimeoutError returns success=False with TIMEOUT error_code."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.side_effect = TimeoutError("Operation timed out")

        handler = HandlerPostgresMarkStale(pool)
        payload = create_mark_stale_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "MARK_STALE_TIMEOUT_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_mark_stale_connection_error(self) -> None:
        """Test InfraConnectionError returns success=False with CONNECTION error_code."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.side_effect = InfraConnectionError("Connection refused")

        handler = HandlerPostgresMarkStale(pool)
        payload = create_mark_stale_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "MARK_STALE_CONNECTION_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_mark_stale_authentication_error(self) -> None:
        """Test InfraAuthenticationError returns success=False with AUTH error_code."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.side_effect = InfraAuthenticationError("Auth failed")

        handler = HandlerPostgresMarkStale(pool)
        payload = create_mark_stale_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "MARK_STALE_AUTH_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_mark_stale_generic_exception(self) -> None:
        """Test generic Exception returns success=False with UNKNOWN error_code."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.side_effect = RuntimeError("Unexpected error")

        handler = HandlerPostgresMarkStale(pool)
        payload = create_mark_stale_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "MARK_STALE_UNKNOWN_ERROR"
        assert result.backend_id == "postgres"


# =============================================================================
# HandlerPostgresHeartbeat Tests
# =============================================================================


class TestHandlerPostgresHeartbeatSuccess:
    """Test successful heartbeat update operations."""

    @pytest.mark.asyncio
    async def test_heartbeat_success_row_found(self) -> None:
        """Test successful heartbeat returns ModelBackendResult with success=True."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.return_value = "UPDATE 1"

        handler = HandlerPostgresHeartbeat(pool)
        payload = create_update_heartbeat_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_heartbeat_success_no_row_found(self) -> None:
        """Test heartbeat with no row found still returns success (idempotent)."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.return_value = "UPDATE 0"

        handler = HandlerPostgresHeartbeat(pool)
        payload = create_update_heartbeat_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert - operation still succeeds even if no row found
        assert result.success is True
        assert result.backend_id == "postgres"


class TestHandlerPostgresHeartbeatErrors:
    """Test heartbeat error handling."""

    @pytest.mark.asyncio
    async def test_heartbeat_timeout_error(self) -> None:
        """Test TimeoutError returns success=False with TIMEOUT error_code."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.side_effect = TimeoutError("Operation timed out")

        handler = HandlerPostgresHeartbeat(pool)
        payload = create_update_heartbeat_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "HEARTBEAT_TIMEOUT_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_heartbeat_connection_error(self) -> None:
        """Test InfraConnectionError returns success=False with CONNECTION error_code."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.side_effect = InfraConnectionError("Connection refused")

        handler = HandlerPostgresHeartbeat(pool)
        payload = create_update_heartbeat_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "HEARTBEAT_CONNECTION_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_heartbeat_authentication_error(self) -> None:
        """Test InfraAuthenticationError returns success=False with AUTH error_code."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.side_effect = InfraAuthenticationError("Auth failed")

        handler = HandlerPostgresHeartbeat(pool)
        payload = create_update_heartbeat_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "HEARTBEAT_AUTH_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_heartbeat_generic_exception(self) -> None:
        """Test generic Exception returns success=False with UNKNOWN error_code."""
        # Arrange
        pool = create_mock_pool_direct_execute()
        pool.execute.side_effect = RuntimeError("Unexpected error")

        handler = HandlerPostgresHeartbeat(pool)
        payload = create_update_heartbeat_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "HEARTBEAT_UNKNOWN_ERROR"
        assert result.backend_id == "postgres"


# =============================================================================
# HandlerPostgresDeactivate Tests
# =============================================================================


class TestHandlerPostgresDeactivateSuccess:
    """Test successful deactivate operations."""

    @pytest.mark.asyncio
    async def test_deactivate_success_row_found(self) -> None:
        """Test successful deactivate returns ModelBackendResult with success=True."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = "test-node:1.0.0"

        handler = HandlerPostgresDeactivate(pool)
        payload = create_deactivate_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_deactivate_success_no_row_found(self) -> None:
        """Test deactivate with no row found returns success with message (idempotent)."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = None

        handler = HandlerPostgresDeactivate(pool)
        payload = create_deactivate_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert - operation still succeeds even if no row found (idempotent)
        assert result.success is True
        assert result.backend_id == "postgres"
        # error field contains "not found" message but success is still True
        assert result.error is not None
        assert "not found" in result.error


class TestHandlerPostgresDeactivateErrors:
    """Test deactivate error handling."""

    @pytest.mark.asyncio
    async def test_deactivate_timeout_error(self) -> None:
        """Test TimeoutError returns success=False with TIMEOUT error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.side_effect = TimeoutError("Operation timed out")

        handler = HandlerPostgresDeactivate(pool)
        payload = create_deactivate_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_DEACTIVATE_TIMEOUT_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_deactivate_connection_error(self) -> None:
        """Test InfraConnectionError returns success=False with CONNECTION error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.side_effect = InfraConnectionError("Connection refused")

        handler = HandlerPostgresDeactivate(pool)
        payload = create_deactivate_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_DEACTIVATE_CONNECTION_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_deactivate_authentication_error(self) -> None:
        """Test InfraAuthenticationError returns success=False with AUTH error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.side_effect = InfraAuthenticationError("Auth failed")

        handler = HandlerPostgresDeactivate(pool)
        payload = create_deactivate_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_DEACTIVATE_AUTH_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_deactivate_generic_exception(self) -> None:
        """Test generic Exception returns success=False with UNKNOWN error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.side_effect = RuntimeError("Unexpected error")

        handler = HandlerPostgresDeactivate(pool)
        payload = create_deactivate_contract_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_DEACTIVATE_UNKNOWN_ERROR"
        assert result.backend_id == "postgres"


# =============================================================================
# HandlerPostgresCleanupTopics Tests
# =============================================================================


class TestHandlerPostgresCleanupTopicsSuccess:
    """Test successful cleanup topics operations."""

    @pytest.mark.asyncio
    async def test_cleanup_topics_success(self) -> None:
        """Test successful cleanup returns ModelBackendResult with success=True."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.return_value = "UPDATE 3"

        handler = HandlerPostgresCleanupTopics(pool)
        payload = create_cleanup_topics_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_cleanup_topics_zero_rows_still_success(self) -> None:
        """Test cleanup with zero affected rows still returns success (idempotent)."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.return_value = "UPDATE 0"

        handler = HandlerPostgresCleanupTopics(pool)
        payload = create_cleanup_topics_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        assert result.backend_id == "postgres"


class TestHandlerPostgresCleanupTopicsErrors:
    """Test cleanup topics error handling."""

    @pytest.mark.asyncio
    async def test_cleanup_topics_timeout_error(self) -> None:
        """Test TimeoutError returns success=False with TIMEOUT error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.side_effect = TimeoutError("Operation timed out")

        handler = HandlerPostgresCleanupTopics(pool)
        payload = create_cleanup_topics_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_CLEANUP_TOPICS_TIMEOUT_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_cleanup_topics_connection_error(self) -> None:
        """Test InfraConnectionError returns success=False with CONNECTION error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.side_effect = InfraConnectionError("Connection refused")

        handler = HandlerPostgresCleanupTopics(pool)
        payload = create_cleanup_topics_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_CLEANUP_TOPICS_CONNECTION_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_cleanup_topics_authentication_error(self) -> None:
        """Test InfraAuthenticationError returns success=False with AUTH error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.side_effect = InfraAuthenticationError("Auth failed")

        handler = HandlerPostgresCleanupTopics(pool)
        payload = create_cleanup_topics_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_CLEANUP_TOPICS_AUTH_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_cleanup_topics_generic_exception(self) -> None:
        """Test generic Exception returns success=False with UNKNOWN error_code."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.side_effect = RuntimeError("Unexpected error")

        handler = HandlerPostgresCleanupTopics(pool)
        payload = create_cleanup_topics_payload()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is False
        assert result.error_code == "POSTGRES_CLEANUP_TOPICS_UNKNOWN_ERROR"
        assert result.backend_id == "postgres"


# =============================================================================
# Additional Tests: Topic Normalization
# =============================================================================


class TestTopicNormalization:
    """Test topic suffix normalization in HandlerPostgresTopicUpdate."""

    @pytest.mark.asyncio
    async def test_topic_normalization_strips_env_placeholder(self) -> None:
        """Test that {env}. prefix is stripped from topic suffix."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = {
            "topic_suffix": "onex.evt.test.topic.v1",
            "direction": "publish",
            "contract_ids": ["test-node:1.0.0"],
        }

        handler = HandlerPostgresTopicUpdate(pool)
        payload = ModelPayloadUpdateTopic(
            correlation_id=uuid4(),
            topic_suffix="{env}.onex.evt.test.topic.v1",
            direction="publish",
            contract_id="test-node:1.0.0",
            node_name="test-node",
            last_seen_at=TEST_NOW,
        )
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        # Verify the normalized topic was passed to the query
        call_args = mock_conn.fetchrow.call_args
        # First positional arg after SQL is the topic suffix
        normalized_topic = call_args[0][1]
        assert normalized_topic == "onex.evt.test.topic.v1"

    @pytest.mark.asyncio
    async def test_topic_normalization_strips_dev_prefix(self) -> None:
        """Test that dev. prefix is stripped from topic suffix."""
        # Arrange
        pool = create_mock_pool_with_acquire()
        mock_conn = pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = {
            "topic_suffix": "onex.evt.test.topic.v1",
            "direction": "subscribe",
            "contract_ids": ["test-node:1.0.0"],
        }

        handler = HandlerPostgresTopicUpdate(pool)
        payload = ModelPayloadUpdateTopic(
            correlation_id=uuid4(),
            topic_suffix="dev.onex.evt.test.topic.v1",
            direction="subscribe",
            contract_id="test-node:1.0.0",
            node_name="test-node",
            last_seen_at=TEST_NOW,
        )
        correlation_id = uuid4()

        # Act
        result = await handler.handle(payload, correlation_id)

        # Assert
        assert result.success is True
        call_args = mock_conn.fetchrow.call_args
        normalized_topic = call_args[0][1]
        assert normalized_topic == "onex.evt.test.topic.v1"


__all__: list[str] = [
    "TestHandlerPostgresContractUpsertSuccess",
    "TestHandlerPostgresContractUpsertErrors",
    "TestHandlerPostgresTopicUpdateSuccess",
    "TestHandlerPostgresTopicUpdateErrors",
    "TestHandlerPostgresMarkStaleSuccess",
    "TestHandlerPostgresMarkStaleErrors",
    "TestHandlerPostgresHeartbeatSuccess",
    "TestHandlerPostgresHeartbeatErrors",
    "TestHandlerPostgresDeactivateSuccess",
    "TestHandlerPostgresDeactivateErrors",
    "TestHandlerPostgresCleanupTopicsSuccess",
    "TestHandlerPostgresCleanupTopicsErrors",
    "TestTopicNormalization",
]
