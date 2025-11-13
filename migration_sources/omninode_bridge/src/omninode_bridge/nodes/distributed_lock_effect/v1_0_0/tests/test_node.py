#!/usr/bin/env python3
"""
Unit tests for NodeDistributedLockEffect.

Tests all lock operations with mocked PostgreSQL:
- acquire_lock
- release_lock
- extend_lock
- query_lock
- cleanup_expired_locks
"""

import time
from datetime import UTC, datetime

import pytest
from omnibase_core import ModelOnexError

from ..models.enum_lock_operation import EnumLockOperation
from ..models.enum_lock_status import EnumLockStatus


@pytest.mark.asyncio
class TestNodeDistributedLockEffect:
    """Test suite for NodeDistributedLockEffect."""

    async def test_initialize(self, node):
        """Test node initialization with PostgreSQL pool."""
        assert node is not None
        assert node._pool is not None
        assert node.config.postgres_database == "test_db"

    async def test_acquire_lock_success(
        self, node, acquire_contract, mock_postgres_pool
    ):
        """Test successful lock acquisition."""
        # Mock successful lock acquisition
        mock_conn = mock_postgres_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = [
            None,  # No existing lock
            {  # Inserted lock
                "lock_id": "test_lock",
                "owner_id": "test_owner",
                "acquired_at": int(time.time()),
                "expires_at": int(time.time() + 30),
                "lease_duration": 30.0,
                "metadata": {},
                "status": "acquired",
                "acquisition_count": 1,
                "extension_count": 0,
            },
        ]

        # Execute
        response = await node.execute_effect(acquire_contract)

        # Assertions
        assert response.success is True
        assert response.operation == EnumLockOperation.ACQUIRE
        assert response.lock_info is not None
        assert response.lock_info.lock_name == "test_lock"
        assert response.lock_info.owner_id == "test_owner"
        assert response.lock_info.status == EnumLockStatus.ACQUIRED
        assert response.acquire_attempts == 1
        assert response.duration_ms > 0

    async def test_acquire_lock_already_held(
        self, node, acquire_contract, mock_postgres_pool
    ):
        """Test lock acquisition when lock is already held."""
        # Mock existing lock held by another owner
        mock_conn = mock_postgres_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = {
            "lock_id": "test_lock",
            "owner_id": "other_owner",
            "acquired_at": int(time.time()),
            "expires_at": int(time.time() + 30),
            "lease_duration": 30.0,
            "metadata": {},
            "status": "acquired",
            "acquisition_count": 1,
            "extension_count": 0,
        }

        # Execute
        response = await node.execute_effect(acquire_contract)

        # Assertions
        assert response.success is False
        assert response.operation == EnumLockOperation.ACQUIRE
        assert response.error_code == "LOCK_TIMEOUT"
        assert response.acquire_attempts == 3  # Max attempts

    async def test_release_lock_success(
        self, node, release_contract, mock_postgres_pool
    ):
        """Test successful lock release."""
        # Mock successful release
        mock_conn = mock_postgres_pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.return_value = "UPDATE 1"

        # Execute
        response = await node.execute_effect(release_contract)

        # Assertions
        assert response.success is True
        assert response.operation == EnumLockOperation.RELEASE
        assert response.duration_ms > 0

    async def test_release_lock_not_held(
        self, node, release_contract, mock_postgres_pool
    ):
        """Test lock release when lock is not held."""
        # Mock no lock found
        mock_conn = mock_postgres_pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.return_value = "UPDATE 0"

        # Execute
        response = await node.execute_effect(release_contract)

        # Assertions
        assert response.success is False
        assert response.error_code == "LOCK_NOT_HELD"

    async def test_extend_lock_success(self, node, extend_contract, mock_postgres_pool):
        """Test successful lock extension."""
        # Mock successful extension
        mock_conn = mock_postgres_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = {
            "lock_id": "test_lock",
            "owner_id": "test_owner",
            "acquired_at": int(time.time()),
            "expires_at": int(time.time() + 60),  # Extended
            "lease_duration": 30.0,
            "metadata": {},
            "status": "acquired",
            "acquisition_count": 1,
            "extension_count": 1,
        }

        # Execute
        response = await node.execute_effect(extend_contract)

        # Assertions
        assert response.success is True
        assert response.operation == EnumLockOperation.EXTEND
        assert response.lock_info is not None
        assert response.lock_info.extension_count == 1

    async def test_query_lock_found(self, node, query_contract, mock_postgres_pool):
        """Test lock query when lock exists."""
        # Mock existing lock
        mock_conn = mock_postgres_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = {
            "lock_id": "test_lock",
            "owner_id": "test_owner",
            "acquired_at": int(time.time()),
            "expires_at": int(time.time() + 30),
            "lease_duration": 30.0,
            "metadata": {},
            "status": "acquired",
            "acquisition_count": 1,
            "extension_count": 0,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }

        # Execute
        response = await node.execute_effect(query_contract)

        # Assertions
        assert response.success is True
        assert response.operation == EnumLockOperation.QUERY
        assert response.lock_info is not None
        assert response.lock_info.lock_name == "test_lock"

    async def test_query_lock_not_found(self, node, query_contract, mock_postgres_pool):
        """Test lock query when lock doesn't exist."""
        # Mock no lock found
        mock_conn = mock_postgres_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = None

        # Execute
        response = await node.execute_effect(query_contract)

        # Assertions
        assert response.success is False
        assert response.error_code == "LOCK_NOT_FOUND"

    async def test_cleanup_locks(self, node, cleanup_contract, mock_postgres_pool):
        """Test cleanup of expired locks."""
        # Mock cleanup results
        mock_conn = mock_postgres_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = [
            {"lock_id": "expired_lock_1"},
            {"lock_id": "expired_lock_2"},
        ]
        mock_conn.execute.return_value = "DELETE 2"

        # Execute
        response = await node.execute_effect(cleanup_contract)

        # Assertions
        assert response.success is True
        assert response.operation == EnumLockOperation.CLEANUP
        assert response.cleaned_count == 2
        assert len(response.cleaned_lock_names) == 2

    async def test_get_metrics(self, node):
        """Test metrics collection."""
        # Get initial metrics
        metrics = node.get_metrics()

        # Assertions
        assert "total_acquires" in metrics
        assert "total_releases" in metrics
        assert "total_extends" in metrics
        assert "total_queries" in metrics
        assert "total_cleanups" in metrics
        assert "success_rate" in metrics
        assert metrics["success_rate"] >= 0.0
        assert metrics["success_rate"] <= 1.0

    async def test_shutdown(self, node):
        """Test node shutdown."""
        # Shutdown
        await node.shutdown()

        # Verify cleanup task is cancelled
        assert node._cleanup_task.cancelled()


@pytest.mark.asyncio
class TestLockOperationErrors:
    """Test error handling in lock operations."""

    async def test_database_error_acquire(
        self, node, acquire_contract, mock_postgres_pool
    ):
        """Test database error during lock acquisition."""
        from asyncpg.exceptions import PostgresError

        # Mock database error
        mock_conn = mock_postgres_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.side_effect = PostgresError("Connection failed")

        # Execute
        response = await node.execute_effect(acquire_contract)

        # Assertions
        assert response.success is False
        assert response.error_code == "DATABASE_ERROR"

    async def test_invalid_operation(self, node, correlation_id):
        """Test invalid lock operation."""
        from omnibase_core.models.contracts.model_contract_effect import (
            ModelContractEffect,
        )

        # Create contract with invalid operation
        contract = ModelContractEffect(
            name="invalid_op",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Invalid operation",
            node_type="EFFECT",
            input_model="ModelDistributedLockRequest",
            output_model="ModelDistributedLockResponse",
            input_state={
                "operation": "invalid_operation",
                "lock_name": "test_lock",
            },
            correlation_id=correlation_id,
        )

        # Execute and expect error
        with pytest.raises(ModelOnexError) as exc_info:
            await node.execute_effect(contract)

        assert "Unknown lock operation" in str(exc_info.value)


__all__ = ["TestNodeDistributedLockEffect", "TestLockOperationErrors"]
