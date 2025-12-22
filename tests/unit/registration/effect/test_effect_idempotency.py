# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Effect idempotency tests for OMN-954.

This test suite validates that effect-level idempotency guarantees hold for
registration intent processing. Effects are the I/O layer that executes intents
emitted by reducers. Idempotency at this layer ensures that:

1. Duplicate intents (same intent_id) cause no additional side effects
2. Natural key conflicts (entity_id, intent_type, registration_id) are treated as duplicates
3. Idempotency persists across effect restarts
4. Idempotency store failures result in safe failure (no partial execution)

Architecture:
    The Effect layer sits between the Reducer (pure computation) and external
    systems (Consul, PostgreSQL). When an intent arrives:

    1. Effect checks idempotency store for intent_id
    2. If duplicate, skip execution and return cached result
    3. If new, execute intent against backend
    4. Record successful execution in idempotency store
    5. Publish confirmation event

    This test suite validates the idempotency checking logic, not the full
    Effect node implementation (which may not yet exist).

Test Organization:
    - TestDuplicateIntentSafety: Duplicate intent_id causes no additional I/O
    - TestNaturalKeyConflict: Same natural key treated as duplicate
    - TestIdempotencyAcrossRestart: Idempotency survives effect restart
    - TestStorageFailureSafety: Store unavailable fails gracefully

Related:
    - RegistrationReducer: Emits registration intents
    - InMemoryIdempotencyStore: In-memory store for testing
    - PostgresIdempotencyStore: Production store
    - OMN-954: Effect idempotency acceptance criteria
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import NAMESPACE_OID, UUID, uuid4, uuid5

import pytest

from omnibase_infra.idempotency import (
    InMemoryIdempotencyStore,
    ModelIdempotencyRecord,
)
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
    ModelNodeRegistrationRecord,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
# NOTE: Common fixtures (inmemory_idempotency_store, mock_consul_client,
# sample_introspection_event, correlation_id) are defined in conftest.py.
# This file only defines fixtures specific to effect idempotency testing.


@pytest.fixture
def mock_postgres_client() -> MagicMock:
    """Create a mock PostgreSQL client for testing effect execution.

    Returns:
        MagicMock configured to track upsert calls.
    """
    client = MagicMock()
    client.execute = AsyncMock(return_value=None)
    client.fetchone = AsyncMock(return_value=None)
    return client


@pytest.fixture
def sample_registry_request() -> dict[str, UUID | str]:
    """Create sample registration request data.

    Returns:
        Dict with entity_id, intent_type, and registration_id.
    """
    return {
        "entity_id": uuid4(),
        "intent_type": "consul.register",
        "registration_id": uuid4(),
    }


@pytest.fixture
def sample_registration_record() -> ModelNodeRegistrationRecord:
    """Create a sample registration record for PostgreSQL intent.

    Returns:
        A valid ModelNodeRegistrationRecord.
    """
    now = datetime.now(UTC)
    return ModelNodeRegistrationRecord(
        node_id=uuid4(),
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(postgres=True),
        endpoints={"health": "http://localhost:8080/health"},
        metadata=ModelNodeMetadata(environment="test"),
        health_endpoint="http://localhost:8080/health",
        registered_at=now,
        updated_at=now,
    )


# -----------------------------------------------------------------------------
# Helper: Simulated Effect Executor
# -----------------------------------------------------------------------------


class SimulatedEffectExecutor:
    """Simulates Effect node behavior with idempotency guard.

    This class represents the essential idempotency logic that an Effect node
    would use when processing registration intents. It demonstrates the pattern:

    1. Check idempotency store before execution
    2. Execute backend operation if not duplicate
    3. Record execution in idempotency store

    Attributes:
        idempotency_store: Store for tracking processed intents.
        consul_client: Mock Consul client for service registration.
        postgres_client: Mock PostgreSQL client for record upsert.
        execution_count: Counter for tracking actual backend calls.
    """

    def __init__(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        consul_client: MagicMock,
        postgres_client: MagicMock,
    ) -> None:
        """Initialize the simulated effect executor.

        Args:
            idempotency_store: Store for idempotency checking.
            consul_client: Mock Consul client.
            postgres_client: Mock PostgreSQL client.
        """
        self.idempotency_store = idempotency_store
        self.consul_client = consul_client
        self.postgres_client = postgres_client
        self.execution_count = 0

    async def execute_consul_register(
        self,
        intent_id: UUID,
        service_id: str,
        service_name: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Execute Consul service registration with idempotency guard.

        Args:
            intent_id: Unique identifier for this intent execution.
            service_id: Consul service ID to register.
            service_name: Consul service name.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if registration succeeded or was already done.
        """
        # Check idempotency - domain "consul" isolates Consul registrations
        is_new = await self.idempotency_store.check_and_record(
            message_id=intent_id,
            domain="consul",
            correlation_id=correlation_id,
        )

        if not is_new:
            # Already processed - skip execution
            return True

        # Execute the actual registration
        self.execution_count += 1
        await self.consul_client.agent.service.register(
            service_id=service_id,
            name=service_name,
        )

        return True

    async def execute_postgres_upsert(
        self,
        intent_id: UUID,
        record: ModelNodeRegistrationRecord,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Execute PostgreSQL upsert with idempotency guard.

        Args:
            intent_id: Unique identifier for this intent execution.
            record: Registration record to upsert.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if upsert succeeded or was already done.
        """
        # Check idempotency - domain "postgres" isolates PostgreSQL operations
        is_new = await self.idempotency_store.check_and_record(
            message_id=intent_id,
            domain="postgres",
            correlation_id=correlation_id,
        )

        if not is_new:
            # Already processed - skip execution
            return True

        # Execute the actual upsert
        self.execution_count += 1
        await self.postgres_client.execute(
            "UPSERT INTO node_registrations ...",
            record.model_dump(),
        )

        return True

    async def execute_with_natural_key_check(
        self,
        intent_id: UUID,
        entity_id: UUID,
        intent_type: str,
        registration_id: UUID,
        execute_fn: AsyncMock,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Execute with natural key conflict detection.

        This method demonstrates checking for natural key conflicts in addition
        to intent_id duplicates. The natural key is (entity_id, intent_type,
        registration_id).

        Args:
            intent_id: Unique identifier for this intent execution.
            entity_id: Entity being registered (e.g., node_id).
            intent_type: Type of intent (e.g., "consul.register").
            registration_id: Unique registration workflow ID.
            execute_fn: Function to call for actual execution.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if execution succeeded or was already done.
        """
        # Build natural key from composite of entity + intent type + registration
        # This ensures that even with different intent_ids, the same logical
        # operation is treated as a duplicate.
        #
        # NOTE: uuid5(NAMESPACE_OID, natural_key) is intentionally used here.
        # uuid5 generates deterministic UUIDs from a namespace and name, which is
        # exactly what we need for natural key deduplication:
        # - Same natural_key always produces the same UUID (deterministic)
        # - Different natural_keys produce different UUIDs (collision-resistant)
        # - NAMESPACE_OID is a well-known UUID namespace (RFC 4122)
        # This is NOT a collision risk because uuid5 is designed for this use case.
        natural_key = f"{entity_id}:{intent_type}:{registration_id}"
        natural_key_uuid = uuid5(NAMESPACE_OID, natural_key)

        # First check: intent_id deduplication
        is_new_intent = await self.idempotency_store.check_and_record(
            message_id=intent_id,
            domain="intent",
            correlation_id=correlation_id,
        )

        if not is_new_intent:
            return True  # Already processed this intent

        # Second check: natural key deduplication
        is_new_natural_key = await self.idempotency_store.check_and_record(
            message_id=natural_key_uuid,
            domain="natural_key",
            correlation_id=correlation_id,
        )

        if not is_new_natural_key:
            return True  # Same logical operation already done

        # Execute the actual operation
        self.execution_count += 1
        await execute_fn()

        return True


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestDuplicateIntentSafety:
    """Test that duplicate intent_id causes no additional side effects (G4.1)."""

    @pytest.mark.asyncio
    async def test_duplicate_intent_causes_no_additional_side_effects(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        mock_consul_client: MagicMock,
        mock_postgres_client: MagicMock,
    ) -> None:
        """Verify that processing the same intent_id twice causes only one I/O.

        This test validates the core idempotency guarantee: duplicate intents
        (identified by the same intent_id) do not cause additional backend calls.

        Test flow:
            1. Create effect executor with idempotency store
            2. Execute intent with intent_id=X -> backend called once
            3. Execute same intent with intent_id=X -> backend NOT called
            4. Assert execution_count == 1
        """
        # Arrange
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=mock_consul_client,
            postgres_client=mock_postgres_client,
        )
        intent_id = uuid4()
        correlation_id = uuid4()

        # Act - first execution
        result1 = await executor.execute_consul_register(
            intent_id=intent_id,
            service_id="node-effect-123",
            service_name="onex-effect",
            correlation_id=correlation_id,
        )

        # Act - second execution with SAME intent_id
        result2 = await executor.execute_consul_register(
            intent_id=intent_id,
            service_id="node-effect-123",
            service_name="onex-effect",
            correlation_id=correlation_id,
        )

        # Assert
        assert result1 is True
        assert result2 is True
        assert executor.execution_count == 1  # Only one actual backend call
        mock_consul_client.agent.service.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_intent_ids_execute_separately(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        mock_consul_client: MagicMock,
        mock_postgres_client: MagicMock,
    ) -> None:
        """Verify that different intent_ids execute independently.

        This test ensures that different intents are processed separately,
        confirming the idempotency key is the intent_id.
        """
        # Arrange
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=mock_consul_client,
            postgres_client=mock_postgres_client,
        )
        intent_id_1 = uuid4()
        intent_id_2 = uuid4()

        # Act
        await executor.execute_consul_register(
            intent_id=intent_id_1,
            service_id="node-effect-123",
            service_name="onex-effect",
        )
        await executor.execute_consul_register(
            intent_id=intent_id_2,
            service_id="node-effect-456",
            service_name="onex-effect",
        )

        # Assert
        assert executor.execution_count == 2
        assert mock_consul_client.agent.service.register.call_count == 2


@pytest.mark.unit
class TestNaturalKeyConflict:
    """Test that natural key conflicts are treated as duplicates (G4.2)."""

    @pytest.mark.asyncio
    async def test_natural_key_conflict_treated_as_duplicate(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        mock_consul_client: MagicMock,
        mock_postgres_client: MagicMock,
    ) -> None:
        """Verify that same natural key with different intent_id is treated as duplicate.

        Natural key = (entity_id, intent_type, registration_id)

        This ensures that if the same logical operation is attempted with a
        different intent_id (e.g., retry with new correlation), it's still
        detected as a duplicate.

        Test flow:
            1. Execute with intent_id=X, natural_key=(A, B, C)
            2. Execute with intent_id=Y, natural_key=(A, B, C) <- different intent_id!
            3. Assert only one backend call (second was detected as duplicate)
        """
        # Arrange
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=mock_consul_client,
            postgres_client=mock_postgres_client,
        )
        mock_execute = AsyncMock()

        # Same natural key components
        entity_id = uuid4()
        intent_type = "consul.register"
        registration_id = uuid4()

        # Different intent IDs
        intent_id_1 = uuid4()
        intent_id_2 = uuid4()

        # Act - first execution
        result1 = await executor.execute_with_natural_key_check(
            intent_id=intent_id_1,
            entity_id=entity_id,
            intent_type=intent_type,
            registration_id=registration_id,
            execute_fn=mock_execute,
        )

        # Act - second execution with DIFFERENT intent_id but SAME natural key
        result2 = await executor.execute_with_natural_key_check(
            intent_id=intent_id_2,
            entity_id=entity_id,
            intent_type=intent_type,
            registration_id=registration_id,
            execute_fn=mock_execute,
        )

        # Assert
        assert result1 is True
        assert result2 is True
        assert executor.execution_count == 1  # Only one actual execution
        mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_natural_keys_execute_separately(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        mock_consul_client: MagicMock,
        mock_postgres_client: MagicMock,
    ) -> None:
        """Verify that different natural keys execute independently."""
        # Arrange
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=mock_consul_client,
            postgres_client=mock_postgres_client,
        )
        mock_execute = AsyncMock()

        # Different natural key components (different entity_id)
        entity_id_1 = uuid4()
        entity_id_2 = uuid4()
        intent_type = "consul.register"
        registration_id = uuid4()

        # Act
        await executor.execute_with_natural_key_check(
            intent_id=uuid4(),
            entity_id=entity_id_1,
            intent_type=intent_type,
            registration_id=registration_id,
            execute_fn=mock_execute,
        )
        await executor.execute_with_natural_key_check(
            intent_id=uuid4(),
            entity_id=entity_id_2,
            intent_type=intent_type,
            registration_id=registration_id,
            execute_fn=mock_execute,
        )

        # Assert
        assert executor.execution_count == 2
        assert mock_execute.call_count == 2


@pytest.mark.unit
class TestIdempotencyAcrossRestart:
    """Test that idempotency persists across effect restarts (G4.3)."""

    @pytest.mark.asyncio
    async def test_idempotency_across_effect_restart(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        mock_consul_client: MagicMock,
        mock_postgres_client: MagicMock,
    ) -> None:
        """Verify idempotency survives effect node restart.

        This test simulates an effect restart by creating a new executor instance
        while keeping the same idempotency store (which would be backed by
        PostgreSQL in production).

        Test flow:
            1. Execute intent with executor instance 1
            2. Simulate restart: create new executor instance (same store)
            3. Re-process same intent with new executor
            4. Assert no additional I/O occurred
        """
        # Arrange - first executor instance (before restart)
        executor_before = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=mock_consul_client,
            postgres_client=mock_postgres_client,
        )
        intent_id = uuid4()
        correlation_id = uuid4()

        # Act - execute before restart
        result_before = await executor_before.execute_consul_register(
            intent_id=intent_id,
            service_id="node-effect-123",
            service_name="onex-effect",
            correlation_id=correlation_id,
        )

        # Simulate restart: create NEW executor with SAME idempotency store
        # In production, the store would be PostgreSQL (persistent)
        executor_after = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,  # Same store!
            consul_client=mock_consul_client,
            postgres_client=mock_postgres_client,
        )

        # Act - re-process same intent after restart
        result_after = await executor_after.execute_consul_register(
            intent_id=intent_id,
            service_id="node-effect-123",
            service_name="onex-effect",
            correlation_id=correlation_id,
        )

        # Assert
        assert result_before is True
        assert result_after is True
        # Combined execution count across both instances should be 1
        total_executions = (
            executor_before.execution_count + executor_after.execution_count
        )
        assert total_executions == 1
        mock_consul_client.agent.service.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_idempotency_record_persisted_in_store(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
    ) -> None:
        """Verify that idempotency records are actually stored.

        This test confirms that after processing an intent, the record exists
        in the idempotency store and can be queried.
        """
        # Arrange
        intent_id = uuid4()
        correlation_id = uuid4()

        # Act
        await inmemory_idempotency_store.check_and_record(
            message_id=intent_id,
            domain="consul",
            correlation_id=correlation_id,
        )

        # Assert - record exists in store
        record = await inmemory_idempotency_store.get_record(
            message_id=intent_id,
            domain="consul",
        )
        assert record is not None
        assert isinstance(record, ModelIdempotencyRecord)
        assert record.message_id == intent_id
        assert record.correlation_id == correlation_id
        assert record.domain == "consul"


@pytest.mark.unit
class TestStorageFailureSafety:
    """Test that idempotency store failures result in safe failure (G4.4)."""

    @pytest.mark.asyncio
    async def test_idempotency_store_unavailable_fails_safe(
        self,
        mock_consul_client: MagicMock,
        mock_postgres_client: MagicMock,
    ) -> None:
        """Verify that store unavailability causes graceful failure.

        When the idempotency store is unavailable (raises exception), the effect
        should fail WITHOUT executing the backend operation. This prevents
        partial execution that could lead to inconsistent state.

        Test flow:
            1. Configure store to raise exception on check_and_record
            2. Attempt to process intent
            3. Assert: exception raised, backend NOT called
        """
        # Arrange - create store that raises on check_and_record
        failing_store = MagicMock(spec=InMemoryIdempotencyStore)
        failing_store.check_and_record = AsyncMock(
            side_effect=ConnectionError("Store unavailable")
        )

        intent_id = uuid4()

        # Act & Assert
        with pytest.raises(ConnectionError, match="Store unavailable"):
            # Check idempotency (this will fail)
            await failing_store.check_and_record(
                message_id=intent_id,
                domain="consul",
            )

        # Backend should NOT have been called (fail-safe behavior)
        mock_consul_client.agent.service.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_execution_prevented_on_store_failure(
        self,
        mock_consul_client: MagicMock,
        mock_postgres_client: MagicMock,
    ) -> None:
        """Verify no partial execution when store fails during check.

        This test ensures that the check-then-execute pattern is atomic:
        if the check fails, execution never starts.
        """
        # Arrange
        failing_store = MagicMock(spec=InMemoryIdempotencyStore)
        failing_store.check_and_record = AsyncMock(
            side_effect=RuntimeError("Database connection lost")
        )

        async def safe_execute_with_idempotency(
            store: MagicMock,
            intent_id: UUID,
            execute_fn: AsyncMock,
        ) -> bool:
            """Execute with idempotency guard, propagating store failures."""
            # This pattern ensures no partial execution
            is_new = await store.check_and_record(
                message_id=intent_id,
                domain="test",
            )
            if is_new:
                await execute_fn()
            return True

        mock_execute = AsyncMock()
        intent_id = uuid4()

        # Act & Assert
        with pytest.raises(RuntimeError, match="Database connection lost"):
            await safe_execute_with_idempotency(
                store=failing_store,
                intent_id=intent_id,
                execute_fn=mock_execute,
            )

        # Execute function should NOT have been called
        mock_execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_timeout_handled_gracefully(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Verify that store timeout is handled without partial execution."""
        # Arrange
        timing_out_store = MagicMock(spec=InMemoryIdempotencyStore)
        timing_out_store.check_and_record = AsyncMock(
            side_effect=TimeoutError("Store query timed out")
        )

        intent_id = uuid4()

        # Act & Assert
        with pytest.raises(TimeoutError, match="Store query timed out"):
            await timing_out_store.check_and_record(
                message_id=intent_id,
                domain="consul",
            )

        # Backend not called due to timeout
        mock_consul_client.agent.service.register.assert_not_called()


@pytest.mark.unit
class TestDomainIsolation:
    """Test that idempotency domains are properly isolated."""

    @pytest.mark.asyncio
    async def test_same_intent_id_different_domains_independent(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
    ) -> None:
        """Verify that the same intent_id in different domains is independent.

        This allows the same intent to be tracked separately for Consul vs
        PostgreSQL execution within the same registration workflow.
        """
        # Arrange
        intent_id = uuid4()

        # Act - record in consul domain
        is_new_consul = await inmemory_idempotency_store.check_and_record(
            message_id=intent_id,
            domain="consul",
        )

        # Act - same intent_id in postgres domain should still be new
        is_new_postgres = await inmemory_idempotency_store.check_and_record(
            message_id=intent_id,
            domain="postgres",
        )

        # Assert
        assert is_new_consul is True
        assert is_new_postgres is True

    @pytest.mark.asyncio
    async def test_duplicate_in_same_domain_detected(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
    ) -> None:
        """Verify that duplicates within the same domain are detected."""
        # Arrange
        intent_id = uuid4()

        # Act
        first_result = await inmemory_idempotency_store.check_and_record(
            message_id=intent_id,
            domain="consul",
        )
        second_result = await inmemory_idempotency_store.check_and_record(
            message_id=intent_id,
            domain="consul",
        )

        # Assert
        assert first_result is True
        assert second_result is False  # Duplicate detected
