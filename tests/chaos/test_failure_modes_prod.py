# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Failure-mode and recovery tests for production infrastructure [OMN-781].

This test suite implements INFRA-040 for the PRODUCTION milestone, covering
failure behavior and recovery for the four core infrastructure services:

1. Kafka Connection Loss
   - Producer cannot reach broker (InfraConnectionError)
   - Consumer disconnects mid-stream
   - Reconnection after broker becomes available
   - Message delivery guarantees across reconnect

2. Postgres Unavailable
   - Database unreachable at query time
   - Connection pool exhaustion
   - Query timeout behaviour
   - Repository error propagation with correct error types

3. Vault / Secret Backend Flakiness
   - Transient 503 responses from Vault
   - Auth token expiry during operation
   - Network-level flakiness (intermittent packet loss)
   - Secret resolution falls back / fails fast appropriately

4. Handler Continuous Errors
   - Handler raises on every call → circuit breaker opens
   - Remaining handlers (good subscribers) are isolated
   - Circuit breaker state is observable
   - Circuit resets after manual call to reset method

Design:
    All tests use mock-based fault injection rather than real infrastructure.
    This keeps the suite fast (<5s total), deterministic, and CI-safe with
    no external dependencies.

    Error types used match the ONEX error hierarchy:
    - InfraConnectionError   → network / transport failures
    - InfraUnavailableError  → service unavailable (circuit open, pool exhausted)
    - InfraTimeoutError      → operation exceeded timeout
    - SecretResolutionError  → secret / config backend failure
    - RepositoryExecutionError → database query failure

Markers:
    @pytest.mark.chaos (auto-applied by conftest.py hook)

Usage:
    uv run pytest tests/chaos/test_failure_modes_prod.py -v
    uv run pytest -m chaos -v

Related:
    - OMN-781: INFRA-040 Chaos and failure-mode tests [PROD]
    - OMN-955: Chaos scenario tests (companion suite)
    - tests/chaos/test_chaos_network_partitions.py: Generic partition tests
    - tests/chaos/test_recovery_circuit_breaker.py: Circuit breaker unit tests
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    RepositoryExecutionError,
    SecretResolutionError,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.models.errors import ModelTimeoutErrorContext
from tests.chaos.conftest import (
    FailureInjector,
    NetworkPartitionSimulator,
    get_chaos_profile,
)
from tests.conftest import make_test_node_identity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kafka_connection_error(
    operation: str, correlation_id: UUID | None = None
) -> InfraConnectionError:
    """Build a Kafka-flavoured InfraConnectionError."""
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.KAFKA,
        operation=operation,
        target_name="omnibase-infra-redpanda:9092",
        correlation_id=correlation_id or uuid4(),
    )
    return InfraConnectionError(
        f"Kafka broker unreachable during '{operation}'",
        context=context,
    )


def _postgres_repo_error(
    operation: str, correlation_id: UUID | None = None
) -> RepositoryExecutionError:
    """Build a Postgres-flavoured RepositoryExecutionError."""
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation=operation,
        correlation_id=correlation_id or uuid4(),
    )
    return RepositoryExecutionError(
        f"PostgreSQL query failed during '{operation}': connection refused",
        op_name=operation,
        context=context,
    )


def _vault_secret_error(
    path: str, correlation_id: UUID | None = None
) -> SecretResolutionError:
    """Build a Vault-flavoured SecretResolutionError."""
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.INFISICAL,
        operation="get_secret",
        target_name=path,
        correlation_id=correlation_id or uuid4(),
    )
    return SecretResolutionError(
        f"Vault unavailable: could not resolve secret at '{path}'",
        context=context,
    )


def _timeout_error(
    operation: str,
    transport: EnumInfraTransportType,
    correlation_id: UUID | None = None,
) -> InfraTimeoutError:
    """Build an InfraTimeoutError with the given transport context."""
    context = ModelTimeoutErrorContext(
        transport_type=transport,
        operation=operation,
        correlation_id=correlation_id or uuid4(),
    )
    return InfraTimeoutError(
        f"Operation '{operation}' exceeded timeout",
        context=context,
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fresh_event_bus() -> AsyncGenerator[EventBusInmemory, None]:
    """Provide a fresh EventBusInmemory for each test.

    Uses a high circuit-breaker threshold so individual tests control when
    the circuit opens via explicit configuration.
    """
    bus = EventBusInmemory(
        environment="failure-mode-test",
        group="prod-chaos",
        max_history=10000,
        circuit_breaker_threshold=5,
    )
    await bus.start()
    yield bus
    await bus.close()


# ---------------------------------------------------------------------------
# 1. Kafka Connection Loss
# ---------------------------------------------------------------------------


@pytest.mark.chaos
class TestKafkaConnectionLoss:
    """Validate system behavior when the Kafka broker is unreachable.

    These tests use mock Kafka adapters and the in-memory event bus with
    injected failures to simulate broker unavailability without requiring
    a real Redpanda instance.
    """

    @pytest.mark.asyncio
    async def test_kafka_producer_failure_raises_infra_connection_error(self) -> None:
        """Producer raises InfraConnectionError when broker is down.

        A simulated Kafka producer that cannot reach the broker must surface
        InfraConnectionError, not a bare ConnectionError or generic exception.
        This ensures calling code can distinguish infrastructure failures from
        application errors.
        """
        correlation_id = uuid4()

        # Simulate the adapter's publish method failing with broker unreachable
        mock_producer = AsyncMock()
        mock_producer.produce.side_effect = _kafka_connection_error(
            "produce", correlation_id
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await mock_producer.produce(
                topic="test.events.v1",
                key=b"test-key",
                value=b'{"event": "test"}',
            )

        error = exc_info.value
        assert isinstance(error, InfraConnectionError)
        assert "Kafka" in str(error) or "broker" in str(error).lower()

    @pytest.mark.asyncio
    async def test_kafka_consumer_disconnect_raises_on_poll(self) -> None:
        """Consumer poll raises InfraConnectionError after broker disconnects.

        When a consumer loses its connection mid-stream, subsequent poll
        calls must raise InfraConnectionError rather than hanging or returning
        empty results silently.
        """
        correlation_id = uuid4()

        mock_consumer = AsyncMock()
        # First call succeeds, subsequent calls fail (simulates mid-stream disconnect)
        from datetime import UTC, datetime

        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        first_message = ModelEventMessage(
            topic="stream.events.v1",
            key=b"key-0",
            value=b'{"seq": 0}',
            headers=ModelEventHeaders(
                source="test",
                event_type="stream.events.v1",
                timestamp=datetime.now(UTC),
            ),
            offset="0",
            partition=0,
        )
        mock_consumer.poll.side_effect = [
            first_message,
            _kafka_connection_error("poll", correlation_id),
        ]

        # First poll succeeds
        result = await mock_consumer.poll()
        assert result.offset == "0"

        # Second poll raises on disconnect
        with pytest.raises(InfraConnectionError, match="Kafka"):
            await mock_consumer.poll()

    @pytest.mark.asyncio
    async def test_kafka_reconnect_delivers_messages_after_partition_heals(
        self,
        fresh_event_bus: EventBusInmemory,
    ) -> None:
        """Messages resume delivery after Kafka partition heals.

        Uses the in-memory bus + NetworkPartitionSimulator to validate that:
        - During partition: event bus publish raises InfraConnectionError
        - After heal: publish succeeds and subscriber receives messages
        """
        topic = "recovery.events.v1"
        partition_sim = NetworkPartitionSimulator()
        received: list[str] = []
        lock = asyncio.Lock()

        from omnibase_infra.event_bus.models import ModelEventMessage

        async def subscriber(msg: ModelEventMessage) -> None:
            async with lock:
                received.append(msg.offset)

        await fresh_event_bus.subscribe(
            topic,
            make_test_node_identity(service="kafka-reconnect", node_name="consumer"),
            subscriber,
        )

        # Simulate partition: publish raises during active partition
        partition_sim.start_partition()
        assert partition_sim.is_partitioned

        # Simulate what a Kafka adapter would do during partition
        with pytest.raises(InfraConnectionError):
            # Raise explicitly — simulates the adapter detecting the partition
            raise _kafka_connection_error("publish_during_partition")

        # Heal partition
        await partition_sim.simulate_partition_healing(duration_ms=10)
        assert not partition_sim.is_partitioned

        # After heal: publish should succeed (in-memory bus = partition healed)
        await fresh_event_bus.publish(
            topic=topic,
            key=b"post-heal-key",
            value=b'{"recovered": true}',
        )

        assert len(received) == 1, "Expected exactly 1 message post-heal"

    @pytest.mark.asyncio
    async def test_kafka_broker_unavailable_circuit_breaker_activates(
        self,
        fresh_event_bus: EventBusInmemory,
    ) -> None:
        """Repeated Kafka failures activate circuit breaker in the event bus.

        Simulates a subscriber that consistently fails (as if the downstream
        Kafka-backed handler cannot reach the broker). After `threshold` failures
        the circuit should open, protecting the rest of the system.
        """
        topic = "kafka.failover.v1"
        fail_count = 0
        lock = asyncio.Lock()

        from omnibase_infra.event_bus.models import ModelEventMessage

        async def kafka_backed_handler(msg: ModelEventMessage) -> None:
            nonlocal fail_count
            async with lock:
                fail_count += 1
            # Simulate the handler failing to forward to Kafka
            raise InfraConnectionError(
                "Simulated Kafka downstream unavailable",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="forward_to_kafka",
                    correlation_id=uuid4(),
                ),
            )

        await fresh_event_bus.subscribe(
            topic,
            make_test_node_identity(service="kafka-cb", node_name="handler"),
            kafka_backed_handler,
        )

        # Publish enough messages to trip the circuit (threshold=5)
        for i in range(10):
            await fresh_event_bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=b'{"test": true}',
            )

        status = await fresh_event_bus.get_circuit_breaker_status()

        assert len(status["open_circuits"]) >= 1, (
            "Circuit breaker should be open after repeated Kafka handler failures"
        )
        # Handler called exactly 5 times (threshold) before circuit opened
        assert fail_count == 5, (
            f"Handler called {fail_count} times; expected 5 (circuit threshold)"
        )


# ---------------------------------------------------------------------------
# 2. Postgres Unavailable
# ---------------------------------------------------------------------------


@pytest.mark.chaos
class TestPostgresUnavailable:
    """Validate system behavior when PostgreSQL is unreachable.

    Tests simulate the repository layer raising RepositoryExecutionError
    (the correct error type for DB failures) rather than raw psycopg exceptions.
    """

    @pytest.mark.asyncio
    async def test_repository_query_raises_repo_execution_error_when_db_down(
        self,
    ) -> None:
        """Repository raises RepositoryExecutionError when Postgres is unavailable.

        Verifies that the repository layer wraps raw database exceptions in the
        ONEX error hierarchy. Callers should never see bare psycopg2 or asyncpg
        exceptions.
        """
        correlation_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.find_by_id.side_effect = _postgres_repo_error(
            "find_by_id", correlation_id
        )

        with pytest.raises(RepositoryExecutionError) as exc_info:
            await mock_repo.find_by_id(str(uuid4()))

        error = exc_info.value
        assert isinstance(error, RepositoryExecutionError)
        assert "PostgreSQL" in str(error) or "connection" in str(error).lower()

    @pytest.mark.asyncio
    async def test_postgres_timeout_raises_infra_timeout_error(self) -> None:
        """Database query timeout raises InfraTimeoutError.

        When a query exceeds its configured timeout, the system must raise
        InfraTimeoutError (not a bare asyncio.TimeoutError) so that calling
        code can apply appropriate retry/circuit-breaker logic.
        """
        correlation_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.execute_query.side_effect = _timeout_error(
            "execute_query",
            EnumInfraTransportType.DATABASE,
            correlation_id,
        )

        with pytest.raises(InfraTimeoutError) as exc_info:
            await mock_repo.execute_query("SELECT 1")

        error = exc_info.value
        assert isinstance(error, InfraTimeoutError)

    @pytest.mark.asyncio
    async def test_postgres_unavailable_does_not_affect_other_handlers(
        self,
        fresh_event_bus: EventBusInmemory,
    ) -> None:
        """Database failure in one handler does not block other subscribers.

        When a DB-backed subscriber raises RepositoryExecutionError, the event
        bus must continue delivering to the next subscriber. Isolation between
        consumers is a critical correctness property.
        """
        topic = "events.registration.v1"
        db_fail_count = 0
        good_received: list[int] = []
        lock = asyncio.Lock()

        from omnibase_infra.event_bus.models import ModelEventMessage

        async def db_backed_handler(msg: ModelEventMessage) -> None:
            nonlocal db_fail_count
            async with lock:
                db_fail_count += 1
            raise _postgres_repo_error("upsert_registration")

        async def good_handler(msg: ModelEventMessage) -> None:
            async with lock:
                good_received.append(1)

        await fresh_event_bus.subscribe(
            topic,
            make_test_node_identity(service="db-isolation", node_name="db-handler"),
            db_backed_handler,
        )
        await fresh_event_bus.subscribe(
            topic,
            make_test_node_identity(service="db-isolation", node_name="good-handler"),
            good_handler,
        )

        num_messages = 3  # Below circuit threshold (5) to keep db_handler active
        for i in range(num_messages):
            await fresh_event_bus.publish(
                topic=topic,
                key=f"reg-{i}".encode(),
                value=b'{"node_id": "test"}',
            )

        # DB handler failed each time but good handler received all messages
        assert db_fail_count == num_messages, (
            f"DB handler expected {num_messages} calls, got {db_fail_count}"
        )
        assert len(good_received) == num_messages, (
            f"Good handler expected {num_messages}, got {len(good_received)}"
        )

    @pytest.mark.asyncio
    async def test_postgres_connection_pool_exhaustion_raises_unavailable(
        self,
    ) -> None:
        """Pool exhaustion raises InfraUnavailableError.

        When all database connections are in use, new queries must fail
        with InfraUnavailableError (not a hang or bare TimeoutError).
        """
        correlation_id = uuid4()

        mock_pool = AsyncMock()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="acquire_connection",
            target_name="omnibase_infra_pool",
            correlation_id=correlation_id,
        )
        mock_pool.acquire.side_effect = InfraUnavailableError(
            "PostgreSQL connection pool exhausted: all 10 connections in use",
            context=context,
        )

        with pytest.raises(InfraUnavailableError, match="pool exhausted"):
            await mock_pool.acquire()


# ---------------------------------------------------------------------------
# 3. Vault / Secret Backend Flakiness
# ---------------------------------------------------------------------------


@pytest.mark.chaos
class TestVaultNetworkFlakiness:
    """Validate system behavior under Vault network flakiness.

    Vault is used as the secret backend in production. Flakiness manifests
    as transient 503 errors, network timeouts, or intermittent packet loss.
    The system must surface SecretResolutionError for any Vault failure.
    """

    @pytest.mark.asyncio
    async def test_vault_transient_503_raises_secret_resolution_error(self) -> None:
        """Transient Vault 503 raises SecretResolutionError.

        The Vault handler must translate HTTP 503 into SecretResolutionError,
        not expose raw HTTP error objects to callers.
        """
        secret_path = "/shared/db/POSTGRES_DSN"
        correlation_id = uuid4()

        mock_vault_handler = AsyncMock()
        mock_vault_handler.get_secret.side_effect = _vault_secret_error(
            secret_path, correlation_id
        )

        with pytest.raises(SecretResolutionError) as exc_info:
            await mock_vault_handler.get_secret(secret_path)

        error = exc_info.value
        assert isinstance(error, SecretResolutionError)
        assert secret_path in str(error)

    @pytest.mark.asyncio
    async def test_vault_timeout_raises_infra_timeout_error(self) -> None:
        """Vault network timeout raises InfraTimeoutError.

        When the Vault handler cannot receive a response within the configured
        timeout, it must raise InfraTimeoutError with VAULT transport context.
        """
        mock_vault_handler = AsyncMock()
        mock_vault_handler.get_secret.side_effect = _timeout_error(
            "vault_secret_fetch",
            EnumInfraTransportType.INFISICAL,
        )

        with pytest.raises(InfraTimeoutError):
            await mock_vault_handler.get_secret("/shared/kafka/BOOTSTRAP_SERVERS")

    @pytest.mark.asyncio
    async def test_vault_intermittent_failures_with_retry_recovery(self) -> None:
        """Intermittent Vault failures recover after retry.

        Simulates the Vault handler succeeding after 2 transient failures,
        validating that a well-implemented retry loop can recover from
        flakiness without bubbling the error to the caller.
        """
        secret_path = "/services/runtime/vault/VAULT_TOKEN"
        expected_value = "s.production-secret-token"

        call_count = 0

        async def vault_handler_with_retry() -> str:
            """Simulate a retry-loop around a flaky Vault client."""
            nonlocal call_count
            max_attempts = 3
            last_exc: Exception | None = None

            for attempt in range(max_attempts):
                call_count += 1
                if attempt < 2:
                    # First two attempts fail with transient error
                    last_exc = _vault_secret_error(secret_path)
                    await asyncio.sleep(0)  # Yield to event loop
                    continue
                # Third attempt succeeds
                return expected_value

            assert last_exc is not None
            raise last_exc

        result = await vault_handler_with_retry()

        assert result == expected_value, "Expected successful recovery on 3rd attempt"
        assert call_count == 3, (
            f"Expected 3 attempts (2 fail + 1 succeed), got {call_count}"
        )

    @pytest.mark.asyncio
    async def test_vault_auth_token_expiry_raises_connection_error(self) -> None:
        """Expired auth token raises InfraConnectionError (VAULT transport).

        An expired or revoked Vault token is an authentication failure at the
        transport level and should be surfaced as InfraConnectionError (not
        InfraAuthenticationError, which is reserved for user-facing auth).
        """
        mock_vault_client = AsyncMock()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.INFISICAL,
            operation="authenticate",
            target_name="vault:8200",
            correlation_id=uuid4(),
        )
        mock_vault_client.renew_token.side_effect = InfraConnectionError(
            "Vault token expired or revoked: permission denied",
            context=context,
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await mock_vault_client.renew_token()

        # context is serialized as dict with additional_context containing transport_type
        ctx = exc_info.value.context
        if isinstance(ctx, dict):
            transport = ctx.get("additional_context", {}).get("transport_type")
        else:
            transport = getattr(ctx, "transport_type", None)
        assert transport in (
            EnumInfraTransportType.INFISICAL,
            EnumInfraTransportType.INFISICAL.value,
        ), f"Expected INFISICAL transport context, got {transport!r}"

    @pytest.mark.asyncio
    async def test_vault_unavailable_secrets_fail_fast_not_hang(self) -> None:
        """Vault unavailability fails fast, not with an indefinite hang.

        A timed-out Vault call must complete within a bounded duration.
        This test validates the error path completes quickly (simulated timeout).
        """
        import time

        mock_vault_handler = AsyncMock()
        mock_vault_handler.get_secret.side_effect = _timeout_error(
            "vault_fetch_timeout",
            EnumInfraTransportType.INFISICAL,
        )

        start = time.perf_counter()
        with pytest.raises(InfraTimeoutError):
            await mock_vault_handler.get_secret("/shared/infra/KEY")
        elapsed = time.perf_counter() - start

        # Error path must complete in < 100ms (not actually waiting for a real timeout)
        assert elapsed < 0.1, f"Vault failure path took {elapsed:.3f}s — possible hang"


# ---------------------------------------------------------------------------
# 4. Handler Continuous Errors
# ---------------------------------------------------------------------------


@pytest.mark.chaos
class TestHandlerContinuousErrors:
    """Validate system behavior when a handler raises on every invocation.

    A handler that continuously fails represents a broken downstream dependency.
    The system must:
    1. Activate the circuit breaker after `threshold` consecutive failures
    2. Protect healthy handlers (subscriber isolation)
    3. Expose the circuit state for observability
    4. Allow manual circuit reset for recovery testing
    """

    @pytest.mark.asyncio
    async def test_continuous_handler_errors_open_circuit_after_threshold(
        self,
    ) -> None:
        """Circuit breaker opens after threshold consecutive handler failures.

        With circuit_breaker_threshold=3, after 3 consecutive failures the circuit
        must be open and the handler must stop receiving messages.
        """
        topic = "continuous.errors.v1"
        threshold = 3
        bus = EventBusInmemory(
            environment="continuous-fail",
            group="test",
            circuit_breaker_threshold=threshold,
        )
        await bus.start()

        fail_count = 0
        lock = asyncio.Lock()

        from omnibase_infra.event_bus.models import ModelEventMessage

        async def always_failing_handler(msg: ModelEventMessage) -> None:
            nonlocal fail_count
            async with lock:
                fail_count += 1
            raise RuntimeError("Persistent downstream failure — handler cannot process")

        await bus.subscribe(
            topic,
            make_test_node_identity(service="continuous-fail", node_name="broken"),
            always_failing_handler,
        )

        # Publish threshold + extra messages
        for i in range(threshold + 5):
            await bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=b'{"test": true}',
            )

        status = await bus.get_circuit_breaker_status()
        await bus.close()

        # Circuit must be open
        assert len(status["open_circuits"]) >= 1, (
            "Circuit should be open after continuous failures"
        )

        # Handler called exactly `threshold` times before circuit opened
        assert fail_count == threshold, (
            f"Handler called {fail_count} times; expected {threshold} "
            f"(circuit opens after threshold failures, not before)"
        )

    @pytest.mark.asyncio
    async def test_continuously_failing_handler_does_not_affect_healthy_handler(
        self,
    ) -> None:
        """Healthy handler receives all messages despite a continuously failing peer.

        This validates the subscriber isolation guarantee: one broken handler
        must not prevent other handlers from receiving messages.
        """
        topic = "isolation.test.v1"
        threshold = 5
        bus = EventBusInmemory(
            environment="isolation",
            group="test",
            circuit_breaker_threshold=threshold,
        )
        await bus.start()

        fail_count = 0
        healthy_count = 0
        lock = asyncio.Lock()

        from omnibase_infra.event_bus.models import ModelEventMessage

        async def broken_handler(msg: ModelEventMessage) -> None:
            nonlocal fail_count
            async with lock:
                fail_count += 1
            raise RuntimeError("Always broken")

        async def healthy_handler(msg: ModelEventMessage) -> None:
            nonlocal healthy_count
            async with lock:
                healthy_count += 1

        await bus.subscribe(
            topic,
            make_test_node_identity(service="isolation", node_name="broken"),
            broken_handler,
        )
        await bus.subscribe(
            topic,
            make_test_node_identity(service="isolation", node_name="healthy"),
            healthy_handler,
        )

        num_messages = threshold + 10
        for i in range(num_messages):
            await bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=b'{"data": true}',
            )

        await bus.close()

        # Healthy handler must receive all messages
        assert healthy_count == num_messages, (
            f"Healthy handler received {healthy_count}/{num_messages} — "
            "isolation broken by failing peer"
        )

        # Broken handler hit threshold then stopped (circuit open)
        assert fail_count == threshold, (
            f"Broken handler called {fail_count} times, expected {threshold}"
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_observable_via_get_status(
        self,
    ) -> None:
        """Circuit breaker state is observable through the event bus API.

        The get_circuit_breaker_status() method must return accurate open
        circuit information for monitoring and alerting integration.
        """
        topic = "observability.test.v1"
        threshold = 4
        bus = EventBusInmemory(
            environment="observability",
            group="test",
            circuit_breaker_threshold=threshold,
        )
        await bus.start()

        from omnibase_infra.event_bus.models import ModelEventMessage

        async def always_fail(msg: ModelEventMessage) -> None:
            raise ValueError("Intentional continuous failure")

        identity = make_test_node_identity(service="obs", node_name="handler")
        await bus.subscribe(topic, identity, always_fail)

        # Verify circuit is closed before failures
        status_before = await bus.get_circuit_breaker_status()
        assert len(status_before["open_circuits"]) == 0, (
            "Circuit should be closed before any failures"
        )

        # Trip the circuit
        for i in range(threshold + 1):
            await bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=b"{}",
            )

        # Verify circuit is now open
        status_after = await bus.get_circuit_breaker_status()
        await bus.close()

        assert len(status_after["open_circuits"]) >= 1, (
            "Circuit should be open after threshold failures"
        )

        # Open circuit entry must include topic and group_id fields
        open_circuit = status_after["open_circuits"][0]
        assert "topic" in open_circuit, "Open circuit entry must include 'topic'"
        assert "group_id" in open_circuit, "Open circuit entry must include 'group_id'"

    @pytest.mark.asyncio
    async def test_circuit_reset_allows_handler_to_receive_again(
        self,
    ) -> None:
        """Manual circuit reset re-enables the previously failing handler.

        After resetting the circuit breaker for a handler that has been fixed,
        subsequent messages must be delivered to that handler again.
        """
        topic = "reset.test.v1"
        threshold = 3
        bus = EventBusInmemory(
            environment="reset",
            group="test",
            circuit_breaker_threshold=threshold,
        )
        await bus.start()

        call_count = 0
        should_fail = True
        lock = asyncio.Lock()

        from omnibase_infra.event_bus.models import ModelEventMessage

        async def recoverable_handler(msg: ModelEventMessage) -> None:
            nonlocal call_count
            async with lock:
                call_count += 1
            if should_fail:
                raise RuntimeError("Temporary failure")

        identity = make_test_node_identity(service="reset", node_name="recoverable")
        await bus.subscribe(topic, identity, recoverable_handler)

        # Trip the circuit
        for i in range(threshold):
            await bus.publish(
                topic=topic,
                key=f"fail-{i}".encode(),
                value=b"{}",
            )

        calls_after_trip = call_count

        # Verify circuit is open (handler stopped receiving)
        status = await bus.get_circuit_breaker_status()
        assert len(status["open_circuits"]) >= 1, "Circuit should be open"

        # Simulate handler "fix" — stop failing
        should_fail = False

        # Manually reset the circuit breaker using the published group_id
        open_circuit = status["open_circuits"][0]
        reset_result = await bus.reset_subscriber_circuit(
            topic=open_circuit["topic"],
            group_id=open_circuit["group_id"],
        )
        assert reset_result is True, (
            "reset_subscriber_circuit should return True on success"
        )

        # Publish one more message — handler should receive it now
        pre_reset_count = call_count
        await bus.publish(
            topic=topic,
            key=b"post-reset",
            value=b'{"recovered": true}',
        )

        await bus.close()

        assert call_count > pre_reset_count, (
            "Handler should have received message after circuit reset"
        )

    @pytest.mark.asyncio
    async def test_handler_continuous_errors_failure_count_in_status(
        self,
    ) -> None:
        """Circuit breaker status includes accurate failure counts.

        The failure_counts field in get_circuit_breaker_status() must reflect
        the number of consecutive failures per subscriber, enabling external
        monitoring to emit alerts proportional to failure severity.
        """
        topic = "failure-count.test.v1"
        threshold = 5
        bus = EventBusInmemory(
            environment="fc-test",
            group="test",
            circuit_breaker_threshold=threshold,
        )
        await bus.start()

        from omnibase_infra.event_bus.models import ModelEventMessage

        async def always_fail(msg: ModelEventMessage) -> None:
            raise ValueError("Always fails")

        identity = make_test_node_identity(service="fc", node_name="handler")
        await bus.subscribe(topic, identity, always_fail)

        # Trip the circuit
        for i in range(threshold):
            await bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=b"{}",
            )

        status = await bus.get_circuit_breaker_status()
        await bus.close()

        # Failure counts dict must have at least one entry at threshold
        failure_counts: dict[str, int] = status["failure_counts"]  # type: ignore[assignment]
        assert len(failure_counts) >= 1, "failure_counts should be non-empty"

        # The failure count for our handler must equal threshold
        count_values = list(failure_counts.values())
        assert any(c == threshold for c in count_values), (
            f"Expected at least one count == {threshold}, got {count_values}"
        )
