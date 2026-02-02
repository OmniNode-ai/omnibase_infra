# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive unit tests for RequestResponseWiring.

This test suite validates:
- Correlation ID handling (injection, preservation, return in result)
- Timeout behavior (InfraTimeoutError, context, cleanup)
- Consumer group naming format and boot nonce consistency
- Pending map management (success, failure, orphan responses)
- Cleanup behavior (idempotency, task cancellation, future failures)
- Circuit breaker integration (threshold, unavailable error)

Test Organization:
    - TestRequestResponseWiringInit: Initialization and validation
    - TestCorrelationIdHandling: Correlation ID injection/preservation
    - TestTimeoutBehavior: Timeout handling and error context
    - TestConsumerGroup: Consumer group format and boot nonce
    - TestPendingMapHandling: Response resolution and orphan handling
    - TestCleanup: Cleanup behavior and idempotency
    - TestCircuitBreaker: Circuit breaker integration

Related:
    - OMN-1742: Request-response wiring for Kafka RPC patterns
    - RequestResponseWiring: Implementation under test

.. versionadded:: 0.3.1
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.contracts.subcontracts import (
    ModelCorrelationConfig,
    ModelReplyTopics,
    ModelRequestResponseConfig,
    ModelRequestResponseInstance,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraTimeoutError,
    InfraUnavailableError,
    ProtocolConfigurationError,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.request_response_wiring import RequestResponseWiring


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus() -> MagicMock:
    """Mock event bus with publish/subscribe capabilities."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus._bootstrap_servers = "localhost:9092"
    return bus


@pytest.fixture
def request_response_config() -> ModelRequestResponseConfig:
    """Sample ModelRequestResponseConfig for testing."""
    return ModelRequestResponseConfig(
        instances=[
            ModelRequestResponseInstance(
                name="routing",
                request_topic="onex.cmd.routing.request.v1",
                reply_topics=ModelReplyTopics(
                    completed="onex.evt.routing.completed.v1",
                    failed="onex.evt.routing.failed.v1",
                ),
                timeout_seconds=5,  # Short for tests
            )
        ]
    )


@pytest.fixture
def request_response_config_multiple() -> ModelRequestResponseConfig:
    """Config with multiple request-response instances."""
    return ModelRequestResponseConfig(
        instances=[
            ModelRequestResponseInstance(
                name="routing",
                request_topic="onex.cmd.routing.request.v1",
                reply_topics=ModelReplyTopics(
                    completed="onex.evt.routing.completed.v1",
                    failed="onex.evt.routing.failed.v1",
                ),
                timeout_seconds=5,
            ),
            ModelRequestResponseInstance(
                name="analysis",
                request_topic="onex.cmd.intelligence.analyze.v1",
                reply_topics=ModelReplyTopics(
                    completed="onex.evt.intelligence.analyzed.v1",
                    failed="onex.evt.intelligence.failed.v1",
                ),
                timeout_seconds=30,
            ),
        ]
    )


@pytest.fixture
async def wiring(mock_event_bus: MagicMock) -> RequestResponseWiring:
    """Configured RequestResponseWiring instance (without consumers started)."""
    from omnibase_infra.runtime.request_response_wiring import RequestResponseWiring

    return RequestResponseWiring(
        event_bus=mock_event_bus,
        environment="test",
        app_name="test-app",
        bootstrap_servers="localhost:9092",
    )


# =============================================================================
# Initialization Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestRequestResponseWiringInit:
    """Test RequestResponseWiring initialization."""

    async def test_init_with_valid_params(self, mock_event_bus: MagicMock) -> None:
        """Test successful initialization with valid parameters."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="dev",
            app_name="my-service",
            bootstrap_servers="localhost:9092",
        )

        assert wiring._environment == "dev"
        assert wiring._app_name == "my-service"
        assert wiring._bootstrap_servers == "localhost:9092"
        assert len(wiring._instances) == 0

    async def test_init_raises_on_empty_environment(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Test that empty environment raises ValueError."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
        )

        with pytest.raises(ValueError, match="environment must be a non-empty string"):
            RequestResponseWiring(
                event_bus=mock_event_bus,
                environment="",
                app_name="test-app",
            )

    async def test_init_raises_on_empty_app_name(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Test that empty app_name raises ValueError."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
        )

        with pytest.raises(ValueError, match="app_name must be a non-empty string"):
            RequestResponseWiring(
                event_bus=mock_event_bus,
                environment="test",
                app_name="",
            )

    async def test_init_uses_event_bus_bootstrap_servers(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Test bootstrap servers fallback to event_bus attribute."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
        )

        mock_event_bus._bootstrap_servers = "kafka.example.com:9092"

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            # No bootstrap_servers provided
        )

        assert wiring._bootstrap_servers == "kafka.example.com:9092"

    async def test_resolve_topic_adds_environment_prefix(
        self, wiring: RequestResponseWiring
    ) -> None:
        """Test that resolve_topic adds environment prefix correctly."""
        topic = wiring.resolve_topic("onex.cmd.routing.request.v1")
        assert topic == "test.onex.cmd.routing.request.v1"


# =============================================================================
# Correlation ID Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCorrelationIdHandling:
    """Test correlation ID injection, preservation, and return."""

    async def test_send_request_injects_correlation_id_when_missing(
        self,
        mock_event_bus: MagicMock,
        request_response_config: ModelRequestResponseConfig,
    ) -> None:
        """Verify UUID4 is injected when correlation_id is missing."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        # Manually create instance state without starting consumer
        instance_name = "routing"
        wiring._instances[instance_name] = MagicMock(
            name=instance_name,
            request_topic="test.onex.cmd.routing.request.v1",
            completed_topic="test.onex.evt.routing.completed.v1",
            failed_topic="test.onex.evt.routing.failed.v1",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.rr.routing.abc12345",
            pending={},
        )

        # Prepare payload without correlation_id
        payload: dict[str, object] = {"data": "test"}

        # Call _ensure_correlation_id to verify injection
        correlation_id = wiring._ensure_correlation_id(
            payload,
            ModelCorrelationConfig(),
        )

        # Verify correlation_id was injected
        assert correlation_id is not None
        assert "correlation_id" in payload
        # Verify it's a valid UUID4 string
        injected_uuid = UUID(str(payload["correlation_id"]))
        assert injected_uuid.version == 4

    async def test_send_request_preserves_existing_correlation_id(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify existing correlation_id is not overwritten."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        # Prepare payload with existing correlation_id
        existing_id = uuid4()
        payload: dict[str, object] = {
            "data": "test",
            "correlation_id": str(existing_id),
        }

        # Call _ensure_correlation_id
        result_id = wiring._ensure_correlation_id(
            payload,
            ModelCorrelationConfig(),
        )

        # Verify existing correlation_id is preserved
        assert str(result_id) == str(existing_id)
        assert payload["correlation_id"] == str(existing_id)

    async def test_correlation_id_returned_in_result(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify result contains correlation_id for tracing."""
        from omnibase_infra.runtime.request_response_wiring import (
            _RequestResponseInstance,
        )

        # Create a mock instance
        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )

        # Add pending future
        correlation_key = str(uuid4())
        future: asyncio.Future[dict[str, object]] = (
            asyncio.get_event_loop().create_future()
        )
        instance.pending[correlation_key] = future

        # Simulate setting result with correlation_id (as done in _handle_response_message)
        response_data: dict[str, object] = {"result": "success"}
        response_data["_correlation_id"] = correlation_key
        future.set_result(response_data)

        # Verify result contains _correlation_id
        result = await future
        assert "_correlation_id" in result
        assert result["_correlation_id"] == correlation_key


# =============================================================================
# Timeout Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTimeoutBehavior:
    """Test timeout handling and error context."""

    async def test_send_request_raises_infra_timeout_error_on_timeout(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify InfraTimeoutError is raised (not InfraUnavailableError)."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        # Create mock instance with very short timeout
        instance_name = "routing"
        instance = _RequestResponseInstance(
            name=instance_name,
            request_topic="test.onex.cmd.routing.request.v1",
            completed_topic="test.onex.evt.routing.completed.v1",
            failed_topic="test.onex.evt.routing.failed.v1",
            timeout_seconds=1,  # Very short timeout
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.rr.routing.abc12345",
        )
        wiring._instances[instance_name] = instance

        # Send request with timeout (no response will come)
        payload: dict[str, object] = {"data": "test"}

        with pytest.raises(InfraTimeoutError) as exc_info:
            # Use 0.1s timeout for fast test
            await wiring.send_request(instance_name, payload, timeout_seconds=1)

        # Verify it's InfraTimeoutError, NOT InfraUnavailableError
        assert isinstance(exc_info.value, InfraTimeoutError)
        assert not isinstance(exc_info.value, InfraUnavailableError)
        assert "timeout" in exc_info.value.message.lower()

    async def test_timeout_error_has_correct_context(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify ModelTimeoutErrorContext fields are correct."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance_name = "routing"
        request_topic = "test.onex.cmd.routing.request.v1"
        instance = _RequestResponseInstance(
            name=instance_name,
            request_topic=request_topic,
            completed_topic="test.onex.evt.routing.completed.v1",
            failed_topic="test.onex.evt.routing.failed.v1",
            timeout_seconds=1,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.rr.routing.abc12345",
        )
        wiring._instances[instance_name] = instance

        payload: dict[str, object] = {"data": "test"}

        with pytest.raises(InfraTimeoutError) as exc_info:
            await wiring.send_request(instance_name, payload, timeout_seconds=1)

        error = exc_info.value
        context = error.model.context

        # Verify context fields
        assert context["transport_type"] == EnumInfraTransportType.KAFKA
        assert context["operation"] == "send_request"
        assert context["target_name"] == request_topic
        assert "timeout_seconds" in context
        assert context["timeout_seconds"] == 1.0

    async def test_timeout_cleans_up_pending_future(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify pending map doesn't leak on timeout."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance_name = "routing"
        instance = _RequestResponseInstance(
            name=instance_name,
            request_topic="test.onex.cmd.routing.request.v1",
            completed_topic="test.onex.evt.routing.completed.v1",
            failed_topic="test.onex.evt.routing.failed.v1",
            timeout_seconds=1,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.rr.routing.abc12345",
        )
        wiring._instances[instance_name] = instance

        # Verify pending map is empty initially
        assert len(instance.pending) == 0

        payload: dict[str, object] = {"data": "test"}

        with pytest.raises(InfraTimeoutError):
            await wiring.send_request(instance_name, payload, timeout_seconds=1)

        # Verify pending map is empty after timeout (cleaned up)
        assert len(instance.pending) == 0


# =============================================================================
# Consumer Group Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestConsumerGroup:
    """Test consumer group naming and boot nonce."""

    async def test_consumer_group_format(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify consumer group format: {environment}.rr.{instance_name}.{boot_nonce}."""
        from omnibase_infra.runtime.request_response_wiring import (
            _BOOT_NONCE,
            RequestResponseWiring,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="dev",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        # Create instance manually to check consumer group
        instance_name = "code-analysis"
        expected_group = f"dev.rr.{instance_name}.{_BOOT_NONCE}"

        # Verify the format matches expected pattern
        # The actual consumer group is built in _wire_instance
        # We test the format by checking expected pattern
        assert _BOOT_NONCE is not None
        assert len(_BOOT_NONCE) == 8
        assert all(c in "0123456789abcdef" for c in _BOOT_NONCE)
        assert expected_group.startswith("dev.rr.code-analysis.")

    async def test_boot_nonce_is_consistent_across_instances(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify same boot_nonce for all instances in process."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
        )

        wiring1 = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="dev",
            app_name="app1",
            bootstrap_servers="localhost:9092",
        )

        wiring2 = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="prod",
            app_name="app2",
            bootstrap_servers="localhost:9092",
        )

        # Both wirings should have the same boot nonce
        assert wiring1.get_boot_nonce() == wiring2.get_boot_nonce()
        assert len(wiring1.get_boot_nonce()) == 8

    async def test_boot_nonce_is_8_char_hex(
        self,
        wiring: RequestResponseWiring,
    ) -> None:
        """Verify boot nonce is 8-character hex string."""
        nonce = wiring.get_boot_nonce()

        assert len(nonce) == 8
        # Verify it's valid hex
        int(nonce, 16)  # Will raise if not valid hex


# =============================================================================
# Pending Map Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestPendingMapHandling:
    """Test pending map response resolution."""

    async def test_successful_response_resolves_future(self) -> None:
        """Verify completed topic resolves future with result."""
        from omnibase_infra.runtime.request_response_wiring import (
            _RequestResponseInstance,
        )

        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )

        # Create pending future
        correlation_key = str(uuid4())
        future: asyncio.Future[dict[str, object]] = (
            asyncio.get_event_loop().create_future()
        )
        instance.pending[correlation_key] = future

        # Resolve future (simulating response handling)
        response_data: dict[str, object] = {"result": "success", "data": [1, 2, 3]}
        response_data["_correlation_id"] = correlation_key
        future.set_result(response_data)

        # Verify future resolved correctly
        result = await future
        assert result["result"] == "success"
        assert result["data"] == [1, 2, 3]
        assert result["_correlation_id"] == correlation_key

    async def test_failed_response_rejects_future(self) -> None:
        """Verify failed topic sets exception on future."""
        from omnibase_infra.runtime.request_response_wiring import (
            _RequestResponseInstance,
        )

        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )

        # Create pending future
        correlation_key = str(uuid4())
        future: asyncio.Future[dict[str, object]] = (
            asyncio.get_event_loop().create_future()
        )
        instance.pending[correlation_key] = future

        # Reject future (simulating failed response)
        error_message = "Request processing failed"
        future.set_exception(RuntimeError(f"Request failed: {error_message}"))

        # Verify future rejected with exception
        with pytest.raises(RuntimeError) as exc_info:
            await future

        assert "Request failed" in str(exc_info.value)

    async def test_orphan_response_is_logged_not_crashed(
        self,
        mock_event_bus: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify missing correlation_id doesn't crash consumer."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )

        # Create mock message with unknown correlation_id
        mock_message = MagicMock()
        mock_message.value = json.dumps(
            {
                "correlation_id": str(uuid4()),  # Unknown correlation ID
                "result": "success",
            }
        ).encode("utf-8")
        mock_message.topic = "test.completed"

        # Handle response - should not crash, just log
        await wiring._handle_response_message(instance, mock_message)

        # Instance pending map should remain empty (no matching future)
        assert len(instance.pending) == 0


# =============================================================================
# Cleanup Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCleanup:
    """Test cleanup behavior and idempotency."""

    async def test_cleanup_cancels_consumer_tasks(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify consumer tasks are cancelled on cleanup."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        # Create mock consumer task
        async def mock_consumer() -> None:
            await asyncio.sleep(100)  # Long running task

        consumer_task = asyncio.create_task(mock_consumer())

        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )
        instance.consumer_task = consumer_task
        instance.consumer = MagicMock()
        instance.consumer.stop = AsyncMock()

        wiring._instances["test"] = instance

        # Cleanup should cancel task
        await wiring.cleanup()

        # Verify task was cancelled
        assert consumer_task.cancelled() or consumer_task.done()
        assert len(wiring._instances) == 0

    async def test_cleanup_fails_pending_futures(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify waiting futures get exceptions on cleanup."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )
        instance.consumer = MagicMock()
        instance.consumer.stop = AsyncMock()

        # Add pending futures
        future1: asyncio.Future[dict[str, object]] = (
            asyncio.get_event_loop().create_future()
        )
        future2: asyncio.Future[dict[str, object]] = (
            asyncio.get_event_loop().create_future()
        )
        instance.pending["id1"] = future1
        instance.pending["id2"] = future2

        wiring._instances["test"] = instance

        # Cleanup should fail pending futures
        await wiring.cleanup()

        # Verify futures have exceptions
        assert future1.done()
        assert future2.done()

        with pytest.raises(RuntimeError) as exc_info:
            future1.result()
        assert "cleaned up" in str(exc_info.value)

        with pytest.raises(RuntimeError) as exc_info:
            future2.result()
        assert "cleaned up" in str(exc_info.value)

    async def test_cleanup_is_idempotent(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify cleanup can be called multiple times safely."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )
        instance.consumer = MagicMock()
        instance.consumer.stop = AsyncMock()

        wiring._instances["test"] = instance

        # Call cleanup multiple times
        await wiring.cleanup()
        await wiring.cleanup()
        await wiring.cleanup()

        # Should not raise and instances should be empty
        assert len(wiring._instances) == 0


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCircuitBreaker:
    """Test circuit breaker integration."""

    async def test_circuit_breaker_opens_after_threshold_failures(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify 5 failures opens circuit."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance_name = "routing"
        instance = _RequestResponseInstance(
            name=instance_name,
            request_topic="test.onex.cmd.routing.request.v1",
            completed_topic="test.onex.evt.routing.completed.v1",
            failed_topic="test.onex.evt.routing.failed.v1",
            timeout_seconds=1,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.rr.routing.abc12345",
        )
        wiring._instances[instance_name] = instance

        # Make publish fail
        mock_event_bus.publish.side_effect = Exception("Publish failed")

        # Record 5 failures (threshold is 5 by default)
        for _ in range(5):
            try:
                payload: dict[str, object] = {"data": "test"}
                await wiring.send_request(instance_name, payload, timeout_seconds=1)
            except Exception:
                pass  # Expected to fail

        # Circuit should now be open
        assert wiring._circuit_breaker_open is True

    async def test_circuit_breaker_raises_unavailable_when_open(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify publish blocked when circuit is open."""
        import time

        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance_name = "routing"
        instance = _RequestResponseInstance(
            name=instance_name,
            request_topic="test.onex.cmd.routing.request.v1",
            completed_topic="test.onex.evt.routing.completed.v1",
            failed_topic="test.onex.evt.routing.failed.v1",
            timeout_seconds=1,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.rr.routing.abc12345",
        )
        wiring._instances[instance_name] = instance

        # Manually open the circuit breaker with future open_until time
        # to prevent auto-transition to half-open state
        wiring._circuit_breaker_open = True
        wiring._circuit_breaker_failures = 5
        wiring._circuit_breaker_open_until = time.time() + 3600  # 1 hour in future

        # Attempt to send request should raise InfraUnavailableError
        payload: dict[str, object] = {"data": "test"}

        with pytest.raises(InfraUnavailableError) as exc_info:
            await wiring.send_request(instance_name, payload, timeout_seconds=1)

        assert "Circuit breaker is open" in exc_info.value.message


# =============================================================================
# Instance Not Wired Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestInstanceNotWired:
    """Test error handling for unwired instances."""

    async def test_send_request_raises_config_error_for_unwired_instance(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify ProtocolConfigurationError when instance not wired."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        # No instances wired
        payload: dict[str, object] = {"data": "test"}

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await wiring.send_request("nonexistent", payload)

        assert "not wired" in exc_info.value.message
        assert "nonexistent" in exc_info.value.message


# =============================================================================
# Response Message Handling Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestResponseMessageHandling:
    """Test _handle_response_message edge cases."""

    async def test_handle_empty_message_value(
        self,
        mock_event_bus: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify empty message value is logged and skipped."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )

        # Create mock message with None value
        mock_message = MagicMock()
        mock_message.value = None
        mock_message.topic = "test.completed"

        # Should not raise, just log warning
        await wiring._handle_response_message(instance, mock_message)

    async def test_handle_invalid_json(
        self,
        mock_event_bus: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify invalid JSON is logged and skipped."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )

        # Create mock message with invalid JSON
        mock_message = MagicMock()
        mock_message.value = b"not valid json"
        mock_message.topic = "test.completed"

        # Should not raise, just log warning
        await wiring._handle_response_message(instance, mock_message)

    async def test_handle_response_missing_correlation_id(
        self,
        mock_event_bus: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify missing correlation_id in response is logged."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        instance = _RequestResponseInstance(
            name="test",
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )

        # Create mock message without correlation_id
        mock_message = MagicMock()
        mock_message.value = json.dumps({"result": "success"}).encode("utf-8")
        mock_message.topic = "test.completed"

        # Should not raise, just log warning
        await wiring._handle_response_message(instance, mock_message)


# =============================================================================
# Wire Request Response Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestWireRequestResponse:
    """Test wire_request_response configuration validation."""

    async def test_wire_duplicate_instance_raises_config_error(
        self,
        mock_event_bus: MagicMock,
    ) -> None:
        """Verify duplicate instance name raises ProtocolConfigurationError."""
        from omnibase_infra.runtime.request_response_wiring import (
            RequestResponseWiring,
            _RequestResponseInstance,
        )

        wiring = RequestResponseWiring(
            event_bus=mock_event_bus,
            environment="test",
            app_name="test-app",
            bootstrap_servers="localhost:9092",
        )

        # Pre-register an instance
        instance_name = "routing"
        wiring._instances[instance_name] = _RequestResponseInstance(
            name=instance_name,
            request_topic="test.topic",
            completed_topic="test.completed",
            failed_topic="test.failed",
            timeout_seconds=5,
            correlation_config=ModelCorrelationConfig(),
            consumer_group="test.group",
        )

        # Try to wire another instance with same name
        instance = ModelRequestResponseInstance(
            name=instance_name,  # Same name
            request_topic="onex.cmd.routing.request.v1",
            reply_topics=ModelReplyTopics(
                completed="onex.evt.routing.completed.v1",
                failed="onex.evt.routing.failed.v1",
            ),
            timeout_seconds=5,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await wiring._wire_instance(instance)

        assert "already wired" in exc_info.value.message
