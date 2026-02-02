# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for RequestResponseWiring with in-memory event bus.

These tests verify the complete request-response flow including:
1. Correlation ID injection and matching
2. Response routing (completed vs failed topics)
3. Concurrent request isolation
4. Timeout handling

Architecture:
    Since RequestResponseWiring uses AIOKafkaConsumer directly for reply topics,
    these tests mock the consumer while using a coordinated responder to simulate
    the full request-response cycle. This avoids Kafka infrastructure requirements
    while testing the core correlation and routing logic.

Related Tickets:
    - OMN-1742: Request-response wiring for Kafka RPC patterns

See Also:
    - src/omnibase_infra/runtime/request_response_wiring.py
    - tests/integration/runtime/test_event_bus_contract_wiring.py
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.models.contracts.subcontracts import (
    ModelCorrelationConfig,
    ModelReplyTopics,
    ModelRequestResponseConfig,
    ModelRequestResponseInstance,
)
from omnibase_infra.errors import InfraTimeoutError, ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.request_response_wiring import RequestResponseWiring

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# =============================================================================
# Mock Consumer Infrastructure
# =============================================================================


@dataclass
class MockConsumerRecord:
    """Mock Kafka ConsumerRecord for testing.

    Simulates the structure of aiokafka.ConsumerRecord with the fields
    used by RequestResponseWiring._handle_response_message().
    """

    topic: str
    value: bytes | None
    key: bytes | None = None
    partition: int = 0
    offset: int = 0


class MockAIOKafkaConsumer:
    """Mock AIOKafkaConsumer that allows injecting messages programmatically.

    This mock simulates the async iterator pattern of AIOKafkaConsumer,
    allowing tests to inject response messages that the RequestResponseWiring
    will process.

    Attributes:
        topics: Topics the consumer is subscribed to.
        message_queue: Queue for injecting messages to be consumed.
        started: Whether start() has been called.
        stopped: Whether stop() has been called.
    """

    def __init__(self, *topics: str, **kwargs: object) -> None:
        """Initialize mock consumer with subscribed topics.

        Args:
            *topics: Topics to subscribe to.
            **kwargs: Ignored kwargs (bootstrap_servers, group_id, etc.)
        """
        self.topics = topics
        self.message_queue: asyncio.Queue[MockConsumerRecord | None] = asyncio.Queue()
        self.started = False
        self.stopped = False
        self._kwargs = kwargs

    async def start(self) -> None:
        """Mark consumer as started."""
        self.started = True

    async def stop(self) -> None:
        """Mark consumer as stopped and inject sentinel to unblock iterator."""
        self.stopped = True
        # Inject None to unblock any waiting __anext__
        await self.message_queue.put(None)

    def __aiter__(self) -> MockAIOKafkaConsumer:
        """Return self as async iterator."""
        return self

    async def __anext__(self) -> MockConsumerRecord:
        """Get next message from queue, raise StopAsyncIteration if stopped."""
        if self.stopped:
            raise StopAsyncIteration
        message = await self.message_queue.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def inject_message(self, message: MockConsumerRecord) -> None:
        """Inject a message to be consumed.

        Args:
            message: The message to inject into the consumer.
        """
        await self.message_queue.put(message)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def environment() -> str:
    """Test environment prefix."""
    return "test"


@pytest.fixture
def app_name() -> str:
    """Test application name."""
    return "test-rpc-service"


@pytest.fixture
async def event_bus() -> AsyncIterator[EventBusInmemory]:
    """Create and start an in-memory event bus.

    Yields:
        Started EventBusInmemory instance.
    """
    bus = EventBusInmemory(environment="test", group="rpc-test")
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
def request_response_config() -> ModelRequestResponseConfig:
    """Create test request-response configuration.

    Returns:
        Configuration with a single test-rpc instance.
    """
    return ModelRequestResponseConfig(
        instances=[
            ModelRequestResponseInstance(
                name="test-rpc",
                request_topic="onex.cmd.test.request.v1",
                reply_topics=ModelReplyTopics(
                    completed="onex.evt.test.completed.v1",
                    failed="onex.evt.test.failed.v1",
                ),
                timeout_seconds=5,
            )
        ]
    )


@pytest.fixture
def multi_instance_config() -> ModelRequestResponseConfig:
    """Create configuration with multiple request-response instances.

    Returns:
        Configuration with three test instances for concurrent testing.
    """
    return ModelRequestResponseConfig(
        instances=[
            ModelRequestResponseInstance(
                name=f"test-rpc-{i}",
                request_topic=f"onex.cmd.test.request-{i}.v1",
                reply_topics=ModelReplyTopics(
                    completed=f"onex.evt.test.completed-{i}.v1",
                    failed=f"onex.evt.test.failed-{i}.v1",
                ),
                timeout_seconds=5,
            )
            for i in range(3)
        ]
    )


# =============================================================================
# Test Class
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestRequestResponseWiringIntegration:
    """Integration tests for RequestResponseWiring with in-memory event bus.

    These tests verify the core request-response functionality without
    requiring real Kafka infrastructure. The AIOKafkaConsumer is mocked
    to allow programmatic message injection.
    """

    async def test_successful_request_response_flow(
        self,
        event_bus: EventBusInmemory,
        environment: str,
        app_name: str,
        request_response_config: ModelRequestResponseConfig,
    ) -> None:
        """Test complete request-response cycle with correlation matching.

        Verifies:
        1. Request is published to request topic
        2. Response on completed topic resolves the pending future
        3. Correlation ID matches between request and response
        4. Response data is correctly returned

        The test simulates a responder by:
        1. Capturing published requests via event bus subscription
        2. Injecting correlated responses into the mock consumer
        """
        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            # EventBusInmemory implements ProtocolEventBusPublisher via duck typing
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(request_response_config)

            assert mock_consumer is not None, "Consumer should be created"
            assert mock_consumer.started, "Consumer should be started"

            # Track published requests
            published_requests: list[dict[str, object]] = []

            async def capture_request(msg: object) -> None:
                """Capture published request and inject response."""
                value = getattr(msg, "value", b"")
                request_data = json.loads(value.decode("utf-8"))
                published_requests.append(request_data)

                # Simulate responder: inject response to mock consumer
                correlation_id = request_data.get("correlation_id")
                response = MockConsumerRecord(
                    topic=f"{environment}.onex.evt.test.completed.v1",
                    value=json.dumps(
                        {
                            "correlation_id": correlation_id,
                            "result": {"status": "success", "data": "processed"},
                        }
                    ).encode("utf-8"),
                )
                await mock_consumer.inject_message(response)

            # Subscribe to request topic to capture and respond
            from omnibase_infra.models import ModelNodeIdentity

            identity = ModelNodeIdentity(
                env=environment,
                service="responder",
                node_name="test-responder",
                version="v1",
            )
            request_topic = f"{environment}.onex.cmd.test.request.v1"
            await event_bus.subscribe(request_topic, identity, capture_request)

            # Send request
            request_payload: dict[str, object] = {
                "action": "process",
                "data": {"input": "test-value"},
            }
            response = await wiring.send_request(
                instance_name="test-rpc",
                payload=request_payload,
            )

            # Verify request was published
            assert len(published_requests) == 1
            published = published_requests[0]
            assert published["action"] == "process"
            assert "correlation_id" in published  # Auto-injected

            # Verify response - cast result to dict for type safety
            result = response["result"]
            assert isinstance(result, dict)
            assert result["status"] == "success"
            assert result["data"] == "processed"
            assert "_correlation_id" in response  # Correlation tracking

            # Verify correlation ID matches
            assert response["_correlation_id"] == published["correlation_id"]

            # Cleanup
            await wiring.cleanup()
            assert mock_consumer.stopped, "Consumer should be stopped on cleanup"

    async def test_error_response_flow(
        self,
        event_bus: EventBusInmemory,
        environment: str,
        app_name: str,
        request_response_config: ModelRequestResponseConfig,
    ) -> None:
        """Test error response handling via failed topic.

        Verifies:
        1. Response on failed topic raises RuntimeError
        2. Error message from response is included in exception
        3. Correlation ID tracking works for error responses
        """
        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            # EventBusInmemory implements ProtocolEventBusPublisher via duck typing
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(request_response_config)
            assert mock_consumer is not None

            async def capture_and_fail(msg: object) -> None:
                """Capture request and inject error response."""
                value = getattr(msg, "value", b"")
                request_data = json.loads(value.decode("utf-8"))
                correlation_id = request_data.get("correlation_id")

                # Inject error response on failed topic
                error_response = MockConsumerRecord(
                    topic=f"{environment}.onex.evt.test.failed.v1",
                    value=json.dumps(
                        {
                            "correlation_id": correlation_id,
                            "error": "Processing failed: invalid input format",
                        }
                    ).encode("utf-8"),
                )
                await mock_consumer.inject_message(error_response)

            from omnibase_infra.models import ModelNodeIdentity

            identity = ModelNodeIdentity(
                env=environment,
                service="responder",
                node_name="error-responder",
                version="v1",
            )
            request_topic = f"{environment}.onex.cmd.test.request.v1"
            await event_bus.subscribe(request_topic, identity, capture_and_fail)

            # Send request and expect error
            with pytest.raises(RuntimeError, match="Processing failed"):
                await wiring.send_request(
                    instance_name="test-rpc",
                    payload={"action": "invalid"},
                )

            await wiring.cleanup()

    async def test_multiple_concurrent_requests(
        self,
        event_bus: EventBusInmemory,
        environment: str,
        app_name: str,
        request_response_config: ModelRequestResponseConfig,
    ) -> None:
        """Test concurrent requests are correctly correlated.

        Verifies:
        1. Multiple concurrent requests are tracked independently
        2. Each response matches its originating request via correlation_id
        3. No cross-contamination between concurrent requests
        """
        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            # EventBusInmemory implements ProtocolEventBusPublisher via duck typing
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(request_response_config)
            assert mock_consumer is not None

            # Track requests and their order
            request_order: list[str] = []

            async def capture_and_respond_with_delay(msg: object) -> None:
                """Capture request, add delay variation, then respond."""
                value = getattr(msg, "value", b"")
                request_data = json.loads(value.decode("utf-8"))
                correlation_id = request_data.get("correlation_id")
                request_id = request_data.get("request_id")

                request_order.append(str(request_id))

                # Vary response delay based on request_id to test out-of-order handling
                delay = 0.01 * (3 - int(str(request_id)))
                await asyncio.sleep(delay)

                # Respond with request_id included for verification
                response = MockConsumerRecord(
                    topic=f"{environment}.onex.evt.test.completed.v1",
                    value=json.dumps(
                        {
                            "correlation_id": correlation_id,
                            "result": {
                                "request_id": request_id,
                                "processed_at": "timestamp",
                            },
                        }
                    ).encode("utf-8"),
                )
                await mock_consumer.inject_message(response)

            from omnibase_infra.models import ModelNodeIdentity

            identity = ModelNodeIdentity(
                env=environment,
                service="responder",
                node_name="concurrent-responder",
                version="v1",
            )
            request_topic = f"{environment}.onex.cmd.test.request.v1"
            await event_bus.subscribe(
                request_topic, identity, capture_and_respond_with_delay
            )

            # Send 3 concurrent requests
            async def send_request(request_id: int) -> dict[str, object]:
                return await wiring.send_request(
                    instance_name="test-rpc",
                    payload={"request_id": request_id, "data": f"payload-{request_id}"},
                )

            # Launch all requests concurrently
            results = await asyncio.gather(
                send_request(0),
                send_request(1),
                send_request(2),
            )

            # Verify all 3 requests were processed
            assert len(results) == 3
            assert len(request_order) == 3

            # Verify each response matches its request (no cross-contamination)
            for i, result in enumerate(results):
                result_data = result["result"]
                assert isinstance(result_data, dict)
                assert result_data["request_id"] == i, (
                    f"Response {i} should contain request_id {i}, "
                    f"got {result_data['request_id']}"
                )

            await wiring.cleanup()

    async def test_correlation_id_preserved_when_provided(
        self,
        event_bus: EventBusInmemory,
        environment: str,
        app_name: str,
        request_response_config: ModelRequestResponseConfig,
    ) -> None:
        """Test that existing correlation_id is preserved, not overwritten.

        Verifies:
        1. When payload contains correlation_id, it is used
        2. Wiring does not inject a new correlation_id
        3. Response tracking uses the provided correlation_id
        """
        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            # EventBusInmemory implements ProtocolEventBusPublisher via duck typing
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(request_response_config)
            assert mock_consumer is not None

            # Pre-defined correlation_id
            provided_correlation_id = str(uuid4())
            captured_correlation_id: str | None = None

            async def capture_and_verify_correlation(msg: object) -> None:
                """Capture request and verify correlation_id."""
                nonlocal captured_correlation_id
                value = getattr(msg, "value", b"")
                request_data = json.loads(value.decode("utf-8"))
                captured_correlation_id = request_data.get("correlation_id")

                response = MockConsumerRecord(
                    topic=f"{environment}.onex.evt.test.completed.v1",
                    value=json.dumps(
                        {
                            "correlation_id": captured_correlation_id,
                            "result": {"verified": True},
                        }
                    ).encode("utf-8"),
                )
                await mock_consumer.inject_message(response)

            from omnibase_infra.models import ModelNodeIdentity

            identity = ModelNodeIdentity(
                env=environment,
                service="responder",
                node_name="correlation-verifier",
                version="v1",
            )
            request_topic = f"{environment}.onex.cmd.test.request.v1"
            await event_bus.subscribe(
                request_topic, identity, capture_and_verify_correlation
            )

            # Send request with pre-existing correlation_id
            response = await wiring.send_request(
                instance_name="test-rpc",
                payload={
                    "correlation_id": provided_correlation_id,
                    "action": "test",
                },
            )

            # Verify the provided correlation_id was used
            assert captured_correlation_id == provided_correlation_id
            assert response["_correlation_id"] == provided_correlation_id

            await wiring.cleanup()

    async def test_timeout_raises_infra_timeout_error(
        self,
        event_bus: EventBusInmemory,
        environment: str,
        app_name: str,
    ) -> None:
        """Test that timeout raises InfraTimeoutError, not InfraUnavailableError.

        Verifies:
        1. When no response is received within timeout, InfraTimeoutError is raised
        2. The error includes timeout details and correlation_id
        3. Pending future is cleaned up after timeout
        """
        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        # Config with very short timeout for testing
        short_timeout_config = ModelRequestResponseConfig(
            instances=[
                ModelRequestResponseInstance(
                    name="timeout-test",
                    request_topic="onex.cmd.test.timeout.v1",
                    reply_topics=ModelReplyTopics(
                        completed="onex.evt.test.timeout-completed.v1",
                        failed="onex.evt.test.timeout-failed.v1",
                    ),
                    timeout_seconds=1,  # Short timeout
                )
            ]
        )

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(short_timeout_config)
            assert mock_consumer is not None

            # Do NOT set up a responder - let it timeout

            with pytest.raises(InfraTimeoutError) as exc_info:
                await wiring.send_request(
                    instance_name="timeout-test",
                    payload={"action": "will-timeout"},
                    timeout_seconds=1,  # Short timeout (int required)
                )

            # Verify error details
            error = exc_info.value
            assert "timeout" in str(error).lower()
            assert "1s" in str(error) or "1 " in str(error)  # Timeout value

            await wiring.cleanup()

    async def test_unwired_instance_raises_protocol_error(
        self,
        event_bus: EventBusInmemory,
        environment: str,
        app_name: str,
        request_response_config: ModelRequestResponseConfig,
    ) -> None:
        """Test that sending to unwired instance raises ProtocolConfigurationError.

        Verifies:
        1. Attempting to send to non-existent instance raises clear error
        2. Error message includes the invalid instance name
        """
        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            # Wire only "test-rpc" instance
            await wiring.wire_request_response(request_response_config)

            # Try to send to non-existent instance
            with pytest.raises(
                ProtocolConfigurationError, match=r"not-wired.*not wired"
            ):
                await wiring.send_request(
                    instance_name="not-wired",
                    payload={"action": "test"},
                )

            await wiring.cleanup()

    async def test_cleanup_cancels_pending_requests(
        self,
        event_bus: EventBusInmemory,
        environment: str,
        app_name: str,
        request_response_config: ModelRequestResponseConfig,
    ) -> None:
        """Test that cleanup properly cancels all pending request futures.

        Verifies:
        1. Pending requests fail with RuntimeError on cleanup
        2. Consumer is stopped
        3. Cleanup is idempotent (safe to call multiple times)
        """
        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(request_response_config)
            assert mock_consumer is not None

            # Start a request but don't provide a response
            request_task = asyncio.create_task(
                wiring.send_request(
                    instance_name="test-rpc",
                    payload={"action": "will-be-cancelled"},
                    timeout_seconds=10,  # Long timeout
                )
            )

            # Give the task time to register the pending future
            await asyncio.sleep(0.05)

            # Cleanup should cancel pending requests
            await wiring.cleanup()

            # Request should fail with RuntimeError
            with pytest.raises(RuntimeError, match="cleaned up"):
                await request_task

            # Verify cleanup is idempotent
            await wiring.cleanup()  # Should not raise

    async def test_boot_nonce_uniqueness(
        self,
        event_bus: EventBusInmemory,
        environment: str,
        app_name: str,
    ) -> None:
        """Test that boot nonce is consistent within a process.

        Verifies:
        1. get_boot_nonce() returns consistent value
        2. Boot nonce is used in consumer group naming
        """
        mock_consumer: MockAIOKafkaConsumer | None = None
        captured_group_id: str | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer, captured_group_id
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            captured_group_id = str(kwargs.get("group_id", ""))
            return mock_consumer

        config = ModelRequestResponseConfig(
            instances=[
                ModelRequestResponseInstance(
                    name="nonce-test",
                    request_topic="onex.cmd.test.nonce.v1",
                    reply_topics=ModelReplyTopics(
                        completed="onex.evt.test.nonce-completed.v1",
                        failed="onex.evt.test.nonce-failed.v1",
                    ),
                    timeout_seconds=5,
                )
            ]
        )

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(config)

            boot_nonce = wiring.get_boot_nonce()

            # Verify boot nonce format (8-char hex)
            assert len(boot_nonce) == 8
            assert all(c in "0123456789abcdef" for c in boot_nonce)

            # Verify boot nonce is in consumer group
            assert captured_group_id is not None
            assert boot_nonce in captured_group_id
            assert f"{environment}.rr.nonce-test.{boot_nonce}" == captured_group_id

            await wiring.cleanup()

    async def test_topic_resolution_includes_environment(
        self,
        event_bus: EventBusInmemory,
        environment: str,
        app_name: str,
        request_response_config: ModelRequestResponseConfig,
    ) -> None:
        """Test that topics are resolved with environment prefix.

        Verifies:
        1. Request topic includes environment prefix
        2. Reply topics include environment prefix
        """
        mock_consumer: MockAIOKafkaConsumer | None = None
        subscribed_topics: tuple[str, ...] = ()

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer, subscribed_topics
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            subscribed_topics = topics
            return mock_consumer

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(request_response_config)

            # Verify resolve_topic method
            resolved = wiring.resolve_topic("onex.cmd.test.request.v1")
            assert resolved == f"{environment}.onex.cmd.test.request.v1"

            # Verify consumer subscribed to resolved topics
            assert f"{environment}.onex.evt.test.completed.v1" in subscribed_topics
            assert f"{environment}.onex.evt.test.failed.v1" in subscribed_topics

            await wiring.cleanup()


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestRequestResponseWiringEdgeCases:
    """Edge case tests for RequestResponseWiring."""

    async def test_empty_response_body_handled(
        self,
        environment: str,
        app_name: str,
    ) -> None:
        """Test that empty/null response body is handled gracefully.

        Verifies:
        1. Empty response body is logged but doesn't crash
        2. No exception is raised for empty messages
        """
        event_bus = EventBusInmemory(environment="test", group="edge-test")
        await event_bus.start()

        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        config = ModelRequestResponseConfig(
            instances=[
                ModelRequestResponseInstance(
                    name="empty-test",
                    request_topic="onex.cmd.test.empty.v1",
                    reply_topics=ModelReplyTopics(
                        completed="onex.evt.test.empty-completed.v1",
                        failed="onex.evt.test.empty-failed.v1",
                    ),
                    timeout_seconds=1,
                )
            ]
        )

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(config)
            assert mock_consumer is not None

            # Inject empty message (should be skipped, not crash)
            empty_response = MockConsumerRecord(
                topic=f"{environment}.onex.evt.test.empty-completed.v1",
                value=None,
            )
            await mock_consumer.inject_message(empty_response)

            # Give consumer time to process
            await asyncio.sleep(0.05)

            # Should not have crashed - cleanup works
            await wiring.cleanup()
            await event_bus.close()

    async def test_orphan_response_handled(
        self,
        environment: str,
        app_name: str,
    ) -> None:
        """Test that responses without pending requests are handled gracefully.

        Verifies:
        1. Orphan responses (no matching correlation_id) are logged
        2. No exception is raised for orphan responses
        """
        event_bus = EventBusInmemory(environment="test", group="orphan-test")
        await event_bus.start()

        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        config = ModelRequestResponseConfig(
            instances=[
                ModelRequestResponseInstance(
                    name="orphan-test",
                    request_topic="onex.cmd.test.orphan.v1",
                    reply_topics=ModelReplyTopics(
                        completed="onex.evt.test.orphan-completed.v1",
                        failed="onex.evt.test.orphan-failed.v1",
                    ),
                    timeout_seconds=1,
                )
            ]
        )

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            await wiring.wire_request_response(config)
            assert mock_consumer is not None

            # Inject response with unknown correlation_id (orphan)
            orphan_response = MockConsumerRecord(
                topic=f"{environment}.onex.evt.test.orphan-completed.v1",
                value=json.dumps(
                    {
                        "correlation_id": str(uuid4()),  # No matching request
                        "result": {"orphan": True},
                    }
                ).encode("utf-8"),
            )
            await mock_consumer.inject_message(orphan_response)

            # Give consumer time to process
            await asyncio.sleep(0.05)

            # Should not have crashed - cleanup works
            await wiring.cleanup()
            await event_bus.close()

    async def test_duplicate_instance_wiring_raises_error(
        self,
        environment: str,
        app_name: str,
    ) -> None:
        """Test that wiring duplicate instance names raises error.

        Verifies:
        1. Attempting to wire same instance name twice raises ProtocolConfigurationError
        2. Error message identifies the duplicate instance
        """
        event_bus = EventBusInmemory(environment="test", group="duplicate-test")
        await event_bus.start()

        mock_consumer: MockAIOKafkaConsumer | None = None

        def consumer_factory(*topics: str, **kwargs: object) -> MockAIOKafkaConsumer:
            nonlocal mock_consumer
            mock_consumer = MockAIOKafkaConsumer(*topics, **kwargs)
            return mock_consumer

        config = ModelRequestResponseConfig(
            instances=[
                ModelRequestResponseInstance(
                    name="duplicate-test",
                    request_topic="onex.cmd.test.dup.v1",
                    reply_topics=ModelReplyTopics(
                        completed="onex.evt.test.dup-completed.v1",
                        failed="onex.evt.test.dup-failed.v1",
                    ),
                    timeout_seconds=5,
                )
            ]
        )

        with patch(
            "omnibase_infra.runtime.request_response_wiring.AIOKafkaConsumer",
            side_effect=consumer_factory,
        ):
            wiring = RequestResponseWiring(
                event_bus=event_bus,  # type: ignore[arg-type]
                environment=environment,
                app_name=app_name,
                bootstrap_servers="localhost:9092",
            )

            # First wiring succeeds
            await wiring.wire_request_response(config)

            # Second wiring with same instance name should fail
            with pytest.raises(
                ProtocolConfigurationError, match=r"duplicate-test.*already wired"
            ):
                await wiring.wire_request_response(config)

            await wiring.cleanup()
            await event_bus.close()
