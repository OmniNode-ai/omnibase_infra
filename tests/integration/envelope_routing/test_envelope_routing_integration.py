# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for end-to-end envelope routing.

This module provides comprehensive integration tests validating the complete flow
of event envelopes through the routing system. Tests cover:

1. End-to-end envelope creation and routing through event bus
2. Envelope payload extraction and handling
3. Error scenarios in envelope routing (validation failures, unknown handlers)
4. Correlation ID propagation through the routing pipeline

Test Patterns:
- Uses EventBusInmemory for deterministic testing without external dependencies
- Tests validate complete routing flow from envelope creation to handler dispatch
- Correlation ID tracking verified at each stage of routing

These tests require no external infrastructure and can run in any environment.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import EnvelopeValidationError, UnknownHandlerTypeError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.envelope_validator import (
    PAYLOAD_REQUIRED_OPERATIONS,
    validate_envelope,
)
from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding


def _make_routing_identity(name: str) -> ModelNodeIdentity:
    """Create test identity for envelope routing tests."""
    return ModelNodeIdentity(
        env="test",
        service="envelope-routing-test",
        node_name=name,
        version="v1",
    )


# =============================================================================
# Test Configuration
# =============================================================================

# Test timing constants
MESSAGE_WAIT_TIMEOUT = 2.0
SUBSCRIBER_SETUP_DELAY = 0.05


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_bus() -> EventBusInmemory:
    """Create EventBusInmemory fixture for testing.

    Returns:
        EventBusInmemory instance configured for test environment.
    """
    return EventBusInmemory(
        environment="integration-test",
        group="envelope-routing-test",
        max_history=500,
    )


@pytest.fixture
def handler_registry() -> RegistryProtocolBinding:
    """Create handler registry with standard handlers registered.

    Registers mock handlers for common protocol types used in envelope routing.

    Returns:
        RegistryProtocolBinding with http, db, kafka, consul, vault handlers.
    """
    registry = RegistryProtocolBinding()

    # Create minimal mock handler class for testing
    class MockHandler:
        """Mock handler for testing envelope routing."""

        async def handle(
            self,
            envelope: dict[str, object],
            correlation_id: UUID | None = None,
        ) -> dict[str, object]:
            """Handle the envelope and return response.

            This method implements the ProtocolHandler protocol requirement.

            Args:
                envelope: The envelope dictionary containing operation and payload.
                correlation_id: Optional correlation ID for tracing (UUID type).

            Returns:
                Response dictionary with status and processed data.
            """
            return {
                "status": "handled",
                "operation": envelope.get("operation"),
                "payload": envelope.get("payload"),
                "correlation_id": str(correlation_id)
                if correlation_id
                else envelope.get("correlation_id"),
            }

        async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
            """Execute the envelope and return success response."""
            return {
                "status": "success",
                "payload": envelope.get("payload"),
                "correlation_id": envelope.get("correlation_id"),
            }

    # Register common handler types
    registry.register("http", MockHandler)
    registry.register("db", MockHandler)
    registry.register("kafka", MockHandler)
    registry.register("consul", MockHandler)
    registry.register("vault", MockHandler)

    return registry


@pytest.fixture
def correlation_id() -> UUID:
    """Generate a unique correlation ID for test tracing.

    Returns:
        UUID for correlation tracking through routing pipeline.
    """
    return uuid4()


# =============================================================================
# End-to-End Envelope Routing Tests
# =============================================================================


class TestEnvelopeRoutingE2E:
    """End-to-end integration tests for envelope routing through event bus."""

    @pytest.mark.asyncio
    async def test_envelope_publish_and_receive_through_event_bus(
        self,
        event_bus: EventBusInmemory,
        correlation_id: UUID,
    ) -> None:
        """Test complete envelope flow from publish to receive through event bus.

        Validates:
        1. Envelope is serialized correctly for transport
        2. Event bus delivers envelope to subscriber
        3. Envelope can be deserialized on receive
        4. Correlation ID is preserved through transport
        """
        await event_bus.start()

        received_envelopes: list[dict[str, object]] = []
        envelope_received = asyncio.Event()

        async def envelope_handler(msg: ModelEventMessage) -> None:
            """Handler that deserializes and captures envelope."""
            envelope = json.loads(msg.value.decode("utf-8"))
            received_envelopes.append(envelope)
            envelope_received.set()

        # Subscribe to envelope topic
        topic = "integration-test.envelopes"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("test-group"), envelope_handler
        )

        # Create and publish envelope
        test_envelope = {
            "operation": "http.get",
            "payload": {"url": "https://api.example.com/data"},
            "correlation_id": str(correlation_id),
            "metadata": {"source": "integration-test"},
        }

        await event_bus.publish_envelope(test_envelope, topic)

        # Wait for envelope delivery
        try:
            await asyncio.wait_for(
                envelope_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT
            )
        except TimeoutError:
            pytest.fail("Envelope not received within timeout")

        # Validate received envelope
        assert len(received_envelopes) == 1
        received = received_envelopes[0]
        assert received["operation"] == "http.get"
        assert received["payload"]["url"] == "https://api.example.com/data"
        assert received["correlation_id"] == str(correlation_id)
        assert received["metadata"]["source"] == "integration-test"

        await unsubscribe()
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_envelope_validation_before_routing(
        self,
        handler_registry: RegistryProtocolBinding,
        correlation_id: UUID,
    ) -> None:
        """Test envelope validation occurs before routing to handlers.

        Validates:
        1. Valid envelopes pass validation
        2. Correlation ID is normalized to UUID
        3. Envelope is ready for handler dispatch after validation
        """
        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": "https://api.example.com"},
            "correlation_id": str(correlation_id),
        }

        # Validate envelope (mutates to normalize correlation_id)
        validate_envelope(envelope, handler_registry)

        # Correlation ID should be normalized to UUID
        assert isinstance(envelope["correlation_id"], UUID)
        assert envelope["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_full_routing_pipeline_with_validation(
        self,
        event_bus: EventBusInmemory,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test complete routing pipeline: validation -> publish -> receive -> dispatch.

        This test simulates the full envelope routing flow:
        1. Create envelope with string correlation_id
        2. Validate envelope (normalizes correlation_id to UUID)
        3. Publish through event bus
        4. Receive and validate on consumer side
        5. Dispatch to appropriate handler
        """
        await event_bus.start()

        # Tracking structures
        dispatched_envelopes: list[dict[str, object]] = []
        dispatch_complete = asyncio.Event()

        async def routing_handler(msg: ModelEventMessage) -> None:
            """Handler that validates and dispatches received envelope."""
            envelope = json.loads(msg.value.decode("utf-8"))

            # Validate envelope before dispatch
            validate_envelope(envelope, handler_registry)

            # Track dispatched envelope
            dispatched_envelopes.append(envelope)
            dispatch_complete.set()

        topic = "integration-test.routing-pipeline"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("pipeline-group"), routing_handler
        )

        # Create envelope with string correlation_id
        original_corr_id = uuid4()
        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {"sql": "SELECT * FROM users"},
            "correlation_id": str(original_corr_id),
        }

        # Validate on publisher side
        validate_envelope(envelope, handler_registry)
        assert isinstance(envelope["correlation_id"], UUID)

        # Serialize for transport (UUID -> string)
        transport_envelope = {
            **envelope,
            "correlation_id": str(envelope["correlation_id"]),
        }

        await event_bus.publish_envelope(transport_envelope, topic)

        try:
            await asyncio.wait_for(
                dispatch_complete.wait(), timeout=MESSAGE_WAIT_TIMEOUT
            )
        except TimeoutError:
            pytest.fail("Routing pipeline did not complete within timeout")

        # Validate dispatched envelope
        assert len(dispatched_envelopes) == 1
        dispatched = dispatched_envelopes[0]
        assert dispatched["operation"] == "db.query"
        assert isinstance(dispatched["correlation_id"], UUID)
        assert dispatched["correlation_id"] == original_corr_id

        await unsubscribe()
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_multiple_envelope_types_routing(
        self,
        event_bus: EventBusInmemory,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test routing of multiple envelope types to appropriate handlers.

        Validates routing for different operation prefixes:
        - http.* -> HTTP handler
        - db.* -> Database handler
        - kafka.* -> Kafka handler
        """
        await event_bus.start()

        received_by_type: dict[str, list[dict[str, object]]] = {
            "http": [],
            "db": [],
            "kafka": [],
        }
        all_received = asyncio.Event()
        expected_count = 3

        async def type_sorting_handler(msg: ModelEventMessage) -> None:
            """Handler that sorts envelopes by operation type."""
            envelope = json.loads(msg.value.decode("utf-8"))
            validate_envelope(envelope, handler_registry)

            operation = str(envelope.get("operation", ""))
            handler_type = operation.split(".")[0]
            if handler_type in received_by_type:
                received_by_type[handler_type].append(envelope)

            total = sum(len(v) for v in received_by_type.values())
            if total >= expected_count:
                all_received.set()

        topic = "integration-test.multi-type"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("multi-group"), type_sorting_handler
        )

        # Publish different envelope types
        envelopes = [
            {"operation": "http.get", "payload": {"url": "https://example.com"}},
            {"operation": "db.query", "payload": {"sql": "SELECT 1"}},
            {
                "operation": "kafka.produce",
                "payload": {"topic": "test", "message": "hi"},
            },
        ]

        for env in envelopes:
            await event_bus.publish_envelope(env, topic)

        try:
            await asyncio.wait_for(
                all_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT * 2
            )
        except TimeoutError:
            pytest.fail("Not all envelope types received")

        assert len(received_by_type["http"]) == 1
        assert len(received_by_type["db"]) == 1
        assert len(received_by_type["kafka"]) == 1

        await unsubscribe()
        await event_bus.close()


# =============================================================================
# Envelope Payload Extraction Tests
# =============================================================================


class TestEnvelopePayloadExtraction:
    """Tests for envelope payload extraction and handling."""

    @pytest.mark.asyncio
    async def test_payload_extraction_for_operations_requiring_payload(
        self,
        event_bus: EventBusInmemory,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test payload extraction for operations that require payload.

        Operations like db.query, http.post, kafka.produce require non-empty
        payloads. This test validates payload is correctly extracted and
        available for handler processing.
        """
        await event_bus.start()

        extracted_payloads: list[dict[str, object]] = []
        all_extracted = asyncio.Event()
        expected_count = 3

        async def payload_extractor(msg: ModelEventMessage) -> None:
            """Extract and validate payloads from envelopes."""
            envelope = json.loads(msg.value.decode("utf-8"))
            validate_envelope(envelope, handler_registry)

            payload = envelope.get("payload")
            if isinstance(payload, dict):
                extracted_payloads.append(
                    {
                        "operation": envelope.get("operation"),
                        "payload": payload,
                    }
                )

            if len(extracted_payloads) >= expected_count:
                all_extracted.set()

        topic = "integration-test.payload-extraction"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("payload-group"), payload_extractor
        )

        # Test various operations with their required payloads
        test_envelopes = [
            {
                "operation": "db.query",
                "payload": {
                    "sql": "SELECT id, name FROM users WHERE active = $1",
                    "parameters": [True],
                },
            },
            {
                "operation": "http.post",
                "payload": {
                    "url": "https://api.example.com/users",
                    "body": {"name": "John"},
                },
            },
            {
                "operation": "kafka.produce",
                "payload": {
                    "topic": "user-events",
                    "message": {"event": "user_created"},
                },
            },
        ]

        for env in test_envelopes:
            await event_bus.publish_envelope(env, topic)

        try:
            await asyncio.wait_for(
                all_extracted.wait(), timeout=MESSAGE_WAIT_TIMEOUT * 2
            )
        except TimeoutError:
            pytest.fail("Not all payloads extracted")

        assert len(extracted_payloads) == 3

        # Verify db.query payload
        db_payload = next(p for p in extracted_payloads if p["operation"] == "db.query")
        assert "sql" in db_payload["payload"]
        assert "parameters" in db_payload["payload"]

        # Verify http.post payload
        http_payload = next(
            p for p in extracted_payloads if p["operation"] == "http.post"
        )
        assert "url" in http_payload["payload"]
        assert "body" in http_payload["payload"]

        # Verify kafka.produce payload
        kafka_payload = next(
            p for p in extracted_payloads if p["operation"] == "kafka.produce"
        )
        assert "topic" in kafka_payload["payload"]
        assert "message" in kafka_payload["payload"]

        await unsubscribe()
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_complex_nested_payload_extraction(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test extraction of complex nested payload structures.

        Validates that deeply nested payload structures are preserved
        through the envelope routing pipeline.
        """
        await event_bus.start()

        received_payload: dict[str, object] | None = None
        payload_received = asyncio.Event()

        async def nested_payload_handler(msg: ModelEventMessage) -> None:
            nonlocal received_payload
            envelope = json.loads(msg.value.decode("utf-8"))
            received_payload = envelope.get("payload")
            payload_received.set()

        topic = "integration-test.nested-payload"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("nested-group"), nested_payload_handler
        )

        # Complex nested payload
        complex_payload = {
            "sql": "SELECT * FROM orders WHERE user_id = $1",
            "parameters": [123],
            "options": {
                "timeout_ms": 5000,
                "retry": {
                    "enabled": True,
                    "max_attempts": 3,
                    "backoff_ms": [100, 500, 1000],
                },
            },
            "metadata": {
                "source": "order-service",
                "tags": ["critical", "user-facing"],
                "context": {
                    "user_id": 123,
                    "session": {"id": "abc-123", "created_at": "2025-01-01T00:00:00Z"},
                },
            },
        }

        envelope = {"operation": "db.query", "payload": complex_payload}
        await event_bus.publish_envelope(envelope, topic)

        try:
            await asyncio.wait_for(
                payload_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT
            )
        except TimeoutError:
            pytest.fail("Nested payload not received")

        assert received_payload is not None
        assert received_payload["sql"] == complex_payload["sql"]
        assert received_payload["parameters"] == [123]
        assert received_payload["options"]["retry"]["max_attempts"] == 3
        assert received_payload["metadata"]["context"]["session"]["id"] == "abc-123"

        await unsubscribe()
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_payload_with_binary_data_as_base64(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test payload containing binary data encoded as base64.

        Binary payloads should be base64 encoded for JSON transport and
        can be decoded by the receiving handler.
        """
        import base64

        await event_bus.start()

        received_data: bytes | None = None
        data_received = asyncio.Event()

        async def binary_handler(msg: ModelEventMessage) -> None:
            nonlocal received_data
            envelope = json.loads(msg.value.decode("utf-8"))
            payload = envelope.get("payload", {})
            if "data_base64" in payload:
                received_data = base64.b64decode(payload["data_base64"])
            data_received.set()

        topic = "integration-test.binary-payload"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("binary-group"), binary_handler
        )

        # Binary data encoded as base64
        original_data = b"\x00\x01\x02\x03\xff\xfe\xfd"
        envelope = {
            "operation": "http.post",
            "payload": {
                "url": "https://api.example.com/upload",
                "data_base64": base64.b64encode(original_data).decode("utf-8"),
                "content_type": "application/octet-stream",
            },
        }

        await event_bus.publish_envelope(envelope, topic)

        try:
            await asyncio.wait_for(data_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT)
        except TimeoutError:
            pytest.fail("Binary data not received")

        assert received_data == original_data

        await unsubscribe()
        await event_bus.close()


# =============================================================================
# Error Scenario Tests
# =============================================================================


class TestEnvelopeRoutingErrors:
    """Tests for error scenarios in envelope routing."""

    def test_missing_operation_raises_validation_error(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that envelope without operation raises EnvelopeValidationError."""
        envelope: dict[str, object] = {"payload": {"url": "https://example.com"}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, handler_registry)

        assert "operation is required" in str(exc_info.value)

    def test_empty_operation_raises_validation_error(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that envelope with empty operation raises EnvelopeValidationError."""
        envelope: dict[str, object] = {"operation": "", "payload": {}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, handler_registry)

        assert "operation is required" in str(exc_info.value)

    def test_unknown_handler_type_raises_error(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that unknown operation prefix raises UnknownHandlerTypeError."""
        envelope: dict[str, object] = {
            "operation": "unknown_protocol.some_action",
            "payload": {"data": "test"},
        }

        with pytest.raises(UnknownHandlerTypeError) as exc_info:
            validate_envelope(envelope, handler_registry)

        assert "unknown_protocol" in str(exc_info.value)
        assert "No handler registered" in str(exc_info.value)

    def test_missing_required_payload_raises_error(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that operations requiring payload fail without payload."""
        # Test each operation that requires payload
        for operation in PAYLOAD_REQUIRED_OPERATIONS:
            envelope: dict[str, object] = {"operation": operation}

            with pytest.raises(EnvelopeValidationError) as exc_info:
                validate_envelope(envelope, handler_registry)

            assert "payload is required" in str(exc_info.value)
            assert operation in str(exc_info.value)

    def test_empty_payload_raises_error_for_required_operations(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that empty payload dict raises error for operations requiring payload."""
        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {},  # Empty dict
        }

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, handler_registry)

        assert "payload is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_envelope_in_routing_pipeline(
        self,
        event_bus: EventBusInmemory,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test handling of invalid envelope in routing pipeline.

        When an invalid envelope arrives, the routing pipeline should
        handle the error gracefully without crashing.
        """
        await event_bus.start()

        validation_errors: list[Exception] = []
        valid_envelopes: list[dict[str, object]] = []
        all_processed = asyncio.Event()
        expected_total = 3

        async def error_handling_router(msg: ModelEventMessage) -> None:
            """Router that catches validation errors."""
            envelope = json.loads(msg.value.decode("utf-8"))
            try:
                validate_envelope(envelope, handler_registry)
                valid_envelopes.append(envelope)
            except (EnvelopeValidationError, UnknownHandlerTypeError) as e:
                validation_errors.append(e)

            if len(valid_envelopes) + len(validation_errors) >= expected_total:
                all_processed.set()

        topic = "integration-test.error-handling"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("error-group"), error_handling_router
        )

        # Mix of valid and invalid envelopes
        envelopes = [
            {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
            },  # Valid
            {"operation": "unknown.action", "payload": {}},  # Invalid handler
            {"operation": "db.query", "payload": {"sql": "SELECT 1"}},  # Valid
        ]

        for env in envelopes:
            await event_bus.publish_envelope(env, topic)

        try:
            await asyncio.wait_for(
                all_processed.wait(), timeout=MESSAGE_WAIT_TIMEOUT * 2
            )
        except TimeoutError:
            pytest.fail("Not all envelopes processed")

        # Should have 2 valid and 1 error
        assert len(valid_envelopes) == 2
        assert len(validation_errors) == 1
        assert isinstance(validation_errors[0], UnknownHandlerTypeError)

        await unsubscribe()
        await event_bus.close()


# =============================================================================
# Correlation ID Propagation Tests
# =============================================================================


class TestCorrelationIdPropagation:
    """Tests for correlation ID propagation through envelope routing."""

    def test_missing_correlation_id_is_generated(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that missing correlation_id is auto-generated as UUID."""
        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": "https://example.com"},
        }

        validate_envelope(envelope, handler_registry)

        assert "correlation_id" in envelope
        assert isinstance(envelope["correlation_id"], UUID)

    def test_string_correlation_id_converted_to_uuid(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that string correlation_id is converted to UUID."""
        original_id = uuid4()
        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": "https://example.com"},
            "correlation_id": str(original_id),
        }

        validate_envelope(envelope, handler_registry)

        assert isinstance(envelope["correlation_id"], UUID)
        assert envelope["correlation_id"] == original_id

    def test_uuid_correlation_id_preserved(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that UUID correlation_id is preserved unchanged."""
        original_id = uuid4()
        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": "https://example.com"},
            "correlation_id": original_id,
        }

        validate_envelope(envelope, handler_registry)

        assert envelope["correlation_id"] is original_id

    def test_invalid_string_correlation_id_replaced(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that invalid string correlation_id is replaced with new UUID."""
        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": "https://example.com"},
            "correlation_id": "not-a-valid-uuid",
        }

        validate_envelope(envelope, handler_registry)

        assert isinstance(envelope["correlation_id"], UUID)

    @pytest.mark.asyncio
    async def test_correlation_id_preserved_through_event_bus_transport(
        self,
        event_bus: EventBusInmemory,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test correlation ID is preserved through event bus transport.

        Validates that correlation_id:
        1. Is serialized as string for JSON transport
        2. Is correctly deserialized on receive
        3. Can be converted back to UUID after validation
        """
        await event_bus.start()

        received_correlation_ids: list[UUID] = []
        all_received = asyncio.Event()
        expected_count = 3

        async def correlation_tracker(msg: ModelEventMessage) -> None:
            """Track correlation IDs through routing."""
            envelope = json.loads(msg.value.decode("utf-8"))
            validate_envelope(envelope, handler_registry)
            received_correlation_ids.append(envelope["correlation_id"])

            if len(received_correlation_ids) >= expected_count:
                all_received.set()

        topic = "integration-test.correlation-tracking"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("corr-group"), correlation_tracker
        )

        # Publish envelopes with specific correlation IDs
        original_ids = [uuid4() for _ in range(3)]
        for corr_id in original_ids:
            envelope = {
                "operation": "http.get",
                "payload": {"url": "https://example.com"},
                "correlation_id": str(corr_id),  # String for JSON transport
            }
            await event_bus.publish_envelope(envelope, topic)

        try:
            await asyncio.wait_for(
                all_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT * 2
            )
        except TimeoutError:
            pytest.fail("Not all correlation IDs tracked")

        # All correlation IDs should be preserved
        assert len(received_correlation_ids) == 3
        for orig, received in zip(original_ids, received_correlation_ids, strict=True):
            assert received == orig

        await unsubscribe()
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_correlation_id_in_event_headers(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test correlation ID is propagated to event message headers.

        When publishing with custom headers that include correlation_id,
        the header correlation_id should match the envelope correlation_id.
        """
        await event_bus.start()

        received_message: ModelEventMessage | None = None
        message_received = asyncio.Event()

        async def header_checker(msg: ModelEventMessage) -> None:
            nonlocal received_message
            received_message = msg
            message_received.set()

        topic = "integration-test.header-correlation"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("header-group"), header_checker
        )

        # Publish with explicit correlation_id in headers
        correlation_id = uuid4()
        headers = ModelEventHeaders(
            source="integration-test",
            event_type="envelope.test",
            correlation_id=correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        envelope = {
            "operation": "http.get",
            "payload": {"url": "https://example.com"},
            "correlation_id": str(correlation_id),
        }

        await event_bus.publish(
            topic, None, json.dumps(envelope).encode("utf-8"), headers
        )

        try:
            await asyncio.wait_for(
                message_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT
            )
        except TimeoutError:
            pytest.fail("Message not received")

        assert received_message is not None
        assert received_message.headers.correlation_id == correlation_id

        await unsubscribe()
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_correlation_id_chain_through_multiple_hops(
        self,
        event_bus: EventBusInmemory,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test correlation ID preserved through multi-hop routing.

        Simulates a scenario where an envelope is processed by multiple
        handlers in sequence, each potentially publishing new envelopes.
        The original correlation_id should be preserved throughout.
        """
        await event_bus.start()

        hop_correlation_ids: list[UUID] = []
        final_hop = asyncio.Event()
        original_correlation_id = uuid4()

        async def hop1_handler(msg: ModelEventMessage) -> None:
            """First hop - validates and forwards."""
            envelope = json.loads(msg.value.decode("utf-8"))
            validate_envelope(envelope, handler_registry)
            hop_correlation_ids.append(envelope["correlation_id"])

            # Forward to hop2 with same correlation_id
            forward_envelope = {
                "operation": "db.query",
                "payload": {"sql": "SELECT 1"},
                "correlation_id": str(envelope["correlation_id"]),
            }
            await event_bus.publish_envelope(forward_envelope, "integration-test.hop2")

        async def hop2_handler(msg: ModelEventMessage) -> None:
            """Second hop - validates and completes."""
            envelope = json.loads(msg.value.decode("utf-8"))
            validate_envelope(envelope, handler_registry)
            hop_correlation_ids.append(envelope["correlation_id"])
            final_hop.set()

        unsub1 = await event_bus.subscribe(
            "integration-test.hop1", _make_routing_identity("hop1-group"), hop1_handler
        )
        unsub2 = await event_bus.subscribe(
            "integration-test.hop2", _make_routing_identity("hop2-group"), hop2_handler
        )

        # Start the chain
        initial_envelope = {
            "operation": "http.get",
            "payload": {"url": "https://example.com"},
            "correlation_id": str(original_correlation_id),
        }
        await event_bus.publish_envelope(initial_envelope, "integration-test.hop1")

        try:
            await asyncio.wait_for(final_hop.wait(), timeout=MESSAGE_WAIT_TIMEOUT * 2)
        except TimeoutError:
            pytest.fail("Multi-hop routing did not complete")

        # Both hops should have the same correlation_id
        assert len(hop_correlation_ids) == 2
        assert hop_correlation_ids[0] == original_correlation_id
        assert hop_correlation_ids[1] == original_correlation_id

        await unsub1()
        await unsub2()
        await event_bus.close()


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


class TestEnvelopeRoutingEdgeCases:
    """Tests for edge cases and boundary conditions in envelope routing."""

    @pytest.mark.asyncio
    async def test_high_volume_envelope_routing(
        self,
        event_bus: EventBusInmemory,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test envelope routing under high volume.

        Validates system stability when routing many envelopes concurrently.
        """
        await event_bus.start()

        received_count = 0
        all_received = asyncio.Event()
        expected_count = 100
        lock = asyncio.Lock()

        async def counter_handler(msg: ModelEventMessage) -> None:
            nonlocal received_count
            envelope = json.loads(msg.value.decode("utf-8"))
            validate_envelope(envelope, handler_registry)

            async with lock:
                received_count += 1
                if received_count >= expected_count:
                    all_received.set()

        topic = "integration-test.high-volume"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("volume-group"), counter_handler
        )

        # Publish many envelopes concurrently
        async def publish_batch(start: int, count: int) -> None:
            for i in range(count):
                envelope = {
                    "operation": "http.get",
                    "payload": {"url": f"https://example.com/{start + i}"},
                    "correlation_id": str(uuid4()),
                }
                await event_bus.publish_envelope(envelope, topic)

        # Publish in parallel batches
        await asyncio.gather(
            publish_batch(0, 25),
            publish_batch(25, 25),
            publish_batch(50, 25),
            publish_batch(75, 25),
        )

        try:
            await asyncio.wait_for(
                all_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT * 3
            )
        except TimeoutError:
            pytest.fail(f"Only received {received_count}/{expected_count} envelopes")

        assert received_count == expected_count

        await unsubscribe()
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_envelope_with_none_values(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test envelope handling with None values in payload.

        Some payloads may contain None values which should be
        preserved through routing.
        """
        await event_bus.start()

        received_payload: dict[str, object] | None = None
        payload_received = asyncio.Event()

        async def none_handler(msg: ModelEventMessage) -> None:
            nonlocal received_payload
            envelope = json.loads(msg.value.decode("utf-8"))
            received_payload = envelope.get("payload")
            payload_received.set()

        topic = "integration-test.none-values"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("none-group"), none_handler
        )

        envelope = {
            "operation": "http.post",
            "payload": {
                "url": "https://example.com",
                "body": {"name": "John", "email": None, "phone": None},
            },
        }
        await event_bus.publish_envelope(envelope, topic)

        try:
            await asyncio.wait_for(
                payload_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT
            )
        except TimeoutError:
            pytest.fail("Envelope with None values not received")

        assert received_payload is not None
        assert received_payload["body"]["name"] == "John"
        assert received_payload["body"]["email"] is None
        assert received_payload["body"]["phone"] is None

        await unsubscribe()
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_envelope_with_unicode_content(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test envelope routing with unicode content.

        Validates that unicode characters in payloads are correctly
        preserved through JSON serialization and event bus transport.
        """
        await event_bus.start()

        received_content: str | None = None
        content_received = asyncio.Event()

        async def unicode_handler(msg: ModelEventMessage) -> None:
            nonlocal received_content
            envelope = json.loads(msg.value.decode("utf-8"))
            received_content = envelope.get("payload", {}).get("message")
            content_received.set()

        topic = "integration-test.unicode"
        unsubscribe = await event_bus.subscribe(
            topic, _make_routing_identity("unicode-group"), unicode_handler
        )

        # Unicode content including various scripts and emoji
        unicode_message = "Hello World! Japanese characters \u65e5\u672c\u8a9e Chinese characters \u4e2d\u6587 Emoji: \U0001f600\U0001f389"
        envelope = {
            "operation": "http.post",
            "payload": {"url": "https://example.com", "message": unicode_message},
        }
        await event_bus.publish_envelope(envelope, topic)

        try:
            await asyncio.wait_for(
                content_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT
            )
        except TimeoutError:
            pytest.fail("Unicode content not received")

        assert received_content == unicode_message

        await unsubscribe()
        await event_bus.close()

    def test_operation_without_dot_uses_whole_string_as_prefix(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that operation without dot uses entire string as handler prefix.

        When operation is just "http" (no dot), it should match the "http" handler.
        """
        envelope: dict[str, object] = {"operation": "http"}

        # Should not raise - "http" is a registered prefix
        validate_envelope(envelope, handler_registry)

    def test_case_sensitive_operation_prefix(
        self,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test that operation prefix matching is case-sensitive.

        "HTTP.get" should not match "http" handler.
        """
        envelope: dict[str, object] = {"operation": "HTTP.get"}

        with pytest.raises(UnknownHandlerTypeError):
            validate_envelope(envelope, handler_registry)
