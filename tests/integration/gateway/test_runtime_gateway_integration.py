# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for runtime host process gateway signing paths.

Tests verify that:
    - _publish_envelope_safe signs dict envelopes when signer is configured
    - _publish_model_safe signs BaseModel payloads when signer is configured
    - Signed envelopes serialize correctly (model_dump produces JSON-safe types)
    - Signing failure degrades gracefully (publishes unsigned)
    - UUID and datetime fields are properly serialized in the output
    - Policy engine rejection prevents publishing
    - _validate_gateway_envelope rejects unsigned envelopes when reject_unsigned=True
    - _validate_gateway_envelope rejects policy-denied inbound messages
    - _validate_gateway_envelope validates signed envelopes and extracts payload

Related Tickets:
    - OMN-1899: Runtime gateway envelope signing
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.gateway.models.model_gateway_config import ModelGatewayConfig
from omnibase_infra.gateway.services.service_envelope_signer import (
    ServiceEnvelopeSigner,
)
from omnibase_infra.gateway.services.service_envelope_validator import (
    ServiceEnvelopeValidator,
    ValidationResult,
)
from omnibase_infra.gateway.services.service_policy_engine import ServicePolicyEngine
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess

pytestmark = pytest.mark.integration


# =============================================================================
# Test Models
# =============================================================================


class ModelTestEvent(BaseModel):
    """Minimal event model for testing publish paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: str = Field(..., description="Action type")
    resource_id: str = Field(..., description="Resource identifier")
    correlation_id: UUID = Field(default_factory=uuid4, description="Correlation ID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )


# =============================================================================
# Fixtures
# =============================================================================


def _make_runtime(
    event_bus: EventBusInmemory | None = None,
) -> RuntimeHostProcess:
    """Create a minimal RuntimeHostProcess for testing.

    Args:
        event_bus: Optional event bus. Creates a new one if not provided.

    Returns:
        RuntimeHostProcess configured for testing.
    """
    bus = event_bus or EventBusInmemory()
    return RuntimeHostProcess(
        event_bus=bus,
        config={
            "service_name": "test-service",
            "node_name": "test-node",
            "env": "test",
        },
    )


def _make_signer() -> ServiceEnvelopeSigner:
    """Create a ServiceEnvelopeSigner with a fresh Ed25519 keypair.

    Returns:
        ServiceEnvelopeSigner configured for testing.
    """
    private_key = Ed25519PrivateKey.generate()
    return ServiceEnvelopeSigner(
        realm="test",
        runtime_id="test-runtime-001",
        private_key=private_key,
    )


# =============================================================================
# Tests: _publish_envelope_safe - Signing Path
# =============================================================================


class TestPublishEnvelopeSafeSigning:
    """Tests for dict envelope signing in _publish_envelope_safe."""

    @pytest.mark.asyncio
    async def test_publish_envelope_safe_signs_when_signer_configured(self) -> None:
        """When envelope signer is configured, dict envelopes are signed.

        Note: The envelope dict must contain JSON-serializable values (strings,
        not UUID objects) because hash_canonical_json uses json.dumps internally.
        UUID objects in the dict would cause signing to fail and degrade gracefully.
        """
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        # Use string values - UUID objects are not JSON-serializable for signing
        envelope = {
            "success": True,
            "status": "ok",
            "correlation_id": str(uuid4()),
            "data": {"key": "value"},
        }

        # Act
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert - publish_envelope was called
        bus.publish_envelope.assert_called_once()
        published_dict = bus.publish_envelope.call_args[0][0]
        published_topic = bus.publish_envelope.call_args[0][1]

        assert published_topic == "test-topic"
        # Signed envelope should have signature and realm fields
        assert "signature" in published_dict
        assert "realm" in published_dict
        assert published_dict["realm"] == "test"
        assert published_dict["runtime_id"] == "test-runtime-001"

    @pytest.mark.asyncio
    async def test_publish_envelope_safe_unsigned_when_no_signer(self) -> None:
        """When no signer is configured, dict envelopes are published raw."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)
        # No signer configured (default)

        envelope = {
            "success": True,
            "status": "ok",
            "data": {"key": "value"},
        }

        # Act
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert
        bus.publish_envelope.assert_called_once()
        published_dict = bus.publish_envelope.call_args[0][0]
        # Raw envelope should NOT have signature fields
        assert "signature" not in published_dict
        assert published_dict["success"] is True

    @pytest.mark.asyncio
    async def test_publish_envelope_safe_graceful_degradation_on_mock_signing_failure(
        self,
    ) -> None:
        """When signing fails (mocked), the original envelope is published unsigned."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        # Create a signer that raises on sign_dict
        broken_signer = MagicMock(spec=ServiceEnvelopeSigner)
        broken_signer.sign_dict.side_effect = RuntimeError("Signing failed")
        broken_signer.realm = "test"
        broken_signer.runtime_id = "test-runtime-001"
        runtime._envelope_signer = broken_signer

        envelope = {
            "success": True,
            "status": "ok",
            "correlation_id": str(uuid4()),
        }

        # Act
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert - should still publish, just unsigned
        bus.publish_envelope.assert_called_once()
        published_dict = bus.publish_envelope.call_args[0][0]
        # The original envelope is published (without signing wrapper)
        assert published_dict["success"] is True
        assert published_dict["status"] == "ok"

    @pytest.mark.asyncio
    async def test_publish_envelope_safe_graceful_degradation_uuid_in_dict(
        self,
    ) -> None:
        """UUID objects in dict cause signing to fail gracefully (publishes unsigned).

        This tests the real-world scenario where _make_error_response creates
        dicts with UUID objects. hash_canonical_json uses json.dumps which cannot
        serialize UUID objects, so signing fails and degrades to unsigned publish.
        """
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        # Envelope with raw UUID object (not JSON-serializable for signing)
        envelope = {
            "success": True,
            "status": "ok",
            "correlation_id": uuid4(),  # UUID object, not string
        }

        # Act
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert - signing fails, publishes unsigned but UUID still serialized
        bus.publish_envelope.assert_called_once()
        published_dict = bus.publish_envelope.call_args[0][0]
        assert published_dict["success"] is True
        # UUID should be serialized to string by _serialize_envelope
        assert isinstance(published_dict["correlation_id"], str)
        # No signing wrapper present (graceful degradation)
        assert "signature" not in published_dict


# =============================================================================
# Tests: _publish_envelope_safe - UUID Serialization
# =============================================================================


class TestPublishEnvelopeSafeSerialization:
    """Tests for UUID/datetime serialization in _publish_envelope_safe."""

    @pytest.mark.asyncio
    async def test_uuid_fields_serialized_to_strings(self) -> None:
        """UUID objects in envelope dict are converted to strings before publishing."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        test_uuid = uuid4()
        envelope = {
            "correlation_id": test_uuid,
            "nested": {"inner_id": uuid4()},
            "items": [uuid4(), uuid4()],
        }

        # Act
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert
        bus.publish_envelope.assert_called_once()
        published = bus.publish_envelope.call_args[0][0]

        # All UUIDs should be strings
        assert isinstance(published["correlation_id"], str)
        assert published["correlation_id"] == str(test_uuid)
        assert isinstance(published["nested"]["inner_id"], str)
        assert all(isinstance(item, str) for item in published["items"])

    @pytest.mark.asyncio
    async def test_signed_envelope_json_serializable(self) -> None:
        """Signed envelope dict is fully JSON-serializable."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        # Use string correlation_id so signing succeeds
        envelope = {
            "success": True,
            "correlation_id": str(uuid4()),
            "data": {"key": "value"},
        }

        # Act
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert - the published dict must be JSON-serializable
        published = bus.publish_envelope.call_args[0][0]
        json_str = json.dumps(published)
        assert isinstance(json_str, str)
        # Round-trip should work
        parsed = json.loads(json_str)
        assert "signature" in parsed
        assert "realm" in parsed

    @pytest.mark.asyncio
    async def test_correlation_id_extraction_from_uuid_object(self) -> None:
        """Correlation ID is correctly extracted when it is a UUID object."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        correlation_id = uuid4()
        envelope = {"correlation_id": correlation_id, "data": "test"}

        # Act - should not raise
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert
        bus.publish_envelope.assert_called_once()

    @pytest.mark.asyncio
    async def test_correlation_id_extraction_from_string(self) -> None:
        """Correlation ID is correctly extracted when it is a string."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        correlation_id = str(uuid4())
        envelope = {"correlation_id": correlation_id, "data": "test"}

        # Act - should not raise
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert
        bus.publish_envelope.assert_called_once()


# =============================================================================
# Tests: _publish_model_safe - Signing Path
# =============================================================================


class TestPublishModelSafeSigning:
    """Tests for BaseModel signing in _publish_model_safe."""

    @pytest.mark.asyncio
    async def test_publish_model_safe_signs_when_signer_configured(self) -> None:
        """When envelope signer is configured, BaseModel payloads are signed."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        model = ModelTestEvent(
            action="created",
            resource_id="resource-123",
        )

        # Act
        await runtime._publish_model_safe(model, "test-topic")

        # Assert
        bus.publish_envelope.assert_called_once()
        published_dict = bus.publish_envelope.call_args[0][0]

        # Signed envelope should have gateway fields
        assert "signature" in published_dict
        assert "realm" in published_dict
        assert published_dict["realm"] == "test"
        assert published_dict["runtime_id"] == "test-runtime-001"

    @pytest.mark.asyncio
    async def test_publish_model_safe_unsigned_when_no_signer(self) -> None:
        """When no signer is configured, model is dumped and published directly."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        model = ModelTestEvent(
            action="updated",
            resource_id="resource-456",
        )

        # Act
        await runtime._publish_model_safe(model, "test-topic")

        # Assert
        bus.publish_envelope.assert_called_once()
        published_dict = bus.publish_envelope.call_args[0][0]

        # Raw model dict should NOT have signing wrapper
        assert "signature" not in published_dict
        assert published_dict["action"] == "updated"
        assert published_dict["resource_id"] == "resource-456"

    @pytest.mark.asyncio
    async def test_publish_model_safe_graceful_degradation_on_signing_failure(
        self,
    ) -> None:
        """When signing fails, the model is published unsigned."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        broken_signer = MagicMock(spec=ServiceEnvelopeSigner)
        broken_signer.sign_envelope.side_effect = RuntimeError("Signing failed")
        broken_signer.realm = "test"
        broken_signer.runtime_id = "test-runtime-001"
        runtime._envelope_signer = broken_signer

        model = ModelTestEvent(
            action="deleted",
            resource_id="resource-789",
        )

        # Act
        await runtime._publish_model_safe(model, "test-topic")

        # Assert - should publish unsigned model dict
        bus.publish_envelope.assert_called_once()
        published_dict = bus.publish_envelope.call_args[0][0]
        assert published_dict["action"] == "deleted"
        assert published_dict["resource_id"] == "resource-789"

    @pytest.mark.asyncio
    async def test_publish_model_safe_extracts_correlation_id_from_model(
        self,
    ) -> None:
        """Correlation ID is extracted from model.correlation_id attribute."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        correlation_id = uuid4()
        model = ModelTestEvent(
            action="created",
            resource_id="resource-123",
            correlation_id=correlation_id,
        )

        # Act
        await runtime._publish_model_safe(model, "test-topic")

        # Assert - signed envelope should contain the trace_id
        bus.publish_envelope.assert_called_once()
        published_dict = bus.publish_envelope.call_args[0][0]
        assert "trace_id" in published_dict
        # trace_id should be the correlation_id from the model
        assert published_dict["trace_id"] == str(correlation_id)

    @pytest.mark.asyncio
    async def test_publish_model_safe_uses_explicit_correlation_id(self) -> None:
        """Explicit correlation_id kwarg takes precedence over model attribute."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        explicit_id = uuid4()
        model = ModelTestEvent(
            action="created",
            resource_id="resource-123",
        )

        # Act
        await runtime._publish_model_safe(
            model, "test-topic", correlation_id=explicit_id
        )

        # Assert
        bus.publish_envelope.assert_called_once()
        published_dict = bus.publish_envelope.call_args[0][0]
        assert published_dict["trace_id"] == str(explicit_id)


# =============================================================================
# Tests: _publish_model_safe - Serialization
# =============================================================================


class TestPublishModelSafeSerialization:
    """Tests for serialization correctness in _publish_model_safe."""

    @pytest.mark.asyncio
    async def test_signed_model_envelope_json_serializable(self) -> None:
        """Signed model envelope is fully JSON-serializable."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        model = ModelTestEvent(
            action="created",
            resource_id="resource-123",
        )

        # Act
        await runtime._publish_model_safe(model, "test-topic")

        # Assert
        published = bus.publish_envelope.call_args[0][0]
        json_str = json.dumps(published)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert "signature" in parsed
        assert "payload" in parsed
        # Payload should contain the model fields
        payload = parsed["payload"]
        assert payload["action"] == "created"
        assert payload["resource_id"] == "resource-123"

    @pytest.mark.asyncio
    async def test_unsigned_model_uuid_and_datetime_serialized(self) -> None:
        """UUID and datetime fields in unsigned model are JSON-serializable."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        model = ModelTestEvent(
            action="created",
            resource_id="resource-123",
        )

        # Act
        await runtime._publish_model_safe(model, "test-topic")

        # Assert - model_dump(mode="json") should handle UUID and datetime
        published = bus.publish_envelope.call_args[0][0]
        json_str = json.dumps(published)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        # correlation_id should be a string (UUID serialized)
        assert isinstance(parsed["correlation_id"], str)
        # created_at should be a string (datetime serialized)
        assert isinstance(parsed["created_at"], str)

    @pytest.mark.asyncio
    async def test_signed_model_datetime_fields_are_strings(self) -> None:
        """DateTime fields in signed envelope are serialized as ISO strings."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        model = ModelTestEvent(
            action="created",
            resource_id="resource-123",
        )

        # Act
        await runtime._publish_model_safe(model, "test-topic")

        # Assert
        published = bus.publish_envelope.call_args[0][0]
        # emitted_at from the signed envelope should be a string
        assert isinstance(published["emitted_at"], str)
        # Verify it parses back to a datetime
        parsed_dt = datetime.fromisoformat(published["emitted_at"])
        assert parsed_dt.tzinfo is not None  # Must be timezone-aware


# =============================================================================
# Tests: _serialize_envelope
# =============================================================================


class TestSerializeEnvelope:
    """Tests for the _serialize_envelope UUID conversion helper."""

    def test_serialize_envelope_converts_top_level_uuids(self) -> None:
        """Top-level UUID values are converted to strings."""
        # Arrange
        runtime = _make_runtime()
        test_id = uuid4()
        envelope: dict[str, object] = {"id": test_id, "name": "test"}

        # Act
        result = runtime._serialize_envelope(envelope)

        # Assert
        assert result["id"] == str(test_id)
        assert result["name"] == "test"

    def test_serialize_envelope_converts_nested_uuids(self) -> None:
        """Nested UUID values in dicts are converted to strings."""
        # Arrange
        runtime = _make_runtime()
        inner_id = uuid4()
        envelope: dict[str, object] = {"nested": {"inner_id": inner_id}}

        # Act
        result = runtime._serialize_envelope(envelope)

        # Assert
        assert result["nested"]["inner_id"] == str(inner_id)

    def test_serialize_envelope_converts_list_uuids(self) -> None:
        """UUID values in lists are converted to strings."""
        # Arrange
        runtime = _make_runtime()
        id1 = uuid4()
        id2 = uuid4()
        envelope: dict[str, object] = {"ids": [id1, id2]}

        # Act
        result = runtime._serialize_envelope(envelope)

        # Assert
        assert result["ids"] == [str(id1), str(id2)]

    def test_serialize_envelope_preserves_non_uuid_types(self) -> None:
        """Non-UUID types (str, int, bool, None) are preserved as-is."""
        # Arrange
        runtime = _make_runtime()
        envelope: dict[str, object] = {
            "name": "test",
            "count": 42,
            "active": True,
            "missing": None,
        }

        # Act
        result = runtime._serialize_envelope(envelope)

        # Assert
        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["active"] is True
        assert result["missing"] is None


# =============================================================================
# Tests: Policy Engine Integration
# =============================================================================


class TestPublishPolicyEnforcement:
    """Tests for policy engine integration in publish paths."""

    @pytest.mark.asyncio
    async def test_publish_envelope_safe_rejected_by_policy(self) -> None:
        """When policy rejects outbound, message is not published."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        # Create a policy engine mock that rejects
        policy_engine = MagicMock()
        rejection = MagicMock()
        rejection.__bool__ = lambda self: False
        rejection.reason = "Topic not in allowlist"
        policy_engine.evaluate_outbound.return_value = rejection
        runtime._policy_engine = policy_engine

        envelope = {"success": True, "data": "test"}

        # Act
        await runtime._publish_envelope_safe(envelope, "blocked-topic")

        # Assert - publish_envelope should NOT be called
        bus.publish_envelope.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_model_safe_rejected_by_policy(self) -> None:
        """When policy rejects outbound model, message is not published."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        policy_engine = MagicMock()
        rejection = MagicMock()
        rejection.__bool__ = lambda self: False
        rejection.reason = "Topic not in allowlist"
        policy_engine.evaluate_outbound.return_value = rejection
        runtime._policy_engine = policy_engine

        model = ModelTestEvent(
            action="created",
            resource_id="resource-123",
        )

        # Act
        await runtime._publish_model_safe(model, "blocked-topic")

        # Assert
        bus.publish_envelope.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_envelope_safe_allowed_by_policy(self) -> None:
        """When policy allows outbound, message is published normally."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        runtime = _make_runtime(event_bus=bus)

        policy_engine = MagicMock()
        approval = MagicMock()
        approval.__bool__ = lambda self: True
        policy_engine.evaluate_outbound.return_value = approval
        runtime._policy_engine = policy_engine

        envelope = {"success": True, "data": "test"}

        # Act
        await runtime._publish_envelope_safe(envelope, "allowed-topic")

        # Assert
        bus.publish_envelope.assert_called_once()


# =============================================================================
# Tests: Bus ID Extraction
# =============================================================================


class TestBusIdExtraction:
    """Tests for bus_id extraction from event bus in signing paths."""

    @pytest.mark.asyncio
    async def test_publish_envelope_safe_uses_bus_id_from_event_bus(self) -> None:
        """When event bus has bus_id attribute, it is used in signing."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        bus.bus_id = "custom-bus-123"
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        envelope = {"success": True, "data": "test"}

        # Act
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert
        published = bus.publish_envelope.call_args[0][0]
        assert published["bus_id"] == "custom-bus-123"

    @pytest.mark.asyncio
    async def test_publish_envelope_safe_uses_default_bus_id(self) -> None:
        """When event bus has no bus_id attribute, 'default' is used."""
        # Arrange
        bus = EventBusInmemory()
        bus.publish_envelope = AsyncMock()
        # Remove bus_id if it exists
        if hasattr(bus, "bus_id"):
            delattr(bus, "bus_id")
        runtime = _make_runtime(event_bus=bus)

        signer = _make_signer()
        runtime._envelope_signer = signer

        envelope = {"success": True, "data": "test"}

        # Act
        await runtime._publish_envelope_safe(envelope, "test-topic")

        # Assert
        published = bus.publish_envelope.call_args[0][0]
        assert published["bus_id"] == "default"


# =============================================================================
# Tests: _validate_gateway_envelope - Inbound Validation
# =============================================================================


class TestValidateGatewayEnvelopeRejectUnsigned:
    """Tests for reject_unsigned enforcement in _validate_gateway_envelope."""

    @pytest.mark.asyncio
    async def test_reject_unsigned_rejects_unsigned_envelope(self) -> None:
        """When reject_unsigned=True, unsigned envelopes are rejected (returns None)."""
        # Arrange
        runtime = _make_runtime()

        # Configure gateway with reject_unsigned=True but NO validator
        # (simulates no public_key_path configured)
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            reject_unsigned=True,
        )
        runtime._gateway_config = config
        # No _envelope_validator set (no public_key_path)
        runtime._envelope_validator = None
        runtime._policy_engine = None

        unsigned_envelope: dict[str, object] = {
            "success": True,
            "data": "test",
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await runtime._validate_gateway_envelope(
            unsigned_envelope, "events.test"
        )

        # Assert - unsigned envelope should be rejected
        assert result is None

    @pytest.mark.asyncio
    async def test_reject_unsigned_false_accepts_unsigned_envelope(self) -> None:
        """When reject_unsigned=False, unsigned envelopes are accepted."""
        # Arrange
        runtime = _make_runtime()

        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            reject_unsigned=False,
        )
        runtime._gateway_config = config
        runtime._envelope_validator = None
        runtime._policy_engine = None

        unsigned_envelope: dict[str, object] = {
            "success": True,
            "data": "test",
        }

        # Act
        result = await runtime._validate_gateway_envelope(
            unsigned_envelope, "events.test"
        )

        # Assert - unsigned envelope should be accepted
        assert result is not None
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_gateway_config_passes_through(self) -> None:
        """When no gateway is configured, envelopes pass through unchanged."""
        # Arrange
        runtime = _make_runtime()
        # Default state: no gateway config, no validator, no policy engine

        envelope: dict[str, object] = {"data": "test", "key": "value"}

        # Act
        result = await runtime._validate_gateway_envelope(envelope, "any-topic")

        # Assert
        assert result is envelope  # Same object, unchanged


class TestValidateGatewayEnvelopePolicyCheck:
    """Tests for policy engine integration in _validate_gateway_envelope."""

    @pytest.mark.asyncio
    async def test_policy_rejection_blocks_inbound(self) -> None:
        """When policy engine rejects inbound topic, envelope is rejected."""
        # Arrange
        runtime = _make_runtime()

        policy = ServicePolicyEngine(
            allowed_topics=["events.*"],
            expected_realm=None,
        )
        runtime._policy_engine = policy
        runtime._gateway_config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            reject_unsigned=False,
        )

        envelope: dict[str, object] = {"data": "test"}

        # Act - topic not in allowlist
        result = await runtime._validate_gateway_envelope(envelope, "internal.secret")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_policy_allows_matching_topic(self) -> None:
        """When policy allows the topic, envelope passes through."""
        # Arrange
        runtime = _make_runtime()

        policy = ServicePolicyEngine(
            allowed_topics=["events.*"],
            expected_realm=None,
        )
        runtime._policy_engine = policy
        runtime._gateway_config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            reject_unsigned=False,
        )

        envelope: dict[str, object] = {"data": "test"}

        # Act
        result = await runtime._validate_gateway_envelope(
            envelope, "events.order.created"
        )

        # Assert
        assert result is not None
        assert result["data"] == "test"

    @pytest.mark.asyncio
    async def test_realm_mismatch_rejects_inbound(self) -> None:
        """When envelope realm doesn't match expected, it is rejected."""
        # Arrange
        runtime = _make_runtime()

        policy = ServicePolicyEngine(
            expected_realm="tenant-123",
        )
        runtime._policy_engine = policy
        runtime._gateway_config = ModelGatewayConfig(
            realm="tenant-123",
            runtime_id="test-runtime",
            reject_unsigned=False,
        )

        envelope: dict[str, object] = {
            "data": "test",
            "realm": "tenant-456",  # Wrong realm
        }

        # Act
        result = await runtime._validate_gateway_envelope(envelope, "events.order")

        # Assert
        assert result is None


class TestValidateGatewayEnvelopeSignedEnvelope:
    """Tests for signed envelope validation in _validate_gateway_envelope."""

    @pytest.mark.asyncio
    async def test_valid_signed_envelope_extracts_payload(self) -> None:
        """A validly signed envelope is validated and inner payload extracted."""
        # Arrange
        runtime = _make_runtime()
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        signer = ServiceEnvelopeSigner(
            realm="test",
            runtime_id="test-runtime-001",
            private_key=private_key,
        )

        validator = ServiceEnvelopeValidator(
            expected_realm="test",
            public_keys={"test-runtime-001": public_key},
            reject_unsigned=True,
        )
        runtime._envelope_validator = validator
        runtime._gateway_config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            reject_unsigned=True,
        )

        # Sign a payload
        payload = ModelTestEvent(
            action="created",
            resource_id="resource-123",
        )
        signed = signer.sign_envelope(payload=payload, bus_id="test-bus")
        signed_dict: dict[str, object] = signed.model_dump(mode="json")

        # Act
        result = await runtime._validate_gateway_envelope(signed_dict, "events.test")

        # Assert - inner payload should be extracted
        assert result is not None
        assert result["action"] == "created"
        assert result["resource_id"] == "resource-123"

    @pytest.mark.asyncio
    async def test_invalid_signature_rejects_envelope(self) -> None:
        """An envelope with an invalid signature is rejected."""
        # Arrange
        runtime = _make_runtime()
        private_key = Ed25519PrivateKey.generate()
        wrong_public_key = Ed25519PrivateKey.generate().public_key()

        signer = ServiceEnvelopeSigner(
            realm="test",
            runtime_id="test-runtime-001",
            private_key=private_key,
        )

        # Use wrong public key for validation
        validator = ServiceEnvelopeValidator(
            expected_realm="test",
            public_keys={"test-runtime-001": wrong_public_key},
            reject_unsigned=True,
        )
        runtime._envelope_validator = validator
        runtime._gateway_config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            reject_unsigned=True,
        )

        payload = ModelTestEvent(
            action="created",
            resource_id="resource-123",
        )
        signed = signer.sign_envelope(payload=payload, bus_id="test-bus")
        signed_dict: dict[str, object] = signed.model_dump(mode="json")

        # Act
        result = await runtime._validate_gateway_envelope(signed_dict, "events.test")

        # Assert - invalid signature should reject
        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_signed_envelope_rejected_gracefully(self) -> None:
        """Malformed signed-looking envelope is handled without crash."""
        # Arrange
        runtime = _make_runtime()
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        validator = ServiceEnvelopeValidator(
            expected_realm="test",
            public_keys={"test-runtime-001": public_key},
        )
        runtime._envelope_validator = validator
        runtime._gateway_config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            reject_unsigned=False,
        )

        # Envelope that looks signed (has signature + required fields)
        # but has malformed data that will fail ModelMessageEnvelope validation
        malformed: dict[str, object] = {
            "realm": "test",
            "runtime_id": "test-runtime-001",
            "bus_id": "test-bus",
            "signature": {"algorithm": "ed25519", "value": "invalid"},
            "payload": "not-a-valid-payload",
        }

        # Act - should not raise
        result = await runtime._validate_gateway_envelope(malformed, "events.test")

        # Assert - malformed envelope is rejected (returns None)
        assert result is None
