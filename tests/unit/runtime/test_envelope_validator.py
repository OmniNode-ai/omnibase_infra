# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for envelope validation.

Tests validate_envelope() function for all validation rules:
1. Operation presence and type validation
2. Handler prefix validation against registry
3. Payload requirement validation for specific operations
4. Correlation ID normalization to UUID
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import EnvelopeValidationError, UnknownHandlerTypeError
from omnibase_infra.runtime.envelope_validator import (
    PAYLOAD_REQUIRED_OPERATIONS,
    validate_envelope,
)
from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry


@pytest.fixture
def mock_registry() -> ProtocolBindingRegistry:
    """Create a mock registry with common handler types registered.

    Note: This fixture uses direct instantiation for unit testing the
    envelope validator. For integration tests that need real container-based
    registries, use container_with_registries from conftest.py.
    """
    registry = ProtocolBindingRegistry()

    # Create a minimal mock handler class
    class MockHandler:
        async def execute(self, envelope: dict) -> dict:
            return {"success": True}

    # Register common handler types
    registry.register("http", MockHandler)  # type: ignore[arg-type]
    registry.register("db", MockHandler)  # type: ignore[arg-type]
    registry.register("kafka", MockHandler)  # type: ignore[arg-type]
    registry.register("consul", MockHandler)  # type: ignore[arg-type]
    registry.register("vault", MockHandler)  # type: ignore[arg-type]

    return registry


class TestOperationValidation:
    """Tests for operation presence and type validation."""

    def test_missing_operation_raises_error(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Envelope without operation field raises EnvelopeValidationError."""
        envelope: dict[str, object] = {"payload": {"data": "test"}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "operation is required" in str(exc_info.value)

    def test_none_operation_raises_error(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Envelope with None operation raises EnvelopeValidationError."""
        envelope: dict[str, object] = {"operation": None, "payload": {}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "operation is required" in str(exc_info.value)

    def test_empty_string_operation_raises_error(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Envelope with empty string operation raises EnvelopeValidationError."""
        envelope: dict[str, object] = {"operation": "", "payload": {}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "operation is required" in str(exc_info.value)

    def test_non_string_operation_raises_error(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Envelope with non-string operation raises EnvelopeValidationError."""
        envelope: dict[str, object] = {"operation": 123, "payload": {}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "operation is required" in str(exc_info.value)
        assert "non-empty string" in str(exc_info.value)


class TestHandlerPrefixValidation:
    """Tests for handler prefix validation against registry."""

    def test_unknown_prefix_raises_error(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Operation with unknown prefix raises UnknownHandlerTypeError."""
        envelope: dict[str, object] = {"operation": "lolnope.query"}

        with pytest.raises(UnknownHandlerTypeError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "lolnope" in str(exc_info.value)
        assert "No handler registered" in str(exc_info.value)

    def test_unknown_prefix_includes_registered_prefixes(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """UnknownHandlerTypeError includes list of registered prefixes."""
        envelope: dict[str, object] = {"operation": "unknown.action"}

        with pytest.raises(UnknownHandlerTypeError) as exc_info:
            validate_envelope(envelope, mock_registry)

        # Error should include context about what IS registered
        error = exc_info.value
        assert hasattr(error, "model")

    def test_valid_http_prefix_passes(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Operation with valid 'http' prefix passes validation."""
        envelope: dict[str, object] = {"operation": "http.get"}
        validate_envelope(envelope, mock_registry)
        # Should not raise

    def test_valid_db_prefix_passes(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Operation with valid 'db' prefix passes validation (with payload)."""
        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1"},
        }
        validate_envelope(envelope, mock_registry)
        # Should not raise

    def test_valid_kafka_prefix_passes(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Operation with valid 'kafka' prefix passes validation."""
        envelope: dict[str, object] = {"operation": "kafka.consume"}
        validate_envelope(envelope, mock_registry)
        # Should not raise (consume doesn't require payload)

    def test_operation_without_dot_uses_whole_string_as_prefix(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Operation without dot uses entire string as prefix."""
        envelope: dict[str, object] = {"operation": "http"}  # No dot
        validate_envelope(envelope, mock_registry)
        # Should not raise - "http" is a registered prefix


class TestPayloadValidation:
    """Tests for payload requirement validation."""

    @pytest.mark.parametrize(
        "operation",
        [
            "db.query",
            "db.execute",
            "http.post",
            "http.put",
            "kafka.produce",
            "consul.kv_put",
            "consul.register",
            "vault.write",
            "vault.encrypt",
            "vault.decrypt",
        ],
    )
    def test_payload_required_operations_without_payload_raises_error(
        self, mock_registry: ProtocolBindingRegistry, operation: str
    ) -> None:
        """Operations that require payload raise error when payload is missing."""
        envelope: dict[str, object] = {"operation": operation}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "payload is required" in str(exc_info.value)
        assert operation in str(exc_info.value)

    @pytest.mark.parametrize(
        "operation",
        [
            "db.query",
            "db.execute",
            "http.post",
            "http.put",
            "kafka.produce",
            "consul.kv_put",
            "vault.write",
        ],
    )
    def test_payload_required_operations_with_empty_dict_raises_error(
        self, mock_registry: ProtocolBindingRegistry, operation: str
    ) -> None:
        """Operations that require payload raise error when payload is empty dict."""
        envelope: dict[str, object] = {"operation": operation, "payload": {}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "payload is required" in str(exc_info.value)

    @pytest.mark.parametrize(
        "operation",
        [
            "db.query",
            "db.execute",
            "http.post",
            "http.put",
            "kafka.produce",
        ],
    )
    def test_payload_required_operations_with_payload_passes(
        self, mock_registry: ProtocolBindingRegistry, operation: str
    ) -> None:
        """Operations that require payload pass when payload is provided."""
        envelope: dict[str, object] = {
            "operation": operation,
            "payload": {"data": "test"},
        }
        validate_envelope(envelope, mock_registry)
        # Should not raise

    @pytest.mark.parametrize(
        "operation",
        [
            "http.get",
            "http.delete",
            "kafka.consume",
            "consul.kv_get",
            "vault.read",
        ],
    )
    def test_operations_without_payload_requirement_pass(
        self, mock_registry: ProtocolBindingRegistry, operation: str
    ) -> None:
        """Operations that don't require payload pass without payload."""
        envelope: dict[str, object] = {"operation": operation}
        validate_envelope(envelope, mock_registry)
        # Should not raise

    def test_payload_required_operations_constant_matches_spec(self) -> None:
        """PAYLOAD_REQUIRED_OPERATIONS matches the specification."""
        expected = {
            "db.query",
            "db.execute",
            "http.post",
            "http.put",
            "kafka.produce",
            "consul.kv_put",
            "consul.register",
            "vault.write",
            "vault.encrypt",
            "vault.decrypt",
        }
        assert expected == PAYLOAD_REQUIRED_OPERATIONS


class TestCorrelationIdNormalization:
    """Tests for correlation_id normalization to UUID."""

    def test_missing_correlation_id_generates_uuid(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Missing correlation_id is generated as UUID."""
        envelope: dict[str, object] = {"operation": "http.get"}
        validate_envelope(envelope, mock_registry)

        assert "correlation_id" in envelope
        assert isinstance(envelope["correlation_id"], UUID)

    def test_none_correlation_id_generates_uuid(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """None correlation_id is replaced with generated UUID."""
        envelope: dict[str, object] = {"operation": "http.get", "correlation_id": None}
        validate_envelope(envelope, mock_registry)

        assert envelope["correlation_id"] is not None
        assert isinstance(envelope["correlation_id"], UUID)

    def test_uuid_correlation_id_preserved(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """UUID correlation_id is preserved."""
        original_id = uuid4()
        envelope: dict[str, object] = {
            "operation": "http.get",
            "correlation_id": original_id,
        }
        validate_envelope(envelope, mock_registry)

        assert envelope["correlation_id"] == original_id

    def test_valid_string_correlation_id_converted_to_uuid(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Valid UUID string is converted to UUID object."""
        original_id = uuid4()
        envelope: dict[str, object] = {
            "operation": "http.get",
            "correlation_id": str(original_id),
        }
        validate_envelope(envelope, mock_registry)

        assert envelope["correlation_id"] == original_id
        assert isinstance(envelope["correlation_id"], UUID)

    def test_invalid_string_correlation_id_replaced_with_new_uuid(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Invalid UUID string is replaced with new UUID."""
        envelope: dict[str, object] = {
            "operation": "http.get",
            "correlation_id": "not-a-uuid",
        }
        validate_envelope(envelope, mock_registry)

        assert isinstance(envelope["correlation_id"], UUID)

    def test_non_string_non_uuid_correlation_id_replaced(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Non-string, non-UUID correlation_id is replaced with new UUID."""
        envelope: dict[str, object] = {"operation": "http.get", "correlation_id": 12345}
        validate_envelope(envelope, mock_registry)

        assert isinstance(envelope["correlation_id"], UUID)


class TestValidationScopeLimit:
    """Tests to ensure validation does NOT inspect handler-specific schemas."""

    def test_validation_does_not_check_sql_in_db_query_payload(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Validation does NOT check for 'sql' field in db.query payload."""
        # This should pass - we only check payload exists, not its contents
        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {"wrong_field": "value"},
        }
        validate_envelope(envelope, mock_registry)
        # Should not raise - handler will check for sql field

    def test_validation_does_not_check_url_in_http_payload(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Validation does NOT check for 'url' field in http payload."""
        envelope: dict[str, object] = {
            "operation": "http.post",
            "payload": {"body": "data"},
        }
        validate_envelope(envelope, mock_registry)
        # Should not raise - handler will check for required fields

    def test_any_non_empty_payload_satisfies_requirement(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Any non-empty payload satisfies the payload requirement."""
        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {"anything": True},
        }
        validate_envelope(envelope, mock_registry)
        # Should not raise


class TestEdgeCases:
    """Edge case tests."""

    def test_envelope_mutation_only_affects_correlation_id(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Validation only mutates correlation_id, preserves other fields."""
        original_payload = {"sql": "SELECT 1"}
        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": original_payload,
            "extra_field": "preserved",
        }
        validate_envelope(envelope, mock_registry)

        assert envelope["payload"] == original_payload
        assert envelope["extra_field"] == "preserved"
        assert envelope["operation"] == "db.query"

    def test_empty_registry_rejects_all_operations(self) -> None:
        """Empty registry rejects all operations."""
        empty_registry = ProtocolBindingRegistry()
        envelope: dict[str, object] = {"operation": "http.get"}

        with pytest.raises(UnknownHandlerTypeError):
            validate_envelope(envelope, empty_registry)

    def test_case_sensitive_prefix_matching(
        self, mock_registry: ProtocolBindingRegistry
    ) -> None:
        """Prefix matching is case-sensitive."""
        envelope: dict[str, object] = {"operation": "HTTP.get"}  # Uppercase

        with pytest.raises(UnknownHandlerTypeError):
            validate_envelope(envelope, mock_registry)
