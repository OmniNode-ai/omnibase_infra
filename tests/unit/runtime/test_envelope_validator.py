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
    normalize_correlation_id,
    validate_envelope,
)
from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding


@pytest.fixture
def mock_registry() -> RegistryProtocolBinding:
    """Create a mock registry with common handler types registered.

    Note: This fixture uses direct instantiation for unit testing the
    envelope validator. For integration tests that need real container-based
    registries, use container_with_registries from conftest.py.
    """
    registry = RegistryProtocolBinding()

    # Create a minimal mock handler class
    class MockHandler:
        async def handle(
            self,
            envelope: dict[str, object],
            correlation_id: UUID | None = None,
        ) -> dict[str, object]:
            """Handle envelope - mock implementation for unit tests."""
            return {"handled": True}

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
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Envelope without operation field raises EnvelopeValidationError."""
        envelope: dict[str, object] = {"payload": {"data": "test"}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "operation is required" in str(exc_info.value)

    def test_none_operation_raises_error(
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Envelope with None operation raises EnvelopeValidationError."""
        envelope: dict[str, object] = {"operation": None, "payload": {}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "operation is required" in str(exc_info.value)

    def test_empty_string_operation_raises_error(
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Envelope with empty string operation raises EnvelopeValidationError."""
        envelope: dict[str, object] = {"operation": "", "payload": {}}

        with pytest.raises(EnvelopeValidationError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "operation is required" in str(exc_info.value)

    def test_non_string_operation_raises_error(
        self, mock_registry: RegistryProtocolBinding
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
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Operation with unknown prefix raises UnknownHandlerTypeError."""
        envelope: dict[str, object] = {"operation": "lolnope.query"}

        with pytest.raises(UnknownHandlerTypeError) as exc_info:
            validate_envelope(envelope, mock_registry)

        assert "lolnope" in str(exc_info.value)
        assert "No handler registered" in str(exc_info.value)

    def test_unknown_prefix_includes_registered_prefixes(
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """UnknownHandlerTypeError includes list of registered prefixes."""
        envelope: dict[str, object] = {"operation": "unknown.action"}

        with pytest.raises(UnknownHandlerTypeError) as exc_info:
            validate_envelope(envelope, mock_registry)

        # Error should include context about what IS registered
        error = exc_info.value
        assert hasattr(error, "model")

        # Verify registered_prefixes is populated with actual registry prefixes
        # This ensures the error provides actionable guidance to developers
        error_model = error.model
        assert hasattr(error_model, "context")
        context = error_model.context
        assert context is not None
        assert "registered_prefixes" in context
        registered_prefixes = context["registered_prefixes"]
        assert isinstance(registered_prefixes, list)
        # Verify all mock registry prefixes are included
        expected_prefixes = {"http", "db", "kafka", "consul", "vault"}
        assert set(registered_prefixes) == expected_prefixes

    def test_valid_http_prefix_passes(
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Operation with valid 'http' prefix passes validation."""
        envelope: dict[str, object] = {"operation": "http.get"}
        validate_envelope(envelope, mock_registry)
        # Should not raise

    def test_valid_db_prefix_passes(
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Operation with valid 'db' prefix passes validation (with payload)."""
        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1"},
        }
        validate_envelope(envelope, mock_registry)
        # Should not raise

    def test_valid_kafka_prefix_passes(
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Operation with valid 'kafka' prefix passes validation."""
        envelope: dict[str, object] = {"operation": "kafka.consume"}
        validate_envelope(envelope, mock_registry)
        # Should not raise (consume doesn't require payload)

    def test_operation_without_dot_uses_whole_string_as_prefix(
        self, mock_registry: RegistryProtocolBinding
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
            "http.patch",
            "kafka.produce",
            "consul.kv_put",
            "consul.register",
            "vault.write",
            "vault.encrypt",
            "vault.decrypt",
        ],
    )
    def test_payload_required_operations_without_payload_raises_error(
        self, mock_registry: RegistryProtocolBinding, operation: str
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
            "http.patch",
            "kafka.produce",
            "consul.kv_put",
            "consul.register",
            "vault.write",
            "vault.encrypt",
            "vault.decrypt",
        ],
    )
    def test_payload_required_operations_with_empty_dict_raises_error(
        self, mock_registry: RegistryProtocolBinding, operation: str
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
            "http.patch",
            "kafka.produce",
            "consul.kv_put",
            "consul.register",
            "vault.write",
            "vault.encrypt",
            "vault.decrypt",
        ],
    )
    def test_payload_required_operations_with_payload_passes(
        self, mock_registry: RegistryProtocolBinding, operation: str
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
        self, mock_registry: RegistryProtocolBinding, operation: str
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
            "http.patch",
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
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Missing correlation_id is generated as UUID."""
        envelope: dict[str, object] = {"operation": "http.get"}
        validate_envelope(envelope, mock_registry)

        assert "correlation_id" in envelope
        assert isinstance(envelope["correlation_id"], UUID)

    def test_none_correlation_id_generates_uuid(
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """None correlation_id is replaced with generated UUID."""
        envelope: dict[str, object] = {"operation": "http.get", "correlation_id": None}
        validate_envelope(envelope, mock_registry)

        assert envelope["correlation_id"] is not None
        assert isinstance(envelope["correlation_id"], UUID)

    def test_uuid_correlation_id_preserved(
        self, mock_registry: RegistryProtocolBinding
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
        self, mock_registry: RegistryProtocolBinding
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
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Invalid UUID string is replaced with new UUID."""
        envelope: dict[str, object] = {
            "operation": "http.get",
            "correlation_id": "not-a-uuid",
        }
        validate_envelope(envelope, mock_registry)

        assert isinstance(envelope["correlation_id"], UUID)

    def test_non_string_non_uuid_correlation_id_replaced(
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Non-string, non-UUID correlation_id is replaced with new UUID."""
        envelope: dict[str, object] = {"operation": "http.get", "correlation_id": 12345}
        validate_envelope(envelope, mock_registry)

        assert isinstance(envelope["correlation_id"], UUID)


class TestValidationScopeLimit:
    """Tests to ensure validation does NOT inspect handler-specific schemas."""

    def test_validation_does_not_check_sql_in_db_query_payload(
        self, mock_registry: RegistryProtocolBinding
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
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Validation does NOT check for 'url' field in http payload."""
        envelope: dict[str, object] = {
            "operation": "http.post",
            "payload": {"body": "data"},
        }
        validate_envelope(envelope, mock_registry)
        # Should not raise - handler will check for required fields

    def test_any_non_empty_payload_satisfies_requirement(
        self, mock_registry: RegistryProtocolBinding
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
        self, mock_registry: RegistryProtocolBinding
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
        empty_registry = RegistryProtocolBinding()
        envelope: dict[str, object] = {"operation": "http.get"}

        with pytest.raises(UnknownHandlerTypeError):
            validate_envelope(envelope, empty_registry)

    def test_case_sensitive_prefix_matching(
        self, mock_registry: RegistryProtocolBinding
    ) -> None:
        """Prefix matching is case-sensitive."""
        envelope: dict[str, object] = {"operation": "HTTP.get"}  # Uppercase

        with pytest.raises(UnknownHandlerTypeError):
            validate_envelope(envelope, mock_registry)


class TestNormalizeCorrelationIdHelper:
    """Tests for normalize_correlation_id helper function.

    This tests the standalone helper function directly, ensuring it
    handles all edge cases properly as the single source of truth
    for correlation_id normalization.
    """

    def test_none_returns_new_uuid(self) -> None:
        """None value generates a new UUID."""
        result = normalize_correlation_id(None)
        assert isinstance(result, UUID)

    def test_uuid_returns_same_uuid(self) -> None:
        """UUID value is returned unchanged."""
        original = uuid4()
        result = normalize_correlation_id(original)
        assert result == original
        assert result is original  # Same object, not a copy

    def test_valid_uuid_string_returns_parsed_uuid(self) -> None:
        """Valid UUID string is parsed and returned as UUID."""
        original = uuid4()
        result = normalize_correlation_id(str(original))
        assert result == original
        assert isinstance(result, UUID)

    def test_invalid_uuid_string_returns_new_uuid(self) -> None:
        """Invalid UUID string generates a new UUID."""
        result = normalize_correlation_id("not-a-valid-uuid")
        assert isinstance(result, UUID)

    def test_empty_string_returns_new_uuid(self) -> None:
        """Empty string generates a new UUID."""
        result = normalize_correlation_id("")
        assert isinstance(result, UUID)

    def test_integer_returns_new_uuid(self) -> None:
        """Integer value generates a new UUID."""
        result = normalize_correlation_id(12345)
        assert isinstance(result, UUID)

    def test_float_returns_new_uuid(self) -> None:
        """Float value generates a new UUID."""
        result = normalize_correlation_id(123.45)
        assert isinstance(result, UUID)

    def test_list_returns_new_uuid(self) -> None:
        """List value generates a new UUID."""
        result = normalize_correlation_id(["a", "b"])
        assert isinstance(result, UUID)

    def test_dict_returns_new_uuid(self) -> None:
        """Dict value generates a new UUID."""
        result = normalize_correlation_id({"key": "value"})
        assert isinstance(result, UUID)

    def test_boolean_returns_new_uuid(self) -> None:
        """Boolean value generates a new UUID."""
        result = normalize_correlation_id(True)
        assert isinstance(result, UUID)

    def test_different_calls_return_different_uuids(self) -> None:
        """Each call with invalid input returns a different UUID."""
        result1 = normalize_correlation_id(None)
        result2 = normalize_correlation_id(None)
        assert result1 != result2

    def test_uuid_string_with_hyphens_parsed(self) -> None:
        """UUID string with hyphens is correctly parsed."""
        uuid_str = "12345678-1234-5678-1234-567812345678"
        result = normalize_correlation_id(uuid_str)
        assert result == UUID(uuid_str)

    def test_uuid_string_without_hyphens_parsed(self) -> None:
        """UUID string without hyphens is correctly parsed."""
        uuid_str = "12345678123456781234567812345678"
        result = normalize_correlation_id(uuid_str)
        assert result == UUID(uuid_str)

    def test_uuid_string_uppercase_parsed(self) -> None:
        """Uppercase UUID string is correctly parsed."""
        original = uuid4()
        result = normalize_correlation_id(str(original).upper())
        assert result == original
