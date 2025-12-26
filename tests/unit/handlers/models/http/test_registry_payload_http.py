# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for RegistryPayloadHttp and ModelPayloadHttp.

This module validates the registry pattern for HTTP payloads,
following the template established in test_registry_intent.py.

The RegistryPayloadHttp provides a decorator-based registration mechanism that
enables dynamic type resolution during Pydantic validation without requiring
explicit union type definitions. This pattern:
- Eliminates duplicate union definitions across modules
- Allows new payload types to be added by implementing ModelPayloadHttp
- Uses the `operation_type` field as a discriminator for type resolution
- Follows ONEX duck typing principles while maintaining type safety

Test Categories:
    1. Registry Tests - decorator and method behavior
    2. Base Model Tests - common fields and configuration
    3. Concrete Model Inheritance Tests - GET, POST, HEALTH_CHECK payloads
    4. Serialization/Deserialization Tests - JSON round-trip validation

Related:
    - EnumHttpOperationType: Enum for operation type values
    - ModelHttpGetPayload, ModelHttpPostPayload, ModelHttpHealthCheckPayload
    - OMN-1007: Union reduction refactoring

.. versionadded:: 0.7.0
    Created as part of OMN-1007 registry pattern implementation.
"""

from __future__ import annotations

from typing import Literal

import pytest
from omnibase_core.types import JsonValue
from pydantic import ValidationError

from omnibase_infra.handlers.models.http import (
    EnumHttpOperationType,
    ModelHttpGetPayload,
    ModelHttpHealthCheckPayload,
    ModelHttpPostPayload,
    ModelPayloadHttp,
    RegistryPayloadHttp,
)

# Rebuild models that use TYPE_CHECKING imports for JsonValue
# This is required because JsonValue is imported under TYPE_CHECKING in the models
ModelHttpGetPayload.model_rebuild()
ModelHttpPostPayload.model_rebuild()


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_get_payload() -> ModelHttpGetPayload:
    """Create a sample HTTP GET payload for testing."""
    return ModelHttpGetPayload(
        status_code=200,
        headers={"content-type": "application/json"},
        body={"message": "success", "data": [1, 2, 3]},
    )


@pytest.fixture
def sample_post_payload() -> ModelHttpPostPayload:
    """Create a sample HTTP POST payload for testing."""
    return ModelHttpPostPayload(
        status_code=201,
        headers={"content-type": "application/json", "location": "/items/42"},
        body={"id": 42, "created": True},
    )


@pytest.fixture
def sample_health_check_payload() -> ModelHttpHealthCheckPayload:
    """Create a sample HTTP health check payload for testing."""
    return ModelHttpHealthCheckPayload(
        healthy=True,
        initialized=True,
        adapter_type="http",
        timeout_seconds=30.0,
        max_request_size=10485760,
        max_response_size=52428800,
    )


# ============================================================================
# Tests for RegistryPayloadHttp
# ============================================================================


@pytest.mark.unit
class TestRegistryPayloadHttp:
    """Tests for RegistryPayloadHttp class methods and registration behavior.

    The RegistryPayloadHttp provides a decorator-based mechanism for registering
    HTTP payload model types, enabling dynamic type resolution during Pydantic
    validation without explicit union type definitions.
    """

    def test_get_type_returns_correct_class_for_get(self) -> None:
        """Registry returns correct class for get operation type."""
        payload_cls = RegistryPayloadHttp.get_type("get")
        assert payload_cls is ModelHttpGetPayload

    def test_get_type_returns_correct_class_for_post(self) -> None:
        """Registry returns correct class for post operation type."""
        payload_cls = RegistryPayloadHttp.get_type("post")
        assert payload_cls is ModelHttpPostPayload

    def test_get_type_returns_correct_class_for_health_check(self) -> None:
        """Registry returns correct class for health_check operation type."""
        payload_cls = RegistryPayloadHttp.get_type("health_check")
        assert payload_cls is ModelHttpHealthCheckPayload

    def test_get_type_unknown_raises_keyerror(self) -> None:
        """Registry raises KeyError for unknown operation type with helpful message."""
        with pytest.raises(KeyError) as exc_info:
            RegistryPayloadHttp.get_type("unknown_operation")

        # Verify error message contains useful information
        error_msg = str(exc_info.value)
        assert "unknown_operation" in error_msg
        assert "Registered types" in error_msg

    def test_get_type_unknown_lists_registered_types(self) -> None:
        """KeyError message includes list of registered operation types."""
        with pytest.raises(KeyError) as exc_info:
            RegistryPayloadHttp.get_type("nonexistent")

        error_msg = str(exc_info.value)
        # Should mention at least get, post, and health_check
        assert "get" in error_msg or "post" in error_msg or "health_check" in error_msg

    def test_get_all_types_returns_all_registered(self) -> None:
        """get_all_types returns dict with all 3 registered types."""
        all_types = RegistryPayloadHttp.get_all_types()

        assert isinstance(all_types, dict)
        assert len(all_types) >= 3
        assert "get" in all_types
        assert "post" in all_types
        assert "health_check" in all_types
        assert all_types["get"] is ModelHttpGetPayload
        assert all_types["post"] is ModelHttpPostPayload
        assert all_types["health_check"] is ModelHttpHealthCheckPayload

    def test_get_all_types_returns_copy_not_reference(self) -> None:
        """get_all_types returns a copy, not the internal registry."""
        all_types = RegistryPayloadHttp.get_all_types()

        # Mutating the returned dict should not affect the registry
        original_count = len(all_types)
        all_types["fake"] = type("FakePayload", (), {})  # type: ignore[assignment]

        # Registry should be unchanged
        assert len(RegistryPayloadHttp.get_all_types()) == original_count

    def test_is_registered_returns_true_for_known(self) -> None:
        """is_registered returns True for all known operation types."""
        assert RegistryPayloadHttp.is_registered("get") is True
        assert RegistryPayloadHttp.is_registered("post") is True
        assert RegistryPayloadHttp.is_registered("health_check") is True

    def test_is_registered_returns_false_for_unknown(self) -> None:
        """is_registered returns False for unknown operation types."""
        assert RegistryPayloadHttp.is_registered("unknown") is False
        assert RegistryPayloadHttp.is_registered("delete") is False
        assert RegistryPayloadHttp.is_registered("patch") is False

    def test_is_registered_empty_string_returns_false(self) -> None:
        """is_registered returns False for empty string."""
        assert RegistryPayloadHttp.is_registered("") is False

    def test_register_decorator_returns_class_unchanged(self) -> None:
        """The @register decorator returns the class unchanged.

        This test creates a temporary test-only payload class to verify
        the decorator mechanism without polluting the registry. We use
        clear() for test isolation (clear() is designed for testing only).
        """
        # Save original registry state
        original_types = RegistryPayloadHttp.get_all_types()

        try:
            # Clear to test registration in isolation
            RegistryPayloadHttp.clear()

            @RegistryPayloadHttp.register("test_operation")
            class TestPayload(ModelPayloadHttp):
                operation_type: Literal["test_operation"] = "test_operation"

            # Verify the class was returned unchanged
            assert TestPayload.__name__ == "TestPayload"
            assert (
                TestPayload.model_fields["operation_type"].default == "test_operation"
            )

            # Verify it was registered
            assert RegistryPayloadHttp.is_registered("test_operation")
            assert RegistryPayloadHttp.get_type("test_operation") is TestPayload

        finally:
            # Restore original registry state
            RegistryPayloadHttp.clear()
            for op_type, cls in original_types.items():
                RegistryPayloadHttp._types[op_type] = cls

    def test_register_duplicate_raises_valueerror(self) -> None:
        """Registering the same operation type twice raises ValueError.

        This prevents accidental overwrites of existing payload types.
        """
        # Save original registry state
        original_types = RegistryPayloadHttp.get_all_types()

        try:
            RegistryPayloadHttp.clear()

            # First registration should succeed
            @RegistryPayloadHttp.register("duplicate_test")
            class FirstPayload(ModelPayloadHttp):
                operation_type: Literal["duplicate_test"] = "duplicate_test"

            # Second registration with same operation type should fail
            with pytest.raises(ValueError) as exc_info:

                @RegistryPayloadHttp.register("duplicate_test")
                class SecondPayload(ModelPayloadHttp):
                    operation_type: Literal["duplicate_test"] = "duplicate_test"

            error_msg = str(exc_info.value)
            assert "duplicate_test" in error_msg
            assert "already registered" in error_msg
            assert "FirstPayload" in error_msg

        finally:
            # Restore original registry state
            RegistryPayloadHttp.clear()
            for op_type, cls in original_types.items():
                RegistryPayloadHttp._types[op_type] = cls

    def test_clear_removes_all_registered_types(self) -> None:
        """clear() removes all registered types from the registry.

        Note: clear() is intended for testing only and should not be
        used in production code.
        """
        # Save original registry state
        original_types = RegistryPayloadHttp.get_all_types()

        try:
            # Registry should have types before clear
            assert len(RegistryPayloadHttp.get_all_types()) > 0

            RegistryPayloadHttp.clear()

            # Registry should be empty after clear
            assert len(RegistryPayloadHttp.get_all_types()) == 0
            assert RegistryPayloadHttp.is_registered("get") is False
            assert RegistryPayloadHttp.is_registered("post") is False
            assert RegistryPayloadHttp.is_registered("health_check") is False

        finally:
            # Restore original registry state
            RegistryPayloadHttp.clear()
            for op_type, cls in original_types.items():
                RegistryPayloadHttp._types[op_type] = cls


# ============================================================================
# Tests for ModelPayloadHttp Base Class
# ============================================================================


@pytest.mark.unit
class TestModelPayloadHttp:
    """Tests for ModelPayloadHttp base class.

    ModelPayloadHttp defines the common interface that all HTTP payloads share.
    It ensures consistent field names and configuration across all payload types.
    """

    def test_has_operation_type_field(self) -> None:
        """Base model defines required operation_type field."""
        fields = ModelPayloadHttp.model_fields
        assert "operation_type" in fields
        assert fields["operation_type"].annotation == str

    def test_model_config_has_frozen_true(self) -> None:
        """Base model config has frozen=True for immutability."""
        config = ModelPayloadHttp.model_config
        assert config.get("frozen") is True

    def test_model_config_has_extra_forbid(self) -> None:
        """Base model config has extra='forbid' to prevent extra fields."""
        config = ModelPayloadHttp.model_config
        assert config.get("extra") == "forbid"

    def test_model_is_frozen_cannot_modify(self) -> None:
        """Frozen model prevents modification of operation_type field."""
        payload = ModelPayloadHttp(operation_type="test")

        with pytest.raises(ValidationError):
            payload.operation_type = "modified"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Model raises ValidationError when extra fields are provided."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPayloadHttp(
                operation_type="test",
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        # Verify error is about extra field
        error_msg = str(exc_info.value)
        assert "extra_field" in error_msg or "Extra" in error_msg

    def test_base_model_can_be_instantiated(self) -> None:
        """Base model can be instantiated with operation_type."""
        payload = ModelPayloadHttp(operation_type="test")
        assert payload.operation_type == "test"


# ============================================================================
# Tests for Concrete HTTP Payload Model Inheritance
# ============================================================================


@pytest.mark.unit
class TestConcreteHttpPayloadModels:
    """Tests for concrete HTTP payload model inheritance and registration.

    These tests verify that ModelHttpGetPayload, ModelHttpPostPayload, and
    ModelHttpHealthCheckPayload correctly inherit from ModelPayloadHttp
    and are properly registered in the RegistryPayloadHttp.
    """

    def test_get_payload_inherits_from_base(self) -> None:
        """ModelHttpGetPayload inherits from ModelPayloadHttp."""
        assert issubclass(ModelHttpGetPayload, ModelPayloadHttp)

    def test_post_payload_inherits_from_base(self) -> None:
        """ModelHttpPostPayload inherits from ModelPayloadHttp."""
        assert issubclass(ModelHttpPostPayload, ModelPayloadHttp)

    def test_health_check_payload_inherits_from_base(self) -> None:
        """ModelHttpHealthCheckPayload inherits from ModelPayloadHttp."""
        assert issubclass(ModelHttpHealthCheckPayload, ModelPayloadHttp)

    def test_all_payloads_registered(self) -> None:
        """All 3 payload types are registered in the registry."""
        assert RegistryPayloadHttp.is_registered("get")
        assert RegistryPayloadHttp.is_registered("post")
        assert RegistryPayloadHttp.is_registered("health_check")

        all_types = RegistryPayloadHttp.get_all_types()
        assert len(all_types) >= 3

    def test_get_payload_operation_type_is_literal_get(
        self, sample_get_payload: ModelHttpGetPayload
    ) -> None:
        """GET payload operation_type is EnumHttpOperationType.GET."""
        assert sample_get_payload.operation_type == EnumHttpOperationType.GET
        assert sample_get_payload.operation_type == "get"

    def test_post_payload_operation_type_is_literal_post(
        self, sample_post_payload: ModelHttpPostPayload
    ) -> None:
        """POST payload operation_type is EnumHttpOperationType.POST."""
        assert sample_post_payload.operation_type == EnumHttpOperationType.POST
        assert sample_post_payload.operation_type == "post"

    def test_health_check_payload_operation_type_is_literal_health_check(
        self, sample_health_check_payload: ModelHttpHealthCheckPayload
    ) -> None:
        """Health check payload operation_type is EnumHttpOperationType.HEALTH_CHECK."""
        assert (
            sample_health_check_payload.operation_type
            == EnumHttpOperationType.HEALTH_CHECK
        )
        assert sample_health_check_payload.operation_type == "health_check"

    def test_get_payload_has_expected_fields(
        self, sample_get_payload: ModelHttpGetPayload
    ) -> None:
        """GET payload has status_code, headers, and body fields."""
        assert hasattr(sample_get_payload, "status_code")
        assert hasattr(sample_get_payload, "headers")
        assert hasattr(sample_get_payload, "body")
        assert sample_get_payload.status_code == 200
        assert sample_get_payload.headers == {"content-type": "application/json"}
        assert sample_get_payload.body == {"message": "success", "data": [1, 2, 3]}

    def test_post_payload_has_expected_fields(
        self, sample_post_payload: ModelHttpPostPayload
    ) -> None:
        """POST payload has status_code, headers, and body fields."""
        assert hasattr(sample_post_payload, "status_code")
        assert hasattr(sample_post_payload, "headers")
        assert hasattr(sample_post_payload, "body")
        assert sample_post_payload.status_code == 201
        assert "location" in sample_post_payload.headers

    def test_health_check_payload_has_expected_fields(
        self, sample_health_check_payload: ModelHttpHealthCheckPayload
    ) -> None:
        """Health check payload has healthy, initialized, and config fields."""
        assert hasattr(sample_health_check_payload, "healthy")
        assert hasattr(sample_health_check_payload, "initialized")
        assert hasattr(sample_health_check_payload, "adapter_type")
        assert hasattr(sample_health_check_payload, "timeout_seconds")
        assert hasattr(sample_health_check_payload, "max_request_size")
        assert hasattr(sample_health_check_payload, "max_response_size")

    def test_get_payload_is_frozen(
        self, sample_get_payload: ModelHttpGetPayload
    ) -> None:
        """GET payload model is frozen and cannot be modified."""
        with pytest.raises(ValidationError):
            sample_get_payload.status_code = 500  # type: ignore[misc]

    def test_post_payload_is_frozen(
        self, sample_post_payload: ModelHttpPostPayload
    ) -> None:
        """POST payload model is frozen and cannot be modified."""
        with pytest.raises(ValidationError):
            sample_post_payload.status_code = 500  # type: ignore[misc]

    def test_health_check_payload_is_frozen(
        self, sample_health_check_payload: ModelHttpHealthCheckPayload
    ) -> None:
        """Health check payload model is frozen and cannot be modified."""
        with pytest.raises(ValidationError):
            sample_health_check_payload.healthy = False  # type: ignore[misc]

    def test_get_payload_forbids_extra_fields(self) -> None:
        """GET payload raises ValidationError for extra fields."""
        with pytest.raises(ValidationError):
            ModelHttpGetPayload(
                status_code=200,
                headers={},
                body=None,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

    def test_post_payload_forbids_extra_fields(self) -> None:
        """POST payload raises ValidationError for extra fields."""
        with pytest.raises(ValidationError):
            ModelHttpPostPayload(
                status_code=201,
                headers={},
                body=None,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

    def test_health_check_payload_forbids_extra_fields(self) -> None:
        """Health check payload raises ValidationError for extra fields."""
        with pytest.raises(ValidationError):
            ModelHttpHealthCheckPayload(
                healthy=True,
                initialized=True,
                adapter_type="http",
                timeout_seconds=30.0,
                max_request_size=0,
                max_response_size=0,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )


# ============================================================================
# Tests for HTTP Payload Serialization and Deserialization
# ============================================================================


@pytest.mark.unit
class TestHttpPayloadSerialization:
    """Tests for HTTP payload serialization and deserialization.

    These tests verify that payload models can be serialized to JSON/dict
    and deserialized back, with the operation_type field enabling type discrimination.
    """

    def test_payload_serializes_to_json(
        self, sample_get_payload: ModelHttpGetPayload
    ) -> None:
        """GET payload can be serialized to JSON string."""
        json_str = sample_get_payload.model_dump_json()

        assert isinstance(json_str, str)
        assert (
            '"operation_type":"get"' in json_str
            or '"operation_type": "get"' in json_str
        )

    def test_payload_deserializes_from_dict(self) -> None:
        """GET payload validates from dict."""
        data = {
            "operation_type": "get",
            "status_code": 200,
            "headers": {"content-type": "application/json"},
            "body": {"result": "success"},
        }

        payload = ModelHttpGetPayload.model_validate(data)

        assert payload.operation_type == "get"
        assert payload.status_code == 200
        assert payload.body == {"result": "success"}

    def test_operation_type_discriminator_works(self) -> None:
        """Operation type enables type lookup from data."""
        get_data = {
            "operation_type": "get",
            "status_code": 200,
            "headers": {},
            "body": None,
        }

        post_data = {
            "operation_type": "post",
            "status_code": 201,
            "headers": {},
            "body": {"created": True},
        }

        health_check_data = {
            "operation_type": "health_check",
            "healthy": True,
            "initialized": True,
            "adapter_type": "http",
            "timeout_seconds": 30.0,
            "max_request_size": 1024,
            "max_response_size": 2048,
        }

        # Use registry to get correct class based on operation_type
        get_cls = RegistryPayloadHttp.get_type(get_data["operation_type"])
        post_cls = RegistryPayloadHttp.get_type(post_data["operation_type"])
        health_cls = RegistryPayloadHttp.get_type(health_check_data["operation_type"])

        get_payload = get_cls.model_validate(get_data)
        post_payload = post_cls.model_validate(post_data)
        health_payload = health_cls.model_validate(health_check_data)

        assert isinstance(get_payload, ModelHttpGetPayload)
        assert isinstance(post_payload, ModelHttpPostPayload)
        assert isinstance(health_payload, ModelHttpHealthCheckPayload)

    def test_get_payload_serializes_to_dict(
        self, sample_get_payload: ModelHttpGetPayload
    ) -> None:
        """GET payload can be serialized to dict."""
        data = sample_get_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "get"
        assert data["status_code"] == 200
        assert "headers" in data
        assert "body" in data

    def test_post_payload_serializes_to_dict(
        self, sample_post_payload: ModelHttpPostPayload
    ) -> None:
        """POST payload can be serialized to dict."""
        data = sample_post_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "post"
        assert data["status_code"] == 201
        assert "headers" in data
        assert "body" in data

    def test_health_check_payload_serializes_to_dict(
        self, sample_health_check_payload: ModelHttpHealthCheckPayload
    ) -> None:
        """Health check payload can be serialized to dict."""
        data = sample_health_check_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "health_check"
        assert data["healthy"] is True
        assert data["initialized"] is True
        assert data["adapter_type"] == "http"

    def test_get_payload_round_trip_preserves_data(
        self, sample_get_payload: ModelHttpGetPayload
    ) -> None:
        """GET payload JSON round-trip preserves all data."""
        json_str = sample_get_payload.model_dump_json()
        restored = ModelHttpGetPayload.model_validate_json(json_str)

        assert restored == sample_get_payload

    def test_post_payload_round_trip_preserves_data(
        self, sample_post_payload: ModelHttpPostPayload
    ) -> None:
        """POST payload JSON round-trip preserves all data."""
        json_str = sample_post_payload.model_dump_json()
        restored = ModelHttpPostPayload.model_validate_json(json_str)

        assert restored == sample_post_payload

    def test_health_check_payload_round_trip_preserves_data(
        self, sample_health_check_payload: ModelHttpHealthCheckPayload
    ) -> None:
        """Health check payload JSON round-trip preserves all data."""
        json_str = sample_health_check_payload.model_dump_json()
        restored = ModelHttpHealthCheckPayload.model_validate_json(json_str)

        assert restored == sample_health_check_payload

    def test_deserialization_with_wrong_operation_type_fails(self) -> None:
        """Deserializing with wrong operation_type field fails validation.

        ModelHttpGetPayload expects operation_type='get' as a Literal.
        Providing a different value should fail validation.
        """
        data = {
            "operation_type": "wrong_type",  # Not 'get'
            "status_code": 200,
            "headers": {},
            "body": None,
        }

        with pytest.raises(ValidationError):
            ModelHttpGetPayload.model_validate(data)


# ============================================================================
# Tests for Thread Safety and Immutability
# ============================================================================


@pytest.mark.unit
class TestHttpPayloadThreadSafety:
    """Tests for thread safety and immutability of HTTP payload models.

    All HTTP payload models and the RegistryPayloadHttp are designed to be thread-safe:
    - RegistryPayloadHttp is populated at module import time and read-only after
    - All payload models are frozen (immutable)
    """

    def test_registry_class_var_is_shared(self) -> None:
        """RegistryPayloadHttp._types is a ClassVar shared across all usage.

        This ensures the registry is consistent regardless of where it's
        accessed in the codebase.
        """
        # Access registry from different calls (would be same in practice)
        types1 = RegistryPayloadHttp.get_all_types()
        types2 = RegistryPayloadHttp.get_all_types()

        # Should return equivalent copies
        assert types1 == types2

    def test_get_payload_not_hashable_due_to_dict_fields(
        self, sample_get_payload: ModelHttpGetPayload
    ) -> None:
        """GET payload is NOT hashable because it contains dict fields.

        Note: Even though the model is frozen, the headers and body fields
        are dicts which are not hashable in Python. This is expected behavior.
        """
        with pytest.raises(TypeError, match="unhashable type"):
            hash(sample_get_payload)

    def test_post_payload_not_hashable_due_to_dict_fields(
        self, sample_post_payload: ModelHttpPostPayload
    ) -> None:
        """POST payload is NOT hashable because it contains dict fields.

        Note: Even though the model is frozen, the headers and body fields
        are dicts which are not hashable in Python. This is expected behavior.
        """
        with pytest.raises(TypeError, match="unhashable type"):
            hash(sample_post_payload)

    def test_health_check_payload_is_hashable(
        self, sample_health_check_payload: ModelHttpHealthCheckPayload
    ) -> None:
        """Frozen health check payload is hashable for use in sets/dicts.

        Health check payload only contains primitive types (bool, str, float, int)
        which are all hashable, unlike GET/POST payloads which contain dicts.
        """
        hash_value = hash(sample_health_check_payload)
        assert isinstance(hash_value, int)

        # Can be used in a set
        payload_set = {sample_health_check_payload}
        assert len(payload_set) == 1

    def test_identical_health_check_payloads_have_same_hash(self) -> None:
        """Identical health check payloads should have the same hash value."""
        payload1 = ModelHttpHealthCheckPayload(
            healthy=True,
            initialized=True,
            adapter_type="http",
            timeout_seconds=30.0,
            max_request_size=1024,
            max_response_size=2048,
        )

        payload2 = ModelHttpHealthCheckPayload(
            healthy=True,
            initialized=True,
            adapter_type="http",
            timeout_seconds=30.0,
            max_request_size=1024,
            max_response_size=2048,
        )

        assert hash(payload1) == hash(payload2)
        assert payload1 == payload2

    def test_health_check_payload_can_be_used_as_dict_key(
        self, sample_health_check_payload: ModelHttpHealthCheckPayload
    ) -> None:
        """Health check payloads can be used as dictionary keys."""
        payload_dict = {sample_health_check_payload: "value"}

        assert payload_dict[sample_health_check_payload] == "value"

    def test_get_post_payloads_still_support_equality(self) -> None:
        """GET and POST payloads support equality comparison despite not being hashable."""
        payload1 = ModelHttpGetPayload(
            status_code=200,
            headers={"x-test": "value"},
            body={"key": "value"},
        )

        payload2 = ModelHttpGetPayload(
            status_code=200,
            headers={"x-test": "value"},
            body={"key": "value"},
        )

        # Equality works even though hash doesn't
        assert payload1 == payload2

        # Different payloads are not equal
        payload3 = ModelHttpGetPayload(
            status_code=404,
            headers={},
            body=None,
        )
        assert payload1 != payload3


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.unit
class TestHttpPayloadEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_get_payload_status_code_minimum(self) -> None:
        """GET payload status_code has minimum value of 100."""
        # Valid minimum
        payload = ModelHttpGetPayload(
            status_code=100,
            headers={},
            body=None,
        )
        assert payload.status_code == 100

        # Below minimum should fail
        with pytest.raises(ValidationError):
            ModelHttpGetPayload(
                status_code=99,
                headers={},
                body=None,
            )

    def test_get_payload_status_code_maximum(self) -> None:
        """GET payload status_code has maximum value of 599."""
        # Valid maximum
        payload = ModelHttpGetPayload(
            status_code=599,
            headers={},
            body=None,
        )
        assert payload.status_code == 599

        # Above maximum should fail
        with pytest.raises(ValidationError):
            ModelHttpGetPayload(
                status_code=600,
                headers={},
                body=None,
            )

    def test_post_payload_status_code_validation(self) -> None:
        """POST payload status_code has same validation as GET."""
        # Valid range
        payload = ModelHttpPostPayload(
            status_code=201,
            headers={},
            body=None,
        )
        assert payload.status_code == 201

        # Below minimum
        with pytest.raises(ValidationError):
            ModelHttpPostPayload(
                status_code=50,
                headers={},
                body=None,
            )

    def test_health_check_max_sizes_non_negative(self) -> None:
        """Health check max_request_size and max_response_size must be >= 0."""
        # Zero is valid
        payload = ModelHttpHealthCheckPayload(
            healthy=True,
            initialized=True,
            adapter_type="http",
            timeout_seconds=30.0,
            max_request_size=0,
            max_response_size=0,
        )
        assert payload.max_request_size == 0
        assert payload.max_response_size == 0

        # Negative should fail
        with pytest.raises(ValidationError):
            ModelHttpHealthCheckPayload(
                healthy=True,
                initialized=True,
                adapter_type="http",
                timeout_seconds=30.0,
                max_request_size=-1,
                max_response_size=0,
            )

    def test_get_payload_empty_headers_is_valid(self) -> None:
        """GET payload accepts empty headers dict."""
        payload = ModelHttpGetPayload(
            status_code=200,
            headers={},
            body=None,
        )
        assert payload.headers == {}

    def test_get_payload_null_body_is_valid(self) -> None:
        """GET payload accepts None as body."""
        payload = ModelHttpGetPayload(
            status_code=204,
            headers={},
            body=None,
        )
        assert payload.body is None

    def test_get_payload_complex_body_is_valid(self) -> None:
        """GET payload accepts complex nested JSON body."""
        complex_body = {
            "users": [
                {"id": 1, "name": "Alice", "tags": ["admin", "active"]},
                {"id": 2, "name": "Bob", "tags": ["user"]},
            ],
            "metadata": {
                "total": 2,
                "page": 1,
                "nested": {"deep": {"value": True}},
            },
        }

        payload = ModelHttpGetPayload(
            status_code=200,
            headers={"content-type": "application/json"},
            body=complex_body,
        )

        assert payload.body == complex_body
        assert payload.body["users"][0]["name"] == "Alice"

    def test_health_check_unhealthy_state(self) -> None:
        """Health check payload can represent unhealthy state."""
        payload = ModelHttpHealthCheckPayload(
            healthy=False,
            initialized=False,
            adapter_type="http",
            timeout_seconds=0.0,
            max_request_size=0,
            max_response_size=0,
        )

        assert payload.healthy is False
        assert payload.initialized is False

    def test_registry_case_sensitive(self) -> None:
        """Registry operation types are case-sensitive."""
        # Lowercase works
        assert RegistryPayloadHttp.is_registered("get") is True

        # Uppercase does not work (not registered that way)
        assert RegistryPayloadHttp.is_registered("GET") is False
        assert RegistryPayloadHttp.is_registered("Get") is False
