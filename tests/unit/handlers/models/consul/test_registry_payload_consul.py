# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for RegistryPayloadConsul and ModelPayloadConsul.

This module validates the registry pattern for Consul handler payloads,
following the template established by test_registry_intent.py.

The RegistryPayloadConsul provides a decorator-based registration mechanism that
enables dynamic type resolution during Pydantic validation without requiring
explicit union type definitions. This pattern:
- Eliminates duplicate union definitions across modules
- Allows new payload types to be added by implementing ModelPayloadConsul
- Uses the `operation_type` field as a discriminator for type resolution
- Follows ONEX duck typing principles while maintaining type safety

Test Categories:
    1. Registry Tests - decorator and method behavior (TestRegistryPayloadConsul)
    2. Base Model Tests - common fields and configuration (TestModelPayloadConsul)
    3. Concrete Model Inheritance Tests - all 7 payload types (TestConcreteConsulPayloadModels)
    4. Serialization/Deserialization Tests - JSON round-trip validation (TestConsulPayloadSerialization)
    5. Thread Safety Tests - immutability and hashability (TestConsulPayloadThreadSafety)
    6. Edge Case Tests - boundary conditions (TestConsulPayloadEdgeCases)

Registered Operation Types (7 total):
    - kv_get_found: ModelConsulKVGetFoundPayload
    - kv_get_not_found: ModelConsulKVGetNotFoundPayload
    - kv_get_recurse: ModelConsulKVGetRecursePayload
    - kv_put: ModelConsulKVPutPayload
    - register: ModelConsulRegisterPayload
    - deregister: ModelConsulDeregisterPayload
    - health_check: ModelConsulHealthCheckPayload

Related:
    - ModelPayloadConsul: Base model for all Consul payloads
    - RegistryPayloadConsul: Registry for payload type discovery
    - OMN-1007: Union reduction refactoring

.. versionadded:: 0.7.0
    Created as part of OMN-1007 registry pattern implementation.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.models.consul import (
    ModelConsulDeregisterPayload,
    ModelConsulHealthCheckPayload,
    ModelConsulKVGetFoundPayload,
    ModelConsulKVGetNotFoundPayload,
    ModelConsulKVGetRecursePayload,
    ModelConsulKVPutPayload,
    ModelConsulRegisterPayload,
    ModelPayloadConsul,
    RegistryPayloadConsul,
)

# ============================================================================
# Constants
# ============================================================================

# All registered operation types and their corresponding classes
EXPECTED_OPERATION_TYPES: dict[str, type[ModelPayloadConsul]] = {
    "kv_get_found": ModelConsulKVGetFoundPayload,
    "kv_get_not_found": ModelConsulKVGetNotFoundPayload,
    "kv_get_recurse": ModelConsulKVGetRecursePayload,
    "kv_put": ModelConsulKVPutPayload,
    "register": ModelConsulRegisterPayload,
    "deregister": ModelConsulDeregisterPayload,
    "health_check": ModelConsulHealthCheckPayload,
}


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_kv_get_found_payload() -> ModelConsulKVGetFoundPayload:
    """Create a sample KV get found payload for testing."""
    return ModelConsulKVGetFoundPayload(
        key="test/key",
        value="test-value",
        flags=0,
        modify_index=123,
        index=456,
    )


@pytest.fixture
def sample_kv_get_not_found_payload() -> ModelConsulKVGetNotFoundPayload:
    """Create a sample KV get not found payload for testing."""
    return ModelConsulKVGetNotFoundPayload(
        key="test/missing-key",
        index=789,
    )


@pytest.fixture
def sample_kv_get_recurse_payload() -> ModelConsulKVGetRecursePayload:
    """Create a sample KV get recurse payload for testing."""
    from omnibase_infra.handlers.models.consul import ModelConsulKVItem

    return ModelConsulKVGetRecursePayload(
        found=True,
        items=[
            ModelConsulKVItem(
                key="test/key1",
                value="value1",
                flags=0,
                modify_index=100,
            ),
            ModelConsulKVItem(
                key="test/key2",
                value="value2",
                flags=0,
                modify_index=101,
            ),
        ],
        count=2,
        index=500,
    )


@pytest.fixture
def sample_kv_put_payload() -> ModelConsulKVPutPayload:
    """Create a sample KV put payload for testing."""
    return ModelConsulKVPutPayload(
        success=True,
        key="test/new-key",
    )


@pytest.fixture
def sample_register_payload() -> ModelConsulRegisterPayload:
    """Create a sample register payload for testing."""
    return ModelConsulRegisterPayload(
        registered=True,
        name="test-service",
        consul_service_id="test-service-001",
    )


@pytest.fixture
def sample_deregister_payload() -> ModelConsulDeregisterPayload:
    """Create a sample deregister payload for testing."""
    return ModelConsulDeregisterPayload(
        deregistered=True,
        consul_service_id="test-service-001",
    )


@pytest.fixture
def sample_health_check_payload() -> ModelConsulHealthCheckPayload:
    """Create a sample health check payload for testing."""
    return ModelConsulHealthCheckPayload(
        healthy=True,
        initialized=True,
        handler_type="consul",
        timeout_seconds=5.0,
        circuit_breaker_state="closed",
        circuit_breaker_failure_count=0,
        thread_pool_active_workers=2,
        thread_pool_max_workers=10,
        thread_pool_max_queue_size=100,
    )


# ============================================================================
# Tests for RegistryPayloadConsul
# ============================================================================


@pytest.mark.unit
class TestRegistryPayloadConsul:
    """Tests for RegistryPayloadConsul class methods and registration behavior.

    The RegistryPayloadConsul provides a decorator-based mechanism for registering
    Consul payload model types, enabling dynamic type resolution during Pydantic
    validation without explicit union type definitions.
    """

    def test_get_type_returns_correct_class_for_kv_get_found(self) -> None:
        """Registry returns correct class for kv_get_found operation type."""
        payload_cls = RegistryPayloadConsul.get_type("kv_get_found")
        assert payload_cls is ModelConsulKVGetFoundPayload

    def test_get_type_returns_correct_class_for_kv_get_not_found(self) -> None:
        """Registry returns correct class for kv_get_not_found operation type."""
        payload_cls = RegistryPayloadConsul.get_type("kv_get_not_found")
        assert payload_cls is ModelConsulKVGetNotFoundPayload

    def test_get_type_returns_correct_class_for_kv_get_recurse(self) -> None:
        """Registry returns correct class for kv_get_recurse operation type."""
        payload_cls = RegistryPayloadConsul.get_type("kv_get_recurse")
        assert payload_cls is ModelConsulKVGetRecursePayload

    def test_get_type_returns_correct_class_for_kv_put(self) -> None:
        """Registry returns correct class for kv_put operation type."""
        payload_cls = RegistryPayloadConsul.get_type("kv_put")
        assert payload_cls is ModelConsulKVPutPayload

    def test_get_type_returns_correct_class_for_register(self) -> None:
        """Registry returns correct class for register operation type."""
        payload_cls = RegistryPayloadConsul.get_type("register")
        assert payload_cls is ModelConsulRegisterPayload

    def test_get_type_returns_correct_class_for_deregister(self) -> None:
        """Registry returns correct class for deregister operation type."""
        payload_cls = RegistryPayloadConsul.get_type("deregister")
        assert payload_cls is ModelConsulDeregisterPayload

    def test_get_type_returns_correct_class_for_health_check(self) -> None:
        """Registry returns correct class for health_check operation type."""
        payload_cls = RegistryPayloadConsul.get_type("health_check")
        assert payload_cls is ModelConsulHealthCheckPayload

    def test_get_type_returns_correct_class_for_all_types(self) -> None:
        """Registry returns correct class for all 7 registered operation types."""
        for operation_type, expected_cls in EXPECTED_OPERATION_TYPES.items():
            payload_cls = RegistryPayloadConsul.get_type(operation_type)
            assert payload_cls is expected_cls, (
                f"Expected {expected_cls.__name__} for '{operation_type}', "
                f"got {payload_cls.__name__}"
            )

    def test_get_type_unknown_raises_keyerror(self) -> None:
        """Registry raises KeyError for unknown operation type with helpful message."""
        with pytest.raises(KeyError) as exc_info:
            RegistryPayloadConsul.get_type("unknown_operation")

        error_msg = str(exc_info.value)
        assert "unknown_operation" in error_msg
        assert "Registered types" in error_msg

    def test_get_type_unknown_lists_registered_types(self) -> None:
        """KeyError message includes list of registered operation types."""
        with pytest.raises(KeyError) as exc_info:
            RegistryPayloadConsul.get_type("nonexistent")

        error_msg = str(exc_info.value)
        # Should mention at least one registered type
        assert any(op in error_msg for op in EXPECTED_OPERATION_TYPES)

    def test_get_all_types_returns_dict_of_registered_types(self) -> None:
        """get_all_types returns dict mapping operation_type strings to classes."""
        all_types = RegistryPayloadConsul.get_all_types()

        assert isinstance(all_types, dict)
        assert len(all_types) == 7

        for operation_type, expected_cls in EXPECTED_OPERATION_TYPES.items():
            assert operation_type in all_types
            assert all_types[operation_type] is expected_cls

    def test_get_all_types_returns_all_registered(self) -> None:
        """get_all_types returns a dict with all 7 registered types."""
        all_types = RegistryPayloadConsul.get_all_types()

        assert len(all_types) == 7
        assert set(all_types.keys()) == set(EXPECTED_OPERATION_TYPES.keys())

    def test_get_all_types_returns_copy_not_reference(self) -> None:
        """get_all_types returns a copy, not the internal registry."""
        all_types = RegistryPayloadConsul.get_all_types()

        # Mutating the returned dict should not affect the registry
        original_count = len(all_types)
        all_types["fake"] = type("FakePayload", (), {})  # type: ignore[assignment]

        # Registry should be unchanged
        assert len(RegistryPayloadConsul.get_all_types()) == original_count

    def test_is_registered_returns_true_for_known_types(self) -> None:
        """is_registered returns True for all 7 registered operation types."""
        for operation_type in EXPECTED_OPERATION_TYPES:
            assert RegistryPayloadConsul.is_registered(operation_type) is True, (
                f"Expected True for registered operation_type '{operation_type}'"
            )

    def test_is_registered_returns_false_for_unknown(self) -> None:
        """is_registered returns False for unknown operation types."""
        assert RegistryPayloadConsul.is_registered("unknown") is False
        assert RegistryPayloadConsul.is_registered("consul_kv") is False
        assert RegistryPayloadConsul.is_registered("get_service") is False

    def test_is_registered_empty_string_returns_false(self) -> None:
        """is_registered returns False for empty string."""
        assert RegistryPayloadConsul.is_registered("") is False

    def test_register_decorator_returns_class_unchanged(self) -> None:
        """The @register decorator returns the class unchanged.

        This test creates a temporary test-only payload class to verify
        the decorator mechanism without polluting the registry. We use
        clear() for test isolation (clear() is designed for testing only).
        """
        # Save original registry state
        original_types = RegistryPayloadConsul.get_all_types()

        try:
            # Clear to test registration in isolation
            RegistryPayloadConsul.clear()

            @RegistryPayloadConsul.register("test_operation")
            class TestPayload(ModelPayloadConsul):
                operation_type: Literal["test_operation"] = "test_operation"

            # Verify the class was returned unchanged
            assert TestPayload.__name__ == "TestPayload"
            assert (
                TestPayload.model_fields["operation_type"].default == "test_operation"
            )

            # Verify it was registered
            assert RegistryPayloadConsul.is_registered("test_operation")
            assert RegistryPayloadConsul.get_type("test_operation") is TestPayload

        finally:
            # Restore original registry state
            RegistryPayloadConsul.clear()
            for op_type, cls in original_types.items():
                RegistryPayloadConsul._types[op_type] = cls

    def test_register_duplicate_operation_type_raises_valueerror(self) -> None:
        """Registering the same operation_type twice raises ValueError.

        This prevents accidental overwrites of existing payload types.
        """
        # Save original registry state
        original_types = RegistryPayloadConsul.get_all_types()

        try:
            RegistryPayloadConsul.clear()

            # First registration should succeed
            @RegistryPayloadConsul.register("duplicate_test")
            class FirstPayload(ModelPayloadConsul):
                operation_type: Literal["duplicate_test"] = "duplicate_test"

            # Second registration with same operation_type should fail
            with pytest.raises(ValueError) as exc_info:

                @RegistryPayloadConsul.register("duplicate_test")
                class SecondPayload(ModelPayloadConsul):
                    operation_type: Literal["duplicate_test"] = "duplicate_test"

            error_msg = str(exc_info.value)
            assert "duplicate_test" in error_msg
            assert "already registered" in error_msg
            assert "FirstPayload" in error_msg

        finally:
            # Restore original registry state
            RegistryPayloadConsul.clear()
            for op_type, cls in original_types.items():
                RegistryPayloadConsul._types[op_type] = cls

    def test_clear_removes_all_registered_types(self) -> None:
        """clear() removes all registered types from the registry.

        Note: clear() is intended for testing only and should not be
        used in production code.
        """
        # Save original registry state
        original_types = RegistryPayloadConsul.get_all_types()

        try:
            # Registry should have types before clear
            assert len(RegistryPayloadConsul.get_all_types()) == 7

            RegistryPayloadConsul.clear()

            # Registry should be empty after clear
            assert len(RegistryPayloadConsul.get_all_types()) == 0
            for operation_type in EXPECTED_OPERATION_TYPES:
                assert RegistryPayloadConsul.is_registered(operation_type) is False

        finally:
            # Restore original registry state
            RegistryPayloadConsul.clear()
            for op_type, cls in original_types.items():
                RegistryPayloadConsul._types[op_type] = cls


# ============================================================================
# Tests for ModelPayloadConsul Base Class
# ============================================================================


@pytest.mark.unit
class TestModelPayloadConsul:
    """Tests for ModelPayloadConsul base class.

    ModelPayloadConsul defines the common interface that all Consul handler
    payloads share. It ensures consistent field names and configuration
    across all payload types.
    """

    def test_has_operation_type_field(self) -> None:
        """Base model defines required operation_type field for type discrimination."""
        fields = ModelPayloadConsul.model_fields
        assert "operation_type" in fields
        assert fields["operation_type"].annotation == str

    def test_model_config_has_frozen_true(self) -> None:
        """Base model config has frozen=True for immutability."""
        config = ModelPayloadConsul.model_config
        assert config.get("frozen") is True

    def test_model_config_has_extra_forbid(self) -> None:
        """Base model config has extra='forbid' to prevent extra fields."""
        config = ModelPayloadConsul.model_config
        assert config.get("extra") == "forbid"

    def test_base_model_can_be_instantiated_with_operation_type(self) -> None:
        """Base model can be instantiated with operation_type field.

        Note: In practice, concrete subclasses should be used, but the
        base class should still be instantiable for testing purposes.
        """
        payload = ModelPayloadConsul(operation_type="test")

        assert payload.operation_type == "test"

    def test_model_is_frozen_cannot_modify_operation_type(self) -> None:
        """Frozen model prevents modification of operation_type field."""
        payload = ModelPayloadConsul(operation_type="test")

        with pytest.raises(ValidationError):
            payload.operation_type = "modified"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Model raises ValidationError when extra fields are provided."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPayloadConsul(
                operation_type="test",
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        # Verify error is about extra field
        error_msg = str(exc_info.value)
        assert "extra_field" in error_msg or "Extra" in error_msg


# ============================================================================
# Tests for Concrete Consul Payload Model Inheritance
# ============================================================================


@pytest.mark.unit
class TestConcreteConsulPayloadModels:
    """Tests for concrete Consul payload model inheritance and registration.

    These tests verify that all 7 concrete payload models correctly inherit
    from ModelPayloadConsul and are properly registered in RegistryPayloadConsul.
    """

    def test_kv_get_found_inherits_from_base(self) -> None:
        """ModelConsulKVGetFoundPayload inherits from ModelPayloadConsul."""
        assert issubclass(ModelConsulKVGetFoundPayload, ModelPayloadConsul)

    def test_kv_get_not_found_inherits_from_base(self) -> None:
        """ModelConsulKVGetNotFoundPayload inherits from ModelPayloadConsul."""
        assert issubclass(ModelConsulKVGetNotFoundPayload, ModelPayloadConsul)

    def test_kv_get_recurse_inherits_from_base(self) -> None:
        """ModelConsulKVGetRecursePayload inherits from ModelPayloadConsul."""
        assert issubclass(ModelConsulKVGetRecursePayload, ModelPayloadConsul)

    def test_kv_put_inherits_from_base(self) -> None:
        """ModelConsulKVPutPayload inherits from ModelPayloadConsul."""
        assert issubclass(ModelConsulKVPutPayload, ModelPayloadConsul)

    def test_register_inherits_from_base(self) -> None:
        """ModelConsulRegisterPayload inherits from ModelPayloadConsul."""
        assert issubclass(ModelConsulRegisterPayload, ModelPayloadConsul)

    def test_deregister_inherits_from_base(self) -> None:
        """ModelConsulDeregisterPayload inherits from ModelPayloadConsul."""
        assert issubclass(ModelConsulDeregisterPayload, ModelPayloadConsul)

    def test_health_check_inherits_from_base(self) -> None:
        """ModelConsulHealthCheckPayload inherits from ModelPayloadConsul."""
        assert issubclass(ModelConsulHealthCheckPayload, ModelPayloadConsul)

    def test_all_payloads_inherit_from_base(self) -> None:
        """All 7 payload models inherit from ModelPayloadConsul."""
        for operation_type, payload_cls in EXPECTED_OPERATION_TYPES.items():
            assert issubclass(payload_cls, ModelPayloadConsul), (
                f"{payload_cls.__name__} should inherit from ModelPayloadConsul"
            )

    def test_all_payloads_registered(self) -> None:
        """All 7 payload models are registered in the registry."""
        for operation_type, expected_cls in EXPECTED_OPERATION_TYPES.items():
            assert RegistryPayloadConsul.is_registered(operation_type), (
                f"Operation type '{operation_type}' should be registered"
            )
            assert RegistryPayloadConsul.get_type(operation_type) is expected_cls, (
                f"Registry should return {expected_cls.__name__} for '{operation_type}'"
            )

    def test_kv_get_found_operation_type_is_literal(
        self, sample_kv_get_found_payload: ModelConsulKVGetFoundPayload
    ) -> None:
        """KV get found payload operation_type is always 'kv_get_found'."""
        assert sample_kv_get_found_payload.operation_type == "kv_get_found"

    def test_kv_get_not_found_operation_type_is_literal(
        self, sample_kv_get_not_found_payload: ModelConsulKVGetNotFoundPayload
    ) -> None:
        """KV get not found payload operation_type is always 'kv_get_not_found'."""
        assert sample_kv_get_not_found_payload.operation_type == "kv_get_not_found"

    def test_kv_get_recurse_operation_type_is_literal(
        self, sample_kv_get_recurse_payload: ModelConsulKVGetRecursePayload
    ) -> None:
        """KV get recurse payload operation_type is always 'kv_get_recurse'."""
        assert sample_kv_get_recurse_payload.operation_type == "kv_get_recurse"

    def test_kv_put_operation_type_is_literal(
        self, sample_kv_put_payload: ModelConsulKVPutPayload
    ) -> None:
        """KV put payload operation_type is always 'kv_put'."""
        assert sample_kv_put_payload.operation_type == "kv_put"

    def test_register_operation_type_is_literal(
        self, sample_register_payload: ModelConsulRegisterPayload
    ) -> None:
        """Register payload operation_type is always 'register'."""
        assert sample_register_payload.operation_type == "register"

    def test_deregister_operation_type_is_literal(
        self, sample_deregister_payload: ModelConsulDeregisterPayload
    ) -> None:
        """Deregister payload operation_type is always 'deregister'."""
        assert sample_deregister_payload.operation_type == "deregister"

    def test_health_check_operation_type_is_literal(
        self, sample_health_check_payload: ModelConsulHealthCheckPayload
    ) -> None:
        """Health check payload operation_type is always 'health_check'."""
        assert sample_health_check_payload.operation_type == "health_check"

    def test_all_payloads_are_frozen(self) -> None:
        """All payload models are frozen and cannot be modified."""
        # Create instances for each type
        payloads: list[ModelPayloadConsul] = [
            ModelConsulKVGetFoundPayload(key="test", value="value", index=1),
            ModelConsulKVGetNotFoundPayload(key="test", index=1),
            ModelConsulKVPutPayload(success=True, key="test"),
            ModelConsulRegisterPayload(
                registered=True, name="test", consul_service_id="id"
            ),
            ModelConsulDeregisterPayload(deregistered=True, consul_service_id="id"),
            ModelConsulHealthCheckPayload(
                healthy=True,
                initialized=True,
                handler_type="consul",
                timeout_seconds=5.0,
                circuit_breaker_state="closed",
                circuit_breaker_failure_count=0,
                thread_pool_active_workers=1,
                thread_pool_max_workers=10,
                thread_pool_max_queue_size=100,
            ),
        ]

        for payload in payloads:
            with pytest.raises(ValidationError):
                payload.operation_type = "modified"  # type: ignore[misc]

    def test_all_payloads_forbid_extra_fields(self) -> None:
        """All payload models reject extra fields."""
        # Test each concrete type
        with pytest.raises(ValidationError):
            ModelConsulKVGetFoundPayload(
                key="test",
                value="value",
                index=1,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        with pytest.raises(ValidationError):
            ModelConsulRegisterPayload(
                registered=True,
                name="test",
                consul_service_id="id",
                extra="not_allowed",  # type: ignore[call-arg]
            )


# ============================================================================
# Tests for Consul Payload Serialization and Deserialization
# ============================================================================


@pytest.mark.unit
class TestConsulPayloadSerialization:
    """Tests for Consul payload serialization and deserialization.

    These tests verify that payload models can be serialized to JSON/dict
    and deserialized back, with the operation_type field enabling type discrimination.
    """

    def test_payload_serializes_to_dict(
        self, sample_kv_get_found_payload: ModelConsulKVGetFoundPayload
    ) -> None:
        """Payload can be serialized to dict."""
        data = sample_kv_get_found_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "kv_get_found"
        assert "key" in data
        assert "value" in data
        assert "index" in data

    def test_payload_serializes_to_json(
        self, sample_kv_get_found_payload: ModelConsulKVGetFoundPayload
    ) -> None:
        """Payload can be serialized to JSON string."""
        json_str = sample_kv_get_found_payload.model_dump_json()

        assert isinstance(json_str, str)
        assert (
            '"operation_type":"kv_get_found"' in json_str
            or '"operation_type": "kv_get_found"' in json_str
        )

    def test_payload_deserializes_from_dict(
        self, sample_kv_get_found_payload: ModelConsulKVGetFoundPayload
    ) -> None:
        """Payload can be deserialized from dict."""
        data = sample_kv_get_found_payload.model_dump()
        restored = ModelConsulKVGetFoundPayload.model_validate(data)

        assert restored.operation_type == "kv_get_found"
        assert restored.key == sample_kv_get_found_payload.key
        assert restored.value == sample_kv_get_found_payload.value
        assert restored.index == sample_kv_get_found_payload.index

    def test_register_payload_serializes_to_dict(
        self, sample_register_payload: ModelConsulRegisterPayload
    ) -> None:
        """Register payload can be serialized to dict."""
        data = sample_register_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "register"
        assert data["registered"] is True
        assert data["name"] == "test-service"
        assert data["consul_service_id"] == "test-service-001"

    def test_health_check_payload_serializes_to_dict(
        self, sample_health_check_payload: ModelConsulHealthCheckPayload
    ) -> None:
        """Health check payload can be serialized to dict."""
        data = sample_health_check_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "health_check"
        assert data["healthy"] is True
        assert data["initialized"] is True
        assert data["handler_type"] == "consul"

    def test_kv_get_found_round_trip_preserves_data(
        self, sample_kv_get_found_payload: ModelConsulKVGetFoundPayload
    ) -> None:
        """KV get found payload JSON round-trip preserves all data."""
        json_str = sample_kv_get_found_payload.model_dump_json()
        restored = ModelConsulKVGetFoundPayload.model_validate_json(json_str)

        assert restored == sample_kv_get_found_payload

    def test_register_payload_round_trip_preserves_data(
        self, sample_register_payload: ModelConsulRegisterPayload
    ) -> None:
        """Register payload JSON round-trip preserves all data."""
        json_str = sample_register_payload.model_dump_json()
        restored = ModelConsulRegisterPayload.model_validate_json(json_str)

        assert restored == sample_register_payload

    def test_operation_type_enables_type_discrimination(self) -> None:
        """The operation_type field enables determining the correct payload type.

        This demonstrates the discriminated union pattern where the operation_type
        field is used to select the appropriate model class.
        """
        kv_get_found_data = {
            "operation_type": "kv_get_found",
            "found": True,
            "key": "test/key",
            "value": "test-value",
            "index": 100,
        }

        register_data = {
            "operation_type": "register",
            "registered": True,
            "name": "my-service",
            "consul_service_id": "my-service-001",
        }

        # Use registry to get correct class based on operation_type
        kv_cls = RegistryPayloadConsul.get_type(kv_get_found_data["operation_type"])
        register_cls = RegistryPayloadConsul.get_type(register_data["operation_type"])

        kv_payload = kv_cls.model_validate(kv_get_found_data)
        register_payload = register_cls.model_validate(register_data)

        assert isinstance(kv_payload, ModelConsulKVGetFoundPayload)
        assert isinstance(register_payload, ModelConsulRegisterPayload)

    def test_deserialization_with_wrong_operation_type_fails(self) -> None:
        """Deserializing with wrong operation_type field fails validation.

        ModelConsulKVGetFoundPayload expects operation_type='kv_get_found' as a Literal.
        Providing a different value should fail validation.
        """
        data = {
            "operation_type": "wrong_type",
            "found": True,
            "key": "test",
            "value": "value",
            "index": 1,
        }

        with pytest.raises(ValidationError):
            ModelConsulKVGetFoundPayload.model_validate(data)

    def test_all_payloads_can_serialize_and_deserialize(self) -> None:
        """All 7 payload types can be serialized and deserialized."""
        payloads: list[ModelPayloadConsul] = [
            ModelConsulKVGetFoundPayload(
                key="test", value="value", flags=0, modify_index=1, index=1
            ),
            ModelConsulKVGetNotFoundPayload(key="missing", index=1),
            ModelConsulKVPutPayload(success=True, key="new-key"),
            ModelConsulRegisterPayload(
                registered=True, name="service", consul_service_id="id-1"
            ),
            ModelConsulDeregisterPayload(deregistered=True, consul_service_id="id-1"),
            ModelConsulHealthCheckPayload(
                healthy=True,
                initialized=True,
                handler_type="consul",
                timeout_seconds=5.0,
                circuit_breaker_state="closed",
                circuit_breaker_failure_count=0,
                thread_pool_active_workers=1,
                thread_pool_max_workers=10,
                thread_pool_max_queue_size=100,
            ),
        ]

        for payload in payloads:
            # Serialize to JSON
            json_str = payload.model_dump_json()

            # Get the correct class from registry
            data = payload.model_dump()
            payload_cls = RegistryPayloadConsul.get_type(data["operation_type"])

            # Deserialize back
            restored = payload_cls.model_validate_json(json_str)

            assert restored == payload, (
                f"Round-trip failed for {type(payload).__name__}"
            )


# ============================================================================
# Tests for Thread Safety and Immutability
# ============================================================================


@pytest.mark.unit
class TestConsulPayloadThreadSafety:
    """Tests for thread safety and immutability of Consul payload models.

    All payload models and the RegistryPayloadConsul are designed to be thread-safe:
    - RegistryPayloadConsul is populated at module import time and read-only after
    - All payload models are frozen (immutable)
    """

    def test_registry_class_var_is_shared(self) -> None:
        """RegistryPayloadConsul._types is a ClassVar shared across all usage.

        This ensures the registry is consistent regardless of where it's
        accessed in the codebase.
        """
        types1 = RegistryPayloadConsul.get_all_types()
        types2 = RegistryPayloadConsul.get_all_types()

        # Should return equivalent copies
        assert types1 == types2

    def test_kv_get_found_payload_is_hashable(
        self, sample_kv_get_found_payload: ModelConsulKVGetFoundPayload
    ) -> None:
        """Frozen KV get found payload is hashable for use in sets/dicts."""
        # Should not raise TypeError
        hash_value = hash(sample_kv_get_found_payload)
        assert isinstance(hash_value, int)

        # Can be used in a set
        payload_set = {sample_kv_get_found_payload}
        assert len(payload_set) == 1

    def test_register_payload_is_hashable(
        self, sample_register_payload: ModelConsulRegisterPayload
    ) -> None:
        """Frozen register payload is hashable for use in sets/dicts."""
        hash_value = hash(sample_register_payload)
        assert isinstance(hash_value, int)

    def test_identical_payloads_have_same_hash(self) -> None:
        """Identical payloads should have the same hash value."""
        payload1 = ModelConsulKVPutPayload(success=True, key="test/key")
        payload2 = ModelConsulKVPutPayload(success=True, key="test/key")

        assert hash(payload1) == hash(payload2)
        assert payload1 == payload2

    def test_payloads_can_be_used_as_dict_keys(
        self, sample_register_payload: ModelConsulRegisterPayload
    ) -> None:
        """Frozen payloads can be used as dictionary keys."""
        payload_dict = {sample_register_payload: "value"}

        assert payload_dict[sample_register_payload] == "value"


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.unit
class TestConsulPayloadEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_kv_get_found_with_none_value(self) -> None:
        """KV get found payload can have None value (key exists but empty)."""
        payload = ModelConsulKVGetFoundPayload(
            key="test/empty-key",
            value=None,
            index=1,
        )

        assert payload.key == "test/empty-key"
        assert payload.value is None
        assert payload.found is True

    def test_kv_get_not_found_value_is_always_none(self) -> None:
        """KV get not found payload value is always None."""
        payload = ModelConsulKVGetNotFoundPayload(
            key="test/missing",
            index=1,
        )

        assert payload.value is None
        assert payload.found is False

    def test_kv_get_recurse_with_empty_items(self) -> None:
        """KV get recurse payload can have empty items list."""
        payload = ModelConsulKVGetRecursePayload(
            found=False,
            items=[],
            count=0,
            index=1,
        )

        assert payload.found is False
        assert payload.items == []
        assert payload.count == 0

    def test_health_check_with_none_circuit_breaker_state(self) -> None:
        """Health check payload can have None circuit_breaker_state."""
        payload = ModelConsulHealthCheckPayload(
            healthy=True,
            initialized=True,
            handler_type="consul",
            timeout_seconds=5.0,
            circuit_breaker_state=None,
            circuit_breaker_failure_count=0,
            thread_pool_active_workers=0,
            thread_pool_max_workers=10,
            thread_pool_max_queue_size=100,
        )

        assert payload.circuit_breaker_state is None

    def test_register_payload_with_special_characters_in_name(self) -> None:
        """Register payload accepts special characters in service name."""
        payload = ModelConsulRegisterPayload(
            registered=True,
            name="my-service_v1.0.0",
            consul_service_id="my-service-abc-123",
        )

        assert payload.name == "my-service_v1.0.0"
        assert payload.consul_service_id == "my-service-abc-123"

    def test_kv_get_found_with_zero_index(self) -> None:
        """KV get found payload can have zero index."""
        payload = ModelConsulKVGetFoundPayload(
            key="test/key",
            value="value",
            index=0,
        )

        assert payload.index == 0

    def test_kv_get_found_with_large_modify_index(self) -> None:
        """KV get found payload can have large modify_index."""
        payload = ModelConsulKVGetFoundPayload(
            key="test/key",
            value="value",
            modify_index=9999999999,
            index=1,
        )

        assert payload.modify_index == 9999999999

    def test_health_check_with_zero_workers(self) -> None:
        """Health check payload can have zero active workers."""
        payload = ModelConsulHealthCheckPayload(
            healthy=True,
            initialized=True,
            handler_type="consul",
            timeout_seconds=5.0,
            circuit_breaker_state="closed",
            circuit_breaker_failure_count=0,
            thread_pool_active_workers=0,
            thread_pool_max_workers=10,
            thread_pool_max_queue_size=100,
        )

        assert payload.thread_pool_active_workers == 0

    def test_deregister_with_long_service_id(self) -> None:
        """Deregister payload accepts long service IDs."""
        long_id = "a" * 256
        payload = ModelConsulDeregisterPayload(
            deregistered=True,
            consul_service_id=long_id,
        )

        assert payload.consul_service_id == long_id

    def test_kv_put_with_special_key_characters(self) -> None:
        """KV put payload accepts special characters in key."""
        payload = ModelConsulKVPutPayload(
            success=True,
            key="test/nested/key-with_special.chars",
        )

        assert payload.key == "test/nested/key-with_special.chars"


# ============================================================================
# KV Get Recurse Specific Tests
# ============================================================================


@pytest.mark.unit
class TestKVGetRecursePayload:
    """Tests specific to ModelConsulKVGetRecursePayload with nested items."""

    def test_recurse_payload_with_multiple_items(
        self, sample_kv_get_recurse_payload: ModelConsulKVGetRecursePayload
    ) -> None:
        """Recurse payload correctly stores multiple KV items."""
        assert sample_kv_get_recurse_payload.found is True
        assert len(sample_kv_get_recurse_payload.items) == 2
        assert sample_kv_get_recurse_payload.count == 2

    def test_recurse_payload_items_are_accessible(
        self, sample_kv_get_recurse_payload: ModelConsulKVGetRecursePayload
    ) -> None:
        """Recurse payload items are accessible with correct data."""
        items = sample_kv_get_recurse_payload.items

        assert items[0].key == "test/key1"
        assert items[0].value == "value1"
        assert items[1].key == "test/key2"
        assert items[1].value == "value2"

    def test_recurse_payload_serialization_includes_items(
        self, sample_kv_get_recurse_payload: ModelConsulKVGetRecursePayload
    ) -> None:
        """Recurse payload serialization includes nested items."""
        data = sample_kv_get_recurse_payload.model_dump()

        assert "items" in data
        assert len(data["items"]) == 2
        assert data["items"][0]["key"] == "test/key1"
        assert data["items"][1]["key"] == "test/key2"

    def test_recurse_payload_round_trip_preserves_items(
        self, sample_kv_get_recurse_payload: ModelConsulKVGetRecursePayload
    ) -> None:
        """Recurse payload round-trip preserves nested items."""
        json_str = sample_kv_get_recurse_payload.model_dump_json()
        restored = ModelConsulKVGetRecursePayload.model_validate_json(json_str)

        assert restored == sample_kv_get_recurse_payload
        assert len(restored.items) == 2
