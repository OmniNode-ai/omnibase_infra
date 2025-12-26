# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for RegistryPayloadVault and ModelPayloadVault.

This module validates the registry pattern for Vault handler payloads,
following the template established by test_registry_intent.py.

The RegistryPayloadVault provides a decorator-based registration mechanism that
enables dynamic type resolution during Pydantic validation without requiring
explicit union type definitions. This pattern:
- Eliminates duplicate union definitions across modules
- Allows new payload types to be added by implementing ModelPayloadVault
- Uses the `operation_type` field as a discriminator for type resolution
- Follows ONEX duck typing principles while maintaining type safety

Test Categories:
    1. Registry Tests - decorator and method behavior
    2. Base Model Tests - common fields and configuration
    3. Concrete Model Inheritance Tests - all 6 Vault payload types
    4. Serialization/Deserialization Tests - JSON round-trip validation
    5. Thread Safety and Immutability Tests

Vault Operation Type Mapping:
    - "read_secret" -> ModelVaultSecretPayload
    - "write_secret" -> ModelVaultWritePayload
    - "delete_secret" -> ModelVaultDeletePayload
    - "list_secrets" -> ModelVaultListPayload
    - "renew_token" -> ModelVaultRenewTokenPayload
    - "health_check" -> ModelVaultHealthCheckPayload

Related:
    - EnumVaultOperationType: Discriminator enum for operation types
    - OMN-1007: Union reduction refactoring

.. versionadded:: 0.7.0
    Created as part of OMN-1007 registry pattern implementation.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.models.vault import (
    EnumVaultOperationType,
    ModelPayloadVault,
    ModelVaultDeletePayload,
    ModelVaultHealthCheckPayload,
    ModelVaultListPayload,
    ModelVaultRenewTokenPayload,
    ModelVaultSecretPayload,
    ModelVaultWritePayload,
    RegistryPayloadVault,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_secret_payload() -> ModelVaultSecretPayload:
    """Create a sample Vault secret payload for testing."""
    return ModelVaultSecretPayload(
        data={"username": "admin", "password": "secret123"},
        metadata={"version": 1, "created_time": "2025-01-01T00:00:00Z"},
    )


@pytest.fixture
def sample_write_payload() -> ModelVaultWritePayload:
    """Create a sample Vault write payload for testing."""
    return ModelVaultWritePayload(
        version=2,
        created_time="2025-01-01T12:00:00Z",
    )


@pytest.fixture
def sample_delete_payload() -> ModelVaultDeletePayload:
    """Create a sample Vault delete payload for testing."""
    return ModelVaultDeletePayload(deleted=True)


@pytest.fixture
def sample_list_payload() -> ModelVaultListPayload:
    """Create a sample Vault list payload for testing."""
    return ModelVaultListPayload(keys=["db/", "api/", "config"])


@pytest.fixture
def sample_renew_token_payload() -> ModelVaultRenewTokenPayload:
    """Create a sample Vault renew token payload for testing."""
    return ModelVaultRenewTokenPayload(
        renewable=True,
        lease_duration=3600,
    )


@pytest.fixture
def sample_health_check_payload() -> ModelVaultHealthCheckPayload:
    """Create a sample Vault health check payload for testing."""
    return ModelVaultHealthCheckPayload(
        healthy=True,
        initialized=True,
        handler_type="vault",
        timeout_seconds=30.0,
        token_ttl_remaining_seconds=1800,
        circuit_breaker_state="closed",
        circuit_breaker_failure_count=0,
        thread_pool_active_workers=2,
        thread_pool_max_workers=10,
    )


# ============================================================================
# Tests for RegistryPayloadVault
# ============================================================================


@pytest.mark.unit
class TestRegistryPayloadVault:
    """Tests for RegistryPayloadVault class methods and registration behavior.

    The RegistryPayloadVault provides a decorator-based mechanism for registering
    Vault payload model types, enabling dynamic type resolution during Pydantic
    validation without explicit union type definitions.
    """

    def test_get_type_returns_correct_class_for_read_secret(self) -> None:
        """Registry returns correct class for read_secret operation type."""
        payload_cls = RegistryPayloadVault.get_type("read_secret")
        assert payload_cls is ModelVaultSecretPayload

    def test_get_type_returns_correct_class_for_write_secret(self) -> None:
        """Registry returns correct class for write_secret operation type."""
        payload_cls = RegistryPayloadVault.get_type("write_secret")
        assert payload_cls is ModelVaultWritePayload

    def test_get_type_returns_correct_class_for_delete_secret(self) -> None:
        """Registry returns correct class for delete_secret operation type."""
        payload_cls = RegistryPayloadVault.get_type("delete_secret")
        assert payload_cls is ModelVaultDeletePayload

    def test_get_type_returns_correct_class_for_list_secrets(self) -> None:
        """Registry returns correct class for list_secrets operation type."""
        payload_cls = RegistryPayloadVault.get_type("list_secrets")
        assert payload_cls is ModelVaultListPayload

    def test_get_type_returns_correct_class_for_renew_token(self) -> None:
        """Registry returns correct class for renew_token operation type."""
        payload_cls = RegistryPayloadVault.get_type("renew_token")
        assert payload_cls is ModelVaultRenewTokenPayload

    def test_get_type_returns_correct_class_for_health_check(self) -> None:
        """Registry returns correct class for health_check operation type."""
        payload_cls = RegistryPayloadVault.get_type("health_check")
        assert payload_cls is ModelVaultHealthCheckPayload

    def test_get_type_unknown_raises_keyerror(self) -> None:
        """Registry raises KeyError for unknown operation type with helpful message."""
        with pytest.raises(KeyError) as exc_info:
            RegistryPayloadVault.get_type("unknown_operation")

        # Verify error message contains useful information
        error_msg = str(exc_info.value)
        assert "unknown_operation" in error_msg
        assert "Registered types" in error_msg

    def test_get_type_unknown_lists_registered_types(self) -> None:
        """KeyError message includes list of registered operation types."""
        with pytest.raises(KeyError) as exc_info:
            RegistryPayloadVault.get_type("nonexistent")

        error_msg = str(exc_info.value)
        # Should mention at least some of the registered types
        assert (
            "read_secret" in error_msg
            or "write_secret" in error_msg
            or "health_check" in error_msg
        )

    def test_get_all_types_returns_all_registered(self) -> None:
        """get_all_types returns dict with all 6 registered Vault payload types."""
        all_types = RegistryPayloadVault.get_all_types()

        assert isinstance(all_types, dict)
        assert len(all_types) == 6

        # Verify all 6 types are present
        assert "read_secret" in all_types
        assert "write_secret" in all_types
        assert "delete_secret" in all_types
        assert "list_secrets" in all_types
        assert "renew_token" in all_types
        assert "health_check" in all_types

        # Verify correct class mappings
        assert all_types["read_secret"] is ModelVaultSecretPayload
        assert all_types["write_secret"] is ModelVaultWritePayload
        assert all_types["delete_secret"] is ModelVaultDeletePayload
        assert all_types["list_secrets"] is ModelVaultListPayload
        assert all_types["renew_token"] is ModelVaultRenewTokenPayload
        assert all_types["health_check"] is ModelVaultHealthCheckPayload

    def test_get_all_types_returns_copy_not_reference(self) -> None:
        """get_all_types returns a copy, not the internal registry."""
        all_types = RegistryPayloadVault.get_all_types()

        # Mutating the returned dict should not affect the registry
        original_count = len(all_types)
        all_types["fake"] = type("FakePayload", (), {})  # type: ignore[assignment]

        # Registry should be unchanged
        assert len(RegistryPayloadVault.get_all_types()) == original_count

    def test_is_registered_returns_true_for_known(self) -> None:
        """is_registered returns True for all 6 known operation types."""
        assert RegistryPayloadVault.is_registered("read_secret") is True
        assert RegistryPayloadVault.is_registered("write_secret") is True
        assert RegistryPayloadVault.is_registered("delete_secret") is True
        assert RegistryPayloadVault.is_registered("list_secrets") is True
        assert RegistryPayloadVault.is_registered("renew_token") is True
        assert RegistryPayloadVault.is_registered("health_check") is True

    def test_is_registered_returns_false_for_unknown(self) -> None:
        """is_registered returns False for unknown operation types."""
        assert RegistryPayloadVault.is_registered("unknown") is False
        assert RegistryPayloadVault.is_registered("consul") is False
        assert RegistryPayloadVault.is_registered("postgres") is False
        assert RegistryPayloadVault.is_registered("") is False

    def test_register_decorator_returns_class_unchanged(self) -> None:
        """The @register decorator returns the class unchanged.

        This test creates a temporary test-only payload class to verify
        the decorator mechanism without polluting the registry. We use
        clear() for test isolation (clear() is designed for testing only).
        """
        # Save original registry state
        original_types = RegistryPayloadVault.get_all_types()

        try:
            # Clear to test registration in isolation
            RegistryPayloadVault.clear()

            @RegistryPayloadVault.register("test_operation")
            class TestPayload(ModelPayloadVault):
                operation_type: Literal["test_operation"] = "test_operation"

            # Verify the class was returned unchanged
            assert TestPayload.__name__ == "TestPayload"
            assert (
                TestPayload.model_fields["operation_type"].default == "test_operation"
            )

            # Verify it was registered
            assert RegistryPayloadVault.is_registered("test_operation")
            assert RegistryPayloadVault.get_type("test_operation") is TestPayload

        finally:
            # Restore original registry state
            RegistryPayloadVault.clear()
            for op_type, cls in original_types.items():
                RegistryPayloadVault._types[op_type] = cls

    def test_register_duplicate_raises_valueerror(self) -> None:
        """Registering the same operation type twice raises ValueError.

        This prevents accidental overwrites of existing payload types.
        """
        # Save original registry state
        original_types = RegistryPayloadVault.get_all_types()

        try:
            RegistryPayloadVault.clear()

            # First registration should succeed
            @RegistryPayloadVault.register("duplicate_test")
            class FirstPayload(ModelPayloadVault):
                operation_type: Literal["duplicate_test"] = "duplicate_test"

            # Second registration with same operation type should fail
            with pytest.raises(ValueError) as exc_info:

                @RegistryPayloadVault.register("duplicate_test")
                class SecondPayload(ModelPayloadVault):
                    operation_type: Literal["duplicate_test"] = "duplicate_test"

            error_msg = str(exc_info.value)
            assert "duplicate_test" in error_msg
            assert "already registered" in error_msg
            assert "FirstPayload" in error_msg

        finally:
            # Restore original registry state
            RegistryPayloadVault.clear()
            for op_type, cls in original_types.items():
                RegistryPayloadVault._types[op_type] = cls

    def test_clear_removes_all_registered_types(self) -> None:
        """clear() removes all registered types from the registry.

        Note: clear() is intended for testing only and should not be
        used in production code.
        """
        # Save original registry state
        original_types = RegistryPayloadVault.get_all_types()

        try:
            # Registry should have types before clear
            assert len(RegistryPayloadVault.get_all_types()) == 6

            RegistryPayloadVault.clear()

            # Registry should be empty after clear
            assert len(RegistryPayloadVault.get_all_types()) == 0
            assert RegistryPayloadVault.is_registered("read_secret") is False
            assert RegistryPayloadVault.is_registered("health_check") is False

        finally:
            # Restore original registry state
            RegistryPayloadVault.clear()
            for op_type, cls in original_types.items():
                RegistryPayloadVault._types[op_type] = cls


# ============================================================================
# Tests for ModelPayloadVault Base Class
# ============================================================================


@pytest.mark.unit
class TestModelPayloadVault:
    """Tests for ModelPayloadVault base class.

    ModelPayloadVault defines the common interface that all Vault payloads
    share. It ensures consistent field names and configuration across all
    payload types.
    """

    def test_has_operation_type_field(self) -> None:
        """Base model defines required operation_type field for type discrimination."""
        fields = ModelPayloadVault.model_fields
        assert "operation_type" in fields
        assert fields["operation_type"].annotation == str

    def test_model_config_has_frozen_true(self) -> None:
        """Base model config has frozen=True for immutability."""
        config = ModelPayloadVault.model_config
        assert config.get("frozen") is True

    def test_model_config_has_extra_forbid(self) -> None:
        """Base model config has extra='forbid' to prevent extra fields."""
        config = ModelPayloadVault.model_config
        assert config.get("extra") == "forbid"

    def test_model_is_frozen(self) -> None:
        """Frozen model prevents modification after creation."""
        payload = ModelPayloadVault(operation_type="test")

        with pytest.raises(ValidationError):
            payload.operation_type = "modified"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Model raises ValidationError when extra fields are provided."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPayloadVault(
                operation_type="test",
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        # Verify error is about extra field
        error_msg = str(exc_info.value)
        assert "extra_field" in error_msg or "Extra" in error_msg

    def test_base_model_can_be_instantiated(self) -> None:
        """Base model can be instantiated with operation_type field.

        Note: In practice, concrete subclasses should be used, but the
        base class should still be instantiable for testing purposes.
        """
        payload = ModelPayloadVault(operation_type="test")

        assert payload.operation_type == "test"


# ============================================================================
# Tests for Concrete Vault Payload Model Inheritance
# ============================================================================


@pytest.mark.unit
class TestConcreteVaultPayloadModels:
    """Tests for concrete Vault payload model inheritance and registration.

    These tests verify that all 6 Vault payload models correctly inherit
    from ModelPayloadVault and are properly registered in RegistryPayloadVault.
    """

    def test_secret_payload_inherits_from_base(self) -> None:
        """ModelVaultSecretPayload inherits from ModelPayloadVault."""
        assert issubclass(ModelVaultSecretPayload, ModelPayloadVault)

    def test_write_payload_inherits_from_base(self) -> None:
        """ModelVaultWritePayload inherits from ModelPayloadVault."""
        assert issubclass(ModelVaultWritePayload, ModelPayloadVault)

    def test_delete_payload_inherits_from_base(self) -> None:
        """ModelVaultDeletePayload inherits from ModelPayloadVault."""
        assert issubclass(ModelVaultDeletePayload, ModelPayloadVault)

    def test_list_payload_inherits_from_base(self) -> None:
        """ModelVaultListPayload inherits from ModelPayloadVault."""
        assert issubclass(ModelVaultListPayload, ModelPayloadVault)

    def test_renew_token_payload_inherits_from_base(self) -> None:
        """ModelVaultRenewTokenPayload inherits from ModelPayloadVault."""
        assert issubclass(ModelVaultRenewTokenPayload, ModelPayloadVault)

    def test_health_check_payload_inherits_from_base(self) -> None:
        """ModelVaultHealthCheckPayload inherits from ModelPayloadVault."""
        assert issubclass(ModelVaultHealthCheckPayload, ModelPayloadVault)

    def test_all_payloads_registered(self) -> None:
        """All 6 Vault payload models are registered in the registry."""
        all_types = RegistryPayloadVault.get_all_types()

        assert len(all_types) == 6
        assert ModelVaultSecretPayload in all_types.values()
        assert ModelVaultWritePayload in all_types.values()
        assert ModelVaultDeletePayload in all_types.values()
        assert ModelVaultListPayload in all_types.values()
        assert ModelVaultRenewTokenPayload in all_types.values()
        assert ModelVaultHealthCheckPayload in all_types.values()

    def test_secret_payload_operation_type_is_read_secret(
        self, sample_secret_payload: ModelVaultSecretPayload
    ) -> None:
        """Secret payload operation_type is always 'read_secret'."""
        assert (
            sample_secret_payload.operation_type == EnumVaultOperationType.READ_SECRET
        )
        assert sample_secret_payload.operation_type == "read_secret"

    def test_write_payload_operation_type_is_write_secret(
        self, sample_write_payload: ModelVaultWritePayload
    ) -> None:
        """Write payload operation_type is always 'write_secret'."""
        assert (
            sample_write_payload.operation_type == EnumVaultOperationType.WRITE_SECRET
        )
        assert sample_write_payload.operation_type == "write_secret"

    def test_delete_payload_operation_type_is_delete_secret(
        self, sample_delete_payload: ModelVaultDeletePayload
    ) -> None:
        """Delete payload operation_type is always 'delete_secret'."""
        assert (
            sample_delete_payload.operation_type == EnumVaultOperationType.DELETE_SECRET
        )
        assert sample_delete_payload.operation_type == "delete_secret"

    def test_list_payload_operation_type_is_list_secrets(
        self, sample_list_payload: ModelVaultListPayload
    ) -> None:
        """List payload operation_type is always 'list_secrets'."""
        assert sample_list_payload.operation_type == EnumVaultOperationType.LIST_SECRETS
        assert sample_list_payload.operation_type == "list_secrets"

    def test_renew_token_payload_operation_type_is_renew_token(
        self, sample_renew_token_payload: ModelVaultRenewTokenPayload
    ) -> None:
        """Renew token payload operation_type is always 'renew_token'."""
        assert (
            sample_renew_token_payload.operation_type
            == EnumVaultOperationType.RENEW_TOKEN
        )
        assert sample_renew_token_payload.operation_type == "renew_token"

    def test_health_check_payload_operation_type_is_health_check(
        self, sample_health_check_payload: ModelVaultHealthCheckPayload
    ) -> None:
        """Health check payload operation_type is always 'health_check'."""
        assert (
            sample_health_check_payload.operation_type
            == EnumVaultOperationType.HEALTH_CHECK
        )
        assert sample_health_check_payload.operation_type == "health_check"

    def test_all_concrete_models_are_frozen(
        self,
        sample_secret_payload: ModelVaultSecretPayload,
        sample_write_payload: ModelVaultWritePayload,
        sample_delete_payload: ModelVaultDeletePayload,
        sample_list_payload: ModelVaultListPayload,
        sample_renew_token_payload: ModelVaultRenewTokenPayload,
        sample_health_check_payload: ModelVaultHealthCheckPayload,
    ) -> None:
        """All concrete payload models are frozen and cannot be modified."""
        with pytest.raises(ValidationError):
            sample_secret_payload.data = {}  # type: ignore[misc]

        with pytest.raises(ValidationError):
            sample_write_payload.version = 999  # type: ignore[misc]

        with pytest.raises(ValidationError):
            sample_delete_payload.deleted = False  # type: ignore[misc]

        with pytest.raises(ValidationError):
            sample_list_payload.keys = []  # type: ignore[misc]

        with pytest.raises(ValidationError):
            sample_renew_token_payload.renewable = False  # type: ignore[misc]

        with pytest.raises(ValidationError):
            sample_health_check_payload.healthy = False  # type: ignore[misc]

    def test_all_concrete_models_forbid_extra_fields(self) -> None:
        """All concrete payload models raise ValidationError for extra fields."""
        with pytest.raises(ValidationError):
            ModelVaultSecretPayload(
                data={"key": "value"},
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        with pytest.raises(ValidationError):
            ModelVaultWritePayload(
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        with pytest.raises(ValidationError):
            ModelVaultDeletePayload(
                deleted=True,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        with pytest.raises(ValidationError):
            ModelVaultListPayload(
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        with pytest.raises(ValidationError):
            ModelVaultRenewTokenPayload(
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        with pytest.raises(ValidationError):
            ModelVaultHealthCheckPayload(
                healthy=True,
                initialized=True,
                handler_type="vault",
                timeout_seconds=30.0,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )


# ============================================================================
# Tests for Vault Payload Serialization and Deserialization
# ============================================================================


@pytest.mark.unit
class TestVaultPayloadSerialization:
    """Tests for Vault payload serialization and deserialization.

    These tests verify that payload models can be serialized to JSON/dict
    and deserialized back, with the operation_type field enabling type
    discrimination.
    """

    def test_payload_serializes_to_json(
        self, sample_secret_payload: ModelVaultSecretPayload
    ) -> None:
        """Vault payload can be serialized to JSON string."""
        json_str = sample_secret_payload.model_dump_json()

        assert isinstance(json_str, str)
        assert (
            '"operation_type":"read_secret"' in json_str
            or '"operation_type": "read_secret"' in json_str
        )
        assert "username" in json_str
        assert "admin" in json_str

    def test_payload_deserializes_from_dict(self) -> None:
        """Vault payload can be deserialized from dict."""
        data = {
            "operation_type": "read_secret",
            "data": {"api_key": "secret123"},
            "metadata": {"version": 5},
        }

        payload = ModelVaultSecretPayload.model_validate(data)

        assert payload.operation_type == "read_secret"
        assert payload.data["api_key"] == "secret123"
        assert payload.metadata["version"] == 5

    def test_secret_payload_serializes_to_dict(
        self, sample_secret_payload: ModelVaultSecretPayload
    ) -> None:
        """Secret payload can be serialized to dict."""
        data = sample_secret_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "read_secret"
        assert "data" in data
        assert "metadata" in data

    def test_write_payload_serializes_to_dict(
        self, sample_write_payload: ModelVaultWritePayload
    ) -> None:
        """Write payload can be serialized to dict."""
        data = sample_write_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "write_secret"
        assert "version" in data
        assert "created_time" in data

    def test_delete_payload_serializes_to_dict(
        self, sample_delete_payload: ModelVaultDeletePayload
    ) -> None:
        """Delete payload can be serialized to dict."""
        data = sample_delete_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "delete_secret"
        assert data["deleted"] is True

    def test_list_payload_serializes_to_dict(
        self, sample_list_payload: ModelVaultListPayload
    ) -> None:
        """List payload can be serialized to dict."""
        data = sample_list_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "list_secrets"
        assert data["keys"] == ["db/", "api/", "config"]

    def test_renew_token_payload_serializes_to_dict(
        self, sample_renew_token_payload: ModelVaultRenewTokenPayload
    ) -> None:
        """Renew token payload can be serialized to dict."""
        data = sample_renew_token_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "renew_token"
        assert data["renewable"] is True
        assert data["lease_duration"] == 3600

    def test_health_check_payload_serializes_to_dict(
        self, sample_health_check_payload: ModelVaultHealthCheckPayload
    ) -> None:
        """Health check payload can be serialized to dict."""
        data = sample_health_check_payload.model_dump()

        assert isinstance(data, dict)
        assert data["operation_type"] == "health_check"
        assert data["healthy"] is True
        assert data["initialized"] is True
        assert data["handler_type"] == "vault"

    def test_secret_payload_round_trip_preserves_data(
        self, sample_secret_payload: ModelVaultSecretPayload
    ) -> None:
        """Secret payload JSON round-trip preserves all data."""
        json_str = sample_secret_payload.model_dump_json()
        restored = ModelVaultSecretPayload.model_validate_json(json_str)

        assert restored == sample_secret_payload

    def test_health_check_payload_round_trip_preserves_data(
        self, sample_health_check_payload: ModelVaultHealthCheckPayload
    ) -> None:
        """Health check payload JSON round-trip preserves all data."""
        json_str = sample_health_check_payload.model_dump_json()
        restored = ModelVaultHealthCheckPayload.model_validate_json(json_str)

        assert restored == sample_health_check_payload

    def test_operation_type_enables_type_discrimination(self) -> None:
        """The operation_type field enables determining the correct payload type.

        This demonstrates the discriminated union pattern where the operation_type
        field is used to select the appropriate model class.
        """
        secret_data = {
            "operation_type": "read_secret",
            "data": {"db_password": "secret"},
            "metadata": {},
        }

        health_data = {
            "operation_type": "health_check",
            "healthy": True,
            "initialized": True,
            "handler_type": "vault",
            "timeout_seconds": 30.0,
        }

        # Use registry to get correct class based on operation_type
        secret_cls = RegistryPayloadVault.get_type(secret_data["operation_type"])
        health_cls = RegistryPayloadVault.get_type(health_data["operation_type"])

        secret_payload = secret_cls.model_validate(secret_data)
        health_payload = health_cls.model_validate(health_data)

        assert isinstance(secret_payload, ModelVaultSecretPayload)
        assert isinstance(health_payload, ModelVaultHealthCheckPayload)

    def test_deserialization_with_wrong_operation_type_fails(self) -> None:
        """Deserializing with wrong operation_type field fails validation.

        ModelVaultSecretPayload expects operation_type='read_secret' as a Literal.
        Providing a different value should fail validation.
        """
        data = {
            "operation_type": "wrong_type",  # Not 'read_secret'
            "data": {"key": "value"},
            "metadata": {},
        }

        with pytest.raises(ValidationError):
            ModelVaultSecretPayload.model_validate(data)


# ============================================================================
# Tests for Thread Safety and Immutability
# ============================================================================


@pytest.mark.unit
class TestVaultPayloadThreadSafety:
    """Tests for thread safety and immutability of Vault payload models.

    All payload models and the RegistryPayloadVault are designed to be thread-safe:
    - RegistryPayloadVault is populated at module import time and read-only after
    - All payload models are frozen (immutable)
    """

    def test_registry_class_var_is_shared(self) -> None:
        """RegistryPayloadVault._types is a ClassVar shared across all usage.

        This ensures the registry is consistent regardless of where it's
        accessed in the codebase.
        """
        # Access registry from different calls
        types1 = RegistryPayloadVault.get_all_types()
        types2 = RegistryPayloadVault.get_all_types()

        # Should return equivalent copies
        assert types1 == types2

    def test_delete_payload_is_hashable(
        self, sample_delete_payload: ModelVaultDeletePayload
    ) -> None:
        """Frozen delete payload with immutable fields is hashable.

        Note: Payloads containing only immutable fields (bool, int, str, None)
        are hashable. Payloads with mutable fields (dict, list) are not.
        """
        # Should not raise TypeError - DeletePayload only has bool field
        hash_value = hash(sample_delete_payload)
        assert isinstance(hash_value, int)

        # Can be used in a set
        payload_set = {sample_delete_payload}
        assert len(payload_set) == 1

    def test_secret_payload_not_hashable_due_to_dict_fields(
        self, sample_secret_payload: ModelVaultSecretPayload
    ) -> None:
        """Secret payload is NOT hashable due to mutable dict fields.

        Payloads containing mutable fields (dict, list) cannot be hashed
        even if the model is frozen. This is expected Pydantic behavior.
        """
        with pytest.raises(TypeError) as exc_info:
            hash(sample_secret_payload)

        assert "unhashable" in str(exc_info.value)

    def test_health_check_payload_is_hashable(
        self, sample_health_check_payload: ModelVaultHealthCheckPayload
    ) -> None:
        """Frozen health check payload with immutable fields is hashable."""
        # Should not raise TypeError - HealthCheckPayload has only immutable fields
        hash_value = hash(sample_health_check_payload)
        assert isinstance(hash_value, int)

    def test_identical_payloads_have_same_hash(self) -> None:
        """Identical payloads should have the same hash value."""
        payload1 = ModelVaultDeletePayload(deleted=True)
        payload2 = ModelVaultDeletePayload(deleted=True)

        assert hash(payload1) == hash(payload2)
        assert payload1 == payload2

    def test_payloads_with_immutable_fields_can_be_dict_keys(
        self, sample_delete_payload: ModelVaultDeletePayload
    ) -> None:
        """Frozen payloads with immutable fields can be used as dict keys."""
        payload_dict = {sample_delete_payload: "value"}

        assert payload_dict[sample_delete_payload] == "value"

    def test_list_payload_not_hashable_due_to_list_field(
        self, sample_list_payload: ModelVaultListPayload
    ) -> None:
        """List payload is NOT hashable due to mutable list field.

        Payloads containing mutable fields (dict, list) cannot be hashed
        even if the model is frozen.
        """
        with pytest.raises(TypeError) as exc_info:
            hash(sample_list_payload)

        assert "unhashable" in str(exc_info.value)


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.unit
class TestVaultPayloadEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_secret_payload_with_empty_data(self) -> None:
        """Secret payload accepts empty data dict."""
        payload = ModelVaultSecretPayload(data={})

        assert payload.data == {}
        assert payload.metadata == {}

    def test_secret_payload_with_complex_nested_data(self) -> None:
        """Secret payload accepts complex nested data structures."""
        complex_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "admin",
                    "password": "secret",
                },
            },
            "api_keys": ["key1", "key2", "key3"],
            "enabled": True,
            "count": 42,
        }

        payload = ModelVaultSecretPayload(data=complex_data)

        assert payload.data["database"]["host"] == "localhost"
        assert payload.data["api_keys"][0] == "key1"
        assert payload.data["enabled"] is True
        assert payload.data["count"] == 42

    def test_write_payload_with_none_values(self) -> None:
        """Write payload accepts None for optional fields."""
        payload = ModelVaultWritePayload()

        assert payload.version is None
        assert payload.created_time is None
        assert payload.operation_type == "write_secret"

    def test_list_payload_with_empty_keys(self) -> None:
        """List payload accepts empty keys list."""
        payload = ModelVaultListPayload(keys=[])

        assert payload.keys == []

    def test_list_payload_with_default_keys(self) -> None:
        """List payload uses empty list as default for keys."""
        payload = ModelVaultListPayload()

        assert payload.keys == []

    def test_renew_token_payload_with_zero_lease_duration(self) -> None:
        """Renew token payload accepts zero lease duration."""
        payload = ModelVaultRenewTokenPayload(
            renewable=False,
            lease_duration=0,
        )

        assert payload.lease_duration == 0

    def test_renew_token_payload_lease_duration_must_be_non_negative(self) -> None:
        """Renew token payload rejects negative lease duration."""
        with pytest.raises(ValidationError):
            ModelVaultRenewTokenPayload(
                renewable=True,
                lease_duration=-1,
            )

    def test_health_check_payload_with_none_optional_fields(self) -> None:
        """Health check payload accepts None for optional fields."""
        payload = ModelVaultHealthCheckPayload(
            healthy=True,
            initialized=True,
            handler_type="vault",
            timeout_seconds=30.0,
            token_ttl_remaining_seconds=None,
            circuit_breaker_state=None,
        )

        assert payload.token_ttl_remaining_seconds is None
        assert payload.circuit_breaker_state is None

    def test_health_check_payload_defaults_for_count_fields(self) -> None:
        """Health check payload has proper defaults for count fields."""
        payload = ModelVaultHealthCheckPayload(
            healthy=True,
            initialized=True,
            handler_type="vault",
            timeout_seconds=30.0,
        )

        assert payload.circuit_breaker_failure_count == 0
        assert payload.thread_pool_active_workers == 0
        assert payload.thread_pool_max_workers == 0

    def test_health_check_payload_count_fields_must_be_non_negative(self) -> None:
        """Health check payload rejects negative count fields."""
        with pytest.raises(ValidationError):
            ModelVaultHealthCheckPayload(
                healthy=True,
                initialized=True,
                handler_type="vault",
                timeout_seconds=30.0,
                circuit_breaker_failure_count=-1,
            )

        with pytest.raises(ValidationError):
            ModelVaultHealthCheckPayload(
                healthy=True,
                initialized=True,
                handler_type="vault",
                timeout_seconds=30.0,
                thread_pool_active_workers=-1,
            )

    def test_enum_values_match_registry_keys(self) -> None:
        """Enum values match the registry operation type keys exactly."""
        assert EnumVaultOperationType.READ_SECRET.value == "read_secret"
        assert EnumVaultOperationType.WRITE_SECRET.value == "write_secret"
        assert EnumVaultOperationType.DELETE_SECRET.value == "delete_secret"
        assert EnumVaultOperationType.LIST_SECRETS.value == "list_secrets"
        assert EnumVaultOperationType.RENEW_TOKEN.value == "renew_token"
        assert EnumVaultOperationType.HEALTH_CHECK.value == "health_check"

        # All enum values should have corresponding registry entries
        for enum_member in EnumVaultOperationType:
            assert RegistryPayloadVault.is_registered(enum_member.value)

    def test_registry_keys_match_enum_values(self) -> None:
        """All registry keys correspond to valid enum values."""
        enum_values = {member.value for member in EnumVaultOperationType}
        registry_keys = set(RegistryPayloadVault.get_all_types().keys())

        assert registry_keys == enum_values
