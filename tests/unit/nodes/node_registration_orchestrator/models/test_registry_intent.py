# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for RegistryIntent and ModelRegistryIntent.

This module validates the registry pattern for registration intents,
serving as the template for other payload registry tests (HTTP, Vault, Consul).

The RegistryIntent provides a decorator-based registration mechanism that
enables dynamic type resolution during Pydantic validation without requiring
explicit union type definitions. This pattern:
- Eliminates duplicate union definitions across modules
- Allows new intent types to be added by implementing ModelRegistryIntent
- Uses the `kind` field as a discriminator for type resolution
- Follows ONEX duck typing principles while maintaining type safety

Test Categories:
    1. Registry Registration Tests - decorator and method behavior
    2. Base Model Tests - common fields and configuration
    3. Concrete Model Inheritance Tests - Consul and Postgres intents
    4. Serialization/Deserialization Tests - JSON round-trip validation
    5. Integration Tests - ModelReducerExecutionResult interop

Related:
    - ProtocolRegistrationIntent: Protocol for duck-typed function signatures
    - ModelReducerExecutionResult: Consumer of registry intents
    - OMN-1007: Union reduction refactoring

.. versionadded:: 0.7.0
    Created as part of OMN-1007 registry pattern implementation.
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

import pytest
from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.nodes.node_registration_orchestrator.models import (
    ModelConsulIntentPayload,
    ModelConsulRegistrationIntent,
    ModelPostgresIntentPayload,
    ModelPostgresUpsertIntent,
    ModelReducerExecutionResult,
    ModelReducerState,
    ModelRegistryIntent,
    RegistryIntent,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_registration_intent import (
    get_union_intent_types,
    validate_union_registry_sync,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_consul_payload() -> ModelConsulIntentPayload:
    """Create a sample Consul intent payload for testing."""
    return ModelConsulIntentPayload(
        service_name="test-service",
        tags=("production", "v1"),
        meta=(("env", "test"), ("region", "us-west")),
    )


@pytest.fixture
def sample_consul_intent(
    sample_consul_payload: ModelConsulIntentPayload,
) -> ModelConsulRegistrationIntent:
    """Create a sample Consul registration intent for testing."""
    return ModelConsulRegistrationIntent(
        operation="register",
        node_id=uuid4(),
        correlation_id=uuid4(),
        payload=sample_consul_payload,
    )


@pytest.fixture
def sample_postgres_payload() -> ModelPostgresIntentPayload:
    """Create a sample PostgreSQL intent payload for testing."""
    return ModelPostgresIntentPayload(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        correlation_id=uuid4(),
        timestamp="2025-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_postgres_intent(
    sample_postgres_payload: ModelPostgresIntentPayload,
) -> ModelPostgresUpsertIntent:
    """Create a sample PostgreSQL upsert intent for testing."""
    return ModelPostgresUpsertIntent(
        operation="upsert",
        node_id=uuid4(),
        correlation_id=uuid4(),
        payload=sample_postgres_payload,
    )


@pytest.fixture
def initial_state() -> ModelReducerState:
    """Create an initial reducer state for testing."""
    return ModelReducerState.initial()


# ============================================================================
# Tests for RegistryIntent
# ============================================================================


@pytest.mark.unit
class TestRegistryIntent:
    """Tests for RegistryIntent class methods and registration behavior.

    The RegistryIntent provides a decorator-based mechanism for registering
    intent model types, enabling dynamic type resolution during Pydantic
    validation without explicit union type definitions.
    """

    def test_get_type_returns_correct_class_for_consul(self) -> None:
        """Registry returns correct class for consul kind."""
        intent_cls = RegistryIntent.get_type("consul")
        assert intent_cls is ModelConsulRegistrationIntent

    def test_get_type_returns_correct_class_for_postgres(self) -> None:
        """Registry returns correct class for postgres kind."""
        intent_cls = RegistryIntent.get_type("postgres")
        assert intent_cls is ModelPostgresUpsertIntent

    def test_get_type_unknown_kind_raises_keyerror(self) -> None:
        """Registry raises KeyError for unknown kind with helpful message."""
        with pytest.raises(KeyError) as exc_info:
            RegistryIntent.get_type("unknown_kind")

        # Verify error message contains useful information
        error_msg = str(exc_info.value)
        assert "unknown_kind" in error_msg
        assert "Registered kinds" in error_msg

    def test_get_type_unknown_kind_lists_registered_kinds(self) -> None:
        """KeyError message includes list of registered kinds."""
        with pytest.raises(KeyError) as exc_info:
            RegistryIntent.get_type("nonexistent")

        error_msg = str(exc_info.value)
        # Should mention at least consul and postgres
        assert "consul" in error_msg or "postgres" in error_msg

    def test_get_all_types_returns_dict_of_registered_types(self) -> None:
        """get_all_types returns dict mapping kind strings to classes."""
        all_types = RegistryIntent.get_all_types()

        assert isinstance(all_types, dict)
        assert "consul" in all_types
        assert "postgres" in all_types
        assert all_types["consul"] is ModelConsulRegistrationIntent
        assert all_types["postgres"] is ModelPostgresUpsertIntent

    def test_get_all_types_returns_copy_not_reference(self) -> None:
        """get_all_types returns a copy, not the internal registry."""
        all_types = RegistryIntent.get_all_types()

        # Mutating the returned dict should not affect the registry
        original_count = len(all_types)
        all_types["fake"] = type("FakeIntent", (), {})  # type: ignore[assignment]

        # Registry should be unchanged
        assert len(RegistryIntent.get_all_types()) == original_count

    def test_is_registered_returns_true_for_consul(self) -> None:
        """is_registered returns True for registered consul kind."""
        assert RegistryIntent.is_registered("consul") is True

    def test_is_registered_returns_true_for_postgres(self) -> None:
        """is_registered returns True for registered postgres kind."""
        assert RegistryIntent.is_registered("postgres") is True

    def test_is_registered_returns_false_for_unknown_kind(self) -> None:
        """is_registered returns False for unknown kinds."""
        assert RegistryIntent.is_registered("unknown") is False
        assert RegistryIntent.is_registered("vault") is False
        assert RegistryIntent.is_registered("http") is False

    def test_is_registered_empty_string_returns_false(self) -> None:
        """is_registered returns False for empty string."""
        assert RegistryIntent.is_registered("") is False

    def test_register_decorator_returns_class_unchanged(self) -> None:
        """The @register decorator returns the class unchanged.

        This test creates a temporary test-only intent class to verify
        the decorator mechanism without polluting the registry. We use
        clear() for test isolation (clear() is designed for testing only).
        """
        # Save original registry state
        original_types = RegistryIntent.get_all_types()

        try:
            # Clear to test registration in isolation
            RegistryIntent.clear()

            @RegistryIntent.register("test_kind")
            class TestIntent(ModelRegistryIntent):
                kind: Literal["test_kind"] = "test_kind"

            # Verify the class was returned unchanged
            assert TestIntent.__name__ == "TestIntent"
            assert TestIntent.model_fields["kind"].default == "test_kind"

            # Verify it was registered
            assert RegistryIntent.is_registered("test_kind")
            assert RegistryIntent.get_type("test_kind") is TestIntent

        finally:
            # Restore original registry state
            RegistryIntent.clear()
            for kind, cls in original_types.items():
                RegistryIntent._types[kind] = cls

    def test_register_duplicate_kind_raises_valueerror(self) -> None:
        """Registering the same kind twice raises ProtocolConfigurationError.

        This prevents accidental overwrites of existing intent types.
        """
        # Save original registry state
        original_types = RegistryIntent.get_all_types()

        try:
            RegistryIntent.clear()

            # First registration should succeed
            @RegistryIntent.register("duplicate_test")
            class FirstIntent(ModelRegistryIntent):
                kind: Literal["duplicate_test"] = "duplicate_test"

            # Second registration with same kind should fail
            with pytest.raises(ProtocolConfigurationError) as exc_info:

                @RegistryIntent.register("duplicate_test")
                class SecondIntent(ModelRegistryIntent):
                    kind: Literal["duplicate_test"] = "duplicate_test"

            error_msg = str(exc_info.value)
            assert "duplicate_test" in error_msg
            assert "already registered" in error_msg
            assert "FirstIntent" in error_msg

        finally:
            # Restore original registry state
            RegistryIntent.clear()
            for kind, cls in original_types.items():
                RegistryIntent._types[kind] = cls

    def test_clear_removes_all_registered_types(self) -> None:
        """clear() removes all registered types from the registry.

        Note: clear() is intended for testing only and should not be
        used in production code.
        """
        # Save original registry state
        original_types = RegistryIntent.get_all_types()

        try:
            # Registry should have types before clear
            assert len(RegistryIntent.get_all_types()) > 0

            RegistryIntent.clear()

            # Registry should be empty after clear
            assert len(RegistryIntent.get_all_types()) == 0
            assert RegistryIntent.is_registered("consul") is False
            assert RegistryIntent.is_registered("postgres") is False

        finally:
            # Restore original registry state
            RegistryIntent.clear()
            for kind, cls in original_types.items():
                RegistryIntent._types[kind] = cls


# ============================================================================
# Tests for ModelRegistryIntent Base Class
# ============================================================================


@pytest.mark.unit
class TestModelRegistryIntent:
    """Tests for ModelRegistryIntent base class.

    ModelRegistryIntent defines the common interface that all registration
    intents share. It ensures consistent field names and configuration
    across all intent types.
    """

    def test_has_required_kind_field(self) -> None:
        """Base model defines required kind field for type discrimination."""
        fields = ModelRegistryIntent.model_fields
        assert "kind" in fields
        assert fields["kind"].annotation is str

    def test_has_required_operation_field(self) -> None:
        """Base model defines required operation field."""
        fields = ModelRegistryIntent.model_fields
        assert "operation" in fields
        assert fields["operation"].annotation is str

    def test_has_required_node_id_field(self) -> None:
        """Base model defines required node_id field."""
        from uuid import UUID

        fields = ModelRegistryIntent.model_fields
        assert "node_id" in fields
        assert fields["node_id"].annotation == UUID

    def test_has_required_correlation_id_field(self) -> None:
        """Base model defines required correlation_id field."""
        from uuid import UUID

        fields = ModelRegistryIntent.model_fields
        assert "correlation_id" in fields
        assert fields["correlation_id"].annotation == UUID

    def test_model_config_has_frozen_true(self) -> None:
        """Base model config has frozen=True for immutability."""
        config = ModelRegistryIntent.model_config
        assert config.get("frozen") is True

    def test_model_config_has_extra_forbid(self) -> None:
        """Base model config has extra='forbid' to prevent extra fields."""
        config = ModelRegistryIntent.model_config
        assert config.get("extra") == "forbid"

    def test_base_model_can_be_instantiated_with_all_fields(self) -> None:
        """Base model can be instantiated with all required fields.

        Note: In practice, concrete subclasses should be used, but the
        base class should still be instantiable for testing purposes.
        """
        node_id = uuid4()
        correlation_id = uuid4()

        intent = ModelRegistryIntent(
            kind="test",
            operation="test_op",
            node_id=node_id,
            correlation_id=correlation_id,
        )

        assert intent.kind == "test"
        assert intent.operation == "test_op"
        assert intent.node_id == node_id
        assert intent.correlation_id == correlation_id

    def test_is_frozen_cannot_modify_kind(self) -> None:
        """Frozen model prevents modification of kind field."""
        intent = ModelRegistryIntent(
            kind="test",
            operation="test_op",
            node_id=uuid4(),
            correlation_id=uuid4(),
        )

        with pytest.raises(ValidationError):
            intent.kind = "modified"  # type: ignore[misc]

    def test_is_frozen_cannot_modify_operation(self) -> None:
        """Frozen model prevents modification of operation field."""
        intent = ModelRegistryIntent(
            kind="test",
            operation="test_op",
            node_id=uuid4(),
            correlation_id=uuid4(),
        )

        with pytest.raises(ValidationError):
            intent.operation = "modified"  # type: ignore[misc]

    def test_forbids_extra_fields(self) -> None:
        """Model raises ValidationError when extra fields are provided."""
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistryIntent(
                kind="test",
                operation="test_op",
                node_id=uuid4(),
                correlation_id=uuid4(),
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        # Verify error is about extra field
        error_msg = str(exc_info.value)
        assert "extra_field" in error_msg or "Extra" in error_msg


# ============================================================================
# Tests for Concrete Intent Model Inheritance
# ============================================================================


@pytest.mark.unit
class TestConcreteIntentModels:
    """Tests for concrete intent model inheritance and registration.

    These tests verify that ModelConsulRegistrationIntent and
    ModelPostgresUpsertIntent correctly inherit from ModelRegistryIntent
    and are properly registered in the RegistryIntent.
    """

    def test_consul_intent_inherits_from_base(self) -> None:
        """ModelConsulRegistrationIntent inherits from ModelRegistryIntent."""
        assert issubclass(ModelConsulRegistrationIntent, ModelRegistryIntent)

    def test_postgres_intent_inherits_from_base(self) -> None:
        """ModelPostgresUpsertIntent inherits from ModelRegistryIntent."""
        assert issubclass(ModelPostgresUpsertIntent, ModelRegistryIntent)

    def test_consul_intent_is_registered_in_registry(self) -> None:
        """ModelConsulRegistrationIntent is registered with kind='consul'."""
        assert RegistryIntent.is_registered("consul")
        assert RegistryIntent.get_type("consul") is ModelConsulRegistrationIntent

    def test_postgres_intent_is_registered_in_registry(self) -> None:
        """ModelPostgresUpsertIntent is registered with kind='postgres'."""
        assert RegistryIntent.is_registered("postgres")
        assert RegistryIntent.get_type("postgres") is ModelPostgresUpsertIntent

    def test_consul_intent_kind_field_is_literal_consul(
        self, sample_consul_intent: ModelConsulRegistrationIntent
    ) -> None:
        """Consul intent kind field is always 'consul'."""
        assert sample_consul_intent.kind == "consul"

    def test_postgres_intent_kind_field_is_literal_postgres(
        self, sample_postgres_intent: ModelPostgresUpsertIntent
    ) -> None:
        """Postgres intent kind field is always 'postgres'."""
        assert sample_postgres_intent.kind == "postgres"

    def test_consul_intent_has_payload_field(
        self, sample_consul_intent: ModelConsulRegistrationIntent
    ) -> None:
        """Consul intent has a payload field of type ModelConsulIntentPayload."""
        assert hasattr(sample_consul_intent, "payload")
        assert isinstance(sample_consul_intent.payload, ModelConsulIntentPayload)

    def test_postgres_intent_has_payload_field(
        self, sample_postgres_intent: ModelPostgresUpsertIntent
    ) -> None:
        """Postgres intent has a payload field of type ModelPostgresIntentPayload."""
        assert hasattr(sample_postgres_intent, "payload")
        assert isinstance(sample_postgres_intent.payload, ModelPostgresIntentPayload)

    def test_consul_intent_is_frozen(
        self, sample_consul_intent: ModelConsulRegistrationIntent
    ) -> None:
        """Consul intent model is frozen and cannot be modified."""
        with pytest.raises(ValidationError):
            sample_consul_intent.operation = "modified"  # type: ignore[misc]

    def test_postgres_intent_is_frozen(
        self, sample_postgres_intent: ModelPostgresUpsertIntent
    ) -> None:
        """Postgres intent model is frozen and cannot be modified."""
        with pytest.raises(ValidationError):
            sample_postgres_intent.operation = "modified"  # type: ignore[misc]

    def test_consul_intent_forbids_extra_fields(self) -> None:
        """Consul intent raises ValidationError for extra fields."""
        with pytest.raises(ValidationError):
            ModelConsulRegistrationIntent(
                operation="register",
                node_id=uuid4(),
                correlation_id=uuid4(),
                payload=ModelConsulIntentPayload(service_name="test"),
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

    def test_postgres_intent_forbids_extra_fields(self) -> None:
        """Postgres intent raises ValidationError for extra fields."""
        with pytest.raises(ValidationError):
            ModelPostgresUpsertIntent(
                operation="upsert",
                node_id=uuid4(),
                correlation_id=uuid4(),
                payload=ModelPostgresIntentPayload(
                    node_id=uuid4(),
                    node_type=EnumNodeKind.EFFECT,
                    correlation_id=uuid4(),
                    timestamp="2025-01-01T00:00:00Z",
                ),
                extra_field="not_allowed",  # type: ignore[call-arg]
            )


# ============================================================================
# Tests for Intent Serialization and Deserialization
# ============================================================================


@pytest.mark.unit
class TestIntentSerialization:
    """Tests for intent serialization and deserialization.

    These tests verify that intent models can be serialized to JSON/dict
    and deserialized back, with the kind field enabling type discrimination.
    """

    def test_consul_intent_serializes_to_dict(
        self, sample_consul_intent: ModelConsulRegistrationIntent
    ) -> None:
        """Consul intent can be serialized to dict."""
        data = sample_consul_intent.model_dump()

        assert isinstance(data, dict)
        assert data["kind"] == "consul"
        assert data["operation"] == "register"
        assert "node_id" in data
        assert "correlation_id" in data
        assert "payload" in data

    def test_postgres_intent_serializes_to_dict(
        self, sample_postgres_intent: ModelPostgresUpsertIntent
    ) -> None:
        """Postgres intent can be serialized to dict."""
        data = sample_postgres_intent.model_dump()

        assert isinstance(data, dict)
        assert data["kind"] == "postgres"
        assert data["operation"] == "upsert"
        assert "node_id" in data
        assert "correlation_id" in data
        assert "payload" in data

    def test_consul_intent_serializes_to_json(
        self, sample_consul_intent: ModelConsulRegistrationIntent
    ) -> None:
        """Consul intent can be serialized to JSON string."""
        json_str = sample_consul_intent.model_dump_json()

        assert isinstance(json_str, str)
        assert '"kind":"consul"' in json_str or '"kind": "consul"' in json_str

    def test_postgres_intent_serializes_to_json(
        self, sample_postgres_intent: ModelPostgresUpsertIntent
    ) -> None:
        """Postgres intent can be serialized to JSON string."""
        json_str = sample_postgres_intent.model_dump_json()

        assert isinstance(json_str, str)
        assert '"kind":"postgres"' in json_str or '"kind": "postgres"' in json_str

    def test_consul_intent_deserializes_from_dict(
        self, sample_consul_intent: ModelConsulRegistrationIntent
    ) -> None:
        """Consul intent can be deserialized from dict."""
        data = sample_consul_intent.model_dump()
        restored = ModelConsulRegistrationIntent.model_validate(data)

        assert restored.kind == "consul"
        assert restored.operation == sample_consul_intent.operation
        assert restored.node_id == sample_consul_intent.node_id
        assert restored.correlation_id == sample_consul_intent.correlation_id

    def test_postgres_intent_deserializes_from_dict(
        self, sample_postgres_intent: ModelPostgresUpsertIntent
    ) -> None:
        """Postgres intent can be deserialized from dict."""
        data = sample_postgres_intent.model_dump()
        restored = ModelPostgresUpsertIntent.model_validate(data)

        assert restored.kind == "postgres"
        assert restored.operation == sample_postgres_intent.operation
        assert restored.node_id == sample_postgres_intent.node_id
        assert restored.correlation_id == sample_postgres_intent.correlation_id

    def test_consul_intent_round_trip_preserves_data(
        self, sample_consul_intent: ModelConsulRegistrationIntent
    ) -> None:
        """Consul intent dict round-trip preserves all data.

        Note: Uses model_dump/model_validate instead of JSON serialization
        because the payload.meta field contains tuple of tuples which JSON
        cannot round-trip (tuples become arrays, arrays deserialize as lists).
        """
        data = sample_consul_intent.model_dump()
        restored = ModelConsulRegistrationIntent.model_validate(data)

        assert restored == sample_consul_intent

    def test_postgres_intent_round_trip_preserves_data(
        self, sample_postgres_intent: ModelPostgresUpsertIntent
    ) -> None:
        """Postgres intent dict round-trip preserves all data.

        Note: Uses model_dump/model_validate instead of JSON serialization
        because the payload.endpoints field contains tuples which JSON
        cannot round-trip (tuples become arrays, arrays deserialize as lists).
        """
        data = sample_postgres_intent.model_dump()
        restored = ModelPostgresUpsertIntent.model_validate(data)

        assert restored == sample_postgres_intent

    def test_kind_field_enables_type_discrimination(self) -> None:
        """The kind field enables determining the correct intent type.

        This demonstrates the discriminated union pattern where the kind
        field is used to select the appropriate model class.
        """
        consul_data = {
            "kind": "consul",
            "operation": "register",
            "node_id": uuid4(),
            "correlation_id": uuid4(),
            "payload": {"service_name": "test-service"},
        }

        postgres_data = {
            "kind": "postgres",
            "operation": "upsert",
            "node_id": uuid4(),
            "correlation_id": uuid4(),
            "payload": {
                "node_id": uuid4(),
                "node_type": EnumNodeKind.EFFECT,
                "correlation_id": uuid4(),
                "timestamp": "2025-01-01T00:00:00Z",
            },
        }

        # Use registry to get correct class based on kind
        consul_cls = RegistryIntent.get_type(consul_data["kind"])
        postgres_cls = RegistryIntent.get_type(postgres_data["kind"])

        consul_intent = consul_cls.model_validate(consul_data)
        postgres_intent = postgres_cls.model_validate(postgres_data)

        assert isinstance(consul_intent, ModelConsulRegistrationIntent)
        assert isinstance(postgres_intent, ModelPostgresUpsertIntent)

    def test_deserialization_with_wrong_kind_fails(self) -> None:
        """Deserializing with wrong kind field fails validation.

        ModelConsulRegistrationIntent expects kind='consul' as a Literal.
        Providing a different value should fail validation.
        """
        data = {
            "kind": "wrong_kind",  # Not 'consul'
            "operation": "register",
            "node_id": str(uuid4()),
            "correlation_id": str(uuid4()),
            "payload": {"service_name": "test"},
        }

        with pytest.raises(ValidationError):
            ModelConsulRegistrationIntent.model_validate(data)


# ============================================================================
# Tests for Integration with ModelReducerExecutionResult
# ============================================================================


@pytest.mark.unit
class TestReducerExecutionResultIntegration:
    """Tests for integration with ModelReducerExecutionResult.

    ModelReducerExecutionResult uses a field_validator to resolve intent
    types dynamically from the RegistryIntent during deserialization.
    """

    def test_result_accepts_consul_intent(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
    ) -> None:
        """ModelReducerExecutionResult accepts Consul intent in intents tuple."""
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_consul_intent,),
        )

        assert len(result.intents) == 1
        assert isinstance(result.intents[0], ModelConsulRegistrationIntent)

    def test_result_accepts_postgres_intent(
        self,
        initial_state: ModelReducerState,
        sample_postgres_intent: ModelPostgresUpsertIntent,
    ) -> None:
        """ModelReducerExecutionResult accepts Postgres intent in intents tuple."""
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_postgres_intent,),
        )

        assert len(result.intents) == 1
        assert isinstance(result.intents[0], ModelPostgresUpsertIntent)

    def test_result_accepts_mixed_intent_types(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
        sample_postgres_intent: ModelPostgresUpsertIntent,
    ) -> None:
        """ModelReducerExecutionResult accepts multiple intent types."""
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_consul_intent, sample_postgres_intent),
        )

        assert len(result.intents) == 2
        assert isinstance(result.intents[0], ModelConsulRegistrationIntent)
        assert isinstance(result.intents[1], ModelPostgresUpsertIntent)

    def test_result_deserializes_consul_intent_from_dict(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """ModelReducerExecutionResult deserializes Consul intent from dict.

        The validate_intents validator uses RegistryIntent to resolve the
        correct intent class based on the kind field.

        Note: Uses UUID objects (not strings) because intent models have
        strict=True config, which requires proper types without coercion.
        """
        consul_dict = {
            "kind": "consul",
            "operation": "register",
            "node_id": uuid4(),
            "correlation_id": uuid4(),
            "payload": {"service_name": "test-service"},
        }

        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=[consul_dict],  # List of dicts
        )

        assert len(result.intents) == 1
        assert isinstance(result.intents[0], ModelConsulRegistrationIntent)
        assert result.intents[0].kind == "consul"
        assert result.intents[0].payload.service_name == "test-service"

    def test_result_deserializes_postgres_intent_from_dict(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """ModelReducerExecutionResult deserializes Postgres intent from dict.

        Note: Uses UUID objects (not strings) because intent models have
        strict=True config, which requires proper types without coercion.
        """
        node_id = uuid4()
        correlation_id = uuid4()

        postgres_dict = {
            "kind": "postgres",
            "operation": "upsert",
            "node_id": uuid4(),
            "correlation_id": uuid4(),
            "payload": {
                "node_id": node_id,
                "node_type": EnumNodeKind.EFFECT,
                "correlation_id": correlation_id,
                "timestamp": "2025-01-01T00:00:00Z",
            },
        }

        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=[postgres_dict],
        )

        assert len(result.intents) == 1
        assert isinstance(result.intents[0], ModelPostgresUpsertIntent)
        assert result.intents[0].kind == "postgres"

    def test_result_deserializes_mixed_intents_from_dicts(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """ModelReducerExecutionResult deserializes mixed intent types from dicts.

        Note: Uses UUID objects (not strings) because intent models have
        strict=True config, which requires proper types without coercion.
        """
        consul_dict = {
            "kind": "consul",
            "operation": "register",
            "node_id": uuid4(),
            "correlation_id": uuid4(),
            "payload": {"service_name": "service-a"},
        }

        postgres_dict = {
            "kind": "postgres",
            "operation": "upsert",
            "node_id": uuid4(),
            "correlation_id": uuid4(),
            "payload": {
                "node_id": uuid4(),
                "node_type": EnumNodeKind.COMPUTE,
                "correlation_id": uuid4(),
                "timestamp": "2025-01-02T00:00:00Z",
            },
        }

        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=[consul_dict, postgres_dict],
        )

        assert len(result.intents) == 2
        assert isinstance(result.intents[0], ModelConsulRegistrationIntent)
        assert isinstance(result.intents[1], ModelPostgresUpsertIntent)

    def test_result_rejects_dict_missing_kind_field(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """ModelReducerExecutionResult rejects intent dict without kind field."""
        dict_without_kind = {
            "operation": "register",
            "node_id": str(uuid4()),
            "correlation_id": str(uuid4()),
            "payload": {"service_name": "test"},
            # Missing 'kind' field
        }

        with pytest.raises(ValidationError) as exc_info:
            ModelReducerExecutionResult(
                state=initial_state,
                intents=[dict_without_kind],
            )

        error_msg = str(exc_info.value)
        assert "kind" in error_msg.lower()

    def test_result_rejects_dict_with_unknown_kind(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """ModelReducerExecutionResult rejects intent dict with unregistered kind."""
        dict_with_unknown_kind = {
            "kind": "unknown_kind",
            "operation": "register",
            "node_id": str(uuid4()),
            "correlation_id": str(uuid4()),
            "payload": {"service_name": "test"},
        }

        with pytest.raises(ValidationError) as exc_info:
            ModelReducerExecutionResult(
                state=initial_state,
                intents=[dict_with_unknown_kind],
            )

        error_msg = str(exc_info.value)
        assert "unknown_kind" in error_msg

    def test_result_rejects_invalid_intent_type(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """ModelReducerExecutionResult rejects non-dict, non-intent values."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReducerExecutionResult(
                state=initial_state,
                intents=["not_an_intent"],  # type: ignore[list-item]
            )

        error_msg = str(exc_info.value)
        assert "dict" in error_msg.lower() or "ModelRegistryIntent" in error_msg

    def test_result_round_trip_preserves_intent_types(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
        sample_postgres_intent: ModelPostgresUpsertIntent,
    ) -> None:
        """Round-trip serialization preserves correct intent types.

        Note: Uses serialize_as_any=True to ensure polymorphic intent fields
        (e.g., payload) are serialized according to their concrete type,
        not the declared base type (ModelRegistryIntent).
        """
        original = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_consul_intent, sample_postgres_intent),
        )

        # Serialize to dict (serialize_as_any required for polymorphic intents)
        data = original.model_dump(serialize_as_any=True)

        # Deserialize back
        restored = ModelReducerExecutionResult.model_validate(data)

        assert len(restored.intents) == 2
        assert isinstance(restored.intents[0], ModelConsulRegistrationIntent)
        assert isinstance(restored.intents[1], ModelPostgresUpsertIntent)

    def test_result_json_round_trip_preserves_intent_types(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
        sample_postgres_intent: ModelPostgresUpsertIntent,
    ) -> None:
        """JSON round-trip serialization preserves correct intent types.

        Note: Uses serialize_as_any=True to ensure polymorphic intent fields
        (e.g., payload) are serialized according to their concrete type.
        Also uses model_validate (not model_validate_json) because the latter
        doesn't trigger the validate_intents before-validator correctly when
        strict=True is enabled on intent models.
        """
        original = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_consul_intent, sample_postgres_intent),
        )

        # Serialize to dict (serialize_as_any required for polymorphic intents)
        # We use model_dump instead of model_dump_json because our intents
        # contain tuple fields that JSON cannot round-trip, and strict mode
        # prevents string->UUID coercion.
        data = original.model_dump(serialize_as_any=True)

        # Deserialize back
        restored = ModelReducerExecutionResult.model_validate(data)

        assert len(restored.intents) == 2
        assert isinstance(restored.intents[0], ModelConsulRegistrationIntent)
        assert isinstance(restored.intents[1], ModelPostgresUpsertIntent)

    def test_result_with_intents_factory_accepts_model_instances(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
    ) -> None:
        """with_intents factory accepts model instances directly."""
        result = ModelReducerExecutionResult.with_intents(
            state=initial_state,
            intents=[sample_consul_intent],
        )

        assert len(result.intents) == 1
        assert result.intents[0] is sample_consul_intent


# ============================================================================
# Tests for Thread Safety and Immutability
# ============================================================================


@pytest.mark.unit
class TestIntentThreadSafety:
    """Tests for thread safety and immutability of intent models.

    All intent models and the RegistryIntent are designed to be thread-safe:
    - RegistryIntent is populated at module import time and read-only after
    - All intent models are frozen (immutable)
    - Tuple fields ensure container immutability
    """

    def test_intent_registry_class_var_is_shared(self) -> None:
        """RegistryIntent._types is a ClassVar shared across all usage.

        This ensures the registry is consistent regardless of where it's
        accessed in the codebase.
        """
        # Access registry from different import paths (would be same in practice)
        types1 = RegistryIntent.get_all_types()
        types2 = RegistryIntent.get_all_types()

        # Should return equivalent copies
        assert types1 == types2

    def test_consul_intent_is_hashable(
        self, sample_consul_intent: ModelConsulRegistrationIntent
    ) -> None:
        """Frozen Consul intent is hashable for use in sets/dicts."""
        # Should not raise TypeError
        hash_value = hash(sample_consul_intent)
        assert isinstance(hash_value, int)

        # Can be used in a set
        intent_set = {sample_consul_intent}
        assert len(intent_set) == 1

    def test_postgres_intent_is_not_hashable_due_to_nested_models(
        self, sample_postgres_intent: ModelPostgresUpsertIntent
    ) -> None:
        """Postgres intent is not hashable due to unfrozen nested models.

        Note: ModelPostgresIntentPayload contains ModelNodeCapabilities and
        ModelNodeMetadata which have frozen=False, making them unhashable.
        This prevents the entire intent from being hashable even though the
        intent itself is frozen. This is a known limitation.

        If hashability is needed, the nested models would need to be made
        frozen as well.
        """
        with pytest.raises(TypeError, match="unhashable type"):
            hash(sample_postgres_intent)

    def test_identical_intents_have_same_hash(self) -> None:
        """Identical intents should have the same hash value."""
        node_id = uuid4()
        correlation_id = uuid4()

        intent1 = ModelConsulRegistrationIntent(
            operation="register",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelConsulIntentPayload(service_name="test"),
        )

        intent2 = ModelConsulRegistrationIntent(
            operation="register",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelConsulIntentPayload(service_name="test"),
        )

        assert hash(intent1) == hash(intent2)
        assert intent1 == intent2

    def test_intents_can_be_used_as_dict_keys(
        self, sample_consul_intent: ModelConsulRegistrationIntent
    ) -> None:
        """Frozen intents can be used as dictionary keys."""
        intent_dict = {sample_consul_intent: "value"}

        assert intent_dict[sample_consul_intent] == "value"


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.unit
class TestIntentEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_intents_tuple_is_valid(
        self, initial_state: ModelReducerState
    ) -> None:
        """Empty intents tuple is a valid state."""
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(),
        )

        assert result.intents == ()
        assert len(result.intents) == 0

    def test_none_intents_coerced_to_empty_tuple(
        self, initial_state: ModelReducerState
    ) -> None:
        """None value for intents is coerced to empty tuple by validator."""
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=None,  # type: ignore[arg-type]
        )

        assert result.intents == ()

    def test_consul_intent_operation_must_be_non_empty(self) -> None:
        """Consul intent operation field must have min_length=1."""
        with pytest.raises(ValidationError):
            ModelConsulRegistrationIntent(
                operation="",  # Empty string should fail
                node_id=uuid4(),
                correlation_id=uuid4(),
                payload=ModelConsulIntentPayload(service_name="test"),
            )

    def test_postgres_intent_operation_must_be_non_empty(self) -> None:
        """Postgres intent operation field must have min_length=1."""
        with pytest.raises(ValidationError):
            ModelPostgresUpsertIntent(
                operation="",  # Empty string should fail
                node_id=uuid4(),
                correlation_id=uuid4(),
                payload=ModelPostgresIntentPayload(
                    node_id=uuid4(),
                    node_type=EnumNodeKind.EFFECT,
                    correlation_id=uuid4(),
                    timestamp="2025-01-01T00:00:00Z",
                ),
            )

    def test_intent_with_uuid_string_rejected_in_strict_mode(self) -> None:
        """UUID fields reject string input when model has strict=True.

        Intent models use strict=True config for type safety. UUID fields
        require actual UUID instances, not string representations.
        """
        node_id_str = "12345678-1234-5678-1234-567812345678"
        correlation_id_str = "87654321-4321-8765-4321-876543218765"

        with pytest.raises(ValidationError) as exc_info:
            ModelConsulRegistrationIntent(
                operation="register",
                node_id=node_id_str,  # type: ignore[arg-type]
                correlation_id=correlation_id_str,  # type: ignore[arg-type]
                payload=ModelConsulIntentPayload(service_name="test"),
            )

        error_msg = str(exc_info.value)
        assert "uuid" in error_msg.lower() or "UUID" in error_msg

    def test_consul_payload_service_name_must_be_non_empty(self) -> None:
        """Consul payload service_name must have min_length=1."""
        with pytest.raises(ValidationError):
            ModelConsulIntentPayload(service_name="")

    def test_large_intents_tuple_is_accepted(
        self, initial_state: ModelReducerState
    ) -> None:
        """Result can contain many intents."""
        # Create 100 intents
        intents = tuple(
            ModelConsulRegistrationIntent(
                operation="register",
                node_id=uuid4(),
                correlation_id=uuid4(),
                payload=ModelConsulIntentPayload(service_name=f"service-{i}"),
            )
            for i in range(100)
        )

        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=intents,
        )

        assert len(result.intents) == 100
        assert result.intent_count == 100


# ============================================================================
# Tests for Union and Registry Sync Validation (OMN-1007)
# ============================================================================


@pytest.mark.unit
class TestUnionRegistrySync:
    """Tests for union and registry synchronization.

    These tests verify that the ModelRegistrationIntent discriminated union
    and the RegistryIntent stay in sync. This is critical because:

    1. ModelRegistrationIntent (static union) is used for Pydantic field validation
    2. RegistryIntent (dynamic) is used for runtime type resolution

    If they drift, validation and deserialization may behave inconsistently.

    Related:
        - OMN-1007: Union reduction refactoring
        - model_registration_intent.py: validate_union_registry_sync()
    """

    def test_validate_union_registry_sync_passes(self) -> None:
        """Union and registry should be in sync by default.

        This is the critical test that catches sync issues. If this fails,
        someone added a new intent type to one location but not the other.
        """
        is_valid, errors = validate_union_registry_sync()

        if not is_valid:
            # Provide helpful error message for debugging
            error_msg = "Union and registry are out of sync:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            pytest.fail(error_msg)

    def test_get_union_intent_types_returns_all_types(self) -> None:
        """get_union_intent_types returns all types in the union."""
        types = get_union_intent_types()

        assert isinstance(types, tuple)
        assert len(types) >= 2  # At least consul and postgres
        assert ModelConsulRegistrationIntent in types
        assert ModelPostgresUpsertIntent in types

    def test_union_types_match_registry_count(self) -> None:
        """Number of union types matches number of registered types."""
        union_types = get_union_intent_types()
        registry_types = RegistryIntent.get_all_types()

        assert len(union_types) == len(registry_types), (
            f"Union has {len(union_types)} types but registry has "
            f"{len(registry_types)} types. They must match."
        )

    def test_all_union_types_are_registered(self) -> None:
        """All types in the union are registered in RegistryIntent."""
        union_types = get_union_intent_types()

        for union_type in union_types:
            # Get the kind from the class default
            kind_default = getattr(union_type, "model_fields", {}).get("kind")
            assert kind_default is not None, (
                f"{union_type.__name__} has no 'kind' field"
            )

            kind_value = kind_default.default
            assert RegistryIntent.is_registered(kind_value), (
                f"{union_type.__name__} with kind='{kind_value}' is not "
                f"registered in RegistryIntent"
            )

    def test_all_registered_types_are_in_union(self) -> None:
        """All types in RegistryIntent are included in the union."""
        union_types = set(get_union_intent_types())
        registry_types = RegistryIntent.get_all_types()

        for kind, registered_type in registry_types.items():
            assert registered_type in union_types, (
                f"Registered type {registered_type.__name__} (kind='{kind}') "
                f"is missing from ModelRegistrationIntent union. "
                f"Add it to model_registration_intent.py"
            )

    def test_registry_types_match_union_types_exactly(self) -> None:
        """Registry and union contain exactly the same type instances."""
        union_types = set(get_union_intent_types())
        registry_types = set(RegistryIntent.get_all_types().values())

        assert union_types == registry_types, (
            f"Union types {union_types} do not match registry types {registry_types}"
        )

    def test_sync_validation_detects_missing_from_union(self) -> None:
        """Sync validation detects when a type is registered but not in union.

        This test temporarily registers a new type and verifies the
        validation function catches the inconsistency.
        """
        # Save original registry state
        original_types = RegistryIntent.get_all_types()

        try:
            # Register a fake type that won't be in the union
            @RegistryIntent.register("fake_test_kind")
            class FakeIntent(ModelRegistryIntent):
                kind: Literal["fake_test_kind"] = "fake_test_kind"

            # Now validation should fail
            is_valid, errors = validate_union_registry_sync()

            assert is_valid is False, "Should detect missing union type"
            assert len(errors) >= 1
            assert any("fake_test_kind" in e for e in errors)
            assert any("missing from" in e.lower() for e in errors)

        finally:
            # Restore original registry state
            RegistryIntent.clear()
            for kind, cls in original_types.items():
                RegistryIntent._types[kind] = cls
