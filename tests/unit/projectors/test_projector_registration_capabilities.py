# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for capability projection models and validation.

This test suite validates the capability extension fields added in OMN-1134:
- contract_type: Node contract type (effect, compute, reducer, orchestrator)
- intent_types: Array of intent types the node handles
- protocols: Array of protocols the node implements
- capability_tags: Array of capability tags for discovery
- contract_version: Contract version string

Test Organization:
    - TestCapabilityFieldsModel: Capability fields on ModelRegistrationProjection
    - TestModelCapabilityFields: ModelCapabilityFields dataclass validation
    - TestCapabilityFieldsExtraction: Extraction from ModelNodeCapabilities
    - TestCapabilityFieldsValidation: contract_type validation rules
    - TestUnknownContractTypeValidation: 'unknown' contract type handling

Related Tickets:
    - OMN-1134: Registry Projection Extensions for Capabilities
    - OMN-1124: ModelContractCapabilities Extension (dependency)
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import ModelCapabilityFields
from omnibase_infra.models.projection.model_registration_projection import (
    VALID_CONTRACT_TYPES,
    ModelRegistrationProjection,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)


def create_test_projection_with_capabilities(
    contract_type: str | None = "effect",
    intent_types: list[str] | None = None,
    protocols: list[str] | None = None,
    capability_tags: list[str] | None = None,
    contract_version: str | None = "1.0.0",
) -> ModelRegistrationProjection:
    """Create a test projection with capability fields."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        domain="registration",
        current_state=EnumRegistrationState.ACTIVE,
        node_type=EnumNodeKind.EFFECT,
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        # New capability fields
        contract_type=contract_type,
        intent_types=intent_types or [],
        protocols=protocols or [],
        capability_tags=capability_tags or [],
        contract_version=contract_version,
        # Standard fields
        ack_deadline=now,
        liveness_deadline=now,
        last_applied_event_id=uuid4(),
        last_applied_offset=100,
        registered_at=now,
        updated_at=now,
    )


@pytest.mark.unit
class TestCapabilityFieldsModel:
    """Test that ModelRegistrationProjection accepts capability fields."""

    def test_projection_accepts_capability_fields(self) -> None:
        """Test that projection model accepts new capability fields."""
        projection = create_test_projection_with_capabilities(
            contract_type="effect",
            intent_types=["postgres.upsert", "postgres.query"],
            protocols=["ProtocolDatabaseAdapter", "ProtocolEventPublisher"],
            capability_tags=["postgres.storage", "kafka.consumer"],
            contract_version="2.1.0",
        )

        assert projection.contract_type == "effect"
        assert projection.intent_types == ["postgres.upsert", "postgres.query"]
        assert projection.protocols == [
            "ProtocolDatabaseAdapter",
            "ProtocolEventPublisher",
        ]
        assert projection.capability_tags == ["postgres.storage", "kafka.consumer"]
        assert projection.contract_version == "2.1.0"

    def test_projection_capability_fields_default_to_empty(self) -> None:
        """Test that capability fields have sensible defaults."""
        now = datetime.now(UTC)
        projection = ModelRegistrationProjection(
            entity_id=uuid4(),
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            node_type=EnumNodeKind.EFFECT,
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )

        # Fields should default to None/empty
        assert projection.contract_type is None
        assert projection.intent_types == []
        assert projection.protocols == []
        assert projection.capability_tags == []
        assert projection.contract_version is None

    def test_projection_serializes_capability_arrays(self) -> None:
        """Test that capability arrays serialize correctly to JSON."""
        projection = create_test_projection_with_capabilities(
            intent_types=["intent.a", "intent.b"],
            protocols=["Protocol1"],
            capability_tags=["tag1", "tag2", "tag3"],
        )

        # Serialize to JSON-compatible dict
        data = projection.model_dump(mode="json")

        assert data["intent_types"] == ["intent.a", "intent.b"]
        assert data["protocols"] == ["Protocol1"]
        assert data["capability_tags"] == ["tag1", "tag2", "tag3"]


@pytest.mark.unit
class TestModelCapabilityFields:
    """Test the ModelCapabilityFields dataclass."""

    def test_capability_fields_defaults_to_none(self) -> None:
        """Test that all fields default to None."""
        fields = ModelCapabilityFields()
        assert fields.contract_type is None
        assert fields.intent_types is None
        assert fields.protocols is None
        assert fields.capability_tags is None
        assert fields.contract_version is None

    def test_capability_fields_accepts_all_values(self) -> None:
        """Test that all fields can be set."""
        fields = ModelCapabilityFields(
            contract_type="effect",
            intent_types=["intent.a", "intent.b"],
            protocols=["Protocol1", "Protocol2"],
            capability_tags=["tag1", "tag2"],
            contract_version="1.0.0",
        )
        assert fields.contract_type == "effect"
        assert fields.intent_types == ["intent.a", "intent.b"]
        assert fields.protocols == ["Protocol1", "Protocol2"]
        assert fields.capability_tags == ["tag1", "tag2"]
        assert fields.contract_version == "1.0.0"

    def test_capability_fields_is_frozen(self) -> None:
        """Test that ModelCapabilityFields is immutable."""
        fields = ModelCapabilityFields(contract_type="effect")
        with pytest.raises(ValidationError):
            fields.contract_type = "compute"  # type: ignore[misc]

    def test_capability_fields_serializes_correctly(self) -> None:
        """Test that capability fields serialize to dict."""
        fields = ModelCapabilityFields(
            contract_type="reducer",
            intent_types=["state.update"],
        )
        data = fields.model_dump()
        assert data["contract_type"] == "reducer"
        assert data["intent_types"] == ["state.update"]
        assert data["protocols"] is None
        assert data["capability_tags"] is None
        assert data["contract_version"] is None


@pytest.mark.unit
class TestCapabilityFieldsExtraction:
    """Test extraction of capability fields from ModelNodeCapabilities."""

    def test_extract_capability_fields_from_extra(self) -> None:
        """Test extracting capability fields from model_extra."""
        # ModelNodeCapabilities uses extra="allow" so we can add custom fields
        capabilities = ModelNodeCapabilities(
            postgres=True,
            read=True,
            # These would be in model_extra
            contract_type="effect",  # type: ignore[call-arg]
            intent_types=["postgres.query"],  # type: ignore[call-arg]
        )

        # Access via model_extra
        extra = capabilities.model_extra or {}
        assert extra.get("contract_type") == "effect"
        assert extra.get("intent_types") == ["postgres.query"]

    def test_get_capability_field_with_fallback(self) -> None:
        """Test getting capability field with fallback to None."""
        capabilities = ModelNodeCapabilities(postgres=True)

        # Using get() method for safe access
        contract_type = capabilities.get("contract_type")
        intent_types = capabilities.get("intent_types", [])

        assert contract_type is None
        assert intent_types == []


@pytest.mark.unit
class TestCapabilityFieldsValidation:
    """Test validation of capability fields."""

    def test_contract_type_accepts_valid_node_kinds(self) -> None:
        """Test that contract_type accepts valid node kind values."""
        for node_type in VALID_CONTRACT_TYPES:
            projection = create_test_projection_with_capabilities(
                contract_type=node_type
            )
            assert projection.contract_type == node_type

    def test_contract_type_accepts_none(self) -> None:
        """Test that contract_type accepts None."""
        projection = create_test_projection_with_capabilities(contract_type=None)
        assert projection.contract_type is None

    def test_contract_type_rejects_invalid_values(self) -> None:
        """Test that contract_type rejects invalid values."""
        invalid_types = ["invalid", "runtime_host", "EFFECT", "Effect", "node", ""]

        for invalid_type in invalid_types:
            with pytest.raises(ValidationError) as exc_info:
                create_test_projection_with_capabilities(contract_type=invalid_type)

            # Verify the error message mentions the valid types
            error_str = str(exc_info.value)
            assert "contract_type" in error_str.lower()

    def test_contract_type_error_message_includes_valid_values(self) -> None:
        """Test that validation error message lists valid contract types."""
        with pytest.raises(ValidationError) as exc_info:
            create_test_projection_with_capabilities(contract_type="invalid_type")

        error_str = str(exc_info.value)
        # Error message should mention the valid types
        for valid_type in VALID_CONTRACT_TYPES:
            assert valid_type in error_str

    def test_intent_types_accepts_list_of_strings(self) -> None:
        """Test that intent_types accepts a list of strings."""
        intents = ["postgres.upsert", "postgres.query", "kafka.publish"]
        projection = create_test_projection_with_capabilities(intent_types=intents)
        assert projection.intent_types == intents
        assert len(projection.intent_types) == 3

    def test_protocols_accepts_list_of_strings(self) -> None:
        """Test that protocols accepts a list of strings."""
        protocols = ["ProtocolDatabaseAdapter", "ProtocolEventPublisher"]
        projection = create_test_projection_with_capabilities(protocols=protocols)
        assert projection.protocols == protocols

    def test_capability_tags_accepts_list_of_strings(self) -> None:
        """Test that capability_tags accepts a list of strings."""
        tags = ["postgres.storage", "kafka.consumer", "http.client"]
        projection = create_test_projection_with_capabilities(capability_tags=tags)
        assert projection.capability_tags == tags


@pytest.mark.unit
class TestUnknownContractTypeValidation:
    """Test validation of 'unknown' contract type at model layer."""

    def test_model_capability_fields_accepts_unknown(self) -> None:
        """Test that ModelCapabilityFields accepts 'unknown' for backfill scenarios.

        The model layer allows 'unknown' to be constructed for backfill
        scenarios where contract_type is not yet determined.
        """
        fields = ModelCapabilityFields(contract_type="unknown")
        assert fields.contract_type == "unknown"

    def test_model_registration_projection_accepts_unknown(self) -> None:
        """Test that ModelRegistrationProjection accepts 'unknown' for backfill."""
        now = datetime.now(UTC)
        projection = ModelRegistrationProjection(
            entity_id=uuid4(),
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            node_type=EnumNodeKind.EFFECT,
            contract_type="unknown",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        assert projection.contract_type == "unknown"


__all__: list[str] = []
