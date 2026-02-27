# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelBusDescriptor.

Tests frozen immutability, extra="forbid" validation, serde round-trip,
and field constraints for the trust-domain-scoped bus descriptor model.

Phase 5 of Authenticated Dependency Resolution epic (OMN-2897).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.topics.model_bus_descriptor import ModelBusDescriptor

pytestmark = [pytest.mark.unit]


def _make_descriptor(**overrides: object) -> ModelBusDescriptor:
    """Create a ModelBusDescriptor with sensible defaults, overriding as needed."""
    defaults: dict[str, object] = {
        "bus_id": "bus.local",
        "trust_domain": "local.default",
        "transport_type": EnumInfraTransportType.KAFKA,
        "namespace_prefix": "",
        "bootstrap_servers": ("localhost:9092",),
        "allowed_classifications": ("public", "internal"),
    }
    defaults.update(overrides)
    return ModelBusDescriptor(**defaults)  # type: ignore[arg-type]


class TestModelBusDescriptorFrozen:
    """Frozen immutability tests."""

    def test_frozen_bus_id(self) -> None:
        """bus_id cannot be mutated after construction."""
        desc = _make_descriptor()
        with pytest.raises(ValidationError):
            desc.bus_id = "changed"  # type: ignore[misc]

    def test_frozen_trust_domain(self) -> None:
        """trust_domain cannot be mutated after construction."""
        desc = _make_descriptor()
        with pytest.raises(ValidationError):
            desc.trust_domain = "changed"  # type: ignore[misc]

    def test_frozen_transport_type(self) -> None:
        """transport_type cannot be mutated after construction."""
        desc = _make_descriptor()
        with pytest.raises(ValidationError):
            desc.transport_type = EnumInfraTransportType.INMEMORY  # type: ignore[misc]

    def test_frozen_namespace_prefix(self) -> None:
        """namespace_prefix cannot be mutated after construction."""
        desc = _make_descriptor()
        with pytest.raises(ValidationError):
            desc.namespace_prefix = "changed."  # type: ignore[misc]


class TestModelBusDescriptorValidation:
    """Validation and extra="forbid" tests."""

    def test_extra_fields_rejected(self) -> None:
        """Extra fields not in the schema are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            _make_descriptor(unknown_field="value")

    def test_bus_id_required(self) -> None:
        """bus_id is a required field."""
        with pytest.raises(ValidationError):
            ModelBusDescriptor(
                trust_domain="local.default",
                transport_type=EnumInfraTransportType.KAFKA,
            )  # type: ignore[call-arg]

    def test_trust_domain_required(self) -> None:
        """trust_domain is a required field."""
        with pytest.raises(ValidationError):
            ModelBusDescriptor(
                bus_id="bus.local",
                transport_type=EnumInfraTransportType.KAFKA,
            )  # type: ignore[call-arg]

    def test_transport_type_required(self) -> None:
        """transport_type is a required field."""
        with pytest.raises(ValidationError):
            ModelBusDescriptor(
                bus_id="bus.local",
                trust_domain="local.default",
            )  # type: ignore[call-arg]

    def test_bus_id_min_length(self) -> None:
        """bus_id cannot be an empty string."""
        with pytest.raises(ValidationError, match="string_too_short"):
            _make_descriptor(bus_id="")

    def test_trust_domain_min_length(self) -> None:
        """trust_domain cannot be an empty string."""
        with pytest.raises(ValidationError, match="string_too_short"):
            _make_descriptor(trust_domain="")

    def test_invalid_transport_type(self) -> None:
        """Invalid transport type string is rejected."""
        with pytest.raises(ValidationError):
            _make_descriptor(transport_type="not_a_transport")


class TestModelBusDescriptorSerde:
    """Serialization and deserialization round-trip tests."""

    def test_round_trip_model_dump_load(self) -> None:
        """model_dump -> ModelBusDescriptor round-trip preserves all fields."""
        original = _make_descriptor(
            bus_id="bus.org.omninode",
            trust_domain="org.omninode",
            transport_type=EnumInfraTransportType.KAFKA,
            namespace_prefix="org.omninode.",
            bootstrap_servers=("kafka1:9092", "kafka2:9092"),
            allowed_classifications=("public",),
        )
        data = original.model_dump()
        restored = ModelBusDescriptor.model_validate(data)
        assert restored == original

    def test_round_trip_json(self) -> None:
        """JSON serialization round-trip preserves all fields."""
        original = _make_descriptor()
        json_str = original.model_dump_json()
        restored = ModelBusDescriptor.model_validate_json(json_str)
        assert restored == original

    def test_model_dump_structure(self) -> None:
        """model_dump returns expected keys."""
        desc = _make_descriptor()
        data = desc.model_dump()
        expected_keys = {
            "bus_id",
            "trust_domain",
            "transport_type",
            "namespace_prefix",
            "bootstrap_servers",
            "allowed_classifications",
        }
        assert set(data.keys()) == expected_keys


class TestModelBusDescriptorDefaults:
    """Default value tests."""

    def test_namespace_prefix_defaults_empty(self) -> None:
        """namespace_prefix defaults to empty string."""
        desc = ModelBusDescriptor(
            bus_id="bus.local",
            trust_domain="local.default",
            transport_type=EnumInfraTransportType.KAFKA,
        )
        assert desc.namespace_prefix == ""

    def test_bootstrap_servers_defaults_empty_tuple(self) -> None:
        """bootstrap_servers defaults to empty tuple."""
        desc = ModelBusDescriptor(
            bus_id="bus.local",
            trust_domain="local.default",
            transport_type=EnumInfraTransportType.KAFKA,
        )
        assert desc.bootstrap_servers == ()

    def test_allowed_classifications_defaults_empty_tuple(self) -> None:
        """allowed_classifications defaults to empty tuple."""
        desc = ModelBusDescriptor(
            bus_id="bus.local",
            trust_domain="local.default",
            transport_type=EnumInfraTransportType.KAFKA,
        )
        assert desc.allowed_classifications == ()


class TestModelBusDescriptorTransportTypes:
    """Transport type coverage tests."""

    def test_kafka_transport(self) -> None:
        """Kafka transport type is accepted."""
        desc = _make_descriptor(transport_type=EnumInfraTransportType.KAFKA)
        assert desc.transport_type == EnumInfraTransportType.KAFKA

    def test_inmemory_transport(self) -> None:
        """In-memory transport type is accepted."""
        desc = _make_descriptor(transport_type=EnumInfraTransportType.INMEMORY)
        assert desc.transport_type == EnumInfraTransportType.INMEMORY

    def test_bridge_transport(self) -> None:
        """Bridge transport type is accepted."""
        desc = _make_descriptor(transport_type=EnumInfraTransportType.BRIDGE)
        assert desc.transport_type == EnumInfraTransportType.BRIDGE
