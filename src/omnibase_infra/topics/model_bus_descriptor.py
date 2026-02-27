# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Bus descriptor model for trust-domain-scoped multi-bus topic routing.

A ``ModelBusDescriptor`` captures the configuration of a single message bus
within a trust domain. The ``TopicResolver`` uses these descriptors to prepend
namespace prefixes when resolving topics for a specific trust domain.

This is Phase 5 of the Authenticated Dependency Resolution epic (OMN-2897).
The model is intentionally self-contained within omnibase_infra and does not
depend on omnibase_core routing types -- it will be connected to those types
in Phase 7 (Contract YAML Integration).

.. versionadded:: 0.10.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType


class ModelBusDescriptor(BaseModel):
    """Descriptor for a trust-domain-scoped message bus.

    Each descriptor defines a single bus endpoint within a trust domain,
    including its transport type, namespace prefix for topic scoping, and
    the set of data classifications allowed to traverse it.

    Attributes:
        bus_id: Unique identifier for this bus instance
            (e.g., ``"bus.local"``, ``"bus.org.omninode"``).
        trust_domain: Trust domain this bus belongs to
            (e.g., ``"local.default"``, ``"org.omninode"``).
        transport_type: Transport protocol used by this bus.
        namespace_prefix: Prefix prepended to topic suffixes when routing
            through this bus (e.g., ``"org.omninode."``). Empty string means
            no prefix is applied.
        bootstrap_servers: Kafka/Redpanda bootstrap server addresses for this
            bus. Empty for in-memory or bridge transports.
        allowed_classifications: Data classification labels that may traverse
            this bus (e.g., ``["public", "internal"]``). Empty list means all
            classifications are allowed.

    Example:
        >>> descriptor = ModelBusDescriptor(
        ...     bus_id="bus.local",
        ...     trust_domain="local.default",
        ...     transport_type=EnumInfraTransportType.KAFKA,
        ...     namespace_prefix="",
        ...     bootstrap_servers=["localhost:9092"],
        ...     allowed_classifications=["public", "internal"],
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    bus_id: str = Field(
        ...,
        description="Unique identifier for this bus instance.",
        min_length=1,
    )
    trust_domain: str = Field(
        ...,
        description="Trust domain this bus belongs to.",
        min_length=1,
    )
    transport_type: EnumInfraTransportType = Field(
        ...,
        description="Transport protocol used by this bus.",
    )
    namespace_prefix: str = Field(
        default="",
        description=(
            "Prefix prepended to topic suffixes when routing through this bus. "
            "Empty string means no prefix."
        ),
    )
    bootstrap_servers: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Kafka/Redpanda bootstrap server addresses for this bus.",
    )
    allowed_classifications: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Data classification labels allowed on this bus. "
            "Empty tuple means all classifications are allowed."
        ),
    )


__all__ = ["ModelBusDescriptor"]
