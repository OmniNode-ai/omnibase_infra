# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Capability Fields Model.

Provides a structured container for capability-related fields used in
projection persistence. Consolidates denormalized capability data that
enables fast discovery queries via GIN-indexed columns.

Related Tickets:
    - OMN-1134: Registry Projection Extensions for Capabilities
    - OMN-944 (F1): Implement Registration Projection Schema
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums import EnumContractType
from omnibase_infra.models.projection.model_registration_projection import ContractType


class ModelCapabilityFields(BaseModel):
    """Container for capability fields used in projection persistence.

    Consolidates the denormalized capability fields that are stored in
    registration projections for fast discovery queries. These fields
    are indexed with GIN indexes for efficient containment queries.

    Use Cases:
        - Pass to persist_state_transition() for consolidated capability data
        - Extract from ModelNodeIntrospectionEvent for persistence
        - Build from ModelNodeCapabilities for denormalization

    Design Notes:
        - All fields are optional to support partial updates
        - Lists default to None (not empty list) to distinguish "not set"
          from "explicitly empty" in persistence logic
        - contract_type is validated to be a valid node contract type

    Attributes:
        contract_type: Node contract type (effect, compute, reducer, orchestrator)
        intent_types: List of intent types this node handles
        protocols: List of protocols this node implements
        capability_tags: Tags for capability-based discovery
        contract_version: Semantic version of the node contract

    Example:
        >>> fields = ModelCapabilityFields(
        ...     contract_type="effect",
        ...     intent_types=["postgres.upsert", "postgres.query"],
        ...     protocols=["ProtocolDatabaseAdapter"],
        ...     capability_tags=["postgres.storage"],
        ...     contract_version="1.0.0",
        ... )
        >>> await projector.persist_state_transition(
        ...     entity_id=node_id,
        ...     capability_fields=fields,
        ...     ...
        ... )
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable for safe passing
        extra="forbid",
    )

    contract_type: ContractType | None = Field(
        default=None,
        description=(
            "Contract type for the node. Valid values: 'effect', 'compute', "
            "'reducer', 'orchestrator'. None indicates unspecified."
        ),
    )

    @field_validator("contract_type", mode="before")
    @classmethod
    def validate_contract_type(cls, v: str | None) -> str | None:
        """Validate contract_type is a valid node contract type.

        Provides fail-fast validation for contract_type to catch invalid values
        at model instantiation time rather than at persistence time.

        Args:
            v: The contract_type value to validate

        Returns:
            The validated value (unchanged if valid)

        Raises:
            ValueError: If v is not None and not a valid contract type
        """
        if v is not None and v not in EnumContractType.valid_type_values():
            raise ValueError(
                f"contract_type must be one of {EnumContractType.valid_type_values()}, "
                f"got: {v!r}"
            )
        return v

    intent_types: list[str] | None = Field(
        default=None,
        description="Intent types this node handles (None = not specified)",
    )
    protocols: list[str] | None = Field(
        default=None,
        description="Protocols this node implements (None = not specified)",
    )
    capability_tags: list[str] | None = Field(
        default=None,
        description="Capability tags for discovery (None = not specified)",
    )
    contract_version: str | None = Field(
        default=None,
        description="Contract version string",
    )


__all__: list[str] = ["ModelCapabilityFields"]
