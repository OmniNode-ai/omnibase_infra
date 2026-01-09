# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Type Enumeration for ONEX Nodes.

Defines the valid contract types for ONEX node contracts. Contract types
correspond to the 4-node architecture (effect, compute, reducer, orchestrator)
plus an 'unknown' fallback for backfill scenarios where the type cannot be
determined from capabilities or node metadata.

This enum is distinct from EnumNodeArchetype in that:
    - EnumNodeArchetype: Execution shape validation (what outputs are allowed)
    - EnumContractType: Contract metadata (contract.yaml node_type field)

The UNKNOWN value exists specifically for data migration/backfill scenarios
where historical records may not have a determinable contract type. It is
NOT a valid type for new node registrations.

Related Tickets:
    - OMN-1134: Registry Projection Extensions for Capabilities

See Also:
    - EnumNodeArchetype: Node archetypes for execution shape validation
    - EnumNodeKind: Core node kind enumeration (omnibase_core)
"""

from enum import StrEnum


class EnumContractType(StrEnum):
    """Contract type for ONEX nodes.

    Defines the valid contract types used in node contract metadata,
    registration projections, and capability discovery queries.

    Attributes:
        EFFECT: External interaction nodes (API calls, DB operations).
        COMPUTE: Pure data processing and transformation nodes.
        REDUCER: State management and persistence nodes.
        ORCHESTRATOR: Workflow coordination nodes.
        UNKNOWN: Fallback for backfill when type cannot be determined.
            NOT valid for new node registrations.

    Example:
        >>> from omnibase_infra.enums import EnumContractType
        >>> contract_type = EnumContractType.EFFECT
        >>> contract_type.value
        'effect'
        >>> EnumContractType("reducer")
        <EnumContractType.REDUCER: 'reducer'>
    """

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"
    UNKNOWN = "unknown"

    @classmethod
    def valid_types(cls) -> tuple["EnumContractType", ...]:
        """Return valid contract types (excludes UNKNOWN).

        UNKNOWN is a fallback for backfill scenarios and is not a valid
        type for new node registrations.

        Returns:
            Tuple of valid EnumContractType values (EFFECT, COMPUTE,
            REDUCER, ORCHESTRATOR).
        """
        return (cls.EFFECT, cls.COMPUTE, cls.REDUCER, cls.ORCHESTRATOR)

    @classmethod
    def valid_type_values(cls) -> tuple[str, ...]:
        """Return valid contract type string values (excludes 'unknown').

        Convenience method for validation contexts where string values
        are needed.

        Returns:
            Tuple of valid contract type strings.
        """
        return tuple(t.value for t in cls.valid_types())


__all__ = ["EnumContractType"]
