# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dependency Specification Model.

Defines the specification for declaring dependencies by capability rather than
by concrete implementation. This enables capability-based auto-configuration
where nodes declare what they need, not who provides it.

Core Principle: "I'm interested in what you do, not what you are."

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery for capability-based discovery
    - Design: DESIGN_CAPABILITY_BASED_AUTO_CONFIGURATION.md

Example:
    >>> spec = ModelDependencySpec(
    ...     name="storage_effect",
    ...     type="node",
    ...     capability="postgres.storage",
    ...     contract_type="effect",
    ...     intent_types=["postgres.upsert_registration"],
    ... )
    >>> # System auto-discovers which node provides this capability
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelDependencySpec(BaseModel):
    """Dependency specification - declares what is needed, not who provides it.

    This model specifies a dependency by capability attributes (capability tags,
    intent types, protocols) rather than by concrete module paths. The runtime
    uses ServiceCapabilityQuery to resolve these specifications to actual nodes.

    Discovery Strategy (in priority order):
        1. If capability specified -> find by capability tag
        2. If intent_types specified -> find by intent type
        3. If protocol specified -> find by protocol

    Selection Strategy (when multiple matches):
        - first: Return first candidate (deterministic, fast)
        - random: Random selection (load distribution)
        - round_robin: Cycle through candidates (even distribution)
        - least_loaded: RESERVED FOR FUTURE USE - not yet implemented

    Attributes:
        name: Dependency name for reference in code
        type: Dependency type (node or protocol)
        capability: Capability tag (e.g., "consul.registration", "postgres.storage")
        intent_types: Intent types the dependency must handle
        protocol: Protocol the dependency must implement
        contract_type: Filter by contract type (effect, compute, reducer, orchestrator)
        state: Filter by registration state (default: ACTIVE)
        selection_strategy: Strategy for selecting among multiple matches
        fallback_module: Reserved for future - fallback module if auto-discovery fails
        description: Human-readable description of the dependency

    Example - Capability-based:
        >>> spec = ModelDependencySpec(
        ...     name="registry_effect",
        ...     type="node",
        ...     capability="consul.registration",
        ...     contract_type="effect",
        ... )

    Example - Intent-based:
        >>> spec = ModelDependencySpec(
        ...     name="storage_effect",
        ...     type="node",
        ...     intent_types=["postgres.upsert_registration", "postgres.query"],
        ... )

    Example - Protocol-based:
        >>> spec = ModelDependencySpec(
        ...     name="reducer",
        ...     type="protocol",
        ...     protocol="ProtocolReducer",
        ...     contract_type="reducer",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # Identity
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Dependency name for reference in code",
    )
    type: Literal["node", "protocol"] = Field(
        ...,
        description="Dependency type (node or protocol)",
    )

    # Capability-based discovery (choose one or combine)
    capability: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Capability tag (e.g., 'consul.registration', 'postgres.storage')",
    )
    intent_types: list[str] | None = Field(
        default=None,
        description="Intent types the dependency must handle",
    )
    protocol: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Protocol the dependency must implement",
    )

    # Filters
    contract_type: Literal["effect", "compute", "reducer", "orchestrator"] | None = (
        Field(
            default=None,
            description="Filter by contract type",
        )
    )
    state: str = Field(
        default="ACTIVE",
        description="Filter by registration state",
    )

    # Selection strategy (when multiple matches)
    selection_strategy: Literal["first", "random", "round_robin", "least_loaded"] = (
        Field(
            default="first",
            description=(
                "Strategy for selecting among multiple matches. "
                "Valid values: 'first', 'random', 'round_robin'. "
                "Note: 'least_loaded' is reserved for future use and not yet implemented."
            ),
        )
    )

    # Fallback (if capability not found)
    # NOTE: Reserved for future implementation. When auto-discovery fails to find
    # any matching nodes, this module path could be dynamically imported and
    # instantiated as a last resort. Currently NOT used by ServiceCapabilityQuery.
    # See: resolve_dependency() returns None when no candidates found.
    # Future implementation would involve: importlib.import_module + class instantiation.
    fallback_module: str | None = Field(
        default=None,
        description=(
            "Reserved for future implementation. Module path to import if "
            "auto-discovery fails (e.g., 'mypackage.adapters.FallbackAdapter'). "
            "Currently not used by ServiceCapabilityQuery - returns None instead."
        ),
    )

    # Documentation
    description: str | None = Field(
        default=None,
        max_length=1024,
        description="Human-readable description of the dependency",
    )

    @field_validator("selection_strategy")
    @classmethod
    def validate_selection_strategy_implemented(cls, v: str) -> str:
        """Validate that the selection strategy is implemented.

        The 'least_loaded' strategy is reserved for future use and requires
        load metrics infrastructure that is not yet available. This validator
        provides fail-fast behavior at model creation time rather than at
        runtime during node selection.

        Args:
            v: The selection strategy value to validate.

        Returns:
            The validated selection strategy value.

        Raises:
            ValueError: If 'least_loaded' is specified.

        Example:
            >>> spec = ModelDependencySpec(
            ...     name="test",
            ...     type="node",
            ...     capability="test.cap",
            ...     selection_strategy="least_loaded",
            ... )
            ValueError: LEAST_LOADED selection strategy is not yet implemented...
        """
        if v == "least_loaded":
            raise ValueError(
                "LEAST_LOADED selection strategy is not yet implemented. "
                "Use 'first', 'random', or 'round_robin' instead."
            )
        return v

    @model_validator(mode="after")
    def validate_has_filter(self) -> ModelDependencySpec:
        """Validate that at least one discovery filter is specified.

        A dependency spec must have at least one of: capability, intent_types, or protocol.
        Without any filter, the dependency cannot be resolved.

        Raises:
            ValueError: If no discovery filter is specified.
        """
        if not self.has_any_filter():
            raise ValueError(
                f"Dependency spec '{self.name}' must have at least one discovery filter: "
                "capability, intent_types, or protocol. "
                "Cannot resolve a dependency without any filter specified."
            )
        return self

    def has_capability_filter(self) -> bool:
        """Check if capability-based discovery is specified.

        Returns:
            True if capability tag is specified, False otherwise.
        """
        return self.capability is not None

    def has_intent_filter(self) -> bool:
        """Check if intent-based discovery is specified.

        Returns:
            True if intent_types are specified, False otherwise.
        """
        return self.intent_types is not None and len(self.intent_types) > 0

    def has_protocol_filter(self) -> bool:
        """Check if protocol-based discovery is specified.

        Returns:
            True if protocol is specified, False otherwise.
        """
        return self.protocol is not None

    def has_any_filter(self) -> bool:
        """Check if any discovery filter is specified.

        Returns:
            True if at least one filter is specified, False otherwise.
        """
        return (
            self.has_capability_filter()
            or self.has_intent_filter()
            or self.has_protocol_filter()
        )


__all__: list[str] = ["ModelDependencySpec"]
