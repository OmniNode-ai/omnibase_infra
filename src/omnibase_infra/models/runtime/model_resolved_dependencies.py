# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for resolved protocol dependencies.

This module provides ModelResolvedDependencies, a container for protocol
instances resolved from the container service_registry at node creation time.

Part of OMN-1732: Runtime dependency injection for zero-code nodes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelResolvedDependencies(BaseModel):
    """Container for resolved protocol dependencies.

    Holds protocol instances resolved from ModelONEXContainer.service_registry
    for injection into node constructors. This model is immutable after creation.

    The protocols dict maps protocol class names to their resolved instances:
    - Key: Protocol class name (e.g., "ProtocolPostgresAdapter")
    - Value: Resolved instance from container

    Example:
        >>> resolved = ModelResolvedDependencies(
        ...     protocols={
        ...         "ProtocolPostgresAdapter": postgres_adapter,
        ...         "ProtocolCircuitBreakerAware": circuit_breaker,
        ...     }
        ... )
        >>> adapter = resolved.get("ProtocolPostgresAdapter")

    .. versionadded:: 0.x.x
        Part of OMN-1732 runtime dependency injection.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Required for protocol instances
    )

    # ONEX_EXCLUDE: any_type - dict[str, Any] required for heterogeneous protocol instances
    # resolved from container.service_registry. Type varies by protocol (ProtocolPostgresAdapter,
    # ProtocolCircuitBreakerAware, etc.). Cannot use Union as protocols are open-ended.
    protocols: dict[str, Any] = Field(
        default_factory=dict,
        description="Map of protocol class names to resolved instances",
    )

    # ONEX_EXCLUDE: any_type - returns heterogeneous protocol instance from protocols dict
    def get(self, protocol_name: str) -> Any:
        """Get a resolved protocol by name.

        Args:
            protocol_name: The protocol class name (e.g., "ProtocolPostgresAdapter")

        Returns:
            The resolved protocol instance.

        Raises:
            KeyError: If protocol_name is not in the resolved protocols.

        Example:
            >>> adapter = resolved.get("ProtocolPostgresAdapter")
        """
        if protocol_name not in self.protocols:
            raise KeyError(
                f"Protocol '{protocol_name}' not found in resolved dependencies. "
                f"Available: {list(self.protocols.keys())}"
            )
        return self.protocols[protocol_name]

    # ONEX_EXCLUDE: any_type - returns heterogeneous protocol instance, default can be any type
    def get_optional(self, protocol_name: str, default: Any = None) -> Any:
        """Get a resolved protocol by name, returning default if not found.

        Args:
            protocol_name: The protocol class name
            default: Value to return if protocol not found

        Returns:
            The resolved protocol instance or default.
        """
        return self.protocols.get(protocol_name, default)

    def has(self, protocol_name: str) -> bool:
        """Check if a protocol is available.

        Args:
            protocol_name: The protocol class name

        Returns:
            True if protocol is resolved, False otherwise.
        """
        return protocol_name in self.protocols

    def __len__(self) -> int:
        """Return number of resolved protocols."""
        return len(self.protocols)

    def __bool__(self) -> bool:
        """Return True if any protocols are resolved.

        Warning:
            **Non-standard __bool__ behavior**: Returns True only when
            at least one protocol is resolved.
        """
        return len(self.protocols) > 0


__all__ = ["ModelResolvedDependencies"]
