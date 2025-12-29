# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Identifier Model for Validation Error Reporting.

This module provides ModelHandlerIdentifier for uniquely identifying handlers
in validation error contexts. Part of OMN-1091 structured validation and error
reporting for handlers.

The identifier supports multiple construction patterns:
- From handler_id alone (minimal identification)
- From node_path and handler_type (structured identification)
- Full context with human-readable name

This model enables precise error reporting by capturing handler identity
information at validation failure points.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumHandlerType


class ModelHandlerIdentifier(BaseModel):
    """Identifies a handler for validation error reporting.

    This model captures the identity and context of a handler when validation
    errors occur, enabling precise error reporting and debugging. It supports
    multiple levels of detail, from minimal handler_id-only identification to
    full context with node paths and types.

    Attributes:
        handler_id: Unique identifier for the handler (e.g., "http-handler", "db-handler").
            This is the minimal required field for handler identification.
        handler_type: Type of handler (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
            Optional, provides architectural context for the handler.
        handler_name: Human-readable name for the handler.
            Optional, used for display purposes in error messages.
        node_path: Path to the node containing this handler.
            Optional, provides location context for debugging.

    Example:
        >>> # Minimal identification
        >>> identifier = ModelHandlerIdentifier.from_handler_id("http-handler")
        >>> identifier.handler_id
        'http-handler'

        >>> # Structured identification with type
        >>> identifier = ModelHandlerIdentifier.from_node(
        ...     node_path="nodes/registration/node.py",
        ...     handler_type=EnumHandlerType.ORCHESTRATOR,
        ... )
        >>> identifier.handler_type
        <EnumHandlerType.ORCHESTRATOR: 'orchestrator'>

        >>> # Full context
        >>> identifier = ModelHandlerIdentifier(
        ...     handler_id="registration-orchestrator",
        ...     handler_type=EnumHandlerType.ORCHESTRATOR,
        ...     handler_name="Registration Orchestrator",
        ...     node_path="nodes/registration/node.py",
        ... )

    Note:
        This model is frozen to ensure immutability in error contexts.
        Use factory methods for common construction patterns.

    .. versionadded:: 0.6.1
        Created as part of OMN-1091 structured validation and error reporting.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    handler_id: str = Field(
        description="Unique identifier for the handler (e.g., 'http-handler', 'db-handler')",
    )
    handler_type: EnumHandlerType | None = Field(
        default=None,
        description="Type of handler (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)",
    )
    handler_name: str | None = Field(
        default=None,
        description="Human-readable name for the handler",
    )
    node_path: str | None = Field(
        default=None,
        description="Path to the node containing this handler",
    )

    @classmethod
    def from_handler_id(cls, handler_id: str) -> Self:
        """Create identifier from handler_id alone (minimal identification).

        Use this factory method when only the handler_id is available,
        typically in contexts where detailed handler metadata is not
        readily accessible.

        Args:
            handler_id: Unique identifier for the handler.

        Returns:
            ModelHandlerIdentifier with only handler_id set.

        Example:
            >>> identifier = ModelHandlerIdentifier.from_handler_id("http-handler")
            >>> identifier.handler_id
            'http-handler'
            >>> identifier.handler_type is None
            True

        .. versionadded:: 0.6.1
        """
        return cls(handler_id=handler_id)

    @classmethod
    def from_node(
        cls,
        node_path: str,
        handler_type: EnumHandlerType,
        handler_name: str | None = None,
    ) -> Self:
        """Create identifier from node path and handler type (structured identification).

        Use this factory method when constructing identifiers from node context,
        typically during contract validation or handler registration. The handler_id
        is derived from the node path and handler type.

        Args:
            node_path: Path to the node containing the handler.
            handler_type: Type of handler (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
            handler_name: Optional human-readable name for the handler.

        Returns:
            ModelHandlerIdentifier with full context.

        Example:
            >>> identifier = ModelHandlerIdentifier.from_node(
            ...     node_path="nodes/registration/node.py",
            ...     handler_type=EnumHandlerType.ORCHESTRATOR,
            ...     handler_name="Registration Orchestrator",
            ... )
            >>> identifier.handler_id
            'nodes/registration/node.py::orchestrator'
            >>> identifier.handler_type
            <EnumHandlerType.ORCHESTRATOR: 'orchestrator'>

        Note:
            The handler_id is constructed as "{node_path}::{handler_type.value}"
            to provide a unique identifier that combines location and type.

        .. versionadded:: 0.6.1
        """
        # Derive handler_id from node_path and handler_type
        handler_id = f"{node_path}::{handler_type.value}"

        return cls(
            handler_id=handler_id,
            handler_type=handler_type,
            handler_name=handler_name,
            node_path=node_path,
        )

    def format_for_error(self) -> str:
        """Format identifier for inclusion in error messages.

        Produces a human-readable string representation suitable for
        error messages and logs. The format varies based on available
        information, prioritizing readability.

        Returns:
            Formatted string representation of the handler identifier.

        Example:
            >>> # Minimal format
            >>> ModelHandlerIdentifier.from_handler_id("http-handler").format_for_error()
            'handler_id=http-handler'

            >>> # Full format
            >>> identifier = ModelHandlerIdentifier(
            ...     handler_id="registration-orchestrator",
            ...     handler_type=EnumHandlerType.ORCHESTRATOR,
            ...     handler_name="Registration Orchestrator",
            ...     node_path="nodes/registration/node.py",
            ... )
            >>> identifier.format_for_error()
            'Registration Orchestrator (orchestrator) at nodes/registration/node.py'

        .. versionadded:: 0.6.1
        """
        # If we have a human-readable name, use it as the primary identifier
        if self.handler_name:
            parts = [self.handler_name]
            if self.handler_type:
                parts.append(f"({self.handler_type.value})")
            if self.node_path:
                parts.append(f"at {self.node_path}")
            return " ".join(parts)

        # Fall back to handler_id if no name is available
        parts = [f"handler_id={self.handler_id}"]
        if self.handler_type:
            parts.append(f"type={self.handler_type.value}")
        if self.node_path:
            parts.append(f"path={self.node_path}")
        return ", ".join(parts)


__all__ = ["ModelHandlerIdentifier"]
