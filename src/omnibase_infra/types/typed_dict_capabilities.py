# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""TypedDict for node capability structures discovered via reflection.

This module provides a strongly-typed structure for representing node
capabilities discovered during introspection. Using TypedDict instead of
generic dict types enables static type checking and IDE support while
maintaining runtime dict compatibility.

The TypedDictCapabilities structure is used by the MixinNodeIntrospection
mixin to report discovered capabilities in a type-safe manner, eliminating
the need for permissive `dict[str, object]` or `Any` types.

Example:
    >>> from omnibase_infra.types.typed_dict_capabilities import TypedDictCapabilities
    >>> caps: TypedDictCapabilities = {
    ...     "operations": ["execute", "query"],
    ...     "protocols": ["ProtocolDatabaseAdapter"],
    ...     "has_fsm": False,
    ...     "method_signatures": {"execute": "(query: str) -> list[dict]"},
    ... }
"""

from typing import TypedDict

__all__ = ["TypedDictCapabilities"]


class TypedDictCapabilities(TypedDict, total=True):
    """Type-safe structure for node capabilities discovered via reflection.

    This TypedDict provides explicit typing for capability fields, eliminating
    the need for permissive `dict[str, object]` or `Any` types.

    All fields are required (total=True is explicitly set). The MixinNodeIntrospection
    mixin always constructs complete capability dicts with all fields populated.

    Attributes:
        operations: List of public method names that may be operations.
            These are methods matching configured operation keywords
            (e.g., execute, handle, process).
        protocols: List of protocol/interface names implemented by the node.
            Discovered from class hierarchy (e.g., ProtocolDatabaseAdapter).
        has_fsm: Boolean indicating if node has FSM state management.
            True if state attributes like _state or current_state are found.
        method_signatures: Dict mapping public method names to their signature
            strings (e.g., {"execute": "(query: str) -> list[dict]"}).

    Example:
        >>> capabilities: TypedDictCapabilities = {
        ...     "operations": ["execute", "query", "batch_execute"],
        ...     "protocols": ["ProtocolDatabaseAdapter", "MixinNodeIntrospection"],
        ...     "has_fsm": True,
        ...     "method_signatures": {
        ...         "execute": "(query: str) -> list[dict]",
        ...         "query": "(sql: str, params: dict) -> list[dict]",
        ...     },
        ... }
    """

    operations: list[str]
    protocols: list[str]
    has_fsm: bool
    method_signatures: dict[str, str]
