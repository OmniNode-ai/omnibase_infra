# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""TypedDict definition for JSON-serialized introspection cache data.

This module provides a TypedDict that represents the JSON-serialized form of
ModelNodeIntrospectionEvent, enabling proper type checking for cache operations
without requiring type: ignore comments.

The TypedDictIntrospectionCache matches the output of
ModelNodeIntrospectionEvent.model_dump(mode="json"), providing a typed interface
for working with cached introspection data.

Key Features:
    - Full type annotations for all introspection event fields
    - Proper handling of serialized datetime (ISO string format)
    - Proper handling of serialized UUID (string format)
    - Integration with TypedDictCapabilities for nested capabilities structure
    - Integration with TypedDictPerformanceMetricsCache for performance metrics

Usage:
    This TypedDict is primarily used for:
    - Typing cache storage for introspection data
    - Type-safe JSON deserialization of introspection events
    - Avoiding type: ignore comments in cache operations

Example:
    ```python
    from omnibase_infra.types.typed_dict import TypedDictIntrospectionCache

    # Cache storage with proper typing
    cache: TypedDictIntrospectionCache | None = None

    def store_introspection(event: ModelNodeIntrospectionEvent) -> None:
        global cache
        # model_dump(mode="json") returns data matching TypedDictIntrospectionCache
        cache = event.model_dump(mode="json")

    def get_cached_node_id() -> str | None:
        if cache is not None:
            return cache["node_id"]  # Type-safe access
        return None
    ```

See Also:
    - TypedDictCapabilities: Nested structure for capability information
    - TypedDictPerformanceMetricsCache: Nested structure for performance metrics
    - ModelNodeIntrospectionEvent: Source Pydantic model for introspection events
    - MixinNodeIntrospection: Primary consumer of this TypedDict
"""

from __future__ import annotations

from typing import TypedDict

from omnibase_infra.types.typed_dict.typed_dict_performance_metrics_cache import (
    TypedDictPerformanceMetricsCache,
)
from omnibase_infra.types.typed_dict_capabilities import TypedDictCapabilities

__all__ = ["TypedDictIntrospectionCache"]


class TypedDictIntrospectionCache(TypedDict):
    """TypedDict representing the JSON-serialized ModelNodeIntrospectionEvent.

    This type matches the output of ModelNodeIntrospectionEvent.model_dump(mode="json"),
    enabling proper type checking for cache operations without requiring type: ignore
    comments.

    The TypedDict provides a typed interface for working with introspection cache data,
    ensuring type safety when accessing fields from cached introspection events.

    Attributes:
        node_id: Unique identifier for the node (UUID serialized to string).
        node_type: The type classification of the node (e.g., "EFFECT", "COMPUTE").
        capabilities: Nested structure containing node capabilities information.
            Uses TypedDictCapabilities which includes operations, protocols,
            has_fsm flag, and method_signatures.
        endpoints: Dictionary mapping endpoint names to their URLs.
            Common keys include "health", "api", and "metrics".
        current_state: Current FSM state if applicable, None otherwise.
            Represents the node's finite state machine state.
        version: Semantic version string of the node.
        reason: The reason for the introspection event (e.g., "startup", "shutdown",
            "heartbeat", "request").
        correlation_id: Optional correlation ID for request tracing.
            UUID serializes to string in JSON mode, may be None.
        timestamp: ISO 8601 formatted datetime string indicating when the
            introspection event was created.
        performance_metrics: Optional performance metrics from the introspection
            operation. Contains timing information for capability discovery,
            endpoint gathering, and state retrieval. May be None if metrics
            were not captured.

    Note:
        The capabilities field uses TypedDictCapabilities for type safety.
        When serialized to JSON, the structure is:
        - operations (list[str]): Discovered operation method names
        - protocols (list[str]): Implemented protocol names
        - has_fsm (bool): Whether the node has FSM capabilities
        - method_signatures (dict[str, str]): Method name to signature mapping

    Example:
        ```python
        # Type-safe cache access
        cache: TypedDictIntrospectionCache = event.model_dump(mode="json")

        # All field access is type-checked
        node_id: str = cache["node_id"]
        capabilities: TypedDictCapabilities = cache["capabilities"]
        operations: list[str] = capabilities["operations"]

        # Nullable fields properly typed
        state: str | None = cache["current_state"]
        metrics: TypedDictPerformanceMetricsCache | None = cache["performance_metrics"]
        ```
    """

    node_id: str
    node_type: str
    capabilities: TypedDictCapabilities
    endpoints: dict[str, str]
    current_state: str | None
    version: str
    reason: str
    correlation_id: str | None  # UUID serializes to string in JSON mode
    timestamp: str  # datetime serializes to ISO string in JSON mode
    performance_metrics: TypedDictPerformanceMetricsCache | None
