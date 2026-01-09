# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TypedDict for JSON-serialized performance metrics cache.

This module provides a TypedDict that matches the JSON output of
ModelIntrospectionPerformanceMetrics.model_dump(mode="json"), enabling
proper type checking for cached performance metrics without runtime
validation overhead.

The TypedDict is used for:
- Type-safe cache operations for performance metrics
- Avoiding type: ignore comments when working with serialized metrics
- Static type checking of JSON-structured performance data
- Documentation of the expected cache entry shape

Related Models:
    - ModelIntrospectionPerformanceMetrics: The Pydantic source model
    - MixinNodeIntrospection: Uses this for caching introspection metrics

Example:
    ```python
    from omnibase_infra.types.typed_dict import TypedDictPerformanceMetricsCache

    def get_cached_metrics() -> TypedDictPerformanceMetricsCache | None:
        cache_entry = cache.get("metrics")
        if cache_entry is not None:
            # Type checker knows the shape without runtime validation
            return cache_entry
        return None

    metrics = get_cached_metrics()
    if metrics:
        print(f"Total time: {metrics['total_introspection_ms']}ms")
        if metrics.get('threshold_exceeded'):
            print(f"Slow ops: {metrics.get('slow_operations', [])}")
    ```

Note:
    This TypedDict uses `total=True` (default) because the source Pydantic
    model `ModelIntrospectionPerformanceMetrics` has defaults for all fields,
    ensuring that `model_dump(mode="json")` always produces complete JSON
    with all fields present. Type checkers will require all fields when
    constructing dict literals, which matches the actual runtime behavior.
"""

from typing import TypedDict


class TypedDictPerformanceMetricsCache(TypedDict):
    """TypedDict for JSON-serialized ModelIntrospectionPerformanceMetrics.

    This type matches the output of ModelIntrospectionPerformanceMetrics.model_dump(mode="json"),
    enabling proper type checking for cached performance metrics.

    All fields are required (total=True, the default) because the source Pydantic
    model has defaults for all fields, ensuring model_dump() always produces
    complete JSON output with all fields present.

    Attributes:
        get_capabilities_ms: Time taken by get_capabilities() in milliseconds.
            Threshold: <50ms for acceptable performance.
        discover_capabilities_ms: Time taken by _discover_capabilities() in milliseconds.
            Threshold: <30ms for acceptable performance.
        get_endpoints_ms: Time taken by get_endpoints() in milliseconds.
            This operation is typically fast (<1ms) as it returns static data.
        get_current_state_ms: Time taken by get_current_state() in milliseconds.
            This operation is typically fast (<1ms) for simple state retrieval.
        total_introspection_ms: Total time for get_introspection_data() in milliseconds.
            Threshold: <50ms for acceptable performance.
        cache_hit: Whether the result was served from cache.
            Cache hits should complete in <1ms.
        method_count: Number of methods discovered during reflection.
            Higher counts may correlate with longer discover_capabilities_ms times.
        threshold_exceeded: Whether any operation exceeded performance thresholds.
            True indicates potential performance degradation.
        slow_operations: List of operation names that exceeded their thresholds.
            Possible values: 'get_capabilities', 'discover_capabilities',
            'total_introspection', 'cache_hit'.
        captured_at: UTC timestamp when metrics were captured (ISO 8601 string).
            Datetime serializes to ISO string format in JSON mode.

    Example:
        ```python
        from omnibase_infra.types.typed_dict import TypedDictPerformanceMetricsCache

        # Type-safe cache dictionary
        cache: dict[str, TypedDictPerformanceMetricsCache] = {}

        # Store metrics
        cache["node-123"] = {
            "get_capabilities_ms": 12.5,
            "discover_capabilities_ms": 8.2,
            "get_endpoints_ms": 0.5,
            "get_current_state_ms": 0.1,
            "total_introspection_ms": 21.3,
            "cache_hit": False,
            "method_count": 15,
            "threshold_exceeded": False,
            "slow_operations": [],
            "captured_at": "2025-01-15T10:30:00Z",
        }

        # Retrieve with full type safety
        if entry := cache.get("node-123"):
            total_ms = entry.get("total_introspection_ms", 0.0)
            if entry.get("threshold_exceeded"):
                slow_ops = entry.get("slow_operations", [])
                print(f"Performance degradation: {slow_ops}")
        ```

    Performance Thresholds:
        The following thresholds are defined as module-level constants in:
        src/omnibase_infra/mixins/mixin_node_introspection.py

        Constants (SOURCE OF TRUTH - update this docstring if values change):
        - PERF_THRESHOLD_GET_CAPABILITIES_MS: 50.0ms
        - PERF_THRESHOLD_DISCOVER_CAPABILITIES_MS: 30.0ms
        - PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS: 50.0ms
        - PERF_THRESHOLD_CACHE_HIT_MS: 1.0ms

        Summary:
        - get_capabilities: <50ms
        - discover_capabilities: <30ms
        - total_introspection: <50ms
        - cache_hit: <1ms
    """

    get_capabilities_ms: float
    discover_capabilities_ms: float
    get_endpoints_ms: float
    get_current_state_ms: float
    total_introspection_ms: float
    cache_hit: bool
    method_count: int
    threshold_exceeded: bool
    slow_operations: list[str]
    captured_at: str  # datetime serializes to ISO string in JSON mode


__all__ = ["TypedDictPerformanceMetricsCache"]
