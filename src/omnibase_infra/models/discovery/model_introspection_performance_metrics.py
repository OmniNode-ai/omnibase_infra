# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Performance metrics model for node introspection operations.

This module provides a Pydantic model for capturing performance metrics from
introspection operations. It mirrors the dataclass IntrospectionPerformanceMetrics
from MixinNodeIntrospection but provides Pydantic serialization for event payloads.

The model enables:
- Performance monitoring of introspection operations
- Detection of slow operations exceeding the <50ms threshold
- Cache hit/miss tracking for performance optimization
- Serialization to JSON for event bus transmission

Related:
    - MixinNodeIntrospection.get_performance_metrics() - Returns the dataclass version
    - ModelNodeIntrospectionEvent - Contains this model as an optional field
    - OMN-926 - Ticket for adding performance metrics to introspection events
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelIntrospectionPerformanceMetrics(BaseModel):
    """Performance metrics captured during node introspection.

    This model provides timing information for introspection operations,
    enabling observability and alerting when operations exceed the
    <50ms target threshold defined in MixinNodeIntrospection.

    Attributes:
        get_capabilities_ms: Time taken by get_capabilities() in milliseconds.
        discover_capabilities_ms: Time taken by _discover_capabilities() in ms.
        get_endpoints_ms: Time taken by get_endpoints() in milliseconds.
        get_current_state_ms: Time taken by get_current_state() in milliseconds.
        total_introspection_ms: Total time for get_introspection_data() in ms.
        cache_hit: Whether the result was served from cache.
        method_count: Number of methods discovered during reflection.
        threshold_exceeded: Whether any operation exceeded performance thresholds.
        slow_operations: List of operation names that exceeded their thresholds.
        captured_at: UTC timestamp when metrics were captured.

    Example:
        ```python
        from omnibase_infra.models.discovery import (
            ModelIntrospectionPerformanceMetrics,
        )

        metrics = ModelIntrospectionPerformanceMetrics(
            get_capabilities_ms=12.5,
            discover_capabilities_ms=8.2,
            get_endpoints_ms=0.5,
            get_current_state_ms=0.1,
            total_introspection_ms=21.3,
            cache_hit=False,
            method_count=15,
            threshold_exceeded=False,
            slow_operations=[],
        )

        # Check if performance is degraded
        if metrics.threshold_exceeded:
            print(f"Slow operations: {metrics.slow_operations}")
        ```

    Performance Thresholds:
        The following thresholds are used (defined in MixinNodeIntrospection):
        - get_capabilities: <50ms
        - discover_capabilities: <30ms
        - total_introspection: <50ms
        - cache_hit: <1ms
    """

    # Timing metrics in milliseconds
    get_capabilities_ms: float = Field(
        default=0.0,
        description="Time taken by get_capabilities() in milliseconds",
        ge=0.0,
    )
    discover_capabilities_ms: float = Field(
        default=0.0,
        description="Time taken by _discover_capabilities() in milliseconds",
        ge=0.0,
    )
    get_endpoints_ms: float = Field(
        default=0.0,
        description="Time taken by get_endpoints() in milliseconds",
        ge=0.0,
    )
    get_current_state_ms: float = Field(
        default=0.0,
        description="Time taken by get_current_state() in milliseconds",
        ge=0.0,
    )
    total_introspection_ms: float = Field(
        default=0.0,
        description="Total time for get_introspection_data() in milliseconds",
        ge=0.0,
    )

    # Cache and discovery information
    cache_hit: bool = Field(
        default=False,
        description="Whether the result was served from cache",
    )
    method_count: int = Field(
        default=0,
        description="Number of methods discovered during reflection",
        ge=0,
    )

    # Threshold violation tracking
    threshold_exceeded: bool = Field(
        default=False,
        description="Whether any operation exceeded performance thresholds",
    )
    slow_operations: list[str] = Field(
        default_factory=list,
        description="List of operation names that exceeded their thresholds",
    )

    # Capture timestamp
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when metrics were captured",
    )

    # Frozen for immutability - metrics are snapshots
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
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
                },
                {
                    "get_capabilities_ms": 55.0,
                    "discover_capabilities_ms": 45.0,
                    "get_endpoints_ms": 0.2,
                    "get_current_state_ms": 0.1,
                    "total_introspection_ms": 100.3,
                    "cache_hit": False,
                    "method_count": 42,
                    "threshold_exceeded": True,
                    "slow_operations": [
                        "get_capabilities",
                        "discover_capabilities",
                        "total_introspection",
                    ],
                    "captured_at": "2025-01-15T10:31:00Z",
                },
            ]
        },
    )


__all__ = ["ModelIntrospectionPerformanceMetrics"]
