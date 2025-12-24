# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Introspection performance metrics model.

This module provides ModelIntrospectionMetrics for capturing timing and
performance information during node introspection operations.

.. versionadded:: 0.6.0
    Created as part of OMN-1002 Union Reduction Phase 2 to replace the
    union pattern ``dict[str, float | bool | int | list[str]]`` with a
    strongly-typed Pydantic model.

Example:
    >>> metrics = ModelIntrospectionMetrics(
    ...     get_capabilities_ms=12.5,
    ...     total_introspection_ms=45.0,
    ...     method_count=15,
    ... )
    >>> metrics.threshold_exceeded
    False
    >>> metrics.to_dict()
    {'get_capabilities_ms': 12.5, ...}
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelIntrospectionMetrics(BaseModel):
    """Performance metrics for introspection operations.

    This Pydantic model captures timing information for introspection operations,
    enabling performance monitoring and alerting when operations exceed
    the <50ms target threshold.

    .. versionadded:: 0.6.0
        Created to replace union-typed dict return from dataclass.to_dict().

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

    Example:
        >>> from omnibase_infra.models.registration import ModelIntrospectionMetrics
        >>> metrics = ModelIntrospectionMetrics(
        ...     get_capabilities_ms=12.5,
        ...     discover_capabilities_ms=8.0,
        ...     get_endpoints_ms=2.0,
        ...     get_current_state_ms=1.0,
        ...     total_introspection_ms=25.0,
        ...     cache_hit=False,
        ...     method_count=15,
        ...     threshold_exceeded=False,
        ...     slow_operations=[],
        ... )
        >>> metrics.total_introspection_ms
        25.0
        >>> metrics.cache_hit
        False

        # Convert to dictionary for logging/serialization
        >>> metrics_dict = metrics.to_dict()
        >>> metrics_dict["method_count"]
        15

        # Create from existing dataclass
        >>> from dataclasses import dataclass, field
        >>> @dataclass
        ... class LegacyMetrics:
        ...     get_capabilities_ms: float = 0.0
        ...     discover_capabilities_ms: float = 0.0
        ...     get_endpoints_ms: float = 0.0
        ...     get_current_state_ms: float = 0.0
        ...     total_introspection_ms: float = 0.0
        ...     cache_hit: bool = False
        ...     method_count: int = 0
        ...     threshold_exceeded: bool = False
        ...     slow_operations: list[str] = field(default_factory=list)
        >>> legacy = LegacyMetrics(total_introspection_ms=30.0)
        >>> model = ModelIntrospectionMetrics.from_dataclass(legacy)
        >>> model.total_introspection_ms
        30.0
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Timing fields (all in milliseconds)
    get_capabilities_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken by get_capabilities() in milliseconds",
    )
    discover_capabilities_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken by _discover_capabilities() in milliseconds",
    )
    get_endpoints_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken by get_endpoints() in milliseconds",
    )
    get_current_state_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken by get_current_state() in milliseconds",
    )
    total_introspection_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time for get_introspection_data() in milliseconds",
    )

    # Status fields
    cache_hit: bool = Field(
        default=False,
        description="Whether the result was served from cache",
    )
    method_count: int = Field(
        default=0,
        ge=0,
        description="Number of methods discovered during reflection",
    )
    threshold_exceeded: bool = Field(
        default=False,
        description="Whether any operation exceeded performance thresholds",
    )

    # Operations that exceeded thresholds
    slow_operations: list[str] = Field(
        default_factory=list,
        description="List of operation names that exceeded their thresholds",
    )

    def to_dict(self) -> dict[str, object]:
        """Convert metrics to dictionary for logging/serialization.

        This method provides a dictionary output for JSON serialization
        and logging purposes. Uses ``object`` as the value type to avoid
        union complexity while maintaining type safety.

        For strongly-typed access to individual fields, use the model
        attributes directly (e.g., ``metrics.total_introspection_ms``).

        .. versionadded:: 0.6.0

        Returns:
            Dictionary with all metric fields. Uses ``object`` value type
            for simplicity; actual values are floats, bools, ints, or lists.

        Example:
            >>> metrics = ModelIntrospectionMetrics(
            ...     total_introspection_ms=25.0,
            ...     method_count=15,
            ... )
            >>> d = metrics.to_dict()
            >>> d["total_introspection_ms"]
            25.0
            >>> d["method_count"]
            15
        """
        return {
            "get_capabilities_ms": self.get_capabilities_ms,
            "discover_capabilities_ms": self.discover_capabilities_ms,
            "get_endpoints_ms": self.get_endpoints_ms,
            "get_current_state_ms": self.get_current_state_ms,
            "total_introspection_ms": self.total_introspection_ms,
            "cache_hit": self.cache_hit,
            "method_count": self.method_count,
            "threshold_exceeded": self.threshold_exceeded,
            "slow_operations": list(self.slow_operations),
        }

    @classmethod
    def from_dataclass(cls, metrics: object) -> ModelIntrospectionMetrics:
        """Create ModelIntrospectionMetrics from a dataclass instance.

        This factory method converts existing IntrospectionPerformanceMetrics
        dataclass instances to the new Pydantic model, enabling gradual
        migration without breaking existing code.

        .. versionadded:: 0.6.0

        Args:
            metrics: A dataclass instance with compatible fields. Expected to have
                the following attributes: get_capabilities_ms, discover_capabilities_ms,
                get_endpoints_ms, get_current_state_ms, total_introspection_ms,
                cache_hit, method_count, threshold_exceeded, slow_operations.

        Returns:
            A new ModelIntrospectionMetrics instance with values copied from
            the input dataclass.

        Raises:
            pydantic.ValidationError: If any attribute has an incompatible type.

        Note:
            Missing attributes fall back to sensible defaults (0.0 / 0 / False / []),
            making this factory tolerant of partially-populated legacy dataclasses.

        Example:
            >>> from dataclasses import dataclass, field
            >>> @dataclass
            ... class IntrospectionPerformanceMetrics:
            ...     get_capabilities_ms: float = 0.0
            ...     discover_capabilities_ms: float = 0.0
            ...     get_endpoints_ms: float = 0.0
            ...     get_current_state_ms: float = 0.0
            ...     total_introspection_ms: float = 0.0
            ...     cache_hit: bool = False
            ...     method_count: int = 0
            ...     threshold_exceeded: bool = False
            ...     slow_operations: list[str] = field(default_factory=list)
            >>> dc = IntrospectionPerformanceMetrics(
            ...     total_introspection_ms=42.0,
            ...     cache_hit=True,
            ... )
            >>> model = ModelIntrospectionMetrics.from_dataclass(dc)
            >>> model.total_introspection_ms
            42.0
            >>> model.cache_hit
            True
        """
        # Extract slow_operations with defensive copy to avoid mutation
        slow_ops = getattr(metrics, "slow_operations", [])
        slow_ops_copy = list(slow_ops) if slow_ops else []

        return cls(
            get_capabilities_ms=getattr(metrics, "get_capabilities_ms", 0.0),
            discover_capabilities_ms=getattr(metrics, "discover_capabilities_ms", 0.0),
            get_endpoints_ms=getattr(metrics, "get_endpoints_ms", 0.0),
            get_current_state_ms=getattr(metrics, "get_current_state_ms", 0.0),
            total_introspection_ms=getattr(metrics, "total_introspection_ms", 0.0),
            cache_hit=getattr(metrics, "cache_hit", False),
            method_count=getattr(metrics, "method_count", 0),
            threshold_exceeded=getattr(metrics, "threshold_exceeded", False),
            slow_operations=slow_ops_copy,
        )


__all__ = ["ModelIntrospectionMetrics"]
