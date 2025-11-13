#!/usr/bin/env python3
"""
EnumAggregationType - Aggregation Strategy Types.

Defines aggregation strategies for the NodeBridgeReducer to group
and reduce stamping metadata across different dimensions.

ONEX v2.0 Compliance:
- Suffix-based naming: EnumAggregationType
- O.N.E. v0.1 aggregation strategies
- Strong typing with comprehensive documentation
"""

from enum import Enum


class EnumAggregationType(str, Enum):
    """
    Aggregation strategy types for metadata reduction.

    Each type defines how stamp metadata should be grouped and aggregated:
    - NAMESPACE_GROUPING: Group by namespace (primary strategy)
    - TIME_WINDOW: Group by time windows (configurable duration)
    - FILE_TYPE_GROUPING: Group by content_type/file_type
    - SIZE_BUCKETS: Group by file size ranges
    - WORKFLOW_GROUPING: Group by workflow_id
    - CUSTOM: Custom aggregation strategy via configuration
    """

    NAMESPACE_GROUPING = "namespace_grouping"
    """Group stamps by namespace for multi-tenant aggregation."""

    TIME_WINDOW = "time_window"
    """Group stamps by time windows (e.g., 5s, 1m, 1h)."""

    FILE_TYPE_GROUPING = "file_type_grouping"
    """Group stamps by content_type for file type statistics."""

    SIZE_BUCKETS = "size_buckets"
    """Group stamps by file size ranges (small/medium/large)."""

    WORKFLOW_GROUPING = "workflow_grouping"
    """Group stamps by workflow_id for workflow-level aggregation."""

    CUSTOM = "custom"
    """Custom aggregation strategy defined in contract configuration."""

    def __str__(self) -> str:
        """Return string representation of the enum value."""
        return self.value
