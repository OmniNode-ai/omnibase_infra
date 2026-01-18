# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for Prometheus metrics sink.

This module defines the configuration model for creating SinkMetricsPrometheus
instances. The model validates configuration parameters and provides sensible
defaults for zero-config usage.

Usage:
    ```python
    from omnibase_infra.observability.models import ModelMetricsSinkConfig

    # Default configuration
    config = ModelMetricsSinkConfig()

    # Custom configuration
    config = ModelMetricsSinkConfig(
        metric_prefix="myservice",
        histogram_buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
    )
    ```
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import from dedicated constants module (ONEX pattern: constants in constants_*.py)
# This decouples config models from sink implementations, preventing circular imports
# and allowing config validation without requiring the full sink dependency chain.
from omnibase_infra.observability.constants_metrics import (
    DEFAULT_HISTOGRAM_BUCKETS,
)


class ModelMetricsSinkConfig(BaseModel):
    """Configuration model for Prometheus metrics sink creation.

    This model defines the configurable parameters for creating a
    SinkMetricsPrometheus instance. All fields have sensible defaults
    allowing zero-config usage.

    Note:
        This config model only contains static configuration (metric_prefix,
        histogram_buckets). Label cardinality enforcement is handled separately
        by the sink's runtime policy, which is passed directly to the
        SinkMetricsPrometheus constructor to enable dynamic policy updates.

    Attributes:
        metric_prefix: Optional prefix added to all metric names. Useful for
            namespacing metrics by service or component. Empty string means
            no prefix is added.
        histogram_buckets: Bucket boundaries for histogram metrics. Defaults
            to Prometheus-standard latency buckets suitable for request
            durations in seconds. Must be positive and monotonically increasing.

    Histogram Bucket Configuration:
        Customize histogram_buckets based on your operation characteristics:

        **API/HTTP Latencies (default)**: The default buckets (0.005s to 10s)
        are optimized for typical web service request durations where most
        requests complete in milliseconds but some may take seconds.

        **Sub-millisecond Operations**: For fast operations like cache lookups,
        in-memory computations, or high-frequency trading:
        ``(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05)``

        **Batch Processing**: For long-running jobs, ETL pipelines, or
        background tasks that may take minutes:
        ``(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0)``

        **SLA-Aligned**: Include buckets at your SLA thresholds for alerting.
        If your SLA is "99% of requests under 200ms", include a 0.2 bucket.

        Bucket boundaries are validated to be positive and strictly ascending.

    Example:
        ```python
        # Default configuration
        config = ModelMetricsSinkConfig()

        # Custom configuration
        config = ModelMetricsSinkConfig(
            metric_prefix="myservice",
            histogram_buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
        )

        # Apply config to sink
        sink = SinkMetricsPrometheus(
            metric_prefix=config.metric_prefix,
            histogram_buckets=config.histogram_buckets,
        )
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    metric_prefix: str = Field(
        default="",
        description="Optional prefix to add to all metric names.",
    )
    histogram_buckets: tuple[float, ...] = Field(
        default=DEFAULT_HISTOGRAM_BUCKETS,
        description=(
            "Bucket boundaries for histogram metrics in seconds. "
            "Customize based on your operation characteristics: "
            "API latencies (0.005-10s default), sub-millisecond operations "
            "(0.0001, 0.0005, 0.001, 0.005, 0.01), or batch processing "
            "(1.0, 5.0, 10.0, 30.0, 60.0, 300.0). Include buckets near "
            "SLA thresholds for effective alerting."
        ),
    )

    @field_validator("histogram_buckets")
    @classmethod
    def _validate_histogram_buckets(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        """Validate histogram bucket boundaries.

        Enforces Prometheus histogram requirements:
        1. All bucket values must be positive (> 0)
        2. Buckets must be in strictly ascending order (monotonicity)

        Args:
            v: Tuple of bucket boundary values.

        Returns:
            The validated bucket tuple.

        Raises:
            ValueError: If any bucket is non-positive or buckets are not monotonic.
        """
        if not v:
            raise ValueError("histogram_buckets cannot be empty")

        # Check positivity: all values must be > 0
        non_positive = [b for b in v if b <= 0]
        if non_positive:
            raise ValueError(
                f"histogram_buckets must all be positive (> 0), "
                f"found non-positive values: {non_positive}"
            )

        # Check monotonicity: buckets must be strictly ascending
        for i in range(1, len(v)):
            if v[i] <= v[i - 1]:
                raise ValueError(
                    f"histogram_buckets must be strictly ascending, "
                    f"found {v[i]} <= {v[i - 1]} at index {i}"
                )

        return v


__all__: list[str] = [
    "ModelMetricsSinkConfig",
]
