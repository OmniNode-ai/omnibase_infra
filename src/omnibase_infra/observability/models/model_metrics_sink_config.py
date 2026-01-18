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

from omnibase_infra.observability.sinks.sink_metrics_prometheus import (
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
            durations in seconds.

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

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_prefix: str = Field(
        default="",
        description="Optional prefix to add to all metric names.",
    )
    histogram_buckets: tuple[float, ...] = Field(
        default=DEFAULT_HISTOGRAM_BUCKETS,
        description="Bucket boundaries for histogram metrics in seconds.",
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
