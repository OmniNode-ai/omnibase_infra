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

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.sinks.sink_metrics_prometheus import (
    DEFAULT_HISTOGRAM_BUCKETS,
)


class ModelMetricsSinkConfig(BaseModel):
    """Configuration model for Prometheus metrics sink creation.

    This model defines the configurable parameters for creating a
    SinkMetricsPrometheus instance. All fields have sensible defaults
    allowing zero-config usage.

    Note:
        The ModelMetricsPolicy (cardinality policy) is intentionally NOT included
        in this config model. Policy is passed separately to the sink constructor
        to allow runtime policy injection without serializing the full policy
        object into configuration. This enables dynamic policy updates and
        cleaner separation between static config and runtime behavior.

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

        # Policy is passed separately to the sink
        from omnibase_core.models.observability import ModelMetricsPolicy
        policy = ModelMetricsPolicy(...)
        sink = SinkMetricsPrometheus(
            policy=policy,
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


__all__: list[str] = [
    "ModelMetricsSinkConfig",
]
