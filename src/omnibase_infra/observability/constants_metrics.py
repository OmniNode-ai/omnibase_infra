# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Metrics constants for ONEX observability.

This module provides shared constants for metrics sinks and handlers.
Constants are defined here to avoid circular imports and to provide a
single source of truth for default values.

Import Pattern:
    Constants should be imported directly from this module, not from
    sink or handler __init__.py files:

    ```python
    # CORRECT: Import from constants module
    from omnibase_infra.observability.constants_metrics import DEFAULT_HISTOGRAM_BUCKETS

    # INCORRECT: Import via __init__.py re-export
    from omnibase_infra.observability.sinks import DEFAULT_HISTOGRAM_BUCKETS
    ```
"""

# Default histogram buckets following Prometheus conventions.
# Suitable for request latencies in seconds.
#
# Bucket Configuration Guide:
# ---------------------------
# These default buckets are optimized for typical HTTP/API request latencies:
#   - Fast operations (5-100ms): 0.005, 0.01, 0.025, 0.05, 0.1
#   - Normal operations (100ms-1s): 0.25, 0.5, 1.0
#   - Slow operations (1-10s): 2.5, 5.0, 10.0
#
# Expected Use Cases:
#   1. HTTP handler execution times
#   2. Database query latencies
#   3. External API call durations
#   4. Message processing times
#
# Custom Bucket Guidelines:
#   - For faster operations (memory caches, local computations):
#     Use smaller buckets: (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
#   - For slow batch operations (file I/O, large data transfers):
#     Use larger buckets: (0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
#   - General rule: Include buckets near your SLA thresholds for alerting
#
# Example custom buckets for sub-millisecond operations:
#   histogram_buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05)
DEFAULT_HISTOGRAM_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

# Histogram buckets for scrape duration (in seconds).
# Covers typical scrape times from 1ms to 5s timeout.
#
# Bucket Configuration for Prometheus Scrape Metrics:
# ---------------------------------------------------
# These buckets are optimized for tracking the time spent generating
# Prometheus metrics output during /metrics endpoint scrapes.
#
# Expected Scrape Duration Ranges:
#   - Healthy (1-50ms): Small metric registries, efficient serialization
#   - Normal (50-250ms): Medium registries, typical production workloads
#   - Slow (250ms-1s): Large registries, many high-cardinality metrics
#   - Critical (1-5s): Metrics generation approaching timeout threshold
SCRAPE_DURATION_BUCKETS: tuple[float, ...] = (
    0.001,  # 1ms - Very fast scrapes (small registries)
    0.005,  # 5ms
    0.010,  # 10ms
    0.025,  # 25ms
    0.050,  # 50ms - Healthy upper bound
    0.100,  # 100ms
    0.250,  # 250ms - Normal upper bound
    0.500,  # 500ms
    1.000,  # 1s - Warning threshold
    2.500,  # 2.5s - Critical threshold
    5.000,  # 5s - Timeout threshold
)

__all__: list[str] = [
    "DEFAULT_HISTOGRAM_BUCKETS",
    "SCRAPE_DURATION_BUCKETS",
]
