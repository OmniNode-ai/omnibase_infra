# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for registry metrics collection.

Defines the interface for optional metrics hooks that can be plugged
into RegistryCompute for production monitoring.

This protocol enables observability for compute plugin registries without
requiring any specific metrics backend. Implementations can integrate with:
- Prometheus/OpenMetrics
- StatsD/Datadog
- OpenTelemetry
- Custom monitoring solutions

Design Philosophy:
    All methods are optional (duck-typed). If a method is not implemented,
    the metric simply won't be recorded. This allows for partial implementations
    and gradual adoption of metrics collection.

Metrics Categories:
    - Latency: Operation timing for get(), register(), unregister()
    - Cache: Semver cache hit/miss rates for performance tuning
    - Size: Registry growth over time for capacity planning
    - Errors: Error frequency by type for alerting and debugging

Thread Safety:
    Implementations must be thread-safe as metrics may be recorded from
    concurrent registry operations.

Example Implementation:
    ```python
    from omnibase_infra.protocols import ProtocolRegistryMetrics
    from prometheus_client import Counter, Histogram, Gauge

    class PrometheusRegistryMetrics:
        '''Prometheus metrics collector for RegistryCompute.'''

        def __init__(self) -> None:
            self._get_latency = Histogram(
                'registry_compute_get_latency_ms',
                'Latency of get() operations in milliseconds',
                ['plugin_id', 'version_filter'],
            )
            self._cache_hits = Counter(
                'registry_compute_cache_hits_total',
                'Total semver cache hits',
            )
            self._cache_misses = Counter(
                'registry_compute_cache_misses_total',
                'Total semver cache misses',
            )
            self._registry_size = Gauge(
                'registry_compute_size',
                'Current number of registered plugins',
            )
            self._errors = Counter(
                'registry_compute_errors_total',
                'Total errors by type',
                ['error_type', 'plugin_id'],
            )

        def record_get_latency(
            self, plugin_id: str, version: str | None, latency_ms: float
        ) -> None:
            version_label = version if version else "latest"
            self._get_latency.labels(
                plugin_id=plugin_id, version_filter=version_label
            ).observe(latency_ms)

        def record_cache_hit(self) -> None:
            self._cache_hits.inc()

        def record_cache_miss(self) -> None:
            self._cache_misses.inc()

        def record_registry_size(self, size: int) -> None:
            self._registry_size.set(size)

        def record_error(
            self, error_type: str, plugin_id: str | None = None
        ) -> None:
            self._errors.labels(
                error_type=error_type, plugin_id=plugin_id or "unknown"
            ).inc()
    ```

Integration with RegistryCompute:
    ```python
    from omnibase_infra.runtime.registry_compute import RegistryCompute

    # Create registry with metrics
    metrics = PrometheusRegistryMetrics()
    registry = RegistryCompute(metrics_collector=metrics)

    # Or set after creation
    registry = RegistryCompute()
    registry.set_metrics_collector(metrics)
    ```

See Also:
    - src/omnibase_infra/runtime/registry_compute.py for integration
    - docs/patterns/observability_patterns.md for monitoring guidelines
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = [
    "ProtocolRegistryMetrics",
]


@runtime_checkable
class ProtocolRegistryMetrics(Protocol):
    """Protocol for registry metrics collection.

    Implementations can hook into registry operations to collect:
    - Operation latency (get, register, unregister)
    - Cache hit/miss rates
    - Registry size over time
    - Error frequency by type

    All methods are optional (duck-typed). If a method is not implemented,
    the metric simply won't be recorded.

    Thread Safety:
        Implementations MUST be thread-safe. Metrics may be recorded from
        concurrent registry operations protected by the registry's lock.

    Performance:
        Metrics recording should be lightweight (<1ms) to avoid impacting
        registry performance. Consider async or buffered recording for
        high-throughput scenarios.
    """

    def record_get_latency(
        self, plugin_id: str, version: str | None, latency_ms: float
    ) -> None:
        """Record latency for a get() operation.

        Called after each get() operation completes, regardless of success
        or failure. The latency includes the full operation time including
        any semver parsing and comparison.

        Args:
            plugin_id: The plugin ID that was looked up
            version: The version filter (None if latest version requested)
            latency_ms: Time in milliseconds for the operation

        Example:
            >>> # After a get() call taking 0.5ms
            >>> metrics.record_get_latency("json_normalizer", "1.0.0", 0.5)
            >>> # After a latest version lookup taking 1.2ms
            >>> metrics.record_get_latency("json_normalizer", None, 1.2)
        """
        ...

    def record_cache_hit(self) -> None:
        """Record a semver cache hit.

        Called when the semver parser finds a version string in the LRU cache.
        Useful for tuning cache size (SEMVER_CACHE_SIZE).

        High cache hit rates (>95%) indicate good cache sizing.
        Low hit rates may indicate cache is too small for the workload.
        """
        ...

    def record_cache_miss(self) -> None:
        """Record a semver cache miss.

        Called when the semver parser must parse a new version string
        not found in the LRU cache.
        """
        ...

    def record_registry_size(self, size: int) -> None:
        """Record current registry size.

        Called after registration and unregistration operations to track
        registry growth over time. Useful for capacity planning.

        Args:
            size: Number of registered plugins (plugin_id, version pairs)

        Example:
            >>> # After registering a new plugin
            >>> metrics.record_registry_size(42)
        """
        ...

    def record_error(self, error_type: str, plugin_id: str | None = None) -> None:
        """Record an error occurrence.

        Called when registry operations fail. Error types are standardized
        strings that can be used for alerting and debugging.

        Standard Error Types:
            - "not_found": Plugin ID not found in registry
            - "version_not_found": Specific version not found
            - "invalid_version": Invalid semver format during registration
            - "async_not_flagged": Async plugin without deterministic_async=True

        Args:
            error_type: Type of error (standardized string)
            plugin_id: Optional plugin ID associated with error

        Example:
            >>> # Plugin not found
            >>> metrics.record_error("not_found", "unknown_plugin")
            >>> # Invalid version format
            >>> metrics.record_error("invalid_version", "json_normalizer")
        """
        ...
