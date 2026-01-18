# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Factory for creating observability sinks.

This module provides a factory for creating observability infrastructure components
including metrics sinks, logging sinks, and observability hooks. The factory
supports configuration-based creation and optional singleton behavior for
resource-efficient sink management.

Factory Pattern Benefits:
    - Centralized sink creation with consistent configuration
    - Singleton support for shared sink instances across components
    - Configuration validation before sink creation
    - Easy testing via factory injection

Configuration Models:
    The factory defines two configuration models for sink creation:
    - ModelMetricsSinkConfig: Configuration for Prometheus metrics sinks
    - ModelLoggingSinkConfig: Configuration for structured logging sinks

    These models can be populated from ONEX contract subcontracts or created
    directly with sensible defaults.

Usage Example:
    ```python
    from omnibase_infra.observability import FactoryObservabilitySink

    # Create factory instance
    factory = FactoryObservabilitySink()

    # Create sinks with defaults
    metrics_sink = factory.create_metrics_sink()
    logging_sink = factory.create_logging_sink()
    hook = factory.create_hook(metrics_sink=metrics_sink)

    # Or use singleton pattern for shared instances
    metrics_sink_1 = factory.get_or_create_metrics_sink()
    metrics_sink_2 = factory.get_or_create_metrics_sink()
    assert metrics_sink_1 is metrics_sink_2  # Same instance

    # With custom configuration
    from omnibase_infra.observability.models import (
        ModelMetricsSinkConfig,
        ModelLoggingSinkConfig,
    )

    metrics_config = ModelMetricsSinkConfig(
        metric_prefix="myapp",
        histogram_buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    )
    custom_metrics_sink = factory.create_metrics_sink(config=metrics_config)

    logging_config = ModelLoggingSinkConfig(
        max_buffer_size=2000,
        output_format="console",
    )
    custom_logging_sink = factory.create_logging_sink(config=logging_config)
    ```

See Also:
    - SinkMetricsPrometheus: Prometheus metrics sink implementation
    - SinkLoggingStructured: Structured logging sink implementation
    - HookObservability: Observability pipeline hook
    - omnibase_core.models.observability.ModelMetricsPolicy: Cardinality policy
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from omnibase_infra.observability.hooks import HookObservability
from omnibase_infra.observability.models.model_logging_sink_config import (
    ModelLoggingSinkConfig,
)
from omnibase_infra.observability.models.model_metrics_sink_config import (
    ModelMetricsSinkConfig,
)
from omnibase_infra.observability.sinks import (
    SinkLoggingStructured,
    SinkMetricsPrometheus,
)

if TYPE_CHECKING:
    from omnibase_core.models.observability import ModelMetricsPolicy
    from omnibase_spi.protocols.observability import ProtocolHotPathMetricsSink

_logger = logging.getLogger(__name__)


class FactoryObservabilitySink:
    """Factory for creating observability sinks and hooks.

    This factory provides centralized creation of observability components
    with support for configuration-based instantiation and singleton patterns.
    It ensures consistent configuration across all observability infrastructure
    and enables easy testing through factory injection.

    Singleton Support:
        The factory maintains optional singleton instances for metrics and
        logging sinks. Use get_or_create_* methods to leverage singleton
        behavior, or create_* methods for independent instances.

        Singletons are useful when:
        - Multiple components share the same metrics endpoint
        - You want to minimize Prometheus metric registry duplication
        - Resource efficiency is important

    Thread Safety:
        The factory uses a threading.Lock to ensure thread-safe singleton
        creation. Multiple threads calling get_or_create_* concurrently
        will receive the same singleton instance.

    Warning - Test Isolation:
        Singleton state persists across tests when using the same factory
        instance. This can cause test pollution where state from one test
        affects another. To ensure test isolation:

        1. Call clear_singletons() in test fixtures (setup/teardown)
        2. Create a fresh factory instance per test
        3. Use create_* methods instead of get_or_create_* for isolation

        Example pytest fixture::

            @pytest.fixture
            def observability_factory():
                factory = FactoryObservabilitySink()
                yield factory
                factory.clear_singletons()  # Clean up after test

    Attributes:
        _metrics_sink_instance: Cached singleton metrics sink (or None).
        _logging_sink_instance: Cached singleton logging sink (or None).
        _lock: Threading lock for thread-safe singleton creation.

    Example:
        ```python
        # Create factory
        factory = FactoryObservabilitySink()

        # Independent instances (new each time)
        sink1 = factory.create_metrics_sink()
        sink2 = factory.create_metrics_sink()
        assert sink1 is not sink2

        # Singleton instances (same each time)
        sink3 = factory.get_or_create_metrics_sink()
        sink4 = factory.get_or_create_metrics_sink()
        assert sink3 is sink4

        # Create hook with metrics sink
        hook = factory.create_hook(metrics_sink=sink3)
        ```
    """

    def __init__(self) -> None:
        """Initialize the observability sink factory.

        Creates an empty factory with no cached singleton instances.
        Singletons are created lazily on first get_or_create_* call.
        """
        self._metrics_sink_instance: SinkMetricsPrometheus | None = None
        self._logging_sink_instance: SinkLoggingStructured | None = None
        self._lock = threading.Lock()

        _logger.debug("FactoryObservabilitySink initialized")

    # =========================================================================
    # METRICS SINK CREATION
    # =========================================================================

    def create_metrics_sink(
        self,
        config: ModelMetricsSinkConfig | None = None,
        policy: ModelMetricsPolicy | None = None,
    ) -> SinkMetricsPrometheus:
        """Create a new Prometheus metrics sink instance.

        Creates a fresh SinkMetricsPrometheus instance with the provided
        configuration. Each call creates an independent sink - use
        get_or_create_metrics_sink() for singleton behavior.

        Args:
            config: Optional configuration for the metrics sink. If None,
                default configuration is used (no prefix, standard buckets).
            policy: Optional cardinality policy for label validation. If None,
                the sink uses its default policy that forbids high-cardinality
                labels and warns on violations.

        Returns:
            A new SinkMetricsPrometheus instance configured according to
            the provided parameters.

        Example:
            ```python
            # With defaults
            sink = factory.create_metrics_sink()

            # With custom config
            config = ModelMetricsSinkConfig(metric_prefix="api")
            sink = factory.create_metrics_sink(config=config)

            # With custom policy
            from omnibase_core.models.observability import ModelMetricsPolicy
            policy = ModelMetricsPolicy(max_label_value_length=64)
            sink = factory.create_metrics_sink(policy=policy)
            ```
        """
        effective_config = config or ModelMetricsSinkConfig()

        _logger.debug(
            "Creating new SinkMetricsPrometheus",
            extra={
                "metric_prefix": effective_config.metric_prefix,
                "histogram_buckets": effective_config.histogram_buckets,
                "has_custom_policy": policy is not None,
            },
        )

        return SinkMetricsPrometheus(
            policy=policy,
            histogram_buckets=effective_config.histogram_buckets,
            metric_prefix=effective_config.metric_prefix,
        )

    def get_or_create_metrics_sink(
        self,
        config: ModelMetricsSinkConfig | None = None,
        policy: ModelMetricsPolicy | None = None,
    ) -> SinkMetricsPrometheus:
        """Get existing or create new singleton metrics sink.

        Returns the cached singleton metrics sink if one exists, otherwise
        creates a new one with the provided configuration and caches it.

        Thread Safety:
            This method is thread-safe. Multiple concurrent calls will
            receive the same singleton instance.

        Note:
            Configuration and policy are only used on first call when the
            singleton is created. Subsequent calls ignore these parameters
            and return the existing instance. To create a sink with different
            configuration, use create_metrics_sink() instead.

        Args:
            config: Optional configuration for sink creation. Only used if
                no singleton exists yet.
            policy: Optional cardinality policy for sink creation. Only used
                if no singleton exists yet.

        Returns:
            The singleton SinkMetricsPrometheus instance.

        Example:
            ```python
            # First call creates the singleton
            sink1 = factory.get_or_create_metrics_sink(
                config=ModelMetricsSinkConfig(metric_prefix="myapp")
            )

            # Subsequent calls return the same instance
            sink2 = factory.get_or_create_metrics_sink()
            assert sink1 is sink2
            ```
        """
        with self._lock:
            if self._metrics_sink_instance is None:
                self._metrics_sink_instance = self.create_metrics_sink(
                    config=config,
                    policy=policy,
                )
                _logger.debug("Created singleton SinkMetricsPrometheus")
            elif config is not None or policy is not None:
                _logger.debug(
                    "Returning existing singleton SinkMetricsPrometheus; "
                    "provided config/policy ignored",
                    extra={
                        "config_provided": config is not None,
                        "policy_provided": policy is not None,
                    },
                )
            return self._metrics_sink_instance

    # =========================================================================
    # LOGGING SINK CREATION
    # =========================================================================

    def create_logging_sink(
        self,
        config: ModelLoggingSinkConfig | None = None,
    ) -> SinkLoggingStructured:
        """Create a new structured logging sink instance.

        Creates a fresh SinkLoggingStructured instance with the provided
        configuration. Each call creates an independent sink - use
        get_or_create_logging_sink() for singleton behavior.

        Args:
            config: Optional configuration for the logging sink. If None,
                default configuration is used (1000 buffer, JSON output).

        Returns:
            A new SinkLoggingStructured instance configured according to
            the provided parameters.

        Example:
            ```python
            # With defaults
            sink = factory.create_logging_sink()

            # With custom config
            config = ModelLoggingSinkConfig(
                max_buffer_size=5000,
                output_format="console",
            )
            sink = factory.create_logging_sink(config=config)
            ```
        """
        effective_config = config or ModelLoggingSinkConfig()

        _logger.debug(
            "Creating new SinkLoggingStructured",
            extra={
                "max_buffer_size": effective_config.max_buffer_size,
                "output_format": effective_config.output_format,
            },
        )

        return SinkLoggingStructured(
            max_buffer_size=effective_config.max_buffer_size,
            output_format=effective_config.output_format,
        )

    def get_or_create_logging_sink(
        self,
        config: ModelLoggingSinkConfig | None = None,
    ) -> SinkLoggingStructured:
        """Get existing or create new singleton logging sink.

        Returns the cached singleton logging sink if one exists, otherwise
        creates a new one with the provided configuration and caches it.

        Thread Safety:
            This method is thread-safe. Multiple concurrent calls will
            receive the same singleton instance.

        Note:
            Configuration is only used on first call when the singleton is
            created. Subsequent calls ignore the config parameter and return
            the existing instance. To create a sink with different
            configuration, use create_logging_sink() instead.

        Args:
            config: Optional configuration for sink creation. Only used if
                no singleton exists yet.

        Returns:
            The singleton SinkLoggingStructured instance.

        Example:
            ```python
            # First call creates the singleton
            sink1 = factory.get_or_create_logging_sink(
                config=ModelLoggingSinkConfig(max_buffer_size=5000)
            )

            # Subsequent calls return the same instance
            sink2 = factory.get_or_create_logging_sink()
            assert sink1 is sink2
            ```
        """
        with self._lock:
            if self._logging_sink_instance is None:
                self._logging_sink_instance = self.create_logging_sink(config=config)
                _logger.debug("Created singleton SinkLoggingStructured")
            elif config is not None:
                _logger.debug(
                    "Returning existing singleton SinkLoggingStructured; "
                    "provided config ignored",
                )
            return self._logging_sink_instance

    # =========================================================================
    # HOOK CREATION
    # =========================================================================

    def create_hook(
        self,
        metrics_sink: ProtocolHotPathMetricsSink | None = None,
    ) -> HookObservability:
        """Create an observability hook with optional metrics sink.

        Creates a HookObservability instance configured with an optional
        metrics sink. The hook provides timing, context tracking, and
        metrics emission for infrastructure operations.

        Args:
            metrics_sink: Optional metrics sink for emitting observability
                data. If None, the hook operates in timing-only mode
                (metrics emission is a no-op). Pass a SinkMetricsPrometheus
                instance to enable full metrics collection.

        Returns:
            A new HookObservability instance.

        Example:
            ```python
            # Hook without metrics (timing only)
            hook = factory.create_hook()

            # Hook with metrics sink
            metrics_sink = factory.get_or_create_metrics_sink()
            hook = factory.create_hook(metrics_sink=metrics_sink)

            # Usage
            with hook.operation_context("handler.process"):
                result = await handler.execute()
            ```
        """
        _logger.debug(
            "Creating HookObservability",
            extra={"has_metrics_sink": metrics_sink is not None},
        )

        return HookObservability(metrics_sink=metrics_sink)

    def create_hook_with_singleton_metrics(
        self,
        metrics_config: ModelMetricsSinkConfig | None = None,
        metrics_policy: ModelMetricsPolicy | None = None,
    ) -> HookObservability:
        """Create an observability hook using the singleton metrics sink.

        Convenience method that creates a hook wired to the singleton
        metrics sink. If no singleton exists, one is created with the
        provided configuration.

        This is the recommended way to create hooks when you want all
        components to share the same metrics endpoint.

        Args:
            metrics_config: Optional configuration for metrics sink creation.
                Only used if no singleton metrics sink exists yet.
            metrics_policy: Optional cardinality policy for metrics sink
                creation. Only used if no singleton metrics sink exists yet.

        Returns:
            A new HookObservability instance configured with the singleton
            metrics sink.

        Example:
            ```python
            # All hooks share the same metrics sink
            hook1 = factory.create_hook_with_singleton_metrics()
            hook2 = factory.create_hook_with_singleton_metrics()
            # Both hooks emit to the same Prometheus metrics
            ```
        """
        metrics_sink = self.get_or_create_metrics_sink(
            config=metrics_config,
            policy=metrics_policy,
        )
        return self.create_hook(metrics_sink=metrics_sink)

    # =========================================================================
    # MANAGEMENT METHODS
    # =========================================================================

    def clear_singletons(self) -> None:
        """Clear all cached singleton instances.

        Removes references to cached singleton sinks, allowing them to be
        garbage collected if no other references exist. Subsequent calls
        to get_or_create_* methods will create new instances.

        Use Cases:
            - Testing: Reset factory state between tests
            - Reconfiguration: Allow new singletons with different config
            - Memory management: Release unused sink resources

        Thread Safety:
            This method is thread-safe.

        Warning:
            Existing references to the old singletons remain valid. Only
            future get_or_create_* calls will create new instances.

        Example:
            ```python
            sink1 = factory.get_or_create_metrics_sink()
            factory.clear_singletons()
            sink2 = factory.get_or_create_metrics_sink()
            assert sink1 is not sink2  # New instance
            # sink1 still works but is no longer the singleton
            ```
        """
        with self._lock:
            self._metrics_sink_instance = None
            self._logging_sink_instance = None
            _logger.debug("Cleared all singleton instances")

    def has_metrics_singleton(self) -> bool:
        """Check if a singleton metrics sink exists.

        Returns:
            True if a singleton metrics sink has been created, False otherwise.
        """
        with self._lock:
            return self._metrics_sink_instance is not None

    def has_logging_singleton(self) -> bool:
        """Check if a singleton logging sink exists.

        Returns:
            True if a singleton logging sink has been created, False otherwise.
        """
        with self._lock:
            return self._logging_sink_instance is not None


__all__ = [
    "FactoryObservabilitySink",
]
