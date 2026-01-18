# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for observability handlers and factory.

This module tests the observability handler lifecycle and factory integration:
- HandlerMetricsPrometheus lifecycle (initialize/shutdown)
- HandlerLoggingStructured lifecycle
- Factory creates working sink instances
- Handler execute operations

Handler Lifecycle:
    ONEX handlers follow a standard lifecycle:
    1. Construction: Handler created in uninitialized state
    2. initialize(): Configure handler with settings, start resources
    3. execute(): Process requests during handler lifetime
    4. shutdown(): Gracefully release resources, flush buffers

Factory Pattern:
    FactoryObservabilitySink provides:
    - Consistent sink creation with configuration validation
    - Singleton support for resource-efficient sink management
    - Hook creation with optional metrics sink injection
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.observability import FactoryObservabilitySink
from omnibase_infra.observability.factory_observability_sink import (
    ModelLoggingSinkConfig,
    ModelMetricsSinkConfig,
)
from omnibase_infra.observability.handlers import (
    HandlerLoggingStructured,
    HandlerMetricsPrometheus,
)

if TYPE_CHECKING:
    from omnibase_infra.observability.sinks import (
        SinkLoggingStructured,
        SinkMetricsPrometheus,
    )


# =============================================================================
# HANDLER METRICS PROMETHEUS TESTS
# =============================================================================


class TestHandlerMetricsPrometheus:
    """Test HandlerMetricsPrometheus lifecycle and operations."""

    @pytest.mark.asyncio
    async def test_initialize_with_default_config(self) -> None:
        """Verify handler initializes with default configuration."""
        handler = HandlerMetricsPrometheus()

        # Initialize without starting server (to avoid port conflicts)
        await handler.initialize({"enable_server": False})

        try:
            assert handler._initialized is True
            assert handler._config is not None
            assert handler._config.enable_server is False
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_config(self) -> None:
        """Verify handler accepts custom configuration."""
        handler = HandlerMetricsPrometheus()

        await handler.initialize(
            {
                "host": "127.0.0.1",
                "port": 19090,  # Non-standard port
                "path": "/custom_metrics",
                "enable_server": False,
            }
        )

        try:
            assert handler._config is not None
            assert handler._config.host == "127.0.0.1"
            assert handler._config.port == 19090
            assert handler._config.path == "/custom_metrics"
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_invalid_config_raises(self) -> None:
        """Verify invalid configuration raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError

        handler = HandlerMetricsPrometheus()

        with pytest.raises(ProtocolConfigurationError):
            await handler.initialize(
                {
                    "port": "not_a_number",  # Invalid port type
                }
            )

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self) -> None:
        """Verify shutdown can be called multiple times safely."""
        handler = HandlerMetricsPrometheus()

        await handler.initialize({"enable_server": False})
        await handler.shutdown()

        # Second shutdown should not raise
        await handler.shutdown()

        assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_execute_metrics_scrape_operation(self) -> None:
        """Verify metrics.scrape operation returns metrics text."""
        handler = HandlerMetricsPrometheus()

        await handler.initialize({"enable_server": False})

        try:
            result = await handler.execute(
                {
                    "operation": "metrics.scrape",
                    "correlation_id": str(uuid.uuid4()),
                    "envelope_id": str(uuid.uuid4()),
                }
            )

            assert result.result.status.value == "success"
            assert result.result.payload.operation_type == "metrics.scrape"
            # Metrics text should be present
            assert result.result.payload.metrics_text is not None
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises(self) -> None:
        """Verify execute raises if handler not initialized."""
        from omnibase_infra.errors import RuntimeHostError

        handler = HandlerMetricsPrometheus()

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(
                {
                    "operation": "metrics.scrape",
                }
            )

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_unsupported_operation_raises(self) -> None:
        """Verify unsupported operation raises error."""
        from omnibase_infra.errors import RuntimeHostError

        handler = HandlerMetricsPrometheus()
        await handler.initialize({"enable_server": False})

        try:
            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(
                    {
                        "operation": "invalid.operation",
                        "correlation_id": str(uuid.uuid4()),
                    }
                )

            assert "not supported" in str(exc_info.value).lower()
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_missing_operation_raises(self) -> None:
        """Verify missing operation raises error."""
        from omnibase_infra.errors import RuntimeHostError

        handler = HandlerMetricsPrometheus()
        await handler.initialize({"enable_server": False})

        try:
            with pytest.raises(RuntimeHostError):
                await handler.execute(
                    {
                        "correlation_id": str(uuid.uuid4()),
                        # Missing 'operation' field
                    }
                )
        finally:
            await handler.shutdown()

    def test_describe_returns_handler_metadata(self) -> None:
        """Verify describe returns handler metadata."""
        handler = HandlerMetricsPrometheus()

        metadata = handler.describe()

        assert "handler_type" in metadata
        assert "handler_category" in metadata
        assert "supported_operations" in metadata
        assert "initialized" in metadata
        assert metadata["initialized"] is False

    def test_handler_type_is_infra_handler(self) -> None:
        """Verify handler_type is INFRA_HANDLER."""
        from omnibase_infra.enums import EnumHandlerType

        handler = HandlerMetricsPrometheus()

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_is_effect(self) -> None:
        """Verify handler_category is EFFECT."""
        from omnibase_infra.enums import EnumHandlerTypeCategory

        handler = HandlerMetricsPrometheus()

        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# =============================================================================
# HANDLER LOGGING STRUCTURED TESTS
# =============================================================================


class TestHandlerLoggingStructured:
    """Test HandlerLoggingStructured lifecycle and operations."""

    @pytest.mark.asyncio
    async def test_initialize_with_default_config(self) -> None:
        """Verify handler initializes with default configuration."""
        handler = HandlerLoggingStructured()

        await handler.initialize(
            {
                "flush_interval_seconds": 0,  # Disable periodic flush for testing
            }
        )

        try:
            assert handler._initialized is True
            assert handler._sink is not None
            assert handler._config is not None
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_config(self) -> None:
        """Verify handler accepts custom configuration."""
        handler = HandlerLoggingStructured()

        await handler.initialize(
            {
                "buffer_size": 500,
                "flush_interval_seconds": 0,
                "output_format": "console",
            }
        )

        try:
            assert handler._config is not None
            assert handler._config.buffer_size == 500
            assert handler._config.output_format == "console"
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_twice_raises(self) -> None:
        """Verify initializing twice raises error."""
        from omnibase_infra.errors import RuntimeHostError

        handler = HandlerLoggingStructured()

        await handler.initialize({"flush_interval_seconds": 0})

        try:
            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.initialize({})

            assert "already initialized" in str(exc_info.value).lower()
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_flushes_buffer(self) -> None:
        """Verify shutdown flushes any remaining buffer entries."""
        handler = HandlerLoggingStructured()

        await handler.initialize(
            {
                "buffer_size": 100,
                "flush_interval_seconds": 0,
            }
        )

        # Emit some entries
        await handler.execute(
            {
                "operation": "logging.emit",
                "payload": {
                    "level": "INFO",
                    "message": "Test message",
                    "context": {},
                },
                "correlation_id": str(uuid.uuid4()),
            }
        )

        # Verify entry is in buffer before shutdown
        assert handler._sink is not None
        buffer_before = handler._sink.buffer_size
        assert buffer_before >= 1

        # Shutdown should flush
        await handler.shutdown()

        assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_execute_emit_operation(self) -> None:
        """Verify logging.emit operation buffers entry."""
        handler = HandlerLoggingStructured()

        await handler.initialize({"flush_interval_seconds": 0})

        try:
            result = await handler.execute(
                {
                    "operation": "logging.emit",
                    "payload": {
                        "level": "WARNING",
                        "message": "Test warning message",
                        "context": {"key": "value"},
                    },
                    "correlation_id": str(uuid.uuid4()),
                }
            )

            assert result.status.value == "success"
            assert result.operation == "logging.emit"
            assert result.buffer_size >= 1
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_flush_operation(self) -> None:
        """Verify logging.flush operation flushes buffer."""
        handler = HandlerLoggingStructured()

        await handler.initialize({"flush_interval_seconds": 0})

        try:
            # Emit some entries
            for i in range(5):
                await handler.execute(
                    {
                        "operation": "logging.emit",
                        "payload": {
                            "level": "DEBUG",
                            "message": f"Message {i}",
                            "context": {},
                        },
                        "correlation_id": str(uuid.uuid4()),
                    }
                )

            # Flush
            result = await handler.execute(
                {
                    "operation": "logging.flush",
                    "payload": {},
                    "correlation_id": str(uuid.uuid4()),
                }
            )

            assert result.status.value == "success"
            assert result.operation == "logging.flush"
            assert result.buffer_size == 0  # Buffer should be empty after flush
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises(self) -> None:
        """Verify execute raises if handler not initialized."""
        from omnibase_infra.errors import RuntimeHostError

        handler = HandlerLoggingStructured()

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(
                {
                    "operation": "logging.emit",
                    "payload": {},
                }
            )

        assert "not initialized" in str(exc_info.value).lower()

    def test_describe_returns_handler_metadata(self) -> None:
        """Verify describe returns handler metadata."""
        handler = HandlerLoggingStructured()

        metadata = handler.describe()

        assert "handler_type" in metadata
        assert "handler_category" in metadata
        assert "supported_operations" in metadata
        assert metadata["initialized"] is False

    def test_handler_type_is_infra_handler(self) -> None:
        """Verify handler_type is INFRA_HANDLER."""
        from omnibase_infra.enums import EnumHandlerType

        handler = HandlerLoggingStructured()

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_is_effect(self) -> None:
        """Verify handler_category is EFFECT."""
        from omnibase_infra.enums import EnumHandlerTypeCategory

        handler = HandlerLoggingStructured()

        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# =============================================================================
# FACTORY TESTS
# =============================================================================


class TestFactoryObservabilitySink:
    """Test FactoryObservabilitySink creation and management."""

    def test_create_metrics_sink_default_config(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify factory creates metrics sink with defaults."""
        sink = factory.create_metrics_sink()

        assert sink is not None
        assert sink._metric_prefix == ""

    def test_create_metrics_sink_custom_config(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify factory creates metrics sink with custom config."""
        config = ModelMetricsSinkConfig(
            metric_prefix="custom_prefix",
            histogram_buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
        )

        sink = factory.create_metrics_sink(config=config)

        assert sink._metric_prefix == "custom_prefix"
        assert sink._histogram_buckets == (0.01, 0.05, 0.1, 0.5, 1.0)

    def test_create_logging_sink_default_config(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify factory creates logging sink with defaults."""
        sink = factory.create_logging_sink()

        assert sink is not None
        assert sink.max_buffer_size == 1000
        assert sink.output_format == "json"

    def test_create_logging_sink_custom_config(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify factory creates logging sink with custom config."""
        config = ModelLoggingSinkConfig(
            max_buffer_size=500,
            output_format="console",
        )

        sink = factory.create_logging_sink(config=config)

        assert sink.max_buffer_size == 500
        assert sink.output_format == "console"

    def test_get_or_create_metrics_sink_returns_singleton(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify get_or_create returns same instance (singleton)."""
        sink1 = factory.get_or_create_metrics_sink()
        sink2 = factory.get_or_create_metrics_sink()

        assert sink1 is sink2

    def test_get_or_create_logging_sink_returns_singleton(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify get_or_create returns same instance (singleton)."""
        sink1 = factory.get_or_create_logging_sink()
        sink2 = factory.get_or_create_logging_sink()

        assert sink1 is sink2

    def test_create_hook_without_metrics_sink(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify factory creates hook without metrics sink."""
        hook = factory.create_hook(metrics_sink=None)

        assert hook is not None
        assert hook.metrics_sink is None

    def test_create_hook_with_metrics_sink(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify factory creates hook with metrics sink."""
        metrics_sink = factory.create_metrics_sink()
        hook = factory.create_hook(metrics_sink=metrics_sink)

        assert hook.metrics_sink is metrics_sink

    def test_create_hook_with_singleton_metrics(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify create_hook_with_singleton_metrics uses singleton."""
        hook1 = factory.create_hook_with_singleton_metrics()
        hook2 = factory.create_hook_with_singleton_metrics()

        # Both hooks should use the same singleton metrics sink
        assert hook1.metrics_sink is hook2.metrics_sink

    def test_clear_singletons(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify clear_singletons removes cached instances."""
        sink1 = factory.get_or_create_metrics_sink()
        factory.clear_singletons()
        sink2 = factory.get_or_create_metrics_sink()

        assert sink1 is not sink2

    def test_has_metrics_singleton(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify has_metrics_singleton reports correctly."""
        assert factory.has_metrics_singleton() is False

        factory.get_or_create_metrics_sink()

        assert factory.has_metrics_singleton() is True

    def test_has_logging_singleton(
        self,
        factory: FactoryObservabilitySink,
    ) -> None:
        """Verify has_logging_singleton reports correctly."""
        assert factory.has_logging_singleton() is False

        factory.get_or_create_logging_sink()

        assert factory.has_logging_singleton() is True


class TestFactoryThreadSafety:
    """Test factory thread-safety for singleton creation."""

    def test_concurrent_singleton_creation_returns_same_instance(self) -> None:
        """Verify concurrent singleton creation returns same instance."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        factory = FactoryObservabilitySink()
        sinks: list[SinkMetricsPrometheus] = []
        lock = threading.Lock()

        def get_singleton() -> None:
            """Get singleton and record it."""
            sink = factory.get_or_create_metrics_sink()
            with lock:
                sinks.append(sink)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_singleton) for _ in range(20)]
            for f in as_completed(futures):
                f.result()

        # All should be the same instance
        first_sink = sinks[0]
        for sink in sinks:
            assert sink is first_sink

        factory.clear_singletons()


# =============================================================================
# HANDLER PERIODIC FLUSH TESTS
# =============================================================================


class TestHandlerPeriodicFlush:
    """Test periodic flush behavior in logging handler."""

    @pytest.mark.asyncio
    async def test_periodic_flush_task_created(self) -> None:
        """Verify periodic flush task is created when interval > 0."""
        handler = HandlerLoggingStructured()

        await handler.initialize(
            {
                "flush_interval_seconds": 1.0,
            }
        )

        try:
            assert handler._flush_task is not None
            assert not handler._flush_task.done()
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_periodic_flush_disabled_when_interval_zero(self) -> None:
        """Verify no flush task when interval is 0."""
        handler = HandlerLoggingStructured()

        await handler.initialize(
            {
                "flush_interval_seconds": 0,
            }
        )

        try:
            assert handler._flush_task is None
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_flush_task(self) -> None:
        """Verify shutdown cancels periodic flush task."""
        handler = HandlerLoggingStructured()

        await handler.initialize(
            {
                "flush_interval_seconds": 10.0,  # Long interval
            }
        )

        flush_task = handler._flush_task
        assert flush_task is not None

        await handler.shutdown()

        # Task should be cancelled or done
        assert flush_task.done() or flush_task.cancelled()
