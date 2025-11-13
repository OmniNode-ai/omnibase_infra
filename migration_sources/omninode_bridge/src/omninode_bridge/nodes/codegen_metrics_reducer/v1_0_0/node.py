#!/usr/bin/env python3
"""
NodeCodegenMetricsReducer - Code Generation Metrics Aggregator.

Aggregates code generation metrics from event streams for analytics,
monitoring, and trend analysis. Part of the omninode_bridge MVP.

ONEX v2.0 Compliance:
- Suffix-based naming: NodeCodegenMetricsReducer
- Import from omnibase_core infrastructure
- Subcontract composition for aggregation
- ModelContainer for dependency injection
- Pure aggregation logic with async streaming

Performance Targets:
- >1000 events/second aggregation throughput
- <100ms aggregation latency for 1000 items
- Streaming aggregation with windowing
"""

import os
import time
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_reducer import ModelContractReducer
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_reducer import NodeReducer
from pydantic import ValidationError

from omninode_bridge.events.models.codegen_events import (
    TOPIC_CODEGEN_METRICS_RECORDED,
    ModelEventCodegenCompleted,
    ModelEventCodegenFailed,
    ModelEventCodegenStageCompleted,
    ModelEventCodegenStarted,
    ModelEventMetricsRecorded,
)
from omninode_bridge.mixins import MixinIntentPublisher

# TypeAlias for codegen event union
CodegenEvent = (
    ModelEventCodegenStarted
    | ModelEventCodegenStageCompleted
    | ModelEventCodegenCompleted
    | ModelEventCodegenFailed
)

# Import aggregator and models
from .aggregator import MetricsAggregator
from .models.enum_metrics_window import EnumMetricsWindow
from .models.model_metrics_state import ModelMetricsState


class NodeCodegenMetricsReducer(NodeReducer, MixinIntentPublisher):
    """
    Code generation metrics reducer for streaming aggregation.

    Consumes code generation events and aggregates metrics:
    1. Consume CODEGEN_* events from Kafka
    2. Aggregate by time window (hourly/daily/weekly)
    3. Compute performance, quality, and cost metrics
    4. Publish intent to publish GENERATION_METRICS_RECORDED events
    5. Return aggregated metrics state

    Architecture:
    - Pure aggregation logic (MetricsAggregator)
    - Coordination I/O via MixinIntentPublisher
    - Intent executor publishes final events via EFFECT

    Mixins:
    - MixinIntentPublisher: Provides publish_event_intent() for coordination I/O

    Subcontracts:
    - intent_publisher.yaml: Intent publishing capability

    Performance:
    - >1000 events/second throughput
    - <100ms aggregation latency
    - Streaming with windowed aggregation
    """

    def __init__(self, container: ModelContainer) -> None:
        """
        Initialize NodeCodegenMetricsReducer with dependency injection container.

        Args:
            container: ONEX DI container with service dependencies
        """
        super().__init__(container)

        # Initialize MixinIntentPublisher capability
        self._init_intent_publisher(container)

        # Metrics aggregator (pure logic, no I/O)
        self.aggregator = MetricsAggregator()

        # Configuration
        self.default_window_type = EnumMetricsWindow(
            container.config.get("metrics_window_type", "hourly")
        )
        self.batch_size = container.config.get("metrics_batch_size", 100)

        # Consul configuration for service discovery
        self.consul_host: str = container.config.get(
            "consul_host", os.getenv("CONSUL_HOST", "omninode-bridge-consul")
        )
        self.consul_port: int = container.config.get(
            "consul_port", int(os.getenv("CONSUL_PORT", "28500"))
        )
        self.consul_enable_registration: bool = container.config.get(
            "consul_enable_registration", True
        )

        # Internal state for windowed aggregation
        self._current_window_events: list[CodegenEvent] = []
        self._window_start_time: Optional[datetime] = None

        # Register with Consul for service discovery (skip in health check mode)
        health_check_mode = container.config.get("health_check_mode", False)
        if not health_check_mode and self.consul_enable_registration:
            self._register_with_consul_sync()

    async def execute_reduction(
        self,
        contract: ModelContractReducer,
    ) -> ModelMetricsState:
        """
        Execute pure metrics aggregation from event stream.

        PURE DOMAIN LOGIC - COORDINATION I/O VIA INTENT PATTERN
        This method performs pure data aggregation and publishes coordination
        intents (not direct domain I/O). Actual event publishing is delegated
        to IntentExecutor EFFECT nodes for separation of concerns.

        Intent Pattern Architecture:
            1. Aggregate events (pure domain logic)
            2. Build event payload (pure data construction)
            3. Publish intent to coordination topic (MixinIntentPublisher)
            4. IntentExecutor EFFECT consumes intent → publishes domain event

        Aggregation Strategy:
        1. Stream events from Kafka consumer (async iterator)
        2. Group events by time window
        3. Aggregate using MetricsAggregator (pure logic)
        4. Publish coordination intent for event publishing
        5. Return aggregated metrics state

        Args:
            contract: Reducer contract with streaming configuration

        Returns:
            ModelMetricsState with aggregated metrics

        Raises:
            ModelOnexError: If reduction fails or validation errors occur
        """
        start_time = time.perf_counter()
        correlation_id = getattr(contract, "correlation_id", uuid4())

        try:
            # Extract window configuration from contract
            window_type = self._get_window_type(contract)
            batch_size = self._get_batch_size(contract)

            # Collect all events for aggregation
            all_events: list[CodegenEvent] = []

            # Stream events from contract input
            async for event_batch in self._stream_events(
                contract, batch_size=batch_size
            ):
                all_events.extend(event_batch)

            # Aggregate using pure logic
            metrics_state = self.aggregator.aggregate_events(
                events=all_events,
                window_type=window_type,
            )

            # Calculate aggregation performance
            duration_ms = (time.perf_counter() - start_time) * 1000
            items_per_second = (
                len(all_events) / (duration_ms / 1000) if duration_ms > 0 else 0.0
            )

            # Update performance metrics
            metrics_state.aggregation_duration_ms = duration_ms
            metrics_state.items_per_second = items_per_second

            # Publish intent to publish metrics event (coordination I/O via MixinIntentPublisher)
            # Intent will be consumed by IntentExecutor EFFECT and published to domain topic
            #
            # Design rationale for Intent Pattern over alternatives:
            # 1. Returning tuple (state, event): Breaks single responsibility, couples concerns
            # 2. Adding pending_events to state: Pollutes domain model with coordination data
            # 3. Intent Pattern (current): Clean separation - domain logic stays pure, coordination
            #    is explicit via mixin, execution is delegated to EFFECT nodes
            #
            # This allows:
            # - Pure unit testing without Kafka mocks
            # - Reusable domain logic across different execution contexts
            # - Observable coordination via Kafka intent topic
            # - Independent retry/recovery of intent execution
            metrics_event = self._build_metrics_recorded_event(metrics_state)
            await self.publish_event_intent(
                target_topic=TOPIC_CODEGEN_METRICS_RECORDED,
                target_key=str(metrics_state.aggregation_id),
                event=metrics_event,
            )

            return metrics_state

        except Exception as e:
            raise ModelOnexError(
                message="Metrics reduction failed",
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                context={"error": str(e), "correlation_id": str(correlation_id)},
            ) from e

    async def _stream_events(
        self,
        contract: ModelContractReducer,
        batch_size: Optional[int] = None,
    ) -> AsyncIterator[list[CodegenEvent]]:
        """
        Stream code generation events from contract input.

        Implements windowed streaming with batching for efficient processing.

        Args:
            contract: Reducer contract with streaming configuration
            batch_size: Number of items per batch

        Yields:
            Batches of code generation events
        """
        # Use configured batch size if not provided
        if batch_size is None:
            batch_size = self.batch_size

        # Check if contract has streaming input
        if hasattr(contract, "input_stream") and contract.input_stream:
            # Stream from async iterator
            batch: list[CodegenEvent] = []

            async for item in contract.input_stream:
                # Convert dict to event model if needed
                event = self._parse_event(item) if isinstance(item, dict) else item

                if event:
                    batch.append(event)

                # Yield batch when full
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            # Yield remaining items
            if batch:
                yield batch

        # Fallback: process input_state as a single batch
        elif hasattr(contract, "input_state") and contract.input_state:
            # Handle single item or list of items
            input_data = contract.input_state

            if isinstance(input_data, dict):
                items = input_data.get("items", [input_data])
            elif isinstance(input_data, list):
                items = input_data
            else:
                items = [input_data]

            # Convert to event models and yield in batches
            batch = []
            for item in items:
                event = self._parse_event(item) if isinstance(item, dict) else item

                if event:
                    batch.append(event)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

    def _parse_event(self, event_data: Mapping[str, object]) -> CodegenEvent | None:
        """
        Parse event dict into appropriate event model.

        Args:
            event_data: Event data dict

        Returns:
            Parsed event model or None if parsing fails
        """
        event_type = event_data.get("event_type")

        try:
            if event_type == "CODEGEN_STARTED":
                return ModelEventCodegenStarted(**event_data)
            elif event_type == "CODEGEN_STAGE_COMPLETED":
                return ModelEventCodegenStageCompleted(**event_data)
            elif event_type == "CODEGEN_COMPLETED":
                return ModelEventCodegenCompleted(**event_data)
            elif event_type == "CODEGEN_FAILED":
                return ModelEventCodegenFailed(**event_data)
            else:
                # Unknown event type, skip
                raw_msg = str(event_data)[:100]
                emit_log_event(
                    LogLevel.WARNING,
                    f"Unknown event type, skipping: event_type={event_type}, "
                    f"raw_message={raw_msg}...",
                    {
                        "metric": "parsing_failures",
                        "reason": "unknown_event_type",
                        "event_type": event_type,
                    },
                )
                return None
        except (ValidationError, TypeError, KeyError) as e:
            # Parsing failed due to validation error, type mismatch, or missing required field
            raw_msg = str(event_data)[:100]
            emit_log_event(
                LogLevel.WARNING,
                f"Event parsing failed: event_type={event_type}, "
                f"error={e!r}, raw_message={raw_msg}...",
                {
                    "metric": "parsing_failures",
                    "reason": "validation_error",
                    "event_type": event_type,
                    "error_type": type(e).__name__,
                },
            )
            return None

    def _build_metrics_recorded_event(
        self, metrics_state: ModelMetricsState
    ) -> ModelEventMetricsRecorded:
        """
        Build GENERATION_METRICS_RECORDED event (PURE FUNCTION - NO I/O).

        Args:
            metrics_state: Aggregated metrics state

        Returns:
            ModelEventMetricsRecorded ready for publishing
        """
        return ModelEventMetricsRecorded(
            correlation_id=uuid4(),
            event_id=uuid4(),
            timestamp=datetime.now(UTC),
            window_start=metrics_state.window_start,
            window_end=metrics_state.window_end,
            aggregation_type=metrics_state.window_type.value,
            total_generations=metrics_state.total_generations,
            successful_generations=metrics_state.successful_generations,
            failed_generations=metrics_state.failed_generations,
            avg_duration_seconds=metrics_state.avg_duration_seconds,
            p50_duration_seconds=metrics_state.p50_duration_seconds,
            p95_duration_seconds=metrics_state.p95_duration_seconds,
            p99_duration_seconds=metrics_state.p99_duration_seconds,
            avg_quality_score=metrics_state.avg_quality_score,
            avg_test_coverage=metrics_state.avg_test_coverage,
            total_tokens=metrics_state.total_tokens,
            total_cost_usd=metrics_state.total_cost_usd,
            avg_cost_per_generation=metrics_state.avg_cost_per_generation,
            model_metrics=metrics_state.model_metrics,
            node_type_metrics=metrics_state.node_type_metrics,
        )

    def _get_window_type(self, contract: ModelContractReducer) -> EnumMetricsWindow:
        """
        Extract window type from contract configuration.

        Args:
            contract: Reducer contract

        Returns:
            Metrics window type
        """
        # Check for aggregation subcontract
        if hasattr(contract, "aggregation") and contract.aggregation:
            if hasattr(contract.aggregation, "window_type"):
                return EnumMetricsWindow(contract.aggregation.window_type)

        # Default to configured window type
        return self.default_window_type

    def _get_batch_size(self, contract: ModelContractReducer) -> int:
        """
        Extract batch size from streaming configuration.

        Args:
            contract: Reducer contract

        Returns:
            Batch size
        """
        if hasattr(contract, "streaming") and contract.streaming:
            if hasattr(contract.streaming, "batch_size"):
                return int(contract.streaming.batch_size)

        # Default batch size
        return int(self.batch_size)

    async def startup(self) -> None:
        """
        Node startup lifecycle hook.

        Initializes container services and publishes initial registration.
        Should be called when node is ready to serve requests.
        """
        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenMetricsReducer starting up",
            {"node_id": self.node_id},
        )

        # Initialize container services if available
        if hasattr(self.container, "initialize"):
            try:
                await self.container.initialize()
                emit_log_event(
                    LogLevel.INFO,
                    "Container services initialized successfully",
                    {"node_id": self.node_id},
                )
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Container initialization failed, continuing in degraded mode: {e}",
                    {
                        "node_id": self.node_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenMetricsReducer startup complete",
            {"node_id": self.node_id},
        )

    async def shutdown(self) -> None:
        """
        Node shutdown lifecycle hook.

        Cleans up resources and deregisters from Consul.
        Should be called when node is preparing to exit.
        """
        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenMetricsReducer shutting down",
            {"node_id": self.node_id},
        )

        # Cleanup container services
        if hasattr(self.container, "cleanup"):
            try:
                await self.container.cleanup()
                emit_log_event(
                    LogLevel.INFO,
                    "Container services cleaned up successfully",
                    {"node_id": self.node_id},
                )
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Container cleanup failed: {e}",
                    {
                        "node_id": self.node_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

        # Deregister from Consul for clean service discovery
        self._deregister_from_consul()

        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenMetricsReducer shutdown complete",
            {"node_id": self.node_id},
        )

    def _register_with_consul_sync(self) -> None:
        """
        Register codegen metrics reducer node with Consul for service discovery (synchronous).

        Registers the reducer as a service with health checks pointing to
        the health endpoint. Includes metadata about node capabilities.

        Note:
            This is a non-blocking registration. Failures are logged but don't
            fail node startup. Service will continue without Consul if registration fails.
        """
        try:
            import consul

            # Initialize Consul client
            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)

            # Generate unique service ID
            service_id = f"omninode-bridge-codegen-metrics-reducer-{self.node_id}"

            # Get service port from config (default to 8063 for codegen metrics reducer)
            service_port = int(self.container.config.get("service_port", 8063))

            # Get service host from config (default to localhost)
            service_host = self.container.config.get("service_host", "localhost")

            # Prepare service tags
            service_tags = [
                "onex",
                "bridge",
                "codegen-metrics-reducer",
                f"version:{getattr(self, 'version', '0.1.0')}",
                "omninode_bridge",
            ]

            # Prepare service metadata (encoded in tags for MVP compatibility)
            service_tags.extend(
                [
                    "node_type:codegen-metrics-reducer",
                    f"window_type:{self.default_window_type.value}",
                ]
            )

            # Health check URL (assumes health endpoint is available)
            health_check_url = f"http://{service_host}:{service_port}/health"

            # Register service with Consul
            consul_client.agent.service.register(
                name="omninode-bridge-codegen-metrics-reducer",
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=service_tags,
                http=health_check_url,
                interval="30s",
                timeout="5s",
            )

            emit_log_event(
                LogLevel.INFO,
                "Registered with Consul successfully",
                {
                    "node_id": self.node_id,
                    "service_id": service_id,
                    "consul_host": self.consul_host,
                    "consul_port": self.consul_port,
                    "service_host": service_host,
                    "service_port": service_port,
                },
            )

            # Store service_id for deregistration
            self._consul_service_id = service_id

        except ImportError:
            emit_log_event(
                LogLevel.WARNING,
                "python-consul not installed - Consul registration skipped",
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                "Failed to register with Consul",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def _deregister_from_consul(self) -> None:
        """
        Deregister codegen metrics reducer from Consul on shutdown (synchronous).

        Removes the service registration from Consul to prevent stale entries
        in the service catalog.

        Note:
            This is called during node shutdown. Failures are logged but don't
            prevent shutdown from completing.
        """
        try:
            if not hasattr(self, "_consul_service_id"):
                # Not registered, nothing to deregister
                return

            import consul

            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            consul_client.agent.service.deregister(self._consul_service_id)

            emit_log_event(
                LogLevel.INFO,
                "Deregistered from Consul successfully",
                {
                    "node_id": self.node_id,
                    "service_id": self._consul_service_id,
                },
            )

        except ImportError:
            # python-consul not installed, silently skip
            pass
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                "Failed to deregister from Consul",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get node metrics."""
        return {
            "total_operations": getattr(self, "_total_operations", 0),
            "avg_duration_ms": round(getattr(self, "_avg_duration", 0.0), 2),
            "reducer_type": "codegen_metrics",
        }


def main() -> int:
    """
    Entry point for node execution.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from omnibase_core.infrastructure.node_base import NodeBase

        # Contract filename - standard ONEX pattern
        CONTRACT_FILENAME = "contract.yaml"

        NodeBase(Path(__file__).parent / CONTRACT_FILENAME)
        return 0
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            f"NodeCodegenMetricsReducer execution failed: {e!s}",
            {"error": str(e), "error_type": type(e).__name__},
        )
        return 1


if __name__ == "__main__":
    main()
