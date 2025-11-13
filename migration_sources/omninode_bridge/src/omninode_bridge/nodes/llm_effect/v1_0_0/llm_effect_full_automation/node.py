#!/usr/bin/env python3
"""
NodeLlmEffectEffect - LLM Effect Node for multi-tier LLM generation with Z.ai integration. Supports CLOUD_FAST (GLM-4.5) tier with 128K context window. Includes circuit breaker, retry logic, token tracking, and cost management.

ONEX v2.0 Effect Node with Registration & Introspection
Domain: ai_services
Generated: 2025-11-05T13:31:05.422648+00:00
"""

from typing import Any

from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck

# Import mixins from omnibase_core (ONEX standard)
from omnibase_core.mixins.mixin_introspection import MixinNodeIntrospection
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

# No external dependencies specified

MIXINS_AVAILABLE = True


class NodeLlmEffectEffect(NodeEffect, MixinHealthCheck, MixinNodeIntrospection):
    """
    LLM Effect Node for multi-tier LLM generation with Z.ai integration. Supports CLOUD_FAST (GLM-4.5) tier with 128K context window. Includes circuit breaker, retry logic, token tracking, and cost management.

    Operations:
    - generate_text
    - calculate_cost
    - track_usage

    Features:
    - Multi-tier LLM support (LOCAL, CLOUD_FAST, CLOUD_PREMIUM)
    - Z.ai API integration (Anthropic-compatible endpoint)
    - GLM-4.5 model support (PRIMARY tier)
    - Circuit breaker pattern for fault tolerance
    - Retry logic with exponential backoff
    - HTTP client with connection pooling
    - Token usage tracking (input/output/total)
    - Cost tracking with sub-cent accuracy
    - Comprehensive metrics collection via MixinMetrics
    - Performance monitoring (latency, throughput)
    - Health checks via MixinHealthCheck
    - Z.ai API health monitoring
    - Structured logging with correlation tracking
    - OpenTelemetry tracing support
    - Environment-based credentials (ZAI_API_KEY)
    - Configurable timeouts and thresholds
    - Per-tier pricing configuration

    Capabilities:
    - Automatic node registration via introspection events
    - Health check endpoints
    - Consul service discovery integration
    """

    def __init__(self, container: ModelContainer) -> None:
        """Initialize NodeLlmEffectEffect with registration and introspection."""
        super().__init__(container)

        # Configuration
        # Access config from container.value (ModelContainer stores config in value field)
        self.config = container.value if isinstance(container.value, dict) else {}

        # Initialize health checks (if mixins available)
        if MIXINS_AVAILABLE:
            self.initialize_health_checks()
            self._register_component_checks()

            # Initialize introspection system
            self.initialize_introspection()

        emit_log_event(
            LogLevel.INFO,
            "NodeLlmEffectEffect initialized with registration support",
            {
                "node_id": str(self.node_id),
                "mixins_available": MIXINS_AVAILABLE,
                "operations": ["generate_text", "calculate_cost", "track_usage"],
                "features": [
                    "Multi-tier LLM support (LOCAL, CLOUD_FAST, CLOUD_PREMIUM)",
                    "Z.ai API integration (Anthropic-compatible endpoint)",
                    "GLM-4.5 model support (PRIMARY tier)",
                    "Circuit breaker pattern for fault tolerance",
                    "Retry logic with exponential backoff",
                    "HTTP client with connection pooling",
                    "Token usage tracking (input/output/total)",
                    "Cost tracking with sub-cent accuracy",
                    "Comprehensive metrics collection via MixinMetrics",
                    "Performance monitoring (latency, throughput)",
                    "Health checks via MixinHealthCheck",
                    "Z.ai API health monitoring",
                    "Structured logging with correlation tracking",
                    "OpenTelemetry tracing support",
                    "Environment-based credentials (ZAI_API_KEY)",
                    "Configurable timeouts and thresholds",
                    "Per-tier pricing configuration",
                ],
            },
        )

    def _register_component_checks(self) -> None:
        """
        Register component health checks for this node.

        Override this method to add custom health checks specific to this node's dependencies.
        """
        # Base node runtime check is registered by HealthCheckMixin
        # Add custom checks here as needed
        pass

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        """
        Execute effect operation.

        Args:
            contract: Effect contract with operation parameters

        Returns:
            Operation result

        Raises:
            ModelOnexError: If operation fails
        """
        try:
            # Validate contract
            if not contract or not contract.operation:
                raise ModelOnexError("Invalid contract: operation is required")

            # Extract operation parameters
            operation = contract.operation
            params = operation.parameters or {}

            # Initialize correlation context
            correlation_id = contract.correlation_id or str(uuid.uuid4())
            self._set_correlation_context(correlation_id)

            # Log operation start
            self.emit_log_event(
                level="INFO",
                message=f"Executing LLM effect operation: {operation.name}",
                data={
                    "operation": operation.name,
                    "correlation_id": correlation_id,
                    "params": {
                        k: v
                        for k, v in params.items()
                        if k not in ["api_key", "prompt"]
                    },
                },
            )

            # Route to appropriate operation handler
            result = None
            if operation.name == "generate_text":
                result = await self._generate_text(params, correlation_id)
            elif operation.name == "calculate_cost":
                result = self._calculate_cost(params)
            elif operation.name == "track_usage":
                result = self._track_usage(params)
            else:
                raise ModelOnexError(f"Unsupported operation: {operation.name}")

            # Log successful completion
            self.emit_log_event(
                level="INFO",
                message=f"LLM effect operation completed successfully: {operation.name}",
                data={
                    "operation": operation.name,
                    "correlation_id": correlation_id,
                    "result_type": type(result).__name__,
                },
            )

            return result

        except ModelOnexError:
            # Re-raise ONEX errors directly
            raise
        except Exception as e:
            # Wrap unexpected errors
            error_msg = f"LLM effect operation failed: {e!s}"
            self.emit_log_event(
                level="ERROR",
                message=error_msg,
                data={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "correlation_id": correlation_id,
                },
            )
            raise ModelOnexError(error_msg) from e

    async def _old_exception_handling_example(self) -> None:
        """Example exception handling (remove this method)."""
        try:
            pass
        except (ConnectionError, TimeoutError) as e:
            # Network/connection failures
            emit_log_event(
                LogLevel.ERROR,
                f"Network error during effect execution: {e!s}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.NETWORK_ERROR,
                message=f"Network error: {e!s}",
                details={"original_error": str(e), "error_type": type(e).__name__},
            ) from e

        except ValueError as e:
            # Invalid input/configuration
            emit_log_event(
                LogLevel.ERROR,
                f"Invalid input for effect execution: {e!s}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                },
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message=f"Invalid input: {e!s}",
                details={"original_error": str(e)},
            ) from e

        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            emit_log_event(
                LogLevel.CRITICAL,
                f"Unexpected error during effect execution: {e!s}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Unexpected error during effect execution: {e!s}",
                details={"original_error": str(e), "error_type": type(e).__name__},
            ) from e

    def get_metadata_loader(self) -> Any:
        """
        Get metadata loader for this node.

        Returns:
            Metadata loader instance or None if not applicable
        """
        return None

    async def startup(self) -> None:
        """
        Node startup lifecycle hook.

        Publishes introspection data to registry and starts background tasks.
        Should be called when node is ready to serve requests.
        """
        if not MIXINS_AVAILABLE:
            emit_log_event(
                LogLevel.WARNING,
                "Mixins not available - skipping startup registration",
                {"node_id": str(self.node_id)},
            )
            return

        emit_log_event(
            LogLevel.INFO,
            "NodeLlmEffectEffect starting up",
            {"node_id": str(self.node_id)},
        )

        # Publish introspection broadcast to registry
        await self.publish_introspection(reason="startup")

        # Start introspection background tasks (heartbeat, registry listener)
        await self.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=30,
            enable_registry_listener=True,
        )

        emit_log_event(
            LogLevel.INFO,
            "NodeLlmEffectEffect startup complete - node registered",
            {"node_id": str(self.node_id)},
        )

    async def shutdown(self) -> None:
        """
        Node shutdown lifecycle hook.

        Stops background tasks and cleans up resources.
        Should be called when node is preparing to exit.
        """
        if not MIXINS_AVAILABLE:
            return

        emit_log_event(
            LogLevel.INFO,
            "NodeLlmEffectEffect shutting down",
            {"node_id": str(self.node_id)},
        )

        # Stop introspection background tasks
        await self.stop_introspection_tasks()

        emit_log_event(
            LogLevel.INFO,
            "NodeLlmEffectEffect shutdown complete",
            {"node_id": str(self.node_id)},
        )


__all__ = ["NodeLlmEffectEffect"]
