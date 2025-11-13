#!/usr/bin/env python3
"""
NodeBridgeOrchestrator - Stamping Workflow Coordinator.

Orchestrates metadata stamping workflows with OnexTree intelligence integration,
FSM-driven state management, and Kafka event publishing.

ONEX v2.0 Compliance:
- Suffix-based naming: NodeBridgeOrchestrator
- Import from omnibase_core infrastructure
- Subcontract composition for workflow/routing/FSM/events
- ModelONEXContainer for dependency injection with service resolution
- Strong typing (no Any types except where needed for flexibility)
"""

import asyncio
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, cast
from uuid import UUID, uuid4

# Import with fallback to stubs when omnibase_core is not available
try:
    from omnibase_core import EnumCoreErrorCode, ModelOnexError
    from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
    from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_core.models.contracts.model_contract_orchestrator import (
        ModelContractOrchestrator,
    )
    from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
except ImportError:
    # Fallback to stubs when omnibase_core is not available (testing/demo mode)
    from ._stubs import (
        EnumCoreErrorCode,
        LogLevel,
        ModelContractOrchestrator,
        ModelONEXContainer,
        ModelOnexError,
        NodeOrchestrator,
        emit_log_event,
    )

# Import protocols from omnibase_spi for duck typing (Phase 2: Protocol Type Hints)
try:
    from omnibase_spi.protocols import (
        ProtocolOnexEnvelope,  # For event envelope type hints
    )
    from omnibase_spi.protocols import (
        ProtocolServiceRegistry,  # For DI container type hints
    )

    # Flag to indicate protocol availability
    PROTOCOLS_AVAILABLE = True
except ImportError:
    # Protocol imports are optional - duck typing still works with concrete types
    # This maintains backward compatibility when omnibase_spi is not available
    PROTOCOLS_AVAILABLE = False

# Always use concrete type for type checking to avoid Protocol method access errors
# The runtime duck typing works correctly with both Protocol and concrete types
ProtocolServiceRegistry = ModelONEXContainer  # type: ignore[misc,assignment]
ProtocolOnexEnvelope = dict  # type: ignore[misc,assignment]

# Aliases for compatibility
OnexError = ModelOnexError

# Import workflow execution CRUD operations

# Import performance configuration
from ....config import performance_config

# Import EventBus for event-driven coordination
from ....services.event_bus import EventBusService

# Import Prometheus metrics
from ...metrics.prometheus_metrics import create_orchestrator_metrics

# Import health check mixin
from ...mixins.health_mixin import HealthCheckMixin, HealthStatus, check_http_service

# Import introspection mixin
from ...mixins.introspection_mixin import IntrospectionMixin
from .models.enum_workflow_event import EnumWorkflowEvent
from .models.enum_workflow_state import EnumWorkflowState
from .models.model_stamp_response_output import ModelStampResponseOutput


class NodeBridgeOrchestrator(NodeOrchestrator, HealthCheckMixin, IntrospectionMixin):
    """
    Bridge Orchestrator for stamping workflow coordination.

    Coordinates multi-step stamping workflows:
    1. Receive stamp request from MetadataStampingService
    2. Route to OnexTree for intelligence analysis (optional)
    3. Execute BLAKE3 hash generation
    4. Create stamp with namespace support
    5. Publish events to Kafka
    6. Transition FSM state
    7. Return stamped content

    Subcontracts:
    - WorkflowCoordination: Multi-step workflow execution
    - Routing: Service discovery and load balancing
    - FSM: Workflow state management (pending → processing → completed/failed)
    - EventType: Kafka event publishing

    ONEX Pattern Compliance:
    - Extends NodeOrchestrator from omnibase_core.infrastructure
    - Uses ModelONEXContainer for dependency injection with service resolution
    - Implements strong typing with Pydantic models
    - Follows FSM state management patterns
    - UUID correlation tracking throughout workflow
    """

    def __init__(self, container: ProtocolServiceRegistry) -> None:
        """
        Initialize Bridge Orchestrator with dependency injection container.

        Args:
            container: DI container with service resolution (uses ProtocolServiceRegistry
                      for duck typing - any object with get_service/register_service methods)

        Raises:
            OnexError: If container is invalid or initialization fails

        Note:
            Uses ProtocolServiceRegistry protocol type hint for PUBLIC API duck typing.
            Internal implementation still uses concrete ModelONEXContainer from omnibase_core.
            This enables flexibility while maintaining type safety.
        """
        super().__init__(container)

        # Load configuration using ConfigLoader with fallback to container.config
        # This enables YAML + env var + Vault cascade while maintaining test compatibility
        orchestrator_config = self._load_orchestrator_config(container)

        # Bridge-specific configuration from ConfigLoader or container fallback
        self.metadata_stamping_service_url: str = cast(
            str,
            orchestrator_config.get(
                "metadata_stamping_service_url",
                container.config.get(  # type: ignore[attr-defined]
                    "metadata_stamping_service_url", "http://metadata-stamping:8053"
                ),
            ),
        )
        self.onextree_service_url: str = cast(
            str,
            orchestrator_config.get(
                "onextree_service_url",
                container.config.get("onextree_service_url", "http://onextree:8058"),  # type: ignore[attr-defined]
            ),
        )
        self.onextree_timeout_ms: float = cast(
            float,
            orchestrator_config.get(
                "onextree_timeout_ms", container.config.get("onextree_timeout_ms", 500.0)  # type: ignore[attr-defined]
            ),
        )
        self.kafka_broker_url: str = cast(
            str,
            orchestrator_config.get(
                "kafka_broker_url",
                container.config.get(  # type: ignore[attr-defined]
                    "kafka_broker_url",
                    os.getenv(
                        "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
                    ),
                ),
            ),
        )
        self.default_namespace: str = cast(
            str,
            orchestrator_config.get(
                "default_namespace",
                container.config.get("default_namespace", "omninode.bridge"),  # type: ignore[attr-defined]
            ),
        )
        # Consul configuration for service discovery
        self.consul_host: str = cast(
            str,
            orchestrator_config.get(
                "consul_host",
                container.config.get(  # type: ignore[attr-defined]
                    "consul_host", os.getenv("CONSUL_HOST", "omninode-bridge-consul")
                ),
            ),
        )
        self.consul_port: int = cast(
            int,
            orchestrator_config.get(
                "consul_port",
                container.config.get("consul_port", int(os.getenv("CONSUL_PORT", "28500"))),  # type: ignore[attr-defined]
            ),
        )
        self.consul_enable_registration: bool = cast(
            bool,
            orchestrator_config.get(
                "consul_enable_registration",
                container.config.get("consul_enable_registration", True),  # type: ignore[attr-defined]
            ),
        )
        # Optional ASGI app for testing (bypasses network calls)
        self.onextree_app = container.config.get("onextree_app", None)  # type: ignore[attr-defined]

        # Workflow state tracking (FSM)
        self.workflow_fsm_states: dict[str, EnumWorkflowState] = {}
        self.workflow_correlation_ids: dict[str, UUID] = {}

        # Performance metrics tracking
        self.stamping_metrics: dict[str, dict[str, float]] = {}

        # Get or create KafkaClient from container (skip if in health check mode)
        health_check_mode = container.config.get("health_check_mode", False)  # type: ignore[attr-defined]
        self.kafka_client = container.get_service("kafka_client")  # type: ignore[attr-defined]

        if self.kafka_client is None and not health_check_mode:
            # Import KafkaClient
            try:
                from ....services.kafka_client import KafkaClient

                self.kafka_client = KafkaClient(
                    bootstrap_servers=self.kafka_broker_url,
                    enable_dead_letter_queue=True,
                    max_retry_attempts=3,
                    timeout_seconds=performance_config.KAFKA_CLIENT_TIMEOUT_SECONDS,
                )
                container.register_service("kafka_client", self.kafka_client)  # type: ignore
            except ImportError:
                emit_log_event(
                    LogLevel.WARNING,
                    "KafkaClient not available - events will be logged only",
                    {"node_id": self.node_id},
                )
                self.kafka_client = None
        elif health_check_mode:
            # In health check mode, skip Kafka initialization
            emit_log_event(
                LogLevel.DEBUG,
                "Health check mode enabled - skipping Kafka initialization",
                {"node_id": self.node_id},
            )
            self.kafka_client = None

        # Get or create MetadataStampingClient from container
        self.metadata_client = container.get_service("metadata_stamping_client")  # type: ignore[attr-defined]

        if self.metadata_client is None and not health_check_mode:
            # Import AsyncMetadataStampingClient
            try:
                from ....clients.metadata_stamping_client import (
                    AsyncMetadataStampingClient,
                )

                self.metadata_client = AsyncMetadataStampingClient(
                    base_url=self.metadata_stamping_service_url,
                    timeout=performance_config.METADATA_STAMPING_CLIENT_TIMEOUT_SECONDS,
                    max_retries=3,
                )
                container.register_service(  # type: ignore
                    "metadata_stamping_client", self.metadata_client  # type: ignore[arg-type]
                )

                emit_log_event(
                    LogLevel.INFO,
                    "MetadataStampingClient initialized successfully",
                    {
                        "node_id": self.node_id,
                        "service_url": self.metadata_stamping_service_url,
                    },
                )
            except ImportError as e:
                emit_log_event(
                    LogLevel.WARNING,
                    "MetadataStampingClient not available - will use placeholders",
                    {"node_id": self.node_id, "error": str(e)},
                )
                self.metadata_client = None
        elif health_check_mode:
            # In health check mode, skip client initialization
            emit_log_event(
                LogLevel.DEBUG,
                "Health check mode enabled - skipping MetadataStampingClient initialization",
                {"node_id": self.node_id},
            )
            self.metadata_client = None

        # Get database adapter node from container for persistence operations
        self.db_adapter_node = container.get_service("database_adapter_node")  # type: ignore[attr-defined]
        if self.db_adapter_node is None and not health_check_mode:
            emit_log_event(
                LogLevel.WARNING,
                "DatabaseAdapterNode not available - workflow execution persistence disabled",
                {"node_id": self.node_id},
            )
        elif health_check_mode:
            emit_log_event(
                LogLevel.DEBUG,
                "Health check mode enabled - skipping DatabaseAdapterNode initialization",
                {"node_id": self.node_id},
            )

        # Get or create EventBus from container for event-driven coordination
        self.event_bus = container.get_service("event_bus")  # type: ignore[attr-defined]
        if self.event_bus is None and not health_check_mode:
            # Initialize EventBus service
            if self.kafka_client:
                self.event_bus = EventBusService(
                    kafka_client=self.kafka_client,
                    node_id=self.node_id,
                    namespace=self.default_namespace,
                )
                container.register_service("event_bus", self.event_bus)  # type: ignore
                emit_log_event(
                    LogLevel.INFO,
                    "EventBus service initialized successfully",
                    {
                        "node_id": self.node_id,
                        "namespace": self.default_namespace,
                    },
                )
            else:
                emit_log_event(
                    LogLevel.WARNING,
                    "EventBus not available - Kafka client required for event-driven coordination",
                    {"node_id": self.node_id},
                )
                self.event_bus = None
        elif health_check_mode:
            emit_log_event(
                LogLevel.DEBUG,
                "Health check mode enabled - skipping EventBus initialization",
                {"node_id": self.node_id},
            )
            self.event_bus = None

        # Initialize health check system
        self.initialize_health_checks()

        # Initialize introspection system (skip in health check mode)
        if not health_check_mode:
            self.initialize_introspection()
        else:
            # Initialize minimal introspection state for health check
            self._cached_node_type = "orchestrator"
            self._introspection_cache = {}
            self._cache_timestamps = {}

        # Initialize Prometheus metrics (feature flag controlled)
        enable_prometheus = orchestrator_config.get(
            "enable_prometheus",
            container.config.get("enable_prometheus", True),  # type: ignore[attr-defined]
        )
        self.metrics_collector = create_orchestrator_metrics(
            enable_prometheus=enable_prometheus
        )
        if enable_prometheus:
            emit_log_event(
                LogLevel.INFO,
                "Prometheus metrics enabled for orchestrator",
                {"node_id": self.node_id, "enable_prometheus": enable_prometheus},
            )
        else:
            emit_log_event(
                LogLevel.INFO,
                "Prometheus metrics disabled for orchestrator",
                {"node_id": self.node_id, "enable_prometheus": enable_prometheus},
            )

        emit_log_event(
            LogLevel.INFO,
            "NodeBridgeOrchestrator initialized successfully",
            {
                "node_id": self.node_id,
                "metadata_stamping_url": self.metadata_stamping_service_url,
                "onextree_url": self.onextree_service_url,
                "kafka_broker": self.kafka_broker_url,
                "default_namespace": self.default_namespace,
                "kafka_enabled": self.kafka_client is not None,
                "metadata_client_enabled": self.metadata_client is not None,
                "config_source": (
                    "ConfigLoader" if orchestrator_config else "container.config"
                ),
            },
        )

        # Register with Consul for service discovery (skip in health check mode)
        if not health_check_mode and self.consul_enable_registration:
            # Call synchronous registration wrapper
            self._register_with_consul_sync()

    def _load_orchestrator_config(
        self, container: ProtocolServiceRegistry
    ) -> dict[str, Any]:
        """
        Load orchestrator configuration using ConfigLoader with graceful fallback.

        Configuration hierarchy (highest priority first):
        1. Environment variables (BRIDGE_* prefix)
        2. Environment-specific YAML (development.yaml/production.yaml)
        3. Base YAML (orchestrator.yaml)
        4. Container.config (test/fallback mode)

        Args:
            container: ONEX DI container with configuration and service resolution

        Returns:
            Dictionary with configuration values, or empty dict if ConfigLoader unavailable

        Note:
            Gracefully degrades to container.config if ConfigLoader fails.
            This maintains backward compatibility with tests.
        """
        try:
            # Import ConfigLoader
            from ....config.config_loader import get_orchestrator_config

            # Detect if we're in health check mode (skip ConfigLoader)
            health_check_mode = container.config.get("health_check_mode", False)  # type: ignore[attr-defined]
            if health_check_mode:
                emit_log_event(
                    LogLevel.DEBUG,
                    "Health check mode - skipping ConfigLoader, using container.config",
                    {"node_id": self.node_id},
                )
                return {}

            # Load configuration from ConfigLoader (YAML + env cascade)
            config = get_orchestrator_config()

            emit_log_event(
                LogLevel.INFO,
                "Configuration loaded successfully via ConfigLoader",
                {
                    "node_id": self.node_id,
                    "environment": config.environment,
                    "config_source": "yaml_cascade",
                },
            )

            # Extract configuration values and return as flat dict for easy access
            return {
                "metadata_stamping_service_url": config.services.metadata_stamping.base_url,
                "onextree_service_url": config.services.onextree.base_url,
                "onextree_timeout_ms": config.services.onextree.timeout_seconds
                * 1000.0,
                "kafka_broker_url": config.kafka.bootstrap_servers,
                "default_namespace": config.node.namespace,
                "max_concurrent_workflows": config.orchestrator.max_concurrent_workflows,
                "workflow_timeout_seconds": config.orchestrator.workflow_timeout_seconds,
                "consul_host": config.consul.host,
                "consul_port": config.consul.port,
                "consul_enable_registration": config.consul.enable_registration,
            }

        except ImportError as e:
            # ConfigLoader not available (shouldn't happen in production)
            emit_log_event(
                LogLevel.WARNING,
                "ConfigLoader not available - falling back to container.config",
                {"node_id": self.node_id, "error": str(e)},
            )
            return {}

        except FileNotFoundError as e:
            # Configuration file not found
            emit_log_event(
                LogLevel.WARNING,
                "Configuration file not found - falling back to container.config",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": "FileNotFoundError",
                },
            )
            return {}

        except OSError as e:
            # File system errors reading configuration
            emit_log_event(
                LogLevel.WARNING,
                "File system error loading configuration - falling back to container.config",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return {}

        except (ValueError, KeyError) as e:
            # Configuration validation or parsing errors
            emit_log_event(
                LogLevel.WARNING,
                "Configuration validation failed - falling back to container.config",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return {}

        except Exception as e:
            # Unexpected errors - log and re-raise for visibility
            emit_log_event(
                LogLevel.ERROR,
                "Unexpected error loading configuration - falling back to container.config",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Re-raise unexpected errors for debugging
            # Note: For now we return {} for backward compatibility, but log as ERROR
            return {}

    async def execute_orchestration(
        self, contract: ModelContractOrchestrator
    ) -> ModelStampResponseOutput:
        """
        Execute stamping workflow orchestration with event-driven coordination.

        Supports two modes:
        1. Event-Driven (preferred): Publish Action event to reducer, wait for completion
        2. Legacy (fallback): Direct synchronous workflow execution

        Args:
            contract: Orchestrator contract with workflow configuration

        Returns:
            ModelStampResponseOutput with stamped content and metadata

        Raises:
            OnexError: If workflow execution fails
        """
        # Track workflow execution with Prometheus metrics
        status = "success"
        try:
            # Check if event-driven coordination is available
            if self.event_bus and self.event_bus.is_initialized:
                with self.metrics_collector.time_workflow(status="success"):
                    return await self._execute_event_driven_workflow(contract)
            else:
                # Fallback to legacy synchronous execution
                emit_log_event(
                    LogLevel.WARNING,
                    "EventBus not available - falling back to legacy synchronous workflow",
                    {
                        "node_id": self.node_id,
                        "workflow_id": str(contract.correlation_id),
                    },
                )
                with self.metrics_collector.time_workflow(status="success"):
                    return await self._execute_legacy_workflow(contract)
        except Exception as e:
            status = "error"
            self.metrics_collector.record_error(
                error_type=type(e).__name__, operation="execute_orchestration"
            )
            raise

    async def _execute_event_driven_workflow(
        self, contract: ModelContractOrchestrator
    ) -> ModelStampResponseOutput:
        """
        Execute workflow using event-driven coordination with reducer.

        Flow:
        1. Publish Action event to trigger reducer processing
        2. Wait for StateCommitted or ReducerGaveUp event (30s timeout)
        3. Handle success/failure/timeout via event responses
        4. Apply DAG retry policy for failures

        Args:
            contract: Orchestrator contract with workflow configuration

        Returns:
            ModelStampResponseOutput from event-driven coordination

        Raises:
            OnexError: If event-driven workflow fails
        """
        start_time = time.time()
        workflow_id = contract.correlation_id
        workflow_id_str = str(workflow_id)

        # Initialize FSM state
        current_state = EnumWorkflowState.PENDING
        self.workflow_fsm_states[workflow_id_str] = current_state
        self.workflow_correlation_ids[workflow_id_str] = workflow_id

        try:
            # Validate input
            if not hasattr(contract, "input_data") or contract.input_data is None:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="Contract missing input_data for event-driven workflow",
                    details={
                        "node_id": self.node_id,
                        "workflow_id": workflow_id_str,
                        "contract_type": type(contract).__name__,
                    },
                )

            # Transition to PROCESSING state
            current_state = await self._transition_state(
                workflow_id, current_state, EnumWorkflowState.PROCESSING
            )

            # Publish workflow started event
            await self._publish_event(
                EnumWorkflowEvent.WORKFLOW_STARTED,
                {
                    "workflow_id": workflow_id_str,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "mode": "event_driven",
                },
            )

            # Publish Action event to trigger reducer processing
            emit_log_event(
                LogLevel.INFO,
                "Publishing Action event for event-driven workflow",
                {
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                },
            )

            success = await self.event_bus.publish_action_event(
                correlation_id=workflow_id,
                action_type="ORCHESTRATE_WORKFLOW",
                payload={
                    "operation": "metadata_stamping",
                    "input_data": contract.input_data,
                    "namespace": self.default_namespace,
                    "workflow_type": "metadata_stamping",
                },
            )

            if not success:
                raise OnexError(
                    error_code=EnumCoreErrorCode.OPERATION_FAILED,
                    message="Failed to publish Action event to EventBus",
                    details={
                        "node_id": self.node_id,
                        "workflow_id": workflow_id_str,
                    },
                )

            # Wait for completion event from reducer (configurable timeout)
            emit_log_event(
                LogLevel.INFO,
                "Waiting for reducer completion event",
                {
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "timeout_seconds": performance_config.WORKFLOW_COMPLETION_TIMEOUT_SECONDS,
                },
            )

            event = await self.event_bus.wait_for_completion(
                correlation_id=workflow_id,
                timeout_seconds=performance_config.WORKFLOW_COMPLETION_TIMEOUT_SECONDS,
            )

            # Handle event based on type
            if not event:
                # Timeout - no event received
                return await self._handle_timeout(workflow_id, start_time)

            event_type = event.get("event_type")

            if event_type == "STATE_COMMITTED":
                # Success path
                return await self._handle_success(workflow_id, event, start_time)
            elif event_type == "REDUCER_GAVE_UP":
                # Failure path with retry
                return await self._handle_failure(
                    workflow_id, event, start_time, retry_count=0
                )
            else:
                # Unknown event type
                raise OnexError(
                    error_code=EnumCoreErrorCode.OPERATION_FAILED,
                    message=f"Unexpected event type received: {event_type}",
                    details={
                        "node_id": self.node_id,
                        "workflow_id": workflow_id_str,
                        "event_type": event_type,
                    },
                )

        except OnexError:
            # Re-raise OnexError as-is
            raise

        except Exception as e:
            # Wrap other exceptions
            processing_time_ms = (time.time() - start_time) * 1000

            raise OnexError(
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Event-driven workflow failed: {e!s}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "processing_time_ms": processing_time_ms,
                    "error_type": type(e).__name__,
                },
                cause=e,
            )

    async def _execute_legacy_workflow(
        self, contract: ModelContractOrchestrator
    ) -> ModelStampResponseOutput:
        """
        Execute workflow using legacy synchronous execution.

        Legacy flow (for backward compatibility when EventBus unavailable):
        1. Validate input and transition FSM to 'processing'
        2. Route to OnexTree for intelligence (if enabled)
        3. Route to MetadataStampingService for hash generation
        4. Create metadata stamp with O.N.E. v0.1 compliance
        5. Publish events to Kafka
        6. Transition FSM to 'completed'
        7. Return stamped content with metadata

        Args:
            contract: Orchestrator contract with workflow configuration

        Returns:
            ModelStampResponseOutput with stamped content and metadata

        Raises:
            OnexError: If workflow execution fails
        """
        start_time = time.time()
        workflow_id = contract.correlation_id
        workflow_id_str = str(workflow_id)

        # Initialize FSM state BEFORE validation (so we can track FAILED state if validation fails)
        current_state = EnumWorkflowState.PENDING
        self.workflow_fsm_states[workflow_id_str] = current_state
        self.workflow_correlation_ids[workflow_id_str] = workflow_id

        # Initialize result tracking
        hash_generation_time_ms: float = 0.0
        intelligence_data: Optional[dict[str, str]] = None
        workflow_steps_executed = 0

        try:
            # Extract input data from contract
            # Note: In practice, this would come from contract.input_data or similar
            # For now, we'll create a placeholder input model
            if not hasattr(contract, "input_data") or contract.input_data is None:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="Contract missing input_data for stamping workflow",
                    details={
                        "node_id": self.node_id,
                        "workflow_id": workflow_id_str,
                        "contract_type": type(contract).__name__,
                    },
                )
            # Step 1: Transition to PROCESSING state
            current_state = await self._transition_state(
                workflow_id, current_state, EnumWorkflowState.PROCESSING
            )

            await self._publish_event(
                EnumWorkflowEvent.WORKFLOW_STARTED,
                {
                    "workflow_id": workflow_id_str,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "state": current_state.value,
                    "mode": "legacy",
                },
            )
            workflow_steps_executed += 1

            # Step 2: Execute workflow steps from contract
            results = await self._execute_workflow_steps(contract, workflow_id)
            workflow_steps_executed += len(results)

            # Extract metrics from results
            for result in results:
                if "hash_generation_time_ms" in result:
                    hash_generation_time_ms = result["hash_generation_time_ms"]
                if "intelligence_data" in result:
                    intelligence_data = result["intelligence_data"]

            # Step 3: Aggregate results into final output
            final_output = await self._aggregate_workflow_results(
                results, contract, workflow_id
            )

            # Step 4: Transition to COMPLETED state
            current_state = await self._transition_state(
                workflow_id, current_state, EnumWorkflowState.COMPLETED
            )
            workflow_steps_executed += 1

            # Calculate total processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Step 5: Publish completion event
            await self._publish_event(
                EnumWorkflowEvent.WORKFLOW_COMPLETED,
                {
                    "workflow_id": workflow_id_str,
                    "stamp_id": final_output.stamp_id,
                    "file_hash": final_output.file_hash,
                    "namespace": final_output.namespace,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            # Update metrics
            await self._update_stamping_metrics(
                "workflow_orchestration", processing_time_ms, True
            )

            emit_log_event(
                LogLevel.INFO,
                "Stamping workflow orchestration completed successfully (legacy mode)",
                {
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "stamp_id": final_output.stamp_id,
                    "processing_time_ms": processing_time_ms,
                    "steps_executed": workflow_steps_executed,
                    "final_state": current_state.value,
                },
            )

            # Return complete output with all metrics
            return ModelStampResponseOutput(
                stamp_id=final_output.stamp_id,
                file_hash=final_output.file_hash,
                stamped_content=final_output.stamped_content,
                stamp_metadata=final_output.stamp_metadata,
                namespace=final_output.namespace,
                op_id=final_output.op_id,
                version=final_output.version,
                metadata_version=final_output.metadata_version,
                workflow_state=current_state,
                workflow_id=workflow_id,
                intelligence_data=intelligence_data,
                processing_time_ms=processing_time_ms,
                hash_generation_time_ms=hash_generation_time_ms,
                workflow_steps_executed=workflow_steps_executed,
                created_at=final_output.created_at,
                completed_at=datetime.now(UTC),
            )

        except Exception as e:
            # Transition to FAILED state
            processing_time_ms = (time.time() - start_time) * 1000

            try:
                current_state = await self._transition_state(
                    workflow_id, current_state, EnumWorkflowState.FAILED
                )
            except Exception as transition_error:
                emit_log_event(
                    LogLevel.WARNING,
                    "Failed to transition workflow to FAILED state",
                    {
                        "node_id": self.node_id,
                        "workflow_id": workflow_id_str,
                        "error": str(transition_error),
                    },
                )

            # Publish failure event
            await self._publish_event(
                EnumWorkflowEvent.WORKFLOW_FAILED,
                {
                    "workflow_id": workflow_id_str,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            # Update metrics
            await self._update_stamping_metrics(
                "workflow_orchestration", processing_time_ms, False
            )

            emit_log_event(
                LogLevel.ERROR,
                "Stamping workflow orchestration failed (legacy mode)",
                {
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time_ms": processing_time_ms,
                    "steps_executed": workflow_steps_executed,
                },
            )

            # Re-raise OnexError as-is to preserve error code and message
            if isinstance(e, OnexError):
                raise

            # Wrap other exceptions in OnexError
            raise OnexError(
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Stamping workflow orchestration failed: {e!s}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "processing_time_ms": processing_time_ms,
                    "steps_executed": workflow_steps_executed,
                    "error_type": type(e).__name__,
                },
                cause=e,
            )

        finally:
            # Cleanup workflow state
            if workflow_id_str in self.workflow_fsm_states:
                # Keep final state for audit trail
                pass
            if workflow_id_str in self.workflow_correlation_ids:
                # Keep correlation ID for tracing
                pass

    async def _execute_workflow_steps(
        self, contract: ModelContractOrchestrator, workflow_id: UUID
    ) -> list[dict[str, Any]]:
        """
        Execute workflow steps defined in contract.

        Routes to appropriate services based on step configuration:
        - 'onextree_intelligence': Route to OnexTree for AI analysis
        - 'hash_generation': Route to MetadataStampingService
        - 'stamp_creation': Create stamp with O.N.E. v0.1 compliance

        Args:
            contract: Orchestrator contract with workflow steps
            workflow_id: Workflow correlation ID

        Returns:
            List of step execution results

        Raises:
            OnexError: If workflow step execution fails
        """
        results = []

        # Extract workflow steps from contract
        # In practice, this would read from contract.workflow_coordination.workflow_definition
        # For now, we'll use a default workflow
        if (
            hasattr(contract, "workflow_coordination")
            and contract.workflow_coordination
        ):
            # Use configured workflow steps
            workflow_steps = (
                contract.workflow_coordination.workflow_definition.nodes
                if hasattr(contract.workflow_coordination, "workflow_definition")
                else []
            )
        else:
            # Use default workflow steps
            workflow_steps = self._get_default_workflow_steps()

        # Execute each step sequentially
        for step in workflow_steps:
            step_result = await self._execute_workflow_step(step, contract, workflow_id)
            results.append(step_result)

            # Publish step completion event
            await self._publish_event(
                EnumWorkflowEvent.STEP_COMPLETED,
                {
                    "workflow_id": str(workflow_id),
                    "step_type": step.get("step_type", "unknown"),
                    "step_id": step.get("step_id", "unknown"),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

        return results

    def _get_default_workflow_steps(self) -> list[dict[str, Any]]:
        """
        Get default workflow steps for stamping.

        Returns:
            List of default workflow step configurations
        """
        return [
            {
                "step_id": "validate_input",
                "step_type": "validation",
                "required": True,
            },
            {
                "step_id": "hash_generation",
                "step_type": "hash_generation",
                "required": True,
                "service": "metadata_stamping",
            },
            {
                "step_id": "stamp_creation",
                "step_type": "stamp_creation",
                "required": True,
            },
        ]

    async def _execute_workflow_step(
        self,
        step: dict[str, Any],
        contract: ModelContractOrchestrator,
        workflow_id: UUID,
    ) -> dict[str, Any]:
        """
        Execute a single workflow step.

        Routes to appropriate service based on step configuration:
        - 'validation': Validate input data
        - 'onextree_intelligence': Route to OnexTree for AI analysis
        - 'hash_generation': Route to MetadataStampingService
        - 'stamp_creation': Create stamp locally

        Args:
            step: Workflow step configuration
            contract: Orchestrator contract
            workflow_id: Workflow correlation ID

        Returns:
            Step execution result

        Raises:
            OnexError: If step execution fails
        """
        step_type = step.get("step_type", "unknown")
        step_id = step.get("step_id", "unknown")

        emit_log_event(
            LogLevel.DEBUG,
            f"Executing workflow step: {step_type}",
            {
                "node_id": self.node_id,
                "workflow_id": str(workflow_id),
                "step_type": step_type,
                "step_id": step_id,
            },
        )

        try:
            if step_type == "validation":
                return await self._execute_validation_step(step, contract, workflow_id)
            elif step_type == "onextree_intelligence":
                return await self._route_to_onextree(step, contract, workflow_id)
            elif step_type == "hash_generation":
                return await self._route_to_metadata_stamping(
                    step, contract, workflow_id
                )
            elif step_type == "stamp_creation":
                return await self._create_stamp(step, contract, workflow_id)
            else:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"Unknown workflow step type: {step_type}",
                    details={
                        "node_id": self.node_id,
                        "workflow_id": str(workflow_id),
                        "step_type": step_type,
                        "step_id": step_id,
                    },
                )

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Workflow step execution failed: {step_type}",
                {
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "step_type": step_type,
                    "step_id": step_id,
                    "error": str(e),
                },
            )
            raise

    async def _execute_validation_step(
        self,
        step: dict[str, Any],
        contract: ModelContractOrchestrator,
        workflow_id: UUID,
    ) -> dict[str, Any]:
        """
        Execute input validation step.

        Args:
            step: Validation step configuration
            contract: Orchestrator contract
            workflow_id: Workflow correlation ID

        Returns:
            Validation step result

        Raises:
            OnexError: If validation fails
        """
        # Validate contract input data
        if not hasattr(contract, "input_data") or contract.input_data is None:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="Contract missing input_data for validation",
                details={
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "step_id": step.get("step_id", "unknown"),
                },
            )

        emit_log_event(
            LogLevel.DEBUG,
            "Input validation completed successfully",
            {
                "node_id": self.node_id,
                "workflow_id": str(workflow_id),
                "step_id": step.get("step_id", "unknown"),
            },
        )

        return {
            "step_type": "validation",
            "status": "success",
            "validated_at": datetime.now(UTC).isoformat(),
        }

    async def _route_to_onextree(
        self,
        step: dict[str, Any],
        contract: ModelContractOrchestrator,
        workflow_id: UUID,
    ) -> dict[str, Any]:
        """
        Route to OnexTree service for AI intelligence analysis.

        Uses AsyncOnexTreeClient with 500ms timeout and graceful degradation.
        If OnexTree is unavailable, returns fallback intelligence to continue workflow.

        Args:
            step: OnexTree routing step configuration
            contract: Orchestrator contract
            workflow_id: Workflow correlation ID

        Returns:
            OnexTree intelligence analysis result

        Raises:
            OnexError: If routing fails critically (non-degradable errors)
        """
        start_time = time.time()

        # Publish intelligence request event
        await self._publish_event(
            EnumWorkflowEvent.INTELLIGENCE_REQUESTED,
            {
                "workflow_id": str(workflow_id),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        try:
            # Import AsyncOnexTreeClient
            from ....clients.onextree_client import AsyncOnexTreeClient

            # Create client with timeout configuration (500ms max)
            timeout_seconds = self.onextree_timeout_ms / 1000.0

            async with AsyncOnexTreeClient(
                base_url=self.onextree_service_url,
                timeout=timeout_seconds,
                max_retries=1,  # Single retry for fast failure
                enable_cache=True,
                cache_ttl=300,
                app=self.onextree_app,  # Pass ASGI app if provided (for testing)
            ) as client:
                # Extract context from contract input
                context = "general_content_analysis"
                if hasattr(contract, "input_data") and contract.input_data:
                    if isinstance(contract.input_data, dict):
                        context = contract.input_data.get("context", context)

                # Get intelligence with correlation tracking
                intelligence_data = await client.get_intelligence(
                    context=context,
                    include_patterns=True,
                    include_relationships=True,
                    correlation_id=workflow_id,
                    use_cache=True,
                )

                # Extract intelligence fields
                intelligence_result = intelligence_data.get("intelligence", {})

                # Ensure all values are strings for stamp_metadata compatibility
                intelligence_result = {
                    "analysis_type": str(
                        intelligence_result.get("analysis_type", "unknown")
                    ),
                    "confidence_score": str(
                        intelligence_result.get("confidence_score", "0.0")
                    ),
                    "recommendations": str(
                        intelligence_result.get("recommendations", "")
                    ),
                    "analyzed_at": datetime.now(UTC).isoformat(),
                }

                intelligence_time_ms = (time.time() - start_time) * 1000

                # Publish intelligence received event
                await self._publish_event(
                    EnumWorkflowEvent.INTELLIGENCE_RECEIVED,
                    {
                        "workflow_id": str(workflow_id),
                        "analysis_time_ms": intelligence_time_ms,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                emit_log_event(
                    LogLevel.INFO,
                    "OnexTree intelligence analysis completed",
                    {
                        "node_id": self.node_id,
                        "workflow_id": str(workflow_id),
                        "analysis_time_ms": intelligence_time_ms,
                        "confidence_score": float(
                            intelligence_result["confidence_score"]
                        ),
                        "analysis_type": intelligence_result["analysis_type"],
                    },
                )

                return {
                    "step_type": "onextree_intelligence",
                    "status": "success",
                    "intelligence_data": intelligence_result,
                    "intelligence_time_ms": intelligence_time_ms,
                }

        except Exception as e:
            intelligence_time_ms = (time.time() - start_time) * 1000

            emit_log_event(
                LogLevel.WARNING,
                "OnexTree intelligence analysis failed - using graceful degradation",
                {
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "elapsed_ms": intelligence_time_ms,
                },
            )

            # Graceful degradation - return fallback intelligence to continue workflow
            fallback_intelligence = {
                "analysis_type": "fallback",
                "confidence_score": "0.0",
                "recommendations": f"OnexTree unavailable ({type(e).__name__}) - using fallback intelligence",
                "analyzed_at": datetime.now(UTC).isoformat(),
            }

            return {
                "step_type": "onextree_intelligence",
                "status": "degraded",  # Indicate degraded mode
                "intelligence_data": fallback_intelligence,
                "intelligence_time_ms": intelligence_time_ms,
                "error": str(e),
                "degraded": True,  # Flag for monitoring
            }

    async def _route_to_metadata_stamping(
        self,
        step: dict[str, Any],
        contract: ModelContractOrchestrator,
        workflow_id: UUID,
    ) -> dict[str, Any]:
        """
        Route to MetadataStampingService for BLAKE3 hash generation.

        Args:
            step: Hash generation step configuration
            contract: Orchestrator contract
            workflow_id: Workflow correlation ID

        Returns:
            Hash generation result with performance metrics

        Raises:
            OnexError: If routing or hash generation fails
        """
        start_time = time.time()

        try:
            # Check if MetadataStampingClient is available
            if self.metadata_client is None:
                # Fallback to placeholder if client not available
                emit_log_event(
                    LogLevel.WARNING,
                    "MetadataStampingClient not available, using placeholder hash",
                    {
                        "node_id": self.node_id,
                        "workflow_id": str(workflow_id),
                    },
                )
                file_hash = "blake3_" + uuid4().hex[:32]
                hash_generation_time_ms = (time.time() - start_time) * 1000
            else:
                # Use real HTTP client for hash generation
                # Extract file data from contract
                file_data = b""
                if hasattr(contract, "input_data") and contract.input_data:
                    # Try to get content from input_data
                    if hasattr(contract.input_data, "content"):
                        file_data = (
                            contract.input_data.content.encode()
                            if isinstance(contract.input_data.content, str)
                            else contract.input_data.content
                        )
                    elif (
                        isinstance(contract.input_data, dict)
                        and "content" in contract.input_data
                    ):
                        content = contract.input_data["content"]
                        file_data = (
                            content.encode() if isinstance(content, str) else content
                        )
                    else:
                        # Fail fast when content is missing - do not hash placeholder data
                        raise OnexError(
                            error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                            message="Missing required 'content' field in contract input_data for hash generation",
                            details={
                                "node_id": self.node_id,
                                "workflow_id": str(workflow_id),
                                "step_id": step.get("step_id", "unknown"),
                                "available_fields": (
                                    list(contract.input_data.keys())
                                    if isinstance(contract.input_data, dict)
                                    else []
                                ),
                            },
                        )

                # Initialize client if needed
                if (
                    not hasattr(self.metadata_client, "_http_client")
                    or self.metadata_client._http_client is None
                ):
                    await self.metadata_client.initialize()

                # Generate hash using real HTTP client with circuit breaker protection
                hash_result = await self.metadata_client.generate_hash(
                    file_data=file_data,
                    namespace=self.default_namespace,
                    correlation_id=workflow_id,
                )

                # Extract hash and metrics from result
                file_hash = hash_result.get("hash", "unknown")
                hash_generation_time_ms = hash_result.get("execution_time_ms", 0.0)

                emit_log_event(
                    LogLevel.INFO,
                    "BLAKE3 hash generated via HTTP client",
                    {
                        "node_id": self.node_id,
                        "workflow_id": str(workflow_id),
                        "file_hash": file_hash,
                        "generation_time_ms": hash_generation_time_ms,
                        "performance_grade": hash_result.get(
                            "performance_grade", "unknown"
                        ),
                        "file_size_bytes": hash_result.get("file_size_bytes", 0),
                    },
                )

            # Publish hash generated event
            await self._publish_event(
                EnumWorkflowEvent.HASH_GENERATED,
                {
                    "workflow_id": str(workflow_id),
                    "file_hash": file_hash,
                    "generation_time_ms": hash_generation_time_ms,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            emit_log_event(
                LogLevel.INFO,
                "BLAKE3 hash generation completed",
                {
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "file_hash": file_hash,
                    "generation_time_ms": hash_generation_time_ms,
                },
            )

            return {
                "step_type": "hash_generation",
                "status": "success",
                "file_hash": file_hash,
                "hash_generation_time_ms": hash_generation_time_ms,
            }

        except OnexError:
            # Re-raise OnexError as-is to preserve error context
            raise

        except ConnectionError as e:
            # Network connectivity errors
            raise OnexError(
                error_code=EnumCoreErrorCode.CONNECTION_ERROR,
                message=f"Network connection failed during hash generation: {e!s}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "step_id": step.get("step_id", "unknown"),
                    "error_type": "ConnectionError",
                    "service_url": self.metadata_stamping_service_url,
                },
                cause=e,
            )

        except (TimeoutError, asyncio.TimeoutError) as e:
            # Request timeout errors
            raise OnexError(
                error_code=EnumCoreErrorCode.TIMEOUT,
                message=f"Hash generation request timed out: {e!s}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "step_id": step.get("step_id", "unknown"),
                    "error_type": type(e).__name__,
                    "service_url": self.metadata_stamping_service_url,
                },
                cause=e,
            )

        except (ValueError, KeyError, AttributeError) as e:
            # Data validation/parsing errors
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"Invalid data encountered during hash generation: {e!s}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "step_id": step.get("step_id", "unknown"),
                    "error_type": type(e).__name__,
                },
                cause=e,
            )

        except Exception as e:
            # Unexpected errors - wrap with full context and re-raise
            raise OnexError(
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Unexpected error during hash generation: {e!s}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "step_id": step.get("step_id", "unknown"),
                    "error_type": type(e).__name__,
                },
                cause=e,
            )

    async def _create_stamp(
        self,
        step: dict[str, Any],
        contract: ModelContractOrchestrator,
        workflow_id: UUID,
    ) -> dict[str, Any]:
        """
        Create metadata stamp with O.N.E. v0.1 compliance.

        Args:
            step: Stamp creation step configuration
            contract: Orchestrator contract
            workflow_id: Workflow correlation ID

        Returns:
            Stamp creation result with metadata

        Raises:
            OnexError: If stamp creation fails
        """
        start_time = time.time()

        try:
            # Create stamp with O.N.E. v0.1 compliance
            stamp_id = str(uuid4())
            stamp_data = {
                "stamp_id": stamp_id,
                "created_at": datetime.now(UTC).isoformat(),
                "namespace": self.default_namespace,
                "version": "1",  # Must be string for stamp_metadata
                "metadata_version": "0.1",
            }

            stamp_creation_time_ms = (time.time() - start_time) * 1000

            # Publish stamp created event
            await self._publish_event(
                EnumWorkflowEvent.STAMP_CREATED,
                {
                    "workflow_id": str(workflow_id),
                    "stamp_id": stamp_id,
                    "creation_time_ms": stamp_creation_time_ms,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            emit_log_event(
                LogLevel.INFO,
                "Metadata stamp created successfully",
                {
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "stamp_id": stamp_id,
                    "creation_time_ms": stamp_creation_time_ms,
                },
            )

            return {
                "step_type": "stamp_creation",
                "status": "success",
                "stamp_id": stamp_id,
                "stamp_data": stamp_data,
                "stamp_creation_time_ms": stamp_creation_time_ms,
            }

        except (TypeError, ValueError) as e:
            # Data type or value errors during stamp creation
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"Invalid data type or value during stamp creation: {e!s}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "step_id": step.get("step_id", "unknown"),
                    "error_type": type(e).__name__,
                },
                cause=e,
            )

        except Exception as e:
            # Unexpected errors - wrap with full context and re-raise
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected error during stamp creation: {type(e).__name__}",
                {
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "error": str(e),
                },
            )
            raise OnexError(
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Unexpected error during stamp creation: {e!s}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "step_id": step.get("step_id", "unknown"),
                    "error_type": type(e).__name__,
                },
                cause=e,
            )

    async def _aggregate_workflow_results(
        self,
        results: list[dict[str, Any]],
        contract: ModelContractOrchestrator,
        workflow_id: UUID,
    ) -> ModelStampResponseOutput:
        """
        Aggregate workflow step results into final output.

        Args:
            results: List of workflow step results
            contract: Orchestrator contract
            workflow_id: Workflow correlation ID

        Returns:
            Aggregated stamp response output

        Raises:
            OnexError: If aggregation fails
        """
        # Extract data from results
        file_hash = "unknown"
        stamp_id = "unknown"
        stamp_data = {}
        intelligence_data = None

        for result in results:
            if result.get("step_type") == "hash_generation":
                file_hash = result.get("file_hash", "unknown")
            elif result.get("step_type") == "stamp_creation":
                stamp_id = result.get("stamp_id", "unknown")
                stamp_data = result.get("stamp_data", {})
            elif result.get("step_type") == "onextree_intelligence":
                intelligence_data = result.get("intelligence_data")

        # Create stamp metadata
        stamp_metadata = {
            "workflow_id": str(workflow_id),
            "created_at": datetime.now(UTC).isoformat(),
            **stamp_data,
        }

        # Create stamped content (placeholder)
        stamped_content = f"[STAMP:{stamp_id}] Content with embedded metadata stamp"

        return ModelStampResponseOutput(
            stamp_id=stamp_id,
            file_hash=file_hash,
            stamped_content=stamped_content,
            stamp_metadata=stamp_metadata,
            namespace=self.default_namespace,
            op_id=workflow_id,
            version=1,
            metadata_version="0.1",
            workflow_state=EnumWorkflowState.PROCESSING,  # Will be updated by caller
            workflow_id=workflow_id,
            intelligence_data=intelligence_data,
            processing_time_ms=0.0,  # Will be updated by caller
            hash_generation_time_ms=0.0,  # Will be updated by caller
            workflow_steps_executed=len(results),
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),  # Will be updated by caller
        )

    async def _transition_state(
        self,
        workflow_id: UUID,
        current: EnumWorkflowState,
        target: EnumWorkflowState,
    ) -> EnumWorkflowState:
        """
        Transition FSM state with validation.

        Uses FSM subcontract to validate transitions and execute actions.

        Args:
            workflow_id: Workflow correlation ID
            current: Current FSM state
            target: Target FSM state

        Returns:
            New state after transition

        Raises:
            OnexError: If transition is invalid
        """
        workflow_id_str = str(workflow_id)

        # Validate transition is allowed
        if not current.can_transition_to(target):
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"Invalid FSM state transition: {current.value} → {target.value}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "current_state": current.value,
                    "target_state": target.value,
                },
            )

        # Update workflow state
        self.workflow_fsm_states[workflow_id_str] = target

        # Publish state transition event
        await self._publish_event(
            EnumWorkflowEvent.STATE_TRANSITION,
            {
                "workflow_id": workflow_id_str,
                "from_state": current.value,
                "to_state": target.value,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        emit_log_event(
            LogLevel.DEBUG,
            f"FSM state transition: {current.value} → {target.value}",
            {
                "node_id": self.node_id,
                "workflow_id": workflow_id_str,
                "from_state": current.value,
                "to_state": target.value,
            },
        )

        return target

    async def _publish_event(
        self, event_type: EnumWorkflowEvent, data: dict[str, Any]
    ) -> None:
        """
        Publish event to Kafka using EventType subcontract with OnexEnvelopeV1 wrapping.

        Args:
            event_type: Event type identifier
            data: Event payload data
        """
        try:
            # Get Kafka topic name
            topic_name = event_type.get_topic_name(namespace=self.default_namespace)

            # Publish to Kafka if client is available
            if self.kafka_client and self.kafka_client.is_connected:
                # Extract correlation ID from data (workflow_id)
                correlation_id = data.get("workflow_id")

                # Add node metadata to payload
                payload = {
                    **data,
                    "node_id": self.node_id,
                    "published_at": datetime.now(UTC).isoformat(),
                }

                # Publish with OnexEnvelopeV1 wrapping for standardized event format
                # Include Consul service_id for cross-service event correlation
                event_metadata = {
                    "event_category": "workflow_orchestration",
                    "node_type": "orchestrator",
                    "namespace": self.default_namespace,
                }

                # Add consul_service_id if available (enables cross-service correlation)
                if hasattr(self, "_consul_service_id"):
                    event_metadata["consul_service_id"] = self._consul_service_id

                success = await self.kafka_client.publish_with_envelope(
                    event_type=event_type.value,
                    source_node_id=str(self.node_id),
                    payload=payload,
                    topic=topic_name,
                    correlation_id=correlation_id,
                    metadata=event_metadata,
                )

                if success:
                    emit_log_event(
                        LogLevel.DEBUG,
                        f"Published Kafka event (OnexEnvelopeV1): {event_type.value}",
                        {
                            "node_id": self.node_id,
                            "event_type": event_type.value,
                            "topic_name": topic_name,
                            "correlation_id": correlation_id,
                            "envelope_wrapped": True,
                        },
                    )
                else:
                    emit_log_event(
                        LogLevel.WARNING,
                        f"Failed to publish Kafka event: {event_type.value}",
                        {
                            "node_id": self.node_id,
                            "event_type": event_type.value,
                            "topic_name": topic_name,
                        },
                    )
            else:
                # Kafka not available - log event only
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Kafka unavailable, logging event: {event_type.value}",
                    {
                        "node_id": self.node_id,
                        "event_type": event_type.value,
                        "topic_name": topic_name,
                        "data": data,
                    },
                )

        except Exception as e:
            # Log error but don't fail workflow
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to publish Kafka event: {event_type.value}",
                {
                    "node_id": self.node_id,
                    "event_type": event_type.value,
                    "error": str(e),
                },
            )

    async def _update_stamping_metrics(
        self, operation_type: str, execution_time_ms: float, success: bool
    ) -> None:
        """
        Update stamping performance metrics.

        Args:
            operation_type: Type of stamping operation
            execution_time_ms: Execution time in milliseconds
            success: Whether operation succeeded
        """
        if operation_type not in self.stamping_metrics:
            self.stamping_metrics[operation_type] = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "total_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "min_time_ms": float("inf"),
                "max_time_ms": 0.0,
            }

        metrics = self.stamping_metrics[operation_type]
        metrics["total_operations"] += 1

        if success:
            metrics["successful_operations"] += 1
        else:
            metrics["failed_operations"] += 1

        metrics["total_time_ms"] += execution_time_ms
        metrics["avg_time_ms"] = metrics["total_time_ms"] / metrics["total_operations"]
        metrics["min_time_ms"] = min(metrics["min_time_ms"], execution_time_ms)
        metrics["max_time_ms"] = max(metrics["max_time_ms"], execution_time_ms)

    def get_stamping_metrics(self) -> dict[str, dict[str, float]]:
        """
        Get current stamping performance metrics.

        Returns:
            Dictionary of metrics by operation type
        """
        return self.stamping_metrics.copy()

    def get_workflow_state(self, workflow_id: UUID) -> Optional[EnumWorkflowState]:
        """
        Get current FSM state for a workflow.

        Args:
            workflow_id: Workflow correlation ID

        Returns:
            Current workflow state, or None if not found
        """
        return self.workflow_fsm_states.get(str(workflow_id))

    # Event-Driven Coordination Helper Methods

    async def _handle_success(
        self,
        workflow_id: UUID,
        event: dict[str, Any],
        start_time: float,
    ) -> ModelStampResponseOutput:
        """
        Handle successful StateCommitted event from reducer.

        Args:
            workflow_id: Workflow correlation ID
            event: StateCommitted event data
            start_time: Workflow start timestamp

        Returns:
            Successful workflow response

        Raises:
            OnexError: If success handling fails
        """
        workflow_id_str = str(workflow_id)
        processing_time_ms = (time.time() - start_time) * 1000

        try:
            # Extract payload from StateCommitted event
            payload = event.get("payload", {})
            committed_state = payload.get("state", {})

            # Transition to COMPLETED state
            current_state = self.workflow_fsm_states.get(
                workflow_id_str, EnumWorkflowState.PROCESSING
            )
            current_state = await self._transition_state(
                workflow_id, current_state, EnumWorkflowState.COMPLETED
            )

            # Publish workflow completed event
            await self._publish_event(
                EnumWorkflowEvent.WORKFLOW_COMPLETED,
                {
                    "workflow_id": workflow_id_str,
                    "processing_time_ms": processing_time_ms,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            emit_log_event(
                LogLevel.INFO,
                "Event-driven workflow completed successfully",
                {
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "processing_time_ms": processing_time_ms,
                },
            )

            # Build response from committed state
            return ModelStampResponseOutput(
                stamp_id=committed_state.get("stamp_id", str(uuid4())),
                file_hash=committed_state.get("file_hash", "unknown"),
                stamped_content=committed_state.get("stamped_content", ""),
                stamp_metadata=committed_state.get("stamp_metadata", {}),
                namespace=committed_state.get("namespace", self.default_namespace),
                op_id=workflow_id,
                version=committed_state.get("version", 1),
                metadata_version=committed_state.get("metadata_version", "0.1"),
                workflow_state=current_state,
                workflow_id=workflow_id,
                intelligence_data=committed_state.get("intelligence_data"),
                processing_time_ms=processing_time_ms,
                hash_generation_time_ms=committed_state.get(
                    "hash_generation_time_ms", 0.0
                ),
                workflow_steps_executed=committed_state.get(
                    "workflow_steps_executed", 0
                ),
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
            )

        except Exception as e:
            raise OnexError(
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Failed to handle success event: {e!s}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "error_type": type(e).__name__,
                },
            ) from e

    async def _handle_failure(
        self,
        workflow_id: UUID,
        event: dict[str, Any],
        start_time: float,
        retry_count: int = 0,
    ) -> ModelStampResponseOutput:
        """
        Handle ReducerGaveUp event with DAG retry policy.

        Args:
            workflow_id: Workflow correlation ID
            event: ReducerGaveUp event data
            start_time: Workflow start timestamp
            retry_count: Current retry attempt count

        Returns:
            Failure response or retry result

        Raises:
            OnexError: If failure handling fails or max retries exceeded
        """
        workflow_id_str = str(workflow_id)
        processing_time_ms = (time.time() - start_time) * 1000

        # Extract error from event
        payload = event.get("payload", {})
        error_message = payload.get("error", "Unknown error")

        # DAG retry policy: max 3 retries with exponential backoff
        max_retries = 3
        if retry_count < max_retries:
            # Calculate exponential backoff delay
            backoff_seconds = 2**retry_count  # 1s, 2s, 4s

            emit_log_event(
                LogLevel.WARNING,
                f"Workflow failed, retrying in {backoff_seconds}s (attempt {retry_count + 1}/{max_retries})",
                {
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "error": error_message,
                    "retry_count": retry_count,
                },
            )

            # Wait for backoff period
            await asyncio.sleep(backoff_seconds)

            # Retry workflow - re-publish Action event
            # Note: This would need access to the original contract, which we'd need to cache
            raise OnexError(
                error_code=EnumCoreErrorCode.RETRY_REQUIRED,
                message=f"Workflow failed, retry required: {error_message}",
                details={
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                },
            )

        # Max retries exceeded - transition to FAILED state
        current_state = self.workflow_fsm_states.get(
            workflow_id_str, EnumWorkflowState.PROCESSING
        )
        try:
            current_state = await self._transition_state(
                workflow_id, current_state, EnumWorkflowState.FAILED
            )
        except Exception as transition_error:
            emit_log_event(
                LogLevel.WARNING,
                "Failed to transition to FAILED state",
                {
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "error": str(transition_error),
                },
            )

        # Publish failure event
        await self._publish_event(
            EnumWorkflowEvent.WORKFLOW_FAILED,
            {
                "workflow_id": workflow_id_str,
                "error": error_message,
                "retry_count": retry_count,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        raise OnexError(
            error_code=EnumCoreErrorCode.OPERATION_FAILED,
            message=f"Workflow failed after {retry_count} retries: {error_message}",
            details={
                "node_id": self.node_id,
                "workflow_id": workflow_id_str,
                "processing_time_ms": processing_time_ms,
                "retry_count": retry_count,
            },
        )

    async def _handle_timeout(
        self,
        workflow_id: UUID,
        start_time: float,
    ) -> ModelStampResponseOutput:
        """
        Handle event wait timeout.

        Args:
            workflow_id: Workflow correlation ID
            start_time: Workflow start timestamp

        Returns:
            Timeout error response

        Raises:
            OnexError: Timeout error
        """
        workflow_id_str = str(workflow_id)
        processing_time_ms = (time.time() - start_time) * 1000

        # Transition to FAILED state
        current_state = self.workflow_fsm_states.get(
            workflow_id_str, EnumWorkflowState.PROCESSING
        )
        try:
            current_state = await self._transition_state(
                workflow_id, current_state, EnumWorkflowState.FAILED
            )
        except Exception as transition_error:
            emit_log_event(
                LogLevel.WARNING,
                "Failed to transition to FAILED state on timeout",
                {
                    "node_id": self.node_id,
                    "workflow_id": workflow_id_str,
                    "error": str(transition_error),
                },
            )

        # Publish timeout event
        await self._publish_event(
            EnumWorkflowEvent.WORKFLOW_FAILED,
            {
                "workflow_id": workflow_id_str,
                "error": "Workflow timeout - no response from reducer",
                "error_type": "TimeoutError",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        raise OnexError(
            error_code=EnumCoreErrorCode.TIMEOUT,
            message=f"Workflow timeout - no response from reducer within {performance_config.WORKFLOW_COMPLETION_TIMEOUT_SECONDS} seconds",
            details={
                "node_id": self.node_id,
                "workflow_id": workflow_id_str,
                "processing_time_ms": processing_time_ms,
                "timeout_seconds": performance_config.WORKFLOW_COMPLETION_TIMEOUT_SECONDS,
            },
        )

    # Lifecycle Methods

    async def startup(self) -> None:
        """
        Node startup lifecycle hook.

        Initializes container services (including Kafka), publishes introspection data,
        and starts background tasks. Should be called when node is ready to serve requests.
        """
        emit_log_event(
            LogLevel.INFO,
            "NodeBridgeOrchestrator starting up",
            {"node_id": self.node_id},
        )

        # Initialize container services (connects KafkaClient if available)
        if hasattr(self.container, "initialize"):
            try:
                await self.container.initialize()
                emit_log_event(
                    LogLevel.INFO,
                    "Container services initialized successfully",
                    {
                        "node_id": self.node_id,
                        "kafka_connected": (
                            self.kafka_client.is_connected
                            if self.kafka_client
                            else False
                        ),
                    },
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

        # Initialize EventBus service
        if self.event_bus and not self.event_bus.is_initialized:
            try:
                await self.event_bus.initialize()
                emit_log_event(
                    LogLevel.INFO,
                    "EventBus initialized and ready for event-driven coordination",
                    {"node_id": self.node_id},
                )
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"EventBus initialization failed, continuing without event-driven coordination: {e}",
                    {
                        "node_id": self.node_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

        # Publish initial introspection broadcast
        await self.publish_introspection(reason="startup")

        # Start introspection background tasks (heartbeat, registry listener)
        await self.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=30,
            enable_registry_listener=True,
        )

        emit_log_event(
            LogLevel.INFO,
            "NodeBridgeOrchestrator startup complete",
            {"node_id": self.node_id},
        )

    async def shutdown(self) -> None:
        """
        Node shutdown lifecycle hook.

        Stops background tasks, disconnects Kafka, and cleans up resources.
        Should be called when node is preparing to exit.
        """
        emit_log_event(
            LogLevel.INFO,
            "NodeBridgeOrchestrator shutting down",
            {"node_id": self.node_id},
        )

        # Stop introspection background tasks
        await self.stop_introspection_tasks()

        # Shutdown EventBus service
        if self.event_bus and self.event_bus.is_initialized:
            try:
                await self.event_bus.shutdown()
                emit_log_event(
                    LogLevel.INFO,
                    "EventBus shutdown successfully",
                    {"node_id": self.node_id},
                )
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"EventBus shutdown failed: {e}",
                    {
                        "node_id": self.node_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

        # Cleanup container services (disconnects KafkaClient and other services)
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
            "NodeBridgeOrchestrator shutdown complete",
            {"node_id": self.node_id},
        )

    # Health Check Methods

    def _register_component_checks(self) -> None:
        """
        Register component health checks for orchestrator dependencies.

        Checks:
        - node_runtime: Basic node operational status (critical)
        - metadata_stamping: MetadataStampingService health (critical)
        - onextree: OnexTree intelligence service health (non-critical)
        - kafka: Event publishing health (non-critical, degraded mode available)
        - event_bus: Event-driven coordination health (non-critical, can use legacy mode)
        """
        # Parent class registers node_runtime check
        super()._register_component_checks()

        # Register MetadataStampingService health check (critical)
        self.register_component_check(
            "metadata_stamping",
            self._check_metadata_stamping_health,
            critical=True,
            timeout_seconds=performance_config.METADATA_STAMPING_HEALTH_CHECK_TIMEOUT_SECONDS,
        )

        # Register OnexTree health check (non-critical, graceful degradation)
        self.register_component_check(
            "onextree",
            self._check_onextree_health,
            critical=False,
            timeout_seconds=performance_config.ONEXTREE_HEALTH_CHECK_TIMEOUT_SECONDS,
        )

        # Register Kafka health check (non-critical, can queue events)
        self.register_component_check(
            "kafka",
            self._check_kafka_health,
            critical=False,
            timeout_seconds=performance_config.KAFKA_HEALTH_CHECK_TIMEOUT_SECONDS,
        )

        # Register EventBus health check (non-critical, event-driven coordination)
        self.register_component_check(
            "event_bus",
            self._check_event_bus_health,
            critical=False,
            timeout_seconds=performance_config.EVENT_BUS_HEALTH_CHECK_TIMEOUT_SECONDS,
        )

    async def _check_metadata_stamping_health(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check MetadataStampingService health.

        Returns:
            Tuple of (status, message, details)
        """
        return await check_http_service(
            self.metadata_stamping_service_url,
            timeout_seconds=performance_config.METADATA_STAMPING_HEALTH_CHECK_TIMEOUT_SECONDS,
        )

    async def _check_onextree_health(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check OnexTree intelligence service health.

        Returns:
            Tuple of (status, message, details)
        """
        return await check_http_service(
            self.onextree_service_url,
            timeout_seconds=performance_config.ONEXTREE_HEALTH_CHECK_TIMEOUT_SECONDS,
        )

    async def _check_kafka_health(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check Kafka event publishing health.

        Checks:
        - KafkaClient instance exists and is connected
        - Can retrieve health status from client
        - Broker connectivity is functional

        Returns:
            Tuple of (status, message, details)
        """
        try:
            # Check if KafkaClient is available
            if not self.kafka_client:
                return (
                    HealthStatus.DEGRADED,
                    "Kafka client not initialized - events will be logged only",
                    {
                        "broker_url": self.kafka_broker_url,
                        "mode": "degraded",
                        "reason": "kafka_client_not_initialized",
                    },
                )

            # Check KafkaClient connection status
            if not self.kafka_client.is_connected:
                return (
                    HealthStatus.DEGRADED,
                    "Kafka client not connected - events will be queued or logged",
                    {
                        "broker_url": self.kafka_broker_url,
                        "mode": "degraded",
                        "connected": False,
                    },
                )

            # Get detailed health status from KafkaClient
            kafka_health = await self.kafka_client.health_check()

            if kafka_health.get("status") == "healthy":
                return (
                    HealthStatus.HEALTHY,
                    "Kafka client connected and operational",
                    {
                        "broker_url": self.kafka_broker_url,
                        "connected": True,
                        "producer_active": kafka_health.get("producer_active", False),
                    },
                )
            else:
                return (
                    HealthStatus.DEGRADED,
                    f"Kafka health check degraded: {kafka_health.get('error', 'unknown')}",
                    {
                        "broker_url": self.kafka_broker_url,
                        "kafka_health": kafka_health,
                    },
                )

        except Exception as e:
            return (
                HealthStatus.DEGRADED,
                f"Kafka health check failed: {e!s}",
                {
                    "broker_url": self.kafka_broker_url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def _check_event_bus_health(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check EventBus service health for event-driven coordination.

        Checks:
        - EventBus instance exists and is initialized
        - Background consumer task is running
        - Can retrieve health status from service

        Returns:
            Tuple of (status, message, details)
        """
        try:
            # Check if EventBus is available
            if not self.event_bus:
                return (
                    HealthStatus.DEGRADED,
                    "EventBus not initialized - using legacy coordination mode",
                    {
                        "mode": "legacy",
                        "reason": "event_bus_not_initialized",
                    },
                )

            # Check EventBus initialization status
            if not self.event_bus.is_initialized:
                return (
                    HealthStatus.DEGRADED,
                    "EventBus not initialized - call startup() first",
                    {
                        "mode": "legacy",
                        "initialized": False,
                    },
                )

            # Get detailed health status from EventBus
            event_bus_health = await self.event_bus.health_check()

            if event_bus_health.get("status") == "healthy":
                return (
                    HealthStatus.HEALTHY,
                    "EventBus initialized and operational",
                    {
                        "initialized": True,
                        "kafka_connected": event_bus_health.get(
                            "kafka_connected", False
                        ),
                        "consumer_running": event_bus_health.get(
                            "consumer_task_running", False
                        ),
                        "active_listeners": event_bus_health.get("active_listeners", 0),
                        "metrics": event_bus_health.get("metrics", {}),
                    },
                )
            else:
                return (
                    HealthStatus.DEGRADED,
                    f"EventBus health check degraded: {event_bus_health.get('status', 'unknown')}",
                    {
                        "event_bus_health": event_bus_health,
                    },
                )

        except Exception as e:
            return (
                HealthStatus.DEGRADED,
                f"EventBus health check failed: {e!s}",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def _register_with_consul_sync(self) -> None:
        """
        Register orchestrator node with Consul for service discovery (synchronous).

        Registers the orchestrator as a service with health checks pointing to
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
            service_id = f"omninode-bridge-orchestrator-{self.node_id}"

            # Get service port from config (default to 8060 for orchestrator)
            service_port = int(self.container.config.get("service_port", 8060))  # type: ignore[attr-defined]

            # Get service host from config (default to localhost)
            service_host = self.container.config.get("service_host", "localhost")  # type: ignore[attr-defined]

            # Prepare service tags
            service_tags = [
                "onex",
                "bridge",
                "orchestrator",
                f"version:{getattr(self, 'version', '0.1.0')}",
                "omninode_bridge",
            ]

            # Prepare service metadata (note: python-consul v1.1.0 doesn't support meta parameter)
            # Metadata is encoded in tags for MVP compatibility
            service_tags.extend(
                [
                    "node_type:orchestrator",
                    f"namespace:{self.default_namespace}",
                    f"kafka_enabled:{self.kafka_client is not None}",
                ]
            )

            # Health check URL (assumes health endpoint is available)
            health_check_url = f"http://{service_host}:{service_port}/health"

            # Register service with Consul
            consul_client.agent.service.register(
                name="omninode-bridge-orchestrator",
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
        Deregister orchestrator from Consul on shutdown (synchronous).

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


def main() -> int:
    """
    Entry point for node execution.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Direct imports - omnibase_core is required
        from omnibase_core.infrastructure.node_base import NodeBase

        # Contract filename - standard ONEX pattern
        CONTRACT_FILENAME = "contract.yaml"

        node_base = NodeBase(Path(__file__).parent / CONTRACT_FILENAME)
        return 0
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            f"NodeBridgeOrchestrator execution failed: {e!s}",
            {"error": str(e), "error_type": type(e).__name__},
        )
        return 1


if __name__ == "__main__":
    exit(main())
