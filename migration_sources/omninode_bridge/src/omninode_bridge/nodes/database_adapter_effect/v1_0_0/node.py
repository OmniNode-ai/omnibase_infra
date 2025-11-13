"""
Bridge Database Adapter Effect Node.

This node consumes events from bridge nodes (Orchestrator, Reducer, Registry) and
persists data to PostgreSQL following ONEX v2.0 event-driven architecture.

ONEX v2.0 Compliance:
- Effect node type for external service interaction
- Contract-driven dependency injection
- UUID correlation tracking across operations
- Proper error handling with OnexError

Implementation: Phase 2, Agent 7
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

# Direct imports - omnibase_core is required
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

from omninode_bridge.nodes.mixins.health_mixin import (
    HealthCheckMixin,
    HealthStatus,
    check_database_connection,
)

from ._generic_crud_handlers import GenericCRUDHandlers
from .circuit_breaker import DatabaseCircuitBreaker
from .enums.enum_database_operation_type import EnumDatabaseOperationType
from .models.inputs.model_database_operation_input import ModelDatabaseOperationInput
from .models.outputs.model_database_operation_output import ModelDatabaseOperationOutput
from .models.outputs.model_health_response import ModelHealthResponse
from .security_validator import DatabaseSecurityValidator
from .structured_logger import DatabaseStructuredLogger

# Kafka imports
try:
    from aiokafka import AIOKafkaProducer
    from aiokafka.structs import TopicPartition

    AIOKAFKA_AVAILABLE = True
except ImportError:
    AIOKAFKA_AVAILABLE = False
    AIOKafkaProducer = Any  # type: ignore
    TopicPartition = Any  # type: ignore

# Type-checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper import (
        KafkaConsumerWrapper,
    )
    from omninode_bridge.services.postgres_client import PostgresClient

# Aliases for compatibility
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode

# Module logger
logger = logging.getLogger(__name__)


class NodeBridgeDatabaseAdapterEffect(
    NodeEffect, HealthCheckMixin, GenericCRUDHandlers
):
    """
    Bridge Database Adapter Effect Node.

    Consumes Kafka events from bridge nodes and persists data to PostgreSQL.
    Follows ONEX v2.0 event-driven architecture with contract-based dependency injection.

    Supported Operations:
        Legacy Operations (6):
            1. persist_workflow_execution - Insert/update workflow execution records
            2. persist_workflow_step - Insert workflow step history
            3. persist_bridge_state - Upsert bridge aggregation state (UPSERT)
            4. persist_fsm_transition - Insert FSM state transition records
            5. persist_metadata_stamp - Insert metadata stamp audit records
            6. update_node_heartbeat - Update node heartbeat timestamps

        Generic CRUD Operations (via GenericCRUDHandlers mixin):
            1. QUERY - Select records with filtering, sorting, and pagination
            2. DELETE - Delete records matching filters
            3. BATCH_INSERT - Insert multiple records in single transaction
            4. COUNT - Count records matching filters
            5. EXISTS - Check if records exist matching filters

    Event Sources:
        - NodeBridgeOrchestrator: WORKFLOW_STARTED, WORKFLOW_COMPLETED, WORKFLOW_FAILED,
                                 STEP_COMPLETED, STAMP_CREATED, STATE_TRANSITION
        - NodeBridgeReducer: STATE_AGGREGATION_COMPLETED, STATE_TRANSITION
        - NodeBridgeRegistry: NODE_HEARTBEAT, NODE_INTROSPECTION
        - Generic Queries: QUERY_REQUESTED (via Kafka events)

    Database Tables:
        - workflow_executions: Workflow execution tracking
        - workflow_steps: Workflow step history
        - bridge_states: Bridge aggregation state (UPSERT)
        - fsm_transitions: FSM state transition history
        - metadata_stamps: Metadata stamp audit trail
        - node_registrations: Node heartbeat and health

    Performance Targets:
        - Database operations: < 10ms (p95)
        - Event processing: < 50ms (p95)
        - Connection pool efficiency: > 90%
        - Throughput: 1000+ events/second

    Implementation Status: ⏳ Skeleton (Phase 2, Agent 7)
    """

    def __init__(self, container: ModelContainer):
        """
        Initialize database adapter with dependency injection.

        Args:
            container: ONEX container for dependency injection

        Dependencies (resolved through container):
            - postgres_client: PostgreSQL connection pooling
            - postgres_query_executor: SQL query execution
            - postgres_transaction_manager: Transaction management
            - kafka_consumer: Kafka event consumption
        """
        # Initialize base NodeEffect class
        super().__init__(container)

        # Store container reference for later use
        self.container = container

        # Dependencies (will be resolved in initialize())
        self._postgres_client: Optional[PostgresClient] = None
        self._query_executor: Optional[object] = None  # Service resolved from container
        self._transaction_manager: Optional[object] = (
            None  # Service resolved from container
        )
        self._kafka_consumer: Optional[KafkaConsumerWrapper] = None
        self._kafka_producer: Optional[object] = None  # AIOKafkaProducer for DLQ

        # Event consumption background task
        self._event_consumption_task: Optional[asyncio.Task] = None
        self._is_consuming_events = False

        # DLQ configuration and tracking
        self._dlq_topic_suffix = ".dlq"
        self._dlq_enabled = True  # Enable DLQ by default for safety
        self._dlq_message_count = 0
        self._dlq_by_error_type: dict[str, int] = defaultdict(int)

        # Circuit breaker for database resilience
        self._circuit_breaker: Optional[DatabaseCircuitBreaker] = None

        # Structured logger for correlation tracking
        self._logger: Optional[DatabaseStructuredLogger] = None

        # Security validator for input validation
        self._security_validator: Optional[DatabaseSecurityValidator] = None

        # Performance metrics collector
        self._metrics_collector: Optional[object] = None  # Metrics collector instance

        # Initialize tracking timestamp
        self._initialized_at: Optional[datetime] = None

        # Metrics storage (Agent 8 implementation)
        self._metrics_lock = asyncio.Lock()

        # Operation counters by type
        self._operation_counts: dict[str, int] = defaultdict(int)
        self._total_operations = 0

        # Error tracking
        self._error_counts: dict[str, int] = defaultdict(int)
        self._total_errors = 0

        # Performance tracking (circular buffer for last 1000 operations)
        self._execution_times: deque = deque(maxlen=1000)
        self._execution_times_by_type: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Throughput tracking (sliding window of 60 seconds)
        self._operation_timestamps: deque = deque(maxlen=10000)
        self._peak_throughput: float = 0.0

        # Cached metrics (refresh every 5 seconds)
        self._cached_metrics: Optional[dict[str, Any]] = None
        self._last_metrics_calculation: float = 0.0
        self._metrics_cache_ttl: float = 5.0  # seconds

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

        # Initialize health check system
        self.initialize_health_checks()

        # Register with Consul for service discovery
        health_check_mode = container.config.get("health_check_mode", False)
        if not health_check_mode and self.consul_enable_registration:
            self._register_with_consul_sync()

    def _register_component_checks(self) -> None:
        """Register health checks for database adapter components."""
        # Register database connection check (critical)
        self.register_component_check(
            "database_connection",
            self._check_database_health,
            critical=True,
            timeout_seconds=3.0,
        )

        # Register node runtime check (from base mixin)
        super()._register_component_checks()

    async def _check_database_health(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check PostgreSQL database connection health.

        Returns:
            Tuple of (status, message, details)
        """
        if not self._postgres_client:
            return (
                HealthStatus.DEGRADED,
                "Database connection manager not initialized",
                {},
            )

        try:
            # Use the helper function from health_mixin
            return await check_database_connection(
                self._postgres_client, timeout_seconds=2.0
            )
        except ConnectionError as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Database connection failed: {e!s}",
                {"error": str(e), "error_type": "ConnectionError"},
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            return (
                HealthStatus.DEGRADED,
                f"Database health check timed out: {e!s}",
                {"error": str(e), "error_type": "TimeoutError"},
            )
        except Exception as e:
            # Unexpected errors - log for debugging
            logger.error(
                f"Unexpected error in database health check: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return (
                HealthStatus.UNHEALTHY,
                f"Database health check failed unexpectedly: {e!s}",
                {"error": str(e), "error_type": type(e).__name__},
            )

    @property
    def kafka_consumer(self) -> Optional["KafkaConsumerWrapper"]:
        """Get the Kafka consumer instance for testing."""
        return self._kafka_consumer

    async def initialize(self) -> None:
        """
        Initialize database connections and dependencies.

        Resolves protocol dependencies through registry:
        - ProtocolConnectionPoolManager
        - ProtocolQueryExecutor
        - ProtocolTransactionManager

        Raises:
            OnexError: If dependency resolution fails or database connection fails
        """
        import os

        # Set initialization timestamp (Agent 8)
        self._initialized_at = datetime.now(UTC)

        try:
            # Step 1: Resolve service dependencies from container
            # Services are pre-registered in the container during initialization
            self._postgres_client = self.container.get_service("postgres_client")
            if self._postgres_client is None:
                raise OnexError(
                    code=CoreErrorCode.DEPENDENCY_ERROR,
                    message="Failed to resolve postgres_client from container",
                    context={
                        "service": "postgres_client",
                        "container_type": type(self.container).__name__,
                    },
                )

            self._query_executor = self.container.get_service("postgres_query_executor")
            if self._query_executor is None:
                raise OnexError(
                    code=CoreErrorCode.DEPENDENCY_ERROR,
                    message="Failed to resolve postgres_query_executor from container",
                    context={
                        "service": "postgres_query_executor",
                        "container_type": type(self.container).__name__,
                    },
                )

            self._transaction_manager = self.container.get_service(
                "postgres_transaction_manager"
            )
            if self._transaction_manager is None:
                raise OnexError(
                    code=CoreErrorCode.DEPENDENCY_ERROR,
                    message="Failed to resolve postgres_transaction_manager from container",
                    context={
                        "service": "postgres_transaction_manager",
                        "container_type": type(self.container).__name__,
                    },
                )

            # Step 2: Initialize circuit breaker for database resilience
            self._circuit_breaker = DatabaseCircuitBreaker(
                failure_threshold=5,  # Open after 5 consecutive failures
                timeout_seconds=60,  # Wait 60s before retry
                half_open_max_calls=3,  # Allow 3 test calls in HALF_OPEN
                half_open_success_threshold=2,  # Close after 2 successes
            )

            # Step 3: Initialize structured logger for correlation tracking
            self._logger = DatabaseStructuredLogger(
                component="database_adapter_effect",
                node_type="effect",
                enable_query_sanitization=True,
                enable_error_sanitization=True,
            )

            # Step 4: Initialize security validator for input validation
            self._security_validator = DatabaseSecurityValidator(
                max_query_size=10240,  # 10KB
                max_parameter_count=100,
                max_parameter_size=1048576,  # 1MB
                complexity_warning_threshold=20,
                complexity_reject_threshold=50,
                enable_strict_validation=True,
            )

            # Step 5: Test database connectivity (if connection manager is available)
            if self._postgres_client is not None:
                try:
                    # Simple connectivity test query
                    test_correlation_id = uuid4()
                    self._logger.log_health_check(
                        correlation_id=test_correlation_id,
                        status="testing",
                        response_time_ms=0.0,
                        additional_info={"test": "database_connectivity"},
                    )

                    # Execute simple SELECT 1 query through circuit breaker
                    start_time = time.perf_counter()
                    await self._circuit_breaker.execute(
                        self._postgres_client.execute_query, "SELECT 1", []
                    )
                    test_time_ms = (time.perf_counter() - start_time) * 1000

                    self._logger.log_health_check(
                        correlation_id=test_correlation_id,
                        status="healthy",
                        response_time_ms=test_time_ms,
                        additional_info={
                            "test": "database_connectivity",
                            "response_time_ms": round(test_time_ms, 2),
                        },
                    )

                except OnexError:
                    # Re-raise OnexError from circuit breaker (e.g., circuit open)
                    raise

                except ConnectionError as e:
                    raise OnexError(
                        code=CoreErrorCode.CONNECTION_ERROR,
                        message="Database connection failed during initialization test",
                        context={
                            "test_query": "SELECT 1",
                            "circuit_breaker_state": self._circuit_breaker.get_state().value,
                            "error_type": "ConnectionError",
                        },
                        original_error=e,
                    )

                except (TimeoutError, asyncio.TimeoutError) as e:
                    raise OnexError(
                        code=CoreErrorCode.TIMEOUT,
                        message="Database connectivity test timed out during initialization",
                        context={
                            "test_query": "SELECT 1",
                            "circuit_breaker_state": self._circuit_breaker.get_state().value,
                            "error_type": type(e).__name__,
                        },
                        original_error=e,
                    )

                except Exception as e:
                    # Unexpected database errors - log and wrap
                    logger.error(
                        f"Unexpected error during database connectivity test: {type(e).__name__}",
                        exc_info=True,
                    )
                    raise OnexError(
                        code=CoreErrorCode.DATABASE_CONNECTION_ERROR,
                        message=f"Database connectivity test failed unexpectedly: {type(e).__name__}",
                        context={
                            "test_query": "SELECT 1",
                            "circuit_breaker_state": self._circuit_breaker.get_state().value,
                            "error_type": type(e).__name__,
                        },
                        original_error=e,
                    )

            # Step 6: Initialize Kafka consumer and start event consumption
            try:
                self._kafka_consumer = self.container.get_service("kafka_consumer")

                # Subscribe to bridge events
                await self._kafka_consumer.subscribe_to_topics(
                    topics=[
                        "workflow-started",
                        "workflow-completed",
                        "workflow-failed",
                        "step-completed",
                        "stamp-created",
                        "state-transition",
                        "state-aggregation-completed",
                        "node-heartbeat",
                        "query-requested",  # QUERY event consumption
                    ],
                    group_id="database_adapter_consumers",
                    topic_class="evt",
                )

                self._logger.logger.info(
                    "Kafka consumer subscribed to bridge events",
                    correlation_id="initialization",
                    component="database_adapter_effect",
                    topics=self._kafka_consumer.subscribed_topics,
                    group_id=self._kafka_consumer.consumer_group,
                )

                # Start background event consumption loop
                self._is_consuming_events = True
                self._event_consumption_task = asyncio.create_task(
                    self._consume_events_loop()
                )

                self._logger.logger.info(
                    "Event consumption background task started",
                    correlation_id="initialization",
                    component="database_adapter_effect",
                )

                # Step 6b: Initialize Kafka producer for DLQ
                if AIOKAFKA_AVAILABLE and self._dlq_enabled:
                    try:
                        bootstrap_servers = os.getenv(
                            "KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"
                        )
                        self._kafka_producer = AIOKafkaProducer(
                            bootstrap_servers=bootstrap_servers.split(","),
                            value_serializer=lambda v: (
                                json.dumps(v).encode("utf-8") if v else b""
                            ),
                            key_serializer=lambda k: (k.encode("utf-8") if k else None),
                        )
                        await self._kafka_producer.start()

                        self._logger.logger.info(
                            "Kafka producer initialized for DLQ",
                            correlation_id="initialization",
                            component="database_adapter_effect",
                            dlq_enabled=True,
                        )
                    except Exception as e:
                        self._logger.logger.warning(
                            f"Failed to initialize Kafka producer for DLQ: {e!s}",
                            correlation_id="initialization",
                            component="database_adapter_effect",
                        )
                        self._kafka_producer = None
                        self._dlq_enabled = False

            except OnexError as e:
                # Kafka is optional - log warning and continue without event consumption
                self._logger.logger.warning(
                    f"Kafka consumer initialization failed (non-critical): {e.message}",
                    correlation_id="initialization",
                    component="database_adapter_effect",
                    error_code=e.error_code,
                )
                self._kafka_consumer = None
                self._is_consuming_events = False
            except Exception as e:
                # Kafka is optional - log warning and continue without event consumption
                self._logger.logger.warning(
                    f"Kafka consumer initialization failed (non-critical): {e!s}",
                    correlation_id="initialization",
                    component="database_adapter_effect",
                )
                self._kafka_consumer = None
                self._is_consuming_events = False

            # Step 7: Log successful initialization
            init_duration_ms = (
                datetime.now(UTC) - self._initialized_at
            ).total_seconds() * 1000

            self._logger.logger.info(
                "Database adapter initialized successfully",
                correlation_id="initialization",
                component="database_adapter_effect",
                node_type="effect",
                initialization_duration_ms=round(init_duration_ms, 2),
                components_initialized={
                    "connection_manager": self._postgres_client is not None,
                    "query_executor": self._query_executor is not None,
                    "transaction_manager": self._transaction_manager is not None,
                    "circuit_breaker": self._circuit_breaker is not None,
                    "logger": self._logger is not None,
                    "security_validator": self._security_validator is not None,
                    "kafka_consumer": self._kafka_consumer is not None,
                    "event_consumption_enabled": self._is_consuming_events,
                },
                circuit_breaker_config={
                    "failure_threshold": 5,
                    "timeout_seconds": 60,
                    "half_open_max_calls": 3,
                },
            )

        except OnexError:
            # Re-raise OnexError as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise OnexError(
                code=CoreErrorCode.INTERNAL_ERROR,
                message="Unexpected error during database adapter initialization",
                context={
                    "initialized_at": (
                        self._initialized_at.isoformat()
                        if self._initialized_at
                        else None
                    ),
                },
                original_error=e,
            )

    async def process(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Process database operation from event data.

        Routes to appropriate operation handler based on operation_type:
        - persist_workflow_execution
        - persist_workflow_step
        - persist_bridge_state
        - persist_fsm_transition
        - persist_metadata_stamp
        - update_node_heartbeat

        Args:
            input_data: ModelDatabaseOperationInput with operation type and data

        Returns:
            ModelDatabaseOperationOutput with operation results

        Raises:
            OnexError: If operation type is unsupported or operation fails
        """
        # Start timing for performance tracking (Agent 8)
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        operation_type = input_data.operation_type

        try:
            # Step 1: Validate input with security validator
            if self._security_validator is not None:
                validation_result = self._security_validator.validate_operation(
                    input_data
                )
                if not validation_result.valid:
                    raise OnexError(
                        code=CoreErrorCode.VALIDATION_FAILED,
                        message=f"Input validation failed: {'; '.join(validation_result.errors)}",
                        context={
                            "operation_type": operation_type.value,
                            "correlation_id": str(correlation_id),
                            "validation_errors": validation_result.errors,
                            "validation_warnings": validation_result.warnings,
                        },
                    )

                # Log warnings if any
                if validation_result.warnings and self._logger is not None:
                    self._logger.logger.warning(
                        "Input validation warnings",
                        correlation_id=str(correlation_id),
                        operation_type=operation_type,
                        warnings=validation_result.warnings,
                    )

            # Step 2: Log operation start
            if self._logger is not None:
                self._logger.log_operation_start(
                    correlation_id=correlation_id,
                    operation_type=operation_type,
                    metadata={
                        "operation_type": operation_type.value,
                        "has_workflow_execution_data": input_data.workflow_execution_data
                        is not None,
                        "has_workflow_step_data": input_data.workflow_step_data
                        is not None,
                        "has_bridge_state_data": input_data.bridge_state_data
                        is not None,
                        "has_fsm_transition_data": input_data.fsm_transition_data
                        is not None,
                        "has_metadata_stamp_data": input_data.metadata_stamp_data
                        is not None,
                        "has_node_heartbeat_data": input_data.node_heartbeat_data
                        is not None,
                    },
                )

            # Step 3: Route to appropriate handler based on operation type
            # Generic CRUD routing based on operation_type + entity context
            result: ModelDatabaseOperationOutput

            if operation_type == EnumDatabaseOperationType.INSERT:
                # Route INSERT operations based on which data field is populated
                if (
                    hasattr(input_data, "workflow_execution_data")
                    and input_data.workflow_execution_data
                ):
                    result = await self._persist_workflow_execution(input_data)
                elif (
                    hasattr(input_data, "workflow_step_data")
                    and input_data.workflow_step_data
                ):
                    result = await self._persist_workflow_step(input_data)
                elif (
                    hasattr(input_data, "fsm_transition_data")
                    and input_data.fsm_transition_data
                ):
                    result = await self._persist_fsm_transition(input_data)
                elif (
                    hasattr(input_data, "metadata_stamp_data")
                    and input_data.metadata_stamp_data
                ):
                    result = await self._persist_metadata_stamp(input_data)
                else:
                    raise OnexError(
                        code=CoreErrorCode.VALIDATION_FAILED,
                        message="INSERT operation requires entity data (workflow_execution_data, workflow_step_data, fsm_transition_data, or metadata_stamp_data)",
                        context={
                            "operation_type": operation_type.value,
                            "correlation_id": str(correlation_id),
                        },
                    )

            elif operation_type == EnumDatabaseOperationType.UPDATE:
                # Route UPDATE operations based on which data field is populated
                if (
                    hasattr(input_data, "workflow_execution_data")
                    and input_data.workflow_execution_data
                ):
                    result = await self._persist_workflow_execution(input_data)
                elif (
                    hasattr(input_data, "node_heartbeat_data")
                    and input_data.node_heartbeat_data
                ):
                    result = await self._update_node_heartbeat(input_data)
                else:
                    raise OnexError(
                        code=CoreErrorCode.VALIDATION_FAILED,
                        message="UPDATE operation requires entity data (workflow_execution_data or node_heartbeat_data)",
                        context={
                            "operation_type": operation_type.value,
                            "correlation_id": str(correlation_id),
                        },
                    )

            elif operation_type == EnumDatabaseOperationType.UPSERT:
                # Route UPSERT operations based on which data field is populated
                if (
                    hasattr(input_data, "bridge_state_data")
                    and input_data.bridge_state_data
                ):
                    result = await self._persist_bridge_state(input_data)
                else:
                    raise OnexError(
                        code=CoreErrorCode.VALIDATION_FAILED,
                        message="UPSERT operation requires bridge_state_data",
                        context={
                            "operation_type": operation_type.value,
                            "correlation_id": str(correlation_id),
                        },
                    )

            elif operation_type == EnumDatabaseOperationType.QUERY:
                # Route QUERY operations to generic handler (via GenericCRUDHandlers mixin)
                result = await self._handle_query(input_data)

            elif operation_type == EnumDatabaseOperationType.DELETE:
                # Route DELETE operations to generic handler (via GenericCRUDHandlers mixin)
                result = await self._handle_delete(input_data)

            elif operation_type == EnumDatabaseOperationType.BATCH_INSERT:
                # Route BATCH_INSERT operations to generic handler (via GenericCRUDHandlers mixin)
                result = await self._handle_batch_insert(input_data)

            elif operation_type == EnumDatabaseOperationType.COUNT:
                # Route COUNT operations to generic handler (via GenericCRUDHandlers mixin)
                result = await self._handle_count(input_data)

            elif operation_type == EnumDatabaseOperationType.EXISTS:
                # Route EXISTS operations to generic handler (via GenericCRUDHandlers mixin)
                result = await self._handle_exists(input_data)

            else:
                # Unknown operation type (should not happen with Literal type validation)
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_FAILED,
                    message=f"Unsupported operation type: {operation_type}",
                    context={
                        "operation_type": operation_type.value,
                        "correlation_id": str(correlation_id),
                        "supported_operations": [
                            EnumDatabaseOperationType.INSERT.value,
                            EnumDatabaseOperationType.UPDATE.value,
                            EnumDatabaseOperationType.UPSERT.value,
                            EnumDatabaseOperationType.QUERY.value,
                            EnumDatabaseOperationType.DELETE.value,
                            EnumDatabaseOperationType.BATCH_INSERT.value,
                            EnumDatabaseOperationType.COUNT.value,
                            EnumDatabaseOperationType.EXISTS.value,
                        ],
                    },
                )

            # Step 4: Track metrics (Agent 8)
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            async with self._metrics_lock:
                self._total_operations += 1
                self._operation_counts[operation_type.value] += 1
                self._execution_times.append(execution_time_ms)
                self._execution_times_by_type[operation_type.value].append(
                    execution_time_ms
                )
                self._operation_timestamps.append(time.time())

                # Update peak throughput
                current_throughput = self._calculate_throughput()
                if current_throughput > self._peak_throughput:
                    self._peak_throughput = current_throughput

            # Step 5: Log operation completion
            if self._logger is not None:
                self._logger.log_operation_complete(
                    correlation_id=correlation_id,
                    execution_time_ms=execution_time_ms,
                    rows_affected=result.rows_affected,
                    operation_type=operation_type,
                    additional_context={
                        "success": result.success,
                        "error_message": result.error_message,
                    },
                )

            return result

        except OnexError as e:
            # Track error metrics (Agent 8)
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            async with self._metrics_lock:
                self._total_errors += 1
                self._error_counts[operation_type.value] += 1

            # Log error
            if self._logger is not None:
                self._logger.log_operation_error(
                    correlation_id=correlation_id,
                    error=e,
                    operation_type=operation_type,
                    sanitized=True,
                    additional_context={
                        "error_code": e.error_code,
                        "execution_time_ms": round(execution_time_ms, 2),
                    },
                )

            # Return error output
            return ModelDatabaseOperationOutput(
                success=False,
                operation_type=operation_type,
                correlation_id=correlation_id,
                execution_time_ms=int(execution_time_ms),
                rows_affected=0,
                error_message=e.message,
            )

        except Exception as e:
            # Handle unexpected errors
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            async with self._metrics_lock:
                self._total_errors += 1
                self._error_counts[operation_type.value] += 1

            # Wrap in OnexError
            error = OnexError(
                code=CoreErrorCode.INTERNAL_ERROR,
                message=f"Unexpected error processing operation: {operation_type}",
                context={
                    "operation_type": operation_type.value,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": round(execution_time_ms, 2),
                },
                original_error=e,
            )

            # Log error
            if self._logger is not None:
                self._logger.log_operation_error(
                    correlation_id=correlation_id,
                    error=error,
                    operation_type=operation_type,
                    sanitized=True,
                    additional_context={
                        "error_code": error.error_code,
                        "error_type": type(e).__name__,
                    },
                )

            # Return error output
            return ModelDatabaseOperationOutput(
                success=False,
                operation_type=operation_type,
                correlation_id=correlation_id,
                execution_time_ms=int(execution_time_ms),
                rows_affected=0,
                error_message=error.message,
            )

    def _calculate_throughput(self) -> float:
        """
        Calculate current throughput (operations per second) from sliding window.

        Returns:
            Current throughput in operations/second

        Note: This is a helper method for Agent 8's metrics implementation.
        """
        if len(self._operation_timestamps) < 2:
            return 0.0

        # Calculate throughput over last 60 seconds
        current_time = time.time()
        window_start = current_time - 60.0

        # Count operations in window
        operations_in_window = sum(
            1 for ts in self._operation_timestamps if ts >= window_start
        )

        # Calculate time span
        if operations_in_window > 0:
            oldest_in_window = min(
                ts for ts in self._operation_timestamps if ts >= window_start
            )
            time_span = current_time - oldest_in_window
            if time_span > 0:
                return operations_in_window / time_span

        return 0.0

    async def _persist_workflow_execution(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Persist workflow execution record (INSERT or UPDATE).

        Consumes events from NodeBridgeOrchestrator:
        - WORKFLOW_STARTED: INSERT new workflow
        - WORKFLOW_COMPLETED: UPDATE with completion time
        - WORKFLOW_FAILED: UPDATE with error message

        Args:
            input_data: Full ModelDatabaseOperationInput with correlation_id and workflow_execution_data

        Returns:
            ModelDatabaseOperationOutput with operation results

        Implementation: Phase 2, Agent 1
        """
        import time

        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        workflow_data = input_data.workflow_execution_data

        if not workflow_data:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_FAILED,
                message="workflow_execution_data is required",
                context={"correlation_id": str(correlation_id)},
            )

        # Determine if INSERT or UPDATE based on operation type
        operation_type = input_data.operation_type

        try:
            if operation_type == EnumDatabaseOperationType.INSERT:
                # INSERT new workflow execution
                query = """
                    INSERT INTO workflow_executions (
                        correlation_id, workflow_type, current_state, namespace,
                        started_at, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """
                params = [
                    workflow_data.correlation_id,
                    workflow_data.workflow_type,
                    workflow_data.current_state,
                    workflow_data.namespace,
                    workflow_data.started_at,
                    workflow_data.metadata,
                ]

                result_rows = await self._circuit_breaker.execute(
                    self._query_executor.execute_query, query, *params
                )

                generated_id = result_rows[0]["id"] if result_rows else None
                rows_affected = 1

            else:  # UPDATE
                # UPDATE existing workflow execution by correlation_id
                query = """
                    UPDATE workflow_executions
                    SET current_state = $1,
                        completed_at = $2,
                        execution_time_ms = $3,
                        error_message = $4,
                        metadata = $5,
                        updated_at = NOW()
                    WHERE correlation_id = $6
                    RETURNING id
                """
                params = [
                    workflow_data.current_state,
                    workflow_data.completed_at,
                    workflow_data.execution_time_ms,
                    workflow_data.error_message,
                    workflow_data.metadata,
                    workflow_data.correlation_id,
                ]

                result_rows = await self._circuit_breaker.execute(
                    self._query_executor.execute_query, query, *params
                )

                generated_id = result_rows[0]["id"] if result_rows else None
                rows_affected = len(result_rows)

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Log success
            if self._logger:
                self._logger.log_operation_complete(
                    correlation_id=correlation_id,
                    execution_time_ms=execution_time_ms,
                    rows_affected=rows_affected,
                    operation_type=operation_type.value,
                    additional_context={
                        "workflow_type": workflow_data.workflow_type,
                        "state": workflow_data.current_state,
                    },
                )

            return ModelDatabaseOperationOutput(
                success=True,
                correlation_id=correlation_id,
                rows_affected=rows_affected,
                generated_id=str(generated_id) if generated_id else None,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            if self._logger:
                self._logger.log_operation_failed(
                    correlation_id=correlation_id,
                    error_code=str(CoreErrorCode.DATABASE_OPERATION_ERROR),
                    error_message=str(e),
                    operation_type=operation_type.value,
                )

            raise OnexError(
                code=CoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to persist workflow execution: {e!s}",
                context={
                    "correlation_id": str(correlation_id),
                    "operation_type": operation_type.value,
                    "execution_time_ms": execution_time_ms,
                },
            ) from e

    async def _persist_workflow_step(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Persist workflow step history (INSERT).

        Consumes events from NodeBridgeOrchestrator:
        - STEP_COMPLETED: INSERT step history with execution time

        Args:
            input_data: Full ModelDatabaseOperationInput with correlation_id and workflow_step_data

        Returns:
            ModelDatabaseOperationOutput with operation results

        Implementation: Phase 2, Agent 2
        """
        # Phase 2 implementation - see migrations/002_create_workflow_steps.sql for schema
        pass

    async def _persist_bridge_state(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Persist bridge aggregation state (UPSERT).

        Consumes events from NodeBridgeReducer:
        - STATE_AGGREGATION_COMPLETED: UPSERT aggregation state

        Uses PostgreSQL ON CONFLICT for UPSERT:
        - INSERT if bridge_id doesn't exist
        - UPDATE counters and metadata if exists

        Args:
            input_data: Full ModelDatabaseOperationInput with correlation_id and bridge_state_data

        Returns:
            ModelDatabaseOperationOutput with operation results

        Implementation: Phase 2, Agent 3
        """
        # Phase 2 implementation - UPSERT pattern with ON CONFLICT, see migrations/004_create_bridge_states.sql
        pass

    async def _persist_fsm_transition(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Persist FSM state transition record (INSERT).

        Consumes events from NodeBridgeOrchestrator and NodeBridgeReducer:
        - STATE_TRANSITION: INSERT transition history

        Args:
            input_data: Full ModelDatabaseOperationInput with correlation_id and fsm_transition_data

        Returns:
            ModelDatabaseOperationOutput with operation results

        Implementation: Phase 2, Agent 4
        """
        import time
        from uuid import uuid4

        start_time = time.perf_counter()

        # Extract FSM transition data from input
        from .models.inputs.model_fsm_transition_input import ModelFSMTransitionInput

        fsm_data = input_data.fsm_transition_data
        if not fsm_data:
            error_msg = "Missing fsm_transition_data in operation input"
            if self._logger:
                self._logger.log_operation_error(
                    correlation_id=input_data.correlation_id,
                    error=error_msg,
                    operation_type=EnumDatabaseOperationType.INSERT.value,
                )
            return self._build_error_output(
                correlation_id=input_data.correlation_id,
                error_message=error_msg,
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                operation_type=EnumDatabaseOperationType.INSERT.value,
            )

        # Convert dict to ModelFSMTransitionInput
        if isinstance(fsm_data, dict):
            try:
                fsm_input = ModelFSMTransitionInput(**fsm_data)
            except Exception as e:
                error_msg = f"Invalid FSM transition data format: {e}"
                if self._logger:
                    self._logger.log_operation_error(
                        correlation_id=input_data.correlation_id,
                        error=error_msg,
                        operation_type=EnumDatabaseOperationType.INSERT.value,
                    )
                return self._build_error_output(
                    correlation_id=input_data.correlation_id,
                    error_message=error_msg,
                    execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                    operation_type=EnumDatabaseOperationType.INSERT.value,
                )
        else:
            fsm_input = fsm_data

        # Step 1: Input validation with SecurityValidator
        if self._logger:
            self._logger.log_operation_start(
                correlation_id=input_data.correlation_id,
                operation_type=EnumDatabaseOperationType.INSERT.value,
                metadata={
                    "entity_id": str(fsm_input.entity_id),
                    "entity_type": fsm_input.entity_type,
                    "from_state": fsm_input.from_state,
                    "to_state": fsm_input.to_state,
                    "transition_event": fsm_input.transition_event,
                },
            )

        # Validate correlation_id
        correlation_validation = self._security_validator.validate_correlation_id(
            input_data.correlation_id
        )
        if not correlation_validation.valid:
            error_msg = "; ".join(correlation_validation.errors)
            if self._logger:
                self._logger.log_operation_error(
                    correlation_id=input_data.correlation_id,
                    error=error_msg,
                    operation_type=EnumDatabaseOperationType.INSERT.value,
                )
            return self._build_error_output(
                correlation_id=input_data.correlation_id,
                error_message=error_msg,
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                operation_type=EnumDatabaseOperationType.INSERT.value,
            )

        # Validate entity_id is valid UUID
        if not fsm_input.entity_id:
            error_msg = "entity_id cannot be null or empty"
            if self._logger:
                self._logger.log_operation_error(
                    correlation_id=input_data.correlation_id,
                    error=error_msg,
                    operation_type=EnumDatabaseOperationType.INSERT.value,
                )
            return self._build_error_output(
                correlation_id=input_data.correlation_id,
                error_message=error_msg,
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                operation_type=EnumDatabaseOperationType.INSERT.value,
            )

        # Validate entity_type is non-empty
        if not fsm_input.entity_type or len(fsm_input.entity_type.strip()) == 0:
            error_msg = "entity_type cannot be empty"
            if self._logger:
                self._logger.log_operation_error(
                    correlation_id=input_data.correlation_id,
                    error=error_msg,
                    operation_type=EnumDatabaseOperationType.INSERT.value,
                )
            return self._build_error_output(
                correlation_id=input_data.correlation_id,
                error_message=error_msg,
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                operation_type=EnumDatabaseOperationType.INSERT.value,
            )

        # Validate to_state is non-empty (from_state can be None for initial state)
        if not fsm_input.to_state or len(fsm_input.to_state.strip()) == 0:
            error_msg = "to_state cannot be empty"
            if self._logger:
                self._logger.log_operation_error(
                    correlation_id=input_data.correlation_id,
                    error=error_msg,
                    operation_type=EnumDatabaseOperationType.INSERT.value,
                )
            return self._build_error_output(
                correlation_id=input_data.correlation_id,
                error_message=error_msg,
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                operation_type=EnumDatabaseOperationType.INSERT.value,
            )

        # Validate transition_event is non-empty
        if (
            not fsm_input.transition_event
            or len(fsm_input.transition_event.strip()) == 0
        ):
            error_msg = "transition_event cannot be empty"
            if self._logger:
                self._logger.log_operation_error(
                    correlation_id=input_data.correlation_id,
                    error=error_msg,
                    operation_type=EnumDatabaseOperationType.INSERT.value,
                )
            return self._build_error_output(
                correlation_id=input_data.correlation_id,
                error_message=error_msg,
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
                operation_type=EnumDatabaseOperationType.INSERT.value,
            )

        # Step 2: Build SQL INSERT query (append-only audit log)
        # Generate transition_id
        transition_id = uuid4()

        # Use from_state if provided, otherwise use "initial" for NULL from_state
        from_state_value = fsm_input.from_state if fsm_input.from_state else "initial"

        # Build SQL query
        sql_query = """
            INSERT INTO fsm_transitions (
                transition_id, entity_id, correlation_id, entity_type,
                from_state, to_state, transition_event, metadata,
                transition_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING transition_id, transition_timestamp;
        """

        # Prepare parameters
        import json

        query_params = [
            transition_id,
            fsm_input.entity_id,
            input_data.correlation_id,
            fsm_input.entity_type,
            from_state_value,
            fsm_input.to_state,
            fsm_input.transition_event,
            json.dumps(fsm_input.transition_data),
            fsm_input.created_at,
        ]

        # Step 3: Execute with circuit breaker and transaction manager
        try:
            # Execute database operation through circuit breaker
            result = await self._circuit_breaker.execute(
                self._execute_fsm_transition_query,
                sql_query,
                query_params,
                input_data.correlation_id,
            )

            # Calculate execution time
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Log success
            if self._logger:
                self._logger.log_operation_complete(
                    correlation_id=input_data.correlation_id,
                    execution_time_ms=execution_time_ms,
                    rows_affected=1,
                    operation_type=EnumDatabaseOperationType.INSERT.value,
                    additional_context={
                        "transition_id": str(transition_id),
                        "entity_type": fsm_input.entity_type,
                        "fsm_transition": f"{from_state_value} → {fsm_input.to_state}",
                        "transition_event": fsm_input.transition_event,
                    },
                )

            # Build success output
            from .models.outputs.model_database_operation_output import (
                ModelDatabaseOperationOutput,
            )

            return ModelDatabaseOperationOutput(
                success=True,
                operation_type=EnumDatabaseOperationType.INSERT.value,
                correlation_id=input_data.correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=1,
                error_message=None,
            )

        except Exception as e:
            # Calculate execution time
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Log error
            error_message = str(e)
            if self._logger:
                self._logger.log_operation_error(
                    correlation_id=input_data.correlation_id,
                    error=error_message,
                    operation_type=EnumDatabaseOperationType.INSERT.value,
                    additional_context={
                        "entity_type": fsm_input.entity_type,
                        "transition": f"{from_state_value} → {fsm_input.to_state}",
                        "execution_time_ms": execution_time_ms,
                    },
                )

            # Return error output
            return self._build_error_output(
                correlation_id=input_data.correlation_id,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                operation_type=EnumDatabaseOperationType.INSERT.value,
            )

    async def _execute_fsm_transition_query(
        self, sql_query: str, query_params: list, correlation_id: Any
    ) -> dict:
        """
        Execute FSM transition INSERT query with transaction management.

        Args:
            sql_query: SQL INSERT query
            query_params: Query parameters
            correlation_id: Correlation ID for logging

        Returns:
            Query result dictionary

        Raises:
            Exception: If query execution fails
        """
        async with self._transaction_manager.begin():
            result = await self._query_executor.execute(
                sql_query,
                query_params,
                correlation_id=correlation_id,
            )
            return result

    def _build_error_output(
        self,
        correlation_id: Any,
        error_message: str,
        execution_time_ms: int,
        operation_type: str,
    ) -> ModelDatabaseOperationOutput:
        """
        Build error output for database operations.

        Args:
            correlation_id: Correlation ID
            error_message: Error message
            execution_time_ms: Execution time in milliseconds
            operation_type: Operation type (INSERT, UPDATE, UPSERT, etc.)

        Returns:
            ModelDatabaseOperationOutput with error details
        """
        from .models.outputs.model_database_operation_output import (
            ModelDatabaseOperationOutput,
        )

        return ModelDatabaseOperationOutput(
            success=False,
            operation_type=operation_type,
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
            rows_affected=0,
            error_message=error_message,
        )

    async def _persist_metadata_stamp(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Persist metadata stamp audit record (INSERT).

        Consumes events from NodeBridgeOrchestrator:
        - STAMP_CREATED: INSERT stamp audit trail

        Args:
            input_data: Full ModelDatabaseOperationInput with correlation_id and metadata_stamp_data

        Returns:
            ModelDatabaseOperationOutput with operation results

        Implementation: Phase 2, Agent 5
        """
        # Phase 2 implementation - see _persist_metadata_stamp_implementation.py for reference
        pass

    async def _update_node_heartbeat(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Update node heartbeat timestamp (UPDATE).

        Consumes events from all bridge nodes:
        - NODE_HEARTBEAT: UPDATE last_heartbeat and health_status

        Args:
            input_data: Full ModelDatabaseOperationInput with correlation_id and node_heartbeat_data

        Returns:
            ModelDatabaseOperationOutput with operation results

        Implementation: Phase 2, Agent 6
        """
        # Phase 2 implementation - UPDATE node_registrations SET last_heartbeat = NOW() WHERE node_id = $1
        pass

    async def get_health_status(self) -> ModelHealthResponse:
        """
        Get comprehensive health status of database adapter.

        Performs the following checks:
        1. Database connectivity (simple SELECT 1 query)
        2. Connection pool status (available/in-use counts)
        3. Circuit breaker state (CLOSED/OPEN/HALF_OPEN)
        4. Database version
        5. Node uptime

        Performance Target: < 50ms per health check

        Returns:
            ModelHealthResponse with comprehensive health status

        Implementation: Phase 2, Agent 8
        """
        start_time = time.perf_counter()
        correlation_id = uuid4()

        # Initialize response fields
        success = False
        database_status = "UNHEALTHY"
        connection_pool_size = 0
        connection_pool_available = 0
        connection_pool_in_use = 0
        database_version: Optional[str] = None
        uptime_seconds: Optional[int] = None
        error_message: Optional[str] = None

        try:
            # Check 1: Database connectivity with simple SELECT 1 query
            if self._postgres_client is None or self._query_executor is None:
                error_message = (
                    "Database adapter not initialized - "
                    "connection manager or query executor is None"
                )
                database_status = "UNHEALTHY"
            else:
                try:
                    # Check 1: Execute real database connectivity check
                    # Uses circuit breaker for resilience
                    # Will fail fast if database is down
                    await self._circuit_breaker.execute(
                        self._postgres_client.execute_query,
                        "SELECT 1",
                        [],
                    )

                    # Check 2: Get real connection pool statistics
                    # Provides accurate view of pool utilization and capacity
                    pool_stats = await self._postgres_client.get_pool_metrics()
                    connection_pool_size = pool_stats.get("pool_size", 0)
                    connection_pool_available = pool_stats.get("available", 0)
                    connection_pool_in_use = pool_stats.get("in_use", 0)

                    # Check connection pool utilization
                    if connection_pool_size > 0:
                        utilization = connection_pool_in_use / connection_pool_size
                        if utilization > 0.9:
                            database_status = "DEGRADED"
                            error_message = f"Connection pool near capacity ({utilization:.1%} utilization)"
                        elif utilization > 0.8:
                            database_status = "DEGRADED"
                            error_message = (
                                f"Connection pool high utilization ({utilization:.1%})"
                            )
                        else:
                            database_status = "HEALTHY"
                            success = True

                    # Check 3: Circuit breaker state
                    if self._circuit_breaker is not None:
                        circuit_breaker_state = self._circuit_breaker.get_state()
                        if circuit_breaker_state.value == "open":
                            database_status = "UNHEALTHY"
                            error_message = "Circuit breaker is OPEN - database temporarily unavailable"
                            success = False
                        elif circuit_breaker_state.value == "half_open":
                            if database_status == "HEALTHY":
                                database_status = "DEGRADED"
                            error_message = (
                                "Circuit breaker is HALF_OPEN - testing recovery"
                            )

                    # Check 4: Query actual PostgreSQL version
                    # Provides database version information for diagnostics
                    try:
                        version_result = await self._circuit_breaker.execute(
                            self._postgres_client.execute_query,
                            "SELECT version()",
                            [],
                        )
                        if version_result and len(version_result) > 0:
                            database_version = version_result[0].get(
                                "version", "Unknown"
                            )
                        else:
                            database_version = "Unknown (no result)"
                    except Exception as version_error:
                        database_version = f"Unknown (error: {version_error!s})"
                        if self._logger:
                            self._logger.logger.warning(
                                f"Failed to query database version: {version_error!s}"
                            )

                    # Check 5: Node uptime
                    if self._initialized_at is not None:
                        uptime_seconds = int(
                            (datetime.now(UTC) - self._initialized_at).total_seconds()
                        )

                except Exception as e:
                    error_message = f"Database connectivity check failed: {e!s}"
                    database_status = "UNHEALTHY"
                    success = False

        except Exception as e:
            error_message = f"Health check failed: {e!s}"
            database_status = "UNHEALTHY"
            success = False

        # Calculate execution time
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Return comprehensive health response
        return ModelHealthResponse(
            success=success,
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
            database_status=database_status,
            connection_pool_size=connection_pool_size,
            connection_pool_available=connection_pool_available,
            connection_pool_in_use=connection_pool_in_use,
            database_version=database_version,
            uptime_seconds=uptime_seconds,
            last_check_timestamp=datetime.now(UTC),
            error_message=error_message,
        )

    async def get_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for database operations.

        Metrics Categories:
        1. Operation Counters (persist_workflow_execution count, etc.)
        2. Performance Stats (avg execution time, p95, p99)
        3. Circuit Breaker Metrics (open count, closed count, half_open count)
        4. Error Rates (total errors, error rate %)
        5. Throughput (operations per second)

        Performance Target: < 100ms per metrics collection

        Returns:
            Dictionary with comprehensive performance metrics

        Implementation: Phase 2, Agent 8
        """
        # Check if we have cached metrics that are still fresh
        current_time = time.time()
        if (
            self._cached_metrics is not None
            and (current_time - self._last_metrics_calculation)
            < self._metrics_cache_ttl
        ):
            return self._cached_metrics

        # Acquire lock for thread-safe metric calculation
        async with self._metrics_lock:
            # Re-check after acquiring lock (double-checked locking pattern)
            if (
                self._cached_metrics is not None
                and (current_time - self._last_metrics_calculation)
                < self._metrics_cache_ttl
            ):
                return self._cached_metrics

            try:
                # Calculate metrics
                metrics = await self._calculate_metrics()

                # Update cache
                self._cached_metrics = metrics
                self._last_metrics_calculation = current_time

                return metrics

            except Exception as e:
                # Return error metrics on failure
                return {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "metrics_calculation_failed": True,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

    async def _calculate_metrics(self) -> dict[str, Any]:
        """
        Calculate comprehensive metrics (internal method).

        Returns:
            Dictionary with all performance metrics
        """
        # 1. Operation Counters
        operations_by_type = dict(self._operation_counts)

        # 2. Performance Stats
        performance = self._calculate_performance_stats()

        # 3. Circuit Breaker Metrics
        circuit_breaker_metrics = self._get_circuit_breaker_metrics()

        # 4. Error Rates
        error_metrics = self._calculate_error_rates()

        # 5. Throughput
        throughput_metrics = self._calculate_throughput_metrics()

        # 6. Node Uptime
        uptime_seconds = 0
        if self._initialized_at is not None:
            uptime_seconds = int(
                (datetime.now(UTC) - self._initialized_at).total_seconds()
            )

        # 7. Dead Letter Queue Metrics
        dlq_metrics = {
            "enabled": self._dlq_enabled,
            "total_messages_sent": self._dlq_message_count,
            "messages_by_error_type": dict(self._dlq_by_error_type),
        }

        # Build comprehensive metrics dictionary
        return {
            "total_operations": self._total_operations,
            "operations_by_type": operations_by_type,
            "performance": performance,
            "circuit_breaker": circuit_breaker_metrics,
            "errors": error_metrics,
            "throughput": throughput_metrics,
            "dead_letter_queue": dlq_metrics,
            "uptime_seconds": uptime_seconds,
            "metrics_timestamp": datetime.now(UTC).isoformat(),
        }

    def _calculate_performance_stats(self) -> dict[str, Any]:
        """
        Calculate performance statistics from execution times.

        Returns:
            Dictionary with avg, min, max, p95, p99 execution times
        """
        if not self._execution_times:
            return {
                "avg_execution_time_ms": 0.0,
                "min_execution_time_ms": 0.0,
                "max_execution_time_ms": 0.0,
                "p95_execution_time_ms": 0.0,
                "p99_execution_time_ms": 0.0,
                "sample_count": 0,
            }

        # Convert to sorted list for percentile calculations
        sorted_times = sorted(self._execution_times)
        count = len(sorted_times)

        # Calculate basic stats
        avg_time = sum(sorted_times) / count
        min_time = sorted_times[0]
        max_time = sorted_times[-1]

        # Calculate percentiles
        p95_index = int(count * 0.95)
        p99_index = int(count * 0.99)
        p95_time = sorted_times[p95_index] if p95_index < count else max_time
        p99_time = sorted_times[p99_index] if p99_index < count else max_time

        return {
            "avg_execution_time_ms": round(avg_time, 2),
            "min_execution_time_ms": round(min_time, 2),
            "max_execution_time_ms": round(max_time, 2),
            "p95_execution_time_ms": round(p95_time, 2),
            "p99_execution_time_ms": round(p99_time, 2),
            "sample_count": count,
        }

    def _get_circuit_breaker_metrics(self) -> dict[str, Any]:
        """
        Get circuit breaker metrics.

        Returns:
            Dictionary with circuit breaker state and counters
        """
        if self._circuit_breaker is None:
            return {
                "initialized": False,
                "current_state": "UNKNOWN",
            }

        # Get metrics from circuit breaker
        cb_metrics = self._circuit_breaker.get_metrics()

        return {
            "initialized": True,
            "current_state": cb_metrics["state"].upper(),
            "state_duration_seconds": self._calculate_state_duration(cb_metrics),
            "failure_count": cb_metrics["failure_count"],
            "success_count": cb_metrics["success_count"],
            "total_failures": cb_metrics["total_failures"],
            "total_successes": cb_metrics["total_successes"],
            "state_transitions": cb_metrics["state_transitions"],
            "last_failure_time": cb_metrics["last_failure_time"],
            "last_state_change": cb_metrics["last_state_change"],
            "half_open_calls": cb_metrics["half_open_calls"],
            "config": cb_metrics["config"],
        }

    def _calculate_state_duration(self, cb_metrics: dict[str, Any]) -> int:
        """
        Calculate how long circuit breaker has been in current state.

        Args:
            cb_metrics: Circuit breaker metrics dictionary

        Returns:
            Duration in seconds
        """
        if cb_metrics.get("last_state_change") is None:
            return 0

        try:
            last_change = datetime.fromisoformat(cb_metrics["last_state_change"])
            duration = (datetime.now(UTC) - last_change).total_seconds()
            return int(duration)
        except (ValueError, KeyError, TypeError) as e:
            if self._logger:
                self._logger.logger.debug(
                    f"Failed to parse circuit breaker timestamp: {e}"
                )
            return 0
        except Exception as e:
            if self._logger:
                self._logger.logger.warning(
                    f"Unexpected error calculating circuit breaker duration: {e}",
                    exc_info=True,
                )
            return 0

    def _calculate_error_rates(self) -> dict[str, Any]:
        """
        Calculate error rates and error distribution.

        Returns:
            Dictionary with error counts and rates
        """
        errors_by_type = dict(self._error_counts)

        # Calculate overall error rate
        error_rate_percent = 0.0
        if self._total_operations > 0:
            error_rate_percent = (self._total_errors / self._total_operations) * 100

        return {
            "total_errors": self._total_errors,
            "error_rate_percent": round(error_rate_percent, 3),
            "errors_by_type": errors_by_type,
        }

    def _calculate_throughput_metrics(self) -> dict[str, Any]:
        """
        Calculate operations per second using sliding window.

        Returns:
            Dictionary with current and peak throughput
        """
        if not self._operation_timestamps:
            return {
                "operations_per_second": 0.0,
                "peak_operations_per_second": self._peak_throughput,
                "window_size_seconds": 60,
                "sample_count": 0,
            }

        # Remove timestamps older than 60 seconds
        current_time = time.time()
        cutoff_time = current_time - 60.0

        # Filter to recent operations (last 60 seconds)
        recent_operations = [
            ts for ts in self._operation_timestamps if ts >= cutoff_time
        ]

        # Calculate current throughput
        if recent_operations:
            time_span = current_time - min(recent_operations)
            if time_span > 0:
                current_throughput = len(recent_operations) / time_span
            else:
                current_throughput = 0.0
        else:
            current_throughput = 0.0

        # Update peak throughput
        if current_throughput > self._peak_throughput:
            self._peak_throughput = current_throughput

        return {
            "operations_per_second": round(current_throughput, 2),
            "peak_operations_per_second": round(self._peak_throughput, 2),
            "window_size_seconds": 60,
            "sample_count": len(recent_operations),
        }

    def _calculate_safe_offsets(
        self,
        successful_messages: list[dict[str, Any]],
        failed_messages: list[dict[str, Any]],
    ) -> dict[TopicPartition, int]:
        """
        Calculate safe offsets to commit based on successful message processing.

        Strategy:
            - If ALL messages succeeded or went to DLQ: commit all offsets
            - If some messages failed WITHOUT DLQ: calculate per-partition safe offsets

        Args:
            successful_messages: Messages that were successfully processed or sent to DLQ
            failed_messages: Messages that failed AND could not be sent to DLQ

        Returns:
            Dictionary of TopicPartition -> offset to commit (offset + 1 format for Kafka)
        """
        if not AIOKAFKA_AVAILABLE:
            return {}

        offsets_to_commit: dict[TopicPartition, int] = {}

        # If no failures without DLQ, we can commit all consumed offsets
        # This is the simple case - all messages either succeeded or are in DLQ
        if not failed_messages:
            # Return empty dict to signal: commit all consumed offsets
            # The caller will use commit_offsets() without arguments
            return offsets_to_commit

        # Complex case: Some messages failed without DLQ write
        # We need to calculate per-partition safe offsets
        # Group by (topic, partition)
        successful_by_partition: dict[tuple[str, int], list[int]] = defaultdict(list)
        failed_by_partition: dict[tuple[str, int], list[int]] = defaultdict(list)

        for msg in successful_messages:
            key = (msg.get("topic"), msg.get("partition"))
            successful_by_partition[key].append(msg.get("offset"))

        for msg in failed_messages:
            key = (msg.get("topic"), msg.get("partition"))
            failed_by_partition[key].append(msg.get("offset"))

        # For each partition, find the highest contiguous successful offset
        for (topic, partition), successful_offsets in successful_by_partition.items():
            if not successful_offsets:
                continue

            # Sort offsets
            successful_offsets.sort()
            failed_offsets = sorted(failed_by_partition.get((topic, partition), []))

            # Find highest contiguous offset without failures
            highest_safe_offset = None
            for offset in successful_offsets:
                # Check if there are any failed offsets <= this offset
                if any(failed_offset <= offset for failed_offset in failed_offsets):
                    # There's a failure at or before this offset - can't commit
                    continue
                highest_safe_offset = offset

            if highest_safe_offset is not None:
                tp = TopicPartition(topic, partition)
                # Kafka commits offset+1 (next offset to read)
                offsets_to_commit[tp] = highest_safe_offset + 1

        return offsets_to_commit

    async def _publish_query_response(
        self,
        correlation_id: Any,
        event_type: str,
        success: bool,
        result: Optional[ModelDatabaseOperationOutput] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Publish query response to Kafka completion topics.

        Publishes query results to either query-completed or query-failed topics
        based on operation success status.

        Args:
            correlation_id: Correlation ID from original request
            event_type: Original event type (e.g., "query-requested")
            success: Whether operation succeeded
            result: Operation result if successful
            error_message: Error message if failed

        Returns:
            True if publish succeeded, False otherwise
        """
        if not self._kafka_producer:
            self._logger.logger.warning(
                "Cannot publish query response - Kafka producer not initialized",
                correlation_id=str(correlation_id),
                event_type=event_type,
            )
            return False

        try:
            # Determine response topic based on success status
            if success:
                response_topic = "dev.omninode-bridge.database.query-completed.v1"
                response_payload = {
                    "correlation_id": str(correlation_id),
                    "success": True,
                    "rows": (
                        result.result_data.get("items", [])
                        if result and result.result_data
                        else []
                    ),
                    "rows_affected": result.rows_affected if result else 0,
                    "execution_time_ms": result.execution_time_ms if result else 0,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            else:
                response_topic = "dev.omninode-bridge.database.query-failed.v1"
                response_payload = {
                    "correlation_id": str(correlation_id),
                    "success": False,
                    "error": error_message or "Unknown error",
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            # Publish response to topic
            await self._kafka_producer.send_and_wait(
                topic=response_topic,
                value=response_payload,
                key=str(correlation_id),
            )

            self._logger.logger.debug(
                f"Query response published to {response_topic}",
                correlation_id=str(correlation_id),
                success=success,
                topic=response_topic,
            )

            return True

        except Exception as e:
            self._logger.logger.error(
                f"Failed to publish query response: {e!s}",
                correlation_id=str(correlation_id),
                event_type=event_type,
                exc_info=True,
            )
            return False

    async def _publish_to_dlq(
        self, message: dict[str, Any], error: Exception, error_context: dict[str, Any]
    ) -> bool:
        """
        Publish failed message to Dead Letter Queue.

        Args:
            message: Original Kafka message that failed processing
            error: Exception that caused the failure
            error_context: Additional context about the failure

        Returns:
            True if DLQ publish succeeded, False otherwise
        """
        if not self._dlq_enabled or not self._kafka_producer:
            return False

        try:
            # Build DLQ topic name
            original_topic = message.get("topic", "unknown")
            dlq_topic = f"{original_topic}{self._dlq_topic_suffix}"

            # Build DLQ message payload with error information
            dlq_payload = {
                "original_message": message.get("value"),
                "original_topic": original_topic,
                "original_partition": message.get("partition"),
                "original_offset": message.get("offset"),
                "original_timestamp": message.get("timestamp"),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_context": error_context,
                "failed_at": datetime.now(UTC).isoformat(),
                "retry_count": error_context.get("retry_count", 0),
            }

            # Publish to DLQ topic
            await self._kafka_producer.send_and_wait(
                topic=dlq_topic, value=dlq_payload, key=message.get("key")
            )

            # Update DLQ metrics
            self._dlq_message_count += 1
            error_type = type(error).__name__
            self._dlq_by_error_type[error_type] += 1

            self._logger.logger.warning(
                f"Message sent to DLQ: {dlq_topic}",
                correlation_id=error_context.get("correlation_id", uuid4()),
                original_topic=original_topic,
                original_offset=message.get("offset"),
                error_type=error_type,
                dlq_topic=dlq_topic,
            )

            return True

        except Exception as dlq_error:
            self._logger.logger.error(
                f"Failed to publish message to DLQ: {dlq_error!s}",
                correlation_id=error_context.get("correlation_id", uuid4()),
                original_topic=message.get("topic"),
                original_offset=message.get("offset"),
                exc_info=True,
            )
            return False

    async def _consume_events_loop(self) -> None:
        """
        Background event consumption loop with safe offset commit strategy.

        Continuously consumes events from Kafka and routes them to database operations.
        Implements at-least-once delivery semantics with Dead Letter Queue for failures.

        Safe Offset Commit Strategy:
            1. Process each message in batch
            2. Track successful vs failed messages per partition
            3. Send failed messages to DLQ
            4. Commit offsets ONLY for successfully processed messages
            5. Failed messages without DLQ write are NOT committed (will be redelivered)

        Event Routing:
            - WORKFLOW_STARTED → persist_workflow_execution
            - WORKFLOW_COMPLETED → persist_workflow_execution
            - WORKFLOW_FAILED → persist_workflow_execution
            - STEP_COMPLETED → persist_workflow_step
            - STAMP_CREATED → persist_metadata_stamp
            - STATE_TRANSITION → persist_fsm_transition
            - STATE_AGGREGATION_COMPLETED → persist_bridge_state
            - NODE_HEARTBEAT → update_node_heartbeat

        Runs until:
            - self._is_consuming_events is set to False
            - Unrecoverable error occurs
        """
        if not self._kafka_consumer:
            self._logger.logger.error(
                "Cannot start event consumption loop - Kafka consumer not initialized",
                correlation_id="event_consumption_loop",
            )
            return

        self._logger.logger.info(
            "Event consumption loop started",
            correlation_id="event_consumption_loop",
            component="database_adapter_effect",
        )

        try:
            async for messages in self._kafka_consumer.consume_messages_stream(
                batch_timeout_ms=1000
            ):
                if not self._is_consuming_events:
                    self._logger.logger.info(
                        "Event consumption loop shutting down",
                        correlation_id="event_consumption_loop",
                    )
                    break

                # Process batch with per-message tracking for safe offset commit
                successful_messages: list[dict[str, Any]] = []
                failed_messages: list[dict[str, Any]] = []

                for message in messages:
                    message_correlation_id = uuid4()
                    try:
                        # Process message
                        await self._route_event_to_operation(message)
                        successful_messages.append(message)

                    except Exception as e:
                        # Log processing failure
                        self._logger.log_operation_error(
                            correlation_id=message_correlation_id,
                            error=e,
                            operation_type="event_routing",
                            additional_context={
                                "topic": message.get("topic"),
                                "partition": message.get("partition"),
                                "offset": message.get("offset"),
                            },
                        )

                        # Attempt to publish to DLQ
                        dlq_success = await self._publish_to_dlq(
                            message=message,
                            error=e,
                            error_context={
                                "correlation_id": message_correlation_id,
                                "operation_type": "event_routing",
                                "retry_count": 0,
                            },
                        )

                        # Track failure (only mark as "handled" if DLQ succeeded)
                        if dlq_success:
                            # DLQ write succeeded - safe to consider this message handled
                            successful_messages.append(message)
                        else:
                            # DLQ write failed - DO NOT commit offset for this message
                            # It will be redelivered on restart
                            failed_messages.append(message)

                # Calculate safe offsets to commit (only successful messages)
                # Strategy: Commit per-partition up to last successful offset
                if successful_messages:
                    offsets_to_commit = self._calculate_safe_offsets(
                        successful_messages, failed_messages
                    )

                    if offsets_to_commit:
                        try:
                            # Commit only safe offsets
                            await self._kafka_consumer.commit_offsets()

                            self._logger.logger.debug(
                                "Committed Kafka offsets",
                                correlation_id="event_consumption_loop",
                                successful_count=len(successful_messages),
                                failed_count=len(failed_messages),
                                partitions_committed=len(offsets_to_commit),
                            )

                        except Exception as e:
                            self._logger.logger.error(
                                f"Failed to commit Kafka offsets: {e}",
                                correlation_id="event_consumption_loop",
                                exc_info=True,
                            )

                # Log batch processing summary
                if failed_messages:
                    self._logger.logger.warning(
                        f"Batch processed with {len(failed_messages)} failures",
                        correlation_id="event_consumption_loop",
                        total_messages=len(messages),
                        successful=len(successful_messages),
                        failed=len(failed_messages),
                        dlq_message_count=self._dlq_message_count,
                    )

        except asyncio.CancelledError:
            self._logger.logger.info(
                "Event consumption loop cancelled",
                correlation_id="event_consumption_loop",
            )
        except Exception as e:
            self._logger.logger.error(
                f"Event consumption loop failed: {e}",
                correlation_id="event_consumption_loop",
                exc_info=True,
            )
            self._is_consuming_events = False
        finally:
            self._logger.logger.info(
                "Event consumption loop stopped",
                correlation_id="event_consumption_loop",
            )

    async def _route_event_to_operation(self, message: dict[str, Any]) -> None:
        """
        Route Kafka event to appropriate database operation.

        Extracts event type from topic name and routes to handler.

        Args:
            message: Kafka message dictionary with keys:
                - value: Event payload (dict)
                - topic: Full topic name
                - partition: Partition number
                - offset: Message offset

        Event Type Extraction:
            Topic format: {env}.{tenant}.{context}.{class}.{event_type}.{version}
            Example: dev.omninode_bridge.onex.evt.workflow-started.v1
            Extracts: workflow-started
        """
        # Extract event payload and metadata
        event_payload = message.get("value", {})
        topic = message.get("topic", "")
        correlation_id = event_payload.get("correlation_id", uuid4())

        # Extract event type from topic (5th segment in OmniNode topic naming)
        # Format: {env}.{tenant}.{context}.{class}.{event_type}.{version}
        topic_parts = topic.split(".")
        if len(topic_parts) >= 5:
            event_type = topic_parts[4]  # Extract event_type segment
        else:
            self._logger.logger.warning(
                f"Invalid topic format: {topic}",
                correlation_id=str(correlation_id),
                topic=topic,
            )
            return

        # Route to appropriate database operation (using generic CRUD operations)
        operation_input = None

        if event_type == "workflow-started":
            # INSERT new workflow execution
            operation_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.INSERT,
                correlation_id=correlation_id,
                workflow_execution_data=event_payload.get(
                    "workflow_execution_data", {}
                ),
            )

        elif event_type in ["workflow-completed", "workflow-failed"]:
            # UPDATE existing workflow execution with completion status
            operation_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.UPDATE,
                correlation_id=correlation_id,
                workflow_execution_data=event_payload.get(
                    "workflow_execution_data", {}
                ),
            )

        elif event_type == "step-completed":
            # INSERT workflow step history (append-only)
            operation_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.INSERT,
                correlation_id=correlation_id,
                workflow_step_data=event_payload.get("workflow_step_data", {}),
            )

        elif event_type == "state-transition":
            # INSERT FSM state transition (append-only audit log)
            operation_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.INSERT,
                correlation_id=correlation_id,
                fsm_transition_data=event_payload.get("fsm_transition_data", {}),
            )

        elif event_type == "state-aggregation-completed":
            # UPSERT bridge aggregation state (insert or update)
            operation_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.UPSERT,
                correlation_id=correlation_id,
                bridge_state_data=event_payload.get("bridge_state_data", {}),
            )

        elif event_type == "stamp-created":
            # INSERT metadata stamp audit record (append-only)
            operation_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.INSERT,
                correlation_id=correlation_id,
                metadata_stamp_data=event_payload.get("metadata_stamp_data", {}),
            )

        elif event_type == "node-heartbeat":
            # UPDATE node heartbeat timestamp
            operation_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.UPDATE,
                correlation_id=correlation_id,
                node_heartbeat_data=event_payload.get("node_heartbeat_data", {}),
            )

        elif event_type == "query-requested":
            # QUERY database records (generic CRUD operation)
            # Extract query parameters from event payload
            operation_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.QUERY,
                correlation_id=correlation_id,
                entity_type=event_payload.get("entity_type"),
                query_filters=event_payload.get("filters"),
                sort_by=event_payload.get("sort_by"),
                sort_order=event_payload.get("sort_order", "desc"),
                limit=event_payload.get("limit"),
                offset=event_payload.get("offset"),
            )

        else:
            self._logger.logger.warning(
                f"Unknown event type: {event_type}",
                correlation_id=str(correlation_id),
                event_type=event_type,
                topic=topic,
            )
            return

        # Execute database operation
        if operation_input:
            result = await self.process(operation_input)

            # Publish query response for query-requested events
            if event_type == "query-requested":
                await self._publish_query_response(
                    correlation_id=correlation_id,
                    event_type=event_type,
                    success=result.success,
                    result=result if result.success else None,
                    error_message=result.error_message if not result.success else None,
                )

            # Log result
            if result.success:
                self._logger.logger.debug(
                    f"Event processed successfully: {event_type}",
                    correlation_id=str(correlation_id),
                    event_type=event_type,
                    operation_type=result.operation_type,
                    execution_time_ms=result.execution_time_ms,
                )
            else:
                self._logger.logger.error(
                    f"Event processing failed: {event_type}",
                    correlation_id=str(correlation_id),
                    event_type=event_type,
                    error_message=result.error_message,
                )

    async def shutdown(self) -> None:
        """
        Graceful shutdown of database adapter.

        Cleanup tasks:
        - Stop event consumption loop
        - Close Kafka consumer
        - Close database connections
        - Flush pending metrics
        - Close circuit breaker

        Implementation: Phase 2, Agent 7
        """
        if self._logger:
            self._logger.logger.info(
                "Shutting down database adapter",
                correlation_id="shutdown",
                component="database_adapter_effect",
            )

        # Step 1: Stop event consumption loop
        if self._is_consuming_events:
            if self._logger:
                self._logger.logger.info(
                    "Stopping event consumption loop",
                    correlation_id="shutdown",
                )
            self._is_consuming_events = False

            # Cancel event consumption task
            if self._event_consumption_task:
                self._event_consumption_task.cancel()
                try:
                    await self._event_consumption_task
                except asyncio.CancelledError:
                    if self._logger:
                        self._logger.logger.info(
                            "Event consumption task cancelled",
                            correlation_id="shutdown",
                        )

        # Step 2: Close Kafka producer (DLQ)
        if self._kafka_producer:
            try:
                if self._logger:
                    self._logger.logger.info(
                        "Closing Kafka producer (DLQ)",
                        correlation_id="shutdown",
                        dlq_messages_sent=self._dlq_message_count,
                    )
                await self._kafka_producer.stop()
            except Exception as e:
                if self._logger:
                    self._logger.logger.warning(
                        f"Failed to close Kafka producer: {e!s}",
                        correlation_id="shutdown",
                    )

        # Step 3: Close Kafka consumer
        if self._kafka_consumer:
            try:
                if self._logger:
                    self._logger.logger.info(
                        "Closing Kafka consumer",
                        correlation_id="shutdown",
                    )
                await self._kafka_consumer.close_consumer()
                if self._logger:
                    self._logger.logger.info(
                        "Kafka consumer closed",
                        correlation_id="shutdown",
                    )
            except Exception as e:
                if self._logger:
                    self._logger.logger.error(
                        f"Error closing Kafka consumer: {e}",
                        correlation_id="shutdown",
                        exc_info=True,
                    )

        # Step 3: Close database connections with timeout protection
        if self._postgres_client:
            try:
                if self._logger:
                    self._logger.logger.info(
                        "Closing database connections",
                        correlation_id="shutdown",
                    )
                # Close the connection pool gracefully with 5-second timeout
                if hasattr(self._postgres_client, "close"):
                    await asyncio.wait_for(self._postgres_client.close(), timeout=5.0)
                elif hasattr(self._postgres_client, "disconnect"):
                    await asyncio.wait_for(
                        self._postgres_client.disconnect(), timeout=5.0
                    )
                if self._logger:
                    self._logger.logger.info(
                        "Database connections closed",
                        correlation_id="shutdown",
                    )
            except asyncio.TimeoutError:
                if self._logger:
                    self._logger.logger.error(
                        "Timeout closing database connections after 5 seconds",
                        correlation_id="shutdown",
                    )
            except Exception as e:
                if self._logger:
                    self._logger.logger.error(
                        f"Error closing database connections: {e}",
                        correlation_id="shutdown",
                        exc_info=True,
                    )
            finally:
                # Null out connection manager after cleanup attempt
                self._postgres_client = None

        # Step 4: Deregister from Consul for clean service discovery
        self._deregister_from_consul()

        if self._logger:
            self._logger.logger.info(
                "Database adapter shutdown completed",
                correlation_id="shutdown",
                component="database_adapter_effect",
            )

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        """
        Execute database adapter effect operation.

        This is the main entry point for ONEX effect execution.
        Routes operations to appropriate database handlers.

        Args:
            contract: Effect contract with operation configuration

        Returns:
            Effect execution result

        Raises:
            OnexError: If effect execution fails
        """
        operation = contract.input_data.get("operation", "process")

        if operation == "initialize":
            return await self.initialize()
        elif operation == "shutdown":
            await self.shutdown()
            return {
                "status": "shutdown_complete",
                "component": "database_adapter_effect",
            }
        elif operation == "health_check":
            return await self.get_health_status()
        elif operation == "metrics":
            return await self.get_metrics()
        elif operation == "process":
            # Extract database operation input from contract
            operation_input = ModelDatabaseOperationInput(
                operation_type=contract.input_data.get("operation_type"),
                correlation_id=contract.input_data.get("correlation_id", uuid4()),
                workflow_execution_data=contract.input_data.get(
                    "workflow_execution_data"
                ),
                workflow_step_data=contract.input_data.get("workflow_step_data"),
                bridge_state_data=contract.input_data.get("bridge_state_data"),
                fsm_transition_data=contract.input_data.get("fsm_transition_data"),
                metadata_stamp_data=contract.input_data.get("metadata_stamp_data"),
                node_heartbeat_data=contract.input_data.get("node_heartbeat_data"),
            )
            return await self.process(operation_input)
        else:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Unknown operation: {operation}",
                context={
                    "operation": operation,
                    "supported_operations": [
                        "initialize",
                        "shutdown",
                        "health_check",
                        "metrics",
                        "process",
                    ],
                },
            )

    def _register_with_consul_sync(self) -> None:
        """
        Register database adapter node with Consul for service discovery (synchronous).

        Registers the database adapter as a service with health checks pointing to
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
            service_id = f"omninode-bridge-database-adapter-{self.node_id}"

            # Get service port from config (default to 8062 for database adapter)
            service_port = int(self.container.config.get("service_port", 8062))

            # Get service host from config (default to localhost)
            service_host = self.container.config.get("service_host", "localhost")

            # Prepare service tags
            service_tags = [
                "onex",
                "bridge",
                "database_adapter",
                "effect",
                f"version:{getattr(self, 'version', '0.1.0')}",
                "omninode_bridge",
            ]

            # Add metadata as tags
            service_tags.extend(
                [
                    "node_type:database_adapter",
                    f"postgres_available:{self._postgres_client is not None}",
                    f"kafka_consumer_available:{self._kafka_consumer is not None}",
                ]
            )

            # Health check URL (assumes health endpoint is available)
            health_check_url = f"http://{service_host}:{service_port}/health"

            # Register service with Consul
            consul_client.agent.service.register(
                name="omninode-bridge-database-adapter",
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=service_tags,
                http=health_check_url,
                interval="30s",
                timeout="5s",
            )

            if self._logger:
                self._logger.logger.info(
                    "Registered with Consul successfully",
                    extra={
                        "node_id": str(self.node_id),
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
            if self._logger:
                self._logger.logger.warning(
                    "python-consul not installed - Consul registration skipped",
                    extra={"node_id": str(self.node_id)},
                )
        except Exception as e:
            if self._logger:
                self._logger.logger.error(
                    "Failed to register with Consul",
                    extra={
                        "node_id": str(self.node_id),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

    def _deregister_from_consul(self) -> None:
        """
        Deregister database adapter from Consul on shutdown (synchronous).

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

            if self._logger:
                self._logger.logger.info(
                    "Deregistered from Consul successfully",
                    extra={
                        "node_id": str(self.node_id),
                        "service_id": self._consul_service_id,
                    },
                )

        except ImportError:
            # python-consul not installed, silently skip
            pass
        except Exception as e:
            if self._logger:
                self._logger.logger.warning(
                    "Failed to deregister from Consul",
                    extra={
                        "node_id": str(self.node_id),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )


async def main():
    """Main entry point for database adapter effect node."""
    import asyncio
    import logging
    import os

    # Use stub ModelContainer that supports service registry
    from omninode_bridge.nodes.reducer.v1_0_0._stubs import ModelContainer

    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting DatabaseAdapterEffect node...")

    # Create container with configuration from environment
    logger.info("Creating service container with config...")
    container = ModelContainer(
        value={
            "name": "database_adapter_container",
            "version": "1.0.0",
            "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
            "postgres_port": int(os.getenv("POSTGRES_PORT", "5432")),
            "postgres_database": os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
            "postgres_user": os.getenv("POSTGRES_USER", "postgres"),
            "postgres_password": os.getenv("POSTGRES_PASSWORD", ""),
            "postgres_pool_min_size": int(os.getenv("POSTGRES_POOL_MIN_SIZE", "5")),
            "postgres_pool_max_size": int(os.getenv("POSTGRES_POOL_MAX_SIZE", "20")),
            "kafka_bootstrap_servers": os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
            ),
            "kafka_broker_url": os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
            ),
            "environment": os.getenv("ENVIRONMENT", "development"),
        },
        container_type="config",
    )

    logger.info(f"Container created: {type(container).__name__}")

    # Initialize container (this creates kafka_client service)
    if hasattr(container, "initialize") and callable(container.initialize):
        logger.info("Initializing container...")
        await container.initialize()
        logger.info("Container initialized")
    else:
        logger.info("Container has no initialize method")

    # Register required services in container
    logger.info("Registering database and Kafka services...")
    try:
        # Import required service classes
        from omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper import (
            KafkaConsumerWrapper,
        )
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.registry.registry_bridge_database_adapter import (
            QueryExecutorAdapter,
            TransactionManagerAdapter,
        )
        from omninode_bridge.services.postgres_client import PostgresClient

        # Create PostgresClient with configuration from container
        postgres_client = PostgresClient(
            host=container.config.get("postgres_host", "localhost"),
            port=container.config.get("postgres_port", 5432),
            database=container.config.get("postgres_database", "omninode_bridge"),
            user=container.config.get("postgres_user", "postgres"),
            password=container.config.get("postgres_password", ""),
            min_size=container.config.get("postgres_pool_min_size", 5),
            max_size=container.config.get("postgres_pool_max_size", 20),
        )

        # Establish connection pool
        await postgres_client.connect()
        logger.info(
            f"PostgresClient connected to {postgres_client.host}:"
            f"{postgres_client.port}/{postgres_client.database}"
        )

        # Register PostgresClient service
        container.register_service("postgres_client", postgres_client)
        logger.info("Registered postgres_client")

        # Create and register query executor adapter
        query_executor = QueryExecutorAdapter(postgres_client=postgres_client)
        container.register_service("postgres_query_executor", query_executor)
        logger.info("Registered postgres_query_executor")

        # Create and register transaction manager adapter
        transaction_manager = TransactionManagerAdapter(postgres_client=postgres_client)
        container.register_service("postgres_transaction_manager", transaction_manager)
        logger.info("Registered postgres_transaction_manager")

        # Create and register Kafka consumer wrapper
        kafka_bootstrap_servers = container.config.get(
            "kafka_bootstrap_servers",
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"),
        )
        kafka_consumer = KafkaConsumerWrapper(bootstrap_servers=kafka_bootstrap_servers)
        container.register_service("kafka_consumer", kafka_consumer)
        logger.info(
            f"Registered kafka_consumer with servers: {kafka_bootstrap_servers}"
        )

        logger.info("All services registered successfully")
    except ImportError as e:
        logger.error(f"Failed to import required service classes: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to register services: {e}", exc_info=True)
        raise

    # Create and initialize node
    logger.info("Creating NodeBridgeDatabaseAdapterEffect instance...")
    node = NodeBridgeDatabaseAdapterEffect(container)
    logger.info("Node instance created successfully")

    # Initialize node (begins consuming Kafka events)
    logger.info(
        f"Checking if node has initialize method: {hasattr(node, 'initialize')}"
    )
    if hasattr(node, "initialize") and callable(node.initialize):
        logger.info("Calling node.initialize()...")
        try:
            await node.initialize()
            logger.info("DatabaseAdapterEffect node initialized successfully")
        except Exception as e:
            logger.error(f"Error during initialize: {e}", exc_info=True)
            raise
    else:
        logger.info("Node has no initialize method, skipping")

    logger.info("Entering main event loop...")
    # Keep running until interrupted
    try:
        iteration = 0
        while True:
            iteration += 1
            if iteration % 60 == 0:  # Log every minute
                logger.info(f"Node running... (iteration {iteration})")
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
        if hasattr(node, "shutdown") and callable(node.shutdown):
            await node.shutdown()
            logger.info("DatabaseAdapterEffect node shutdown complete")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
