#!/usr/bin/env python3
"""
NodeBridgeReducer - Stamping Metadata Aggregator.

Reduces stamping metadata across workflows, manages FSM state persistence,
and computes aggregation statistics for the omninode_bridge.

ONEX v2.0 Compliance:
- Suffix-based naming: NodeBridgeReducer
- Import from omnibase_core infrastructure
- Subcontract composition for aggregation/state/FSM
- ModelONEXContainer for dependency injection with service resolution
- Streaming data aggregation with async processing

Key Capabilities:
- Stream metadata from orchestrator workflows
- Aggregate by namespace for multi-tenant isolation
- Track FSM states across workflows
- Persist aggregated state to PostgreSQL
- Compute statistics (count, sum, avg, distinct)
- Support windowed aggregation strategies
"""

import logging
import os
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, TypedDict
from uuid import UUID, uuid4

# Import with fallback to stubs when omnibase_core is not available
try:
    from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
    from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_core.models.contracts.model_contract_reducer import (
        ModelContractReducer,
    )
    from omnibase_core.models.contracts.subcontracts import (
        ModelAggregationSubcontract,
        ModelFSMSubcontract,
        ModelStateManagementSubcontract,
    )
    from omnibase_core.nodes.node_reducer import NodeReducer
except ImportError:
    # Fallback to stubs when omnibase_core is not available (testing/demo mode)
    from ._stubs import (
        LogLevel,
        ModelAggregationSubcontract,
        ModelContractReducer,
        ModelFSMSubcontract,
        ModelONEXContainer,
        ModelStateManagementSubcontract,
        NodeReducer,
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

    # Define protocol aliases to concrete types for type hints
    ProtocolServiceRegistry = ModelONEXContainer  # type: ignore[misc,assignment]
    ProtocolOnexEnvelope = dict  # type: ignore[misc,assignment]

# Import node-specific models
# Import performance configuration
from ....config import performance_config
from ....config.batch_sizes import get_batch_manager

# Import Prometheus metrics
from ...metrics.prometheus_metrics import create_reducer_metrics

# Import health check mixin
from ...mixins.health_mixin import HealthCheckMixin, HealthStatus

# Import introspection mixin
from ...mixins.introspection_mixin import IntrospectionMixin

# Import node-specific models
from .models.enum_aggregation_type import EnumAggregationType
from .models.enum_reducer_event import EnumReducerEvent
from .models.model_input_state import ModelReducerInputState
from .models.model_output_state import ModelReducerOutputState

logger = logging.getLogger(__name__)


class WorkflowStateData(TypedDict):
    """
    Type definition for FSM workflow state cache entries.

    Provides strong typing for workflow state data to replace dict[str, Any].
    """

    current_state: str
    previous_state: Optional[str]
    transition_count: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class FSMStateManager:
    """
    FSM State Manager for Workflow State Tracking.

    Manages workflow FSM states by loading state definitions and transitions
    from FSM subcontract YAML (contract-driven architecture).

    Features:
    - State transition validation from subcontract
    - Intent emission for persistence (no direct I/O)
    - State recovery via intents
    - Transition history tracking
    - Guard condition validation

    States and transitions are loaded dynamically from the FSM subcontract,
    not hard-coded.

    ONEX v2.0 Pure Function Pattern:
    - NO direct I/O operations (no PostgreSQL calls)
    - Emits intents for all side effects
    - Pure in-memory state management
    """

    def __init__(
        self,
        container: ProtocolServiceRegistry,
        fsm_config: Optional[ModelFSMSubcontract] = None,
    ) -> None:
        """
        Initialize FSM state manager with FSM subcontract.

        Args:
            container: DI container with service resolution (uses ProtocolServiceRegistry
                      for duck typing - any object with get_service/register_service methods)
            fsm_config: FSM subcontract loaded from contract YAML (optional for now)

        Note:
            Uses ProtocolServiceRegistry protocol for PUBLIC API duck typing.
        """
        self.container = container
        self._fsm_config = fsm_config
        self._state_cache: dict[UUID, WorkflowStateData] = {}
        self._transition_history: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
        self._state_timestamps: dict[UUID, datetime] = {}
        self._pending_intents: list[Any] = []  # Collect intents for batch emission

        # Load states and transitions from FSM subcontract
        if fsm_config:
            # Extract state names from FSM subcontract (normalize to uppercase for consistency)
            self._valid_states = {
                state.state_name.upper() for state in fsm_config.states
            }

            # Build transition map from FSM subcontract (normalize states to uppercase)
            self._valid_transitions: dict[str, set[str]] = defaultdict(set)
            for transition in fsm_config.transitions:
                self._valid_transitions[transition.from_state.upper()].add(
                    transition.to_state.upper()
                )

            # Store terminal states (normalize to uppercase)
            self._terminal_states = {
                state.upper() for state in fsm_config.terminal_states
            }

            logger.info(
                f"FSM State Manager initialized with {len(self._valid_states)} states "
                f"and {len(fsm_config.transitions)} transitions from subcontract"
            )
        else:
            # Fallback to default states when no FSM subcontract is provided
            # This maintains backward compatibility during transition period
            logger.warning(
                "FSM State Manager initialized without FSM subcontract - "
                "using fallback states (PENDING, PROCESSING, COMPLETED, FAILED)"
            )
            self._valid_states = {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}
            self._valid_transitions = {
                "PENDING": {"PROCESSING", "FAILED"},
                "PROCESSING": {"COMPLETED", "FAILED"},
                "COMPLETED": set(),
                "FAILED": set(),
            }
            self._terminal_states = {"COMPLETED", "FAILED"}

    async def initialize_workflow(
        self,
        workflow_id: UUID,
        initial_state: str = "PENDING",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Initialize a new workflow in FSM.

        Args:
            workflow_id: Workflow identifier
            initial_state: Initial FSM state (default: PENDING, case-insensitive)
            metadata: Optional workflow metadata

        Returns:
            True if workflow initialized successfully, False otherwise
        """
        # Check if workflow already exists
        if workflow_id in self._state_cache:
            logger.warning(f"Workflow {workflow_id} already initialized")
            return False

        # Normalize state to uppercase for consistency
        normalized_state = initial_state.upper()

        # Validate state
        if normalized_state not in self._valid_states:
            logger.error(
                f"Invalid initial state: {initial_state}. "
                f"Valid states: {self._valid_states}"
            )
            return False

        self._state_cache[workflow_id] = {
            "current_state": normalized_state,
            "previous_state": None,
            "transition_count": 0,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        self._state_timestamps[workflow_id] = datetime.now(UTC)

        logger.info(
            f"FSM workflow initialized: {workflow_id} in state {normalized_state}"
        )
        return True

    async def transition_state(
        self,
        workflow_id: UUID,
        from_state: str,
        to_state: str,
        trigger: str = "manual",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Transition workflow to a new FSM state with validation.

        Args:
            workflow_id: Workflow identifier
            from_state: Expected current state (for validation, case-insensitive)
            to_state: Target state to transition to (case-insensitive)
            trigger: Event trigger causing transition
            metadata: Optional transition metadata

        Returns:
            True if transition successful, False otherwise
        """
        # Normalize states to uppercase
        normalized_from_state = from_state.upper()
        normalized_to_state = to_state.upper()

        # Get current state
        if workflow_id not in self._state_cache:
            logger.error(f"Workflow {workflow_id} not initialized")
            return False

        current_state = self._state_cache[workflow_id]["current_state"]

        # Validate current state matches expected from_state
        if current_state != normalized_from_state:
            logger.error(
                f"Current state mismatch for {workflow_id}: "
                f"expected {normalized_from_state}, got {current_state}"
            )
            return False

        # Validate transition
        if not self._validate_transition(current_state, normalized_to_state):
            error_msg = (
                f"Invalid transition: {current_state} -> {normalized_to_state} "
                f"for workflow {workflow_id}"
            )
            logger.error(error_msg)
            return False

        # Record transition
        transition_record = {
            "from_state": current_state,
            "to_state": normalized_to_state,
            "trigger": trigger,
            "timestamp": datetime.now(UTC),
            "metadata": metadata or {},
        }
        self._transition_history[workflow_id].append(transition_record)

        # Update state
        old_state = current_state
        self._state_cache[workflow_id]["previous_state"] = old_state
        self._state_cache[workflow_id]["current_state"] = normalized_to_state
        self._state_cache[workflow_id]["transition_count"] += 1
        self._state_cache[workflow_id]["updated_at"] = datetime.now(UTC)
        self._state_timestamps[workflow_id] = datetime.now(UTC)

        # Generate intent for state transition persistence (no direct I/O)
        self._generate_persist_transition_intent(workflow_id, transition_record)

        logger.info(
            f"FSM state transition: {workflow_id} "
            f"{old_state} -> {normalized_to_state} (trigger: {trigger})"
        )

        return True

    def get_state(self, workflow_id: UUID) -> Optional[str]:
        """
        Get current FSM state for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Current state string, or None if workflow not found
        """
        if workflow_id not in self._state_cache:
            return None
        return self._state_cache[workflow_id]["current_state"]

    def get_transition_history(self, workflow_id: UUID) -> list[dict[str, Any]]:
        """
        Get transition history for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of transition records
        """
        return self._transition_history.get(workflow_id, [])

    def get_pending_intents(self) -> list[Any]:
        """
        Retrieve and clear pending intents.

        Returns:
            List of pending intents for EFFECT node execution
        """
        intents = self._pending_intents.copy()
        self._pending_intents.clear()
        return intents

    async def recover_states(self) -> dict[str, int]:
        """
        Generate intent for FSM state recovery from storage.

        ONEX v2.0 Pure Function Pattern:
        - NO direct I/O operations
        - Generates intent for EFFECT node to recover states

        Returns:
            Recovery statistics dict (will be populated by EFFECT node response)

        Note:
            Intent will be consumed by NodeBridgeStoreEffect which performs
            the actual PostgreSQL query to fsm_workflow_states table and returns
            recovered state data for in-memory restoration.

            EFFECT Node Implementation Requirements:
            - Connect to PostgreSQL via connection manager
            - Query fsm_workflow_states table for all active workflows
            - Return recovered state data for in-memory restoration
            - Process response during node startup to restore _state_cache
        """
        # Import ModelIntent here to avoid circular imports
        from .models.enum_intent_type import EnumIntentType
        from .models.model_intent import ModelIntent

        logger.info("Generating FSM state recovery intent")

        # Generate recovery intent for EFFECT node
        intent = ModelIntent(
            intent_type=EnumIntentType.RECOVER_FSM_STATES.value,
            target="store_effect",
            payload={
                "recovery_id": str(uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "request_all_workflows": True,
            },
            priority=2,  # Highest priority for startup recovery
        )

        self._pending_intents.append(intent)

        # Note: Recovery is deferred to NodeBridgeStoreEffect (EFFECT node)
        # See docstring above for EFFECT node implementation requirements

        logger.info(
            "FSM state recovery intent generated - waiting for EFFECT node execution"
        )

        return {
            "recovered": 0,
            "failed": 0,
            "total": 0,
            "deferred": True,  # Indicates recovery is deferred to EFFECT node
        }

    def _validate_transition(self, from_state: str, to_state: str) -> bool:
        """
        Validate FSM state transition.

        Args:
            from_state: Current state
            to_state: Target state

        Returns:
            True if transition is valid, False otherwise
        """
        # Validate states exist
        if from_state not in self._valid_states:
            logger.error(f"Invalid from_state: {from_state}")
            return False

        if to_state not in self._valid_states:
            logger.error(f"Invalid to_state: {to_state}")
            return False

        # Check if transition is allowed
        allowed_transitions = self._valid_transitions.get(from_state, set())
        if to_state not in allowed_transitions:
            logger.warning(
                f"Transition not allowed: {from_state} -> {to_state}. "
                f"Allowed: {allowed_transitions}"
            )
            return False

        return True

    def _generate_persist_transition_intent(
        self,
        workflow_id: UUID,
        transition_record: dict[str, Any],
    ) -> None:
        """
        Generate intent for FSM state transition persistence.

        ONEX v2.0 Pure Function Pattern:
        - NO direct I/O operations
        - Generates intent for EFFECT node to persist state

        Args:
            workflow_id: Workflow identifier
            transition_record: Transition metadata

        Note:
            Intent will be consumed by NodeBridgeStoreEffect which performs
            the actual PostgreSQL upsert to fsm_workflow_states table.

            EFFECT Node Implementation Requirements:
            - Connect to PostgreSQL via query executor
            - Upsert to fsm_workflow_states table with transaction
            - Publish FSM_STATE_TRANSITIONED event to Kafka
            - Return operation status with execution metrics
        """
        # Import ModelIntent here to avoid circular imports
        from .models.enum_intent_type import EnumIntentType
        from .models.model_intent import ModelIntent

        # Generate persistence intent for EFFECT node
        intent = ModelIntent(
            intent_type=EnumIntentType.PERSIST_FSM_TRANSITION.value,
            target="store_effect",
            payload={
                "workflow_id": str(workflow_id),
                "current_state": self._state_cache[workflow_id]["current_state"],
                "previous_state": self._state_cache[workflow_id]["previous_state"],
                "transition_count": self._state_cache[workflow_id]["transition_count"],
                "transition_record": transition_record,
                "transition_history": self._transition_history[workflow_id],
                "metadata": self._state_cache[workflow_id]["metadata"],
                "timestamp": datetime.now(UTC).isoformat(),
            },
            priority=1,  # High priority for FSM state persistence
        )

        self._pending_intents.append(intent)

        # Note: Persistence is deferred to NodeBridgeStoreEffect (EFFECT node)
        # See docstring above for EFFECT node implementation requirements


class NodeBridgeReducer(NodeReducer, HealthCheckMixin, IntrospectionMixin):
    """
    Bridge Reducer for metadata aggregation and state management.

    Aggregates stamping metadata:
    1. Receive stamp metadata stream from orchestrator
    2. Group by aggregation strategy (namespace, time window, etc.)
    3. Compute statistics (total stamps, namespaces, file types)
    4. Track FSM states for workflows
    5. Persist aggregated state to PostgreSQL
    6. Return aggregation results

    Subcontracts:
    - Aggregation: Data aggregation strategies and windowing
    - StateManagement: PostgreSQL persistence with transactions
    - FSM: State tracking and transition persistence

    Streaming Strategy:
    - Windowed aggregation (default: 5000ms windows)
    - Batch processing (default: 100 items per batch)
    - Incremental state updates for performance
    """

    def __init__(self, container: ProtocolServiceRegistry) -> None:
        """
        Initialize NodeBridgeReducer with dependency injection container.

        Args:
            container: DI container with service resolution (uses ProtocolServiceRegistry
                      for duck typing - any object with get_service/register_service methods)

        Note:
            Uses ProtocolServiceRegistry protocol type hint for PUBLIC API duck typing.
            Internal implementation still uses concrete ModelONEXContainer from omnibase_core.
            This enables flexibility while maintaining type safety.
        """
        super().__init__(container)

        # Load subcontracts from contract model (already loaded by NodeReducer base class)

        # Load FSM subcontract
        self._fsm_config: ModelFSMSubcontract | None = None
        if hasattr(self, "contract_model") and hasattr(
            self.contract_model, "state_transitions"
        ):
            state_transitions = self.contract_model.state_transitions
            # Check if it's a ModelFSMSubcontract instance (not just a dict or $ref)
            if isinstance(state_transitions, ModelFSMSubcontract):
                self._fsm_config = state_transitions
                logger.info(
                    f"Loaded FSM subcontract from contract: {self._fsm_config.state_machine_name} "
                    f"with {len(self._fsm_config.states)} states"
                )

        # Load StateManagement subcontract
        self._state_config: ModelStateManagementSubcontract | None = None
        if hasattr(self, "contract_model") and hasattr(
            self.contract_model, "state_management"
        ):
            state_management = self.contract_model.state_management
            if isinstance(state_management, ModelStateManagementSubcontract):
                self._state_config = state_management
                logger.info("Loaded StateManagement subcontract from contract")

        # Load Aggregation subcontract
        self._aggregation_config: ModelAggregationSubcontract | None = None
        if hasattr(self, "contract_model") and hasattr(
            self.contract_model, "aggregation"
        ):
            aggregation = self.contract_model.aggregation
            if isinstance(aggregation, ModelAggregationSubcontract):
                self._aggregation_config = aggregation
                logger.info("Loaded Aggregation subcontract from contract")

        # Internal state for aggregation
        self._aggregation_buffer: dict[str, dict[str, Any]] = defaultdict(dict)
        self._current_window_start: datetime | None = None

        # FSM state manager (initialized with FSM subcontract from contract)
        self._fsm_manager = FSMStateManager(container, self._fsm_config)

        # Pending event publishing intents (ONEX v2.0 pure function pattern)
        self._pending_event_intents: list[Any] = []

        # Configuration loading with ConfigLoader (Vault + Env + YAML cascade)
        # Load configuration from YAML files with environment overrides
        try:
            # Import ConfigLoader for typed configuration
            import os

            from ....config.config_loader import get_reducer_config

            # Load reducer configuration (cached via lru_cache for performance)
            # Configuration cascade: reducer.yaml → development.yaml → env vars → Vault
            config = get_reducer_config(
                environment=os.getenv("ENVIRONMENT", "development")
            )

            # Extract configuration values with type safety
            self.kafka_broker_url = config.kafka.bootstrap_servers
            self.default_namespace = config.node.namespace

            # Consul configuration for service discovery
            self.consul_host = config.consul.host
            self.consul_port = config.consul.port
            self.consul_enable_registration = config.consul.enable_registration

            # Store additional configuration for future use
            self._config: Optional[Any] = (
                config  # Type: Optional to support fallback mode
            )
            self._aggregation_batch_size = config.reducer.aggregation_batch_size
            self._aggregation_window_seconds = config.reducer.aggregation_window_seconds

            emit_log_event(
                LogLevel.INFO,
                "ConfigLoader loaded successfully for reducer",
                {
                    "node_id": getattr(self, "node_id", "reducer"),
                    "environment": config.environment,
                    "kafka_broker": self.kafka_broker_url,
                    "namespace": self.default_namespace,
                    "batch_size": self._aggregation_batch_size,
                },
            )

        except ImportError as e:
            # ConfigLoader module not available
            emit_log_event(
                LogLevel.WARNING,
                "ConfigLoader not available - falling back to container.config",
                {"node_id": getattr(self, "node_id", "reducer"), "error": str(e)},
            )
            # Set fallback values
            self._set_fallback_config(container)

        except FileNotFoundError as e:
            # Configuration file not found
            emit_log_event(
                LogLevel.WARNING,
                "Configuration file not found - falling back to container.config",
                {
                    "node_id": getattr(self, "node_id", "reducer"),
                    "error": str(e),
                    "error_type": "FileNotFoundError",
                },
            )
            # Set fallback values
            self._set_fallback_config(container)

        except OSError as e:
            # File system errors reading configuration
            emit_log_event(
                LogLevel.WARNING,
                "File system error loading configuration - falling back to container.config",
                {
                    "node_id": getattr(self, "node_id", "reducer"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Set fallback values
            self._set_fallback_config(container)

        except (ValueError, KeyError, AttributeError) as e:
            # Configuration validation or attribute access errors
            emit_log_event(
                LogLevel.WARNING,
                "Configuration validation failed - falling back to container.config",
                {
                    "node_id": getattr(self, "node_id", "reducer"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Set fallback values
            self._set_fallback_config(container)

        except Exception as e:
            # Unexpected errors - log as ERROR for visibility but continue with fallback
            emit_log_event(
                LogLevel.ERROR,
                "Unexpected error loading configuration - falling back to container.config",
                {
                    "node_id": getattr(self, "node_id", "reducer"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Set fallback values
            self._set_fallback_config(container)

        # Get or create KafkaClient from container (skip if in health check mode)
        try:
            # ModelONEXContainer stores config in 'value' field
            config_dict = (
                container.value
                if hasattr(container, "value")
                else getattr(container, "config", {})
            )
            health_check_mode = (
                config_dict.get("health_check_mode", False)
                if hasattr(config_dict, "get")
                else False
            )
        except Exception:
            health_check_mode = False

        try:
            self.kafka_client = (
                container.get_service("kafka_client")
                if hasattr(container, "get_service")
                else None
            )
        except Exception:
            self.kafka_client = None

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
                # Register with container if supported
                if hasattr(container, "register_service"):
                    try:
                        container.register_service("kafka_client", self.kafka_client)  # type: ignore[call-arg,arg-type,unused-coroutine]
                    except Exception:
                        pass  # Ignore if registration fails
                emit_log_event(
                    LogLevel.INFO,
                    "KafkaClient initialized successfully in reducer",
                    {"node_id": getattr(self, "node_id", "reducer")},
                )
            except ImportError:
                emit_log_event(
                    LogLevel.WARNING,
                    "KafkaClient not available - events will be logged only",
                    {"node_id": getattr(self, "node_id", "reducer")},
                )
                self.kafka_client = None
        elif health_check_mode:
            # In health check mode, skip Kafka initialization
            emit_log_event(
                LogLevel.DEBUG,
                "Health check mode enabled - skipping Kafka initialization",
                {"node_id": getattr(self, "node_id", "reducer")},
            )
            self.kafka_client = None

        # Initialize health check system
        self.initialize_health_checks()

        # Initialize introspection system
        # ModelONEXContainer stores config in 'value' field
        config_dict = (
            container.value
            if hasattr(container, "value")
            else getattr(container, "config", {})
        )
        health_check_mode = (
            config_dict.get("health_check_mode", False)
            if hasattr(config_dict, "get")
            else False
        )
        if not health_check_mode:
            self.initialize_introspection()
        else:
            # Initialize minimal introspection state for health check
            self._cached_node_type = "reducer"
            self._introspection_cache = {}
            self._cache_timestamps = {}

        # Initialize Prometheus metrics (feature flag controlled)
        enable_prometheus = True  # Default to enabled
        if hasattr(self, "_config") and self._config:
            enable_prometheus = getattr(
                self._config.monitoring, "enable_prometheus", True
            )
        else:
            # Fallback to container config or True
            enable_prometheus = (
                config_dict.get("enable_prometheus", True)
                if hasattr(config_dict, "get")
                else True
            )

        self.metrics_collector = create_reducer_metrics(
            enable_prometheus=enable_prometheus
        )
        if enable_prometheus:
            emit_log_event(
                LogLevel.INFO,
                "Prometheus metrics enabled for reducer",
                {
                    "node_id": getattr(self, "node_id", "reducer"),
                    "enable_prometheus": enable_prometheus,
                },
            )
        else:
            emit_log_event(
                LogLevel.INFO,
                "Prometheus metrics disabled for reducer",
                {
                    "node_id": getattr(self, "node_id", "reducer"),
                    "enable_prometheus": enable_prometheus,
                },
            )

        # Register with Consul for service discovery (skip in health check mode)
        if not health_check_mode and self.consul_enable_registration:
            # Call synchronous registration wrapper
            self._register_with_consul_sync()

    def _set_fallback_config(self, container: ProtocolServiceRegistry) -> None:
        """
        Set fallback configuration values when ConfigLoader fails.

        This method tries to extract configuration from the container using dict-style
        access first, then falls back to environment variables as a last resort.

        Args:
            container: Service registry container with configuration
        """
        # Try dict-style access (for mocks and testing)
        # ModelONEXContainer stores config in 'value' field, not 'config'
        config_dict = (
            container.value
            if hasattr(container, "value")
            else getattr(container, "config", {})
        )

        if hasattr(config_dict, "get") and callable(config_dict.get):
            self.default_namespace = config_dict.get(
                "default_namespace", "omninode.bridge"
            )
            self.kafka_broker_url = config_dict.get(
                "kafka_broker_url",
                os.getenv("KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"),
            )
            # Consul configuration fallback
            self.consul_host = config_dict.get(
                "consul_host", os.getenv("CONSUL_HOST", "omninode-bridge-consul")
            )
            self.consul_port = config_dict.get(
                "consul_port", int(os.getenv("CONSUL_PORT", "28500"))
            )
            self.consul_enable_registration = config_dict.get(
                "consul_enable_registration", True
            )
        else:
            # Final fallback to environment variables (last resort)
            self.default_namespace = "omninode.bridge"
            self.kafka_broker_url = os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
            )
            # Consul configuration defaults
            self.consul_host = os.getenv("CONSUL_HOST", "omninode-bridge-consul")
            self.consul_port = int(os.getenv("CONSUL_PORT", "28500"))
            self.consul_enable_registration = True

        # Set default batch sizes for fallback mode
        self._config = None  # Fallback when config loading fails
        self._aggregation_batch_size = 100  # Default batch size
        self._aggregation_window_seconds = 5000  # Default window size

    @property
    def _fsm_state_tracker(self) -> dict[UUID, str]:
        """
        Backward compatibility property for tests.

        Returns a dict representation of FSM states for testing.
        This property maintains API compatibility while using the new FSM manager internally.

        Returns:
            Dict mapping workflow IDs to their current states
        """
        return {
            workflow_id: state_data["current_state"]
            for workflow_id, state_data in self._fsm_manager._state_cache.items()
        }

    def get_pending_event_intents(self) -> list[Any]:
        """
        Retrieve and clear pending event publishing intents.

        Returns:
            List of pending PublishEvent intents for orchestrator
        """
        intents = self._pending_event_intents.copy()
        self._pending_event_intents.clear()
        return intents

    async def execute_reduction(
        self,
        contract: ModelContractReducer,
    ) -> ModelReducerOutputState:
        """
        Execute pure metadata aggregation and state reduction with Kafka event publishing.

        Aggregation Strategy:
        1. Stream stamp metadata from input (async iterator)
        2. Group by namespace using windowing
        3. Compute aggregations (count, sum, avg, distinct)
        4. Update FSM state for each workflow (in-memory)
        5. Publish Kafka events at lifecycle points
        6. Return aggregation results with intents for side effects

        Args:
            contract: Reducer contract with aggregation configuration

        Returns:
            ModelReducerOutputState with aggregated metadata, state, and intents

        Raises:
            OnexError: If reduction fails or validation errors occur
            ValueError: If contract is missing required input_state or input_stream
        """
        start_time = time.perf_counter()

        # Generate aggregation ID for tracking (needed for error events)
        aggregation_id = str(uuid4())

        # Import ModelIntent and EnumIntentType here to avoid circular imports
        from .models.enum_intent_type import EnumIntentType
        from .models.model_intent import ModelIntent

        # Extract aggregation configuration from contract (needed for error events)
        try:
            aggregation_type = self._get_aggregation_type(contract)
        except Exception:
            # Use default if extraction fails
            from .models.enum_aggregation_type import EnumAggregationType

            aggregation_type = EnumAggregationType.NAMESPACE_GROUPING

        # Track aggregation metrics
        aggregation_type_str = getattr(aggregation_type, "value", str(aggregation_type))

        try:
            # Validate input: require either input_stream or non-None input_state
            if not hasattr(contract, "input_stream") and not hasattr(
                contract, "input_state"
            ):
                raise ValueError(
                    "Contract must have either 'input_stream' or 'input_state' attribute"
                )
            if hasattr(contract, "input_state") and contract.input_state is None:
                if (
                    not hasattr(contract, "input_stream")
                    or contract.input_stream is None
                ):
                    raise ValueError(
                        "Contract input_state is None and no input_stream provided. "
                        "At least one input source must be provided for aggregation."
                    )

            # Extract remaining aggregation configuration
            window_size_ms = self._get_window_size(contract)
            batch_size = self._get_batch_size(contract)

            # Publish AGGREGATION_STARTED event
            await self._publish_event(
                EnumReducerEvent.AGGREGATION_STARTED,
                {
                    "aggregation_id": aggregation_id,
                    "aggregation_type": aggregation_type.value,
                    "batch_size": batch_size,
                    "window_size_ms": window_size_ms,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            # Initialize intents list for side effects
            intents: list[ModelIntent] = []

            # Initialize aggregation state with bounded collections for true streaming (O(1) memory)
            # Use cardinality tracking with bounded samples instead of unbounded sets
            # Maximum samples configurable via environment variable (default: 100)
            MAX_SAMPLES = performance_config.MAX_CARDINALITY_SAMPLES

            aggregated_data: dict[str, dict[str, Any]] = defaultdict(
                lambda: {
                    "total_stamps": 0,
                    "total_size_bytes": 0,
                    "unique_file_types_count": 0,  # Total count of unique file types seen
                    "file_types": set(),  # Bounded sample (max MAX_SAMPLES for backward compat)
                    "unique_workflows_count": 0,  # Total count of unique workflows seen
                    "workflow_ids": set(),  # Bounded sample (max MAX_SAMPLES for backward compat)
                    "_file_types_seen": set(),  # Internal tracking (will be bounded to 2*MAX_SAMPLES)
                    "_workflow_ids_seen": set(),  # Internal tracking (will be bounded to 2*MAX_SAMPLES)
                }
            )
            fsm_states: dict[str, str] = {}
            total_items = 0
            batches_processed = 0

            # Stream and aggregate data (pure computation with O(1) memory)
            async for metadata_batch in self._stream_metadata(
                contract,
                batch_size=batch_size,
            ):
                for metadata in metadata_batch:
                    # Group by namespace (primary aggregation key)
                    namespace = metadata.namespace
                    total_items += 1
                    ns_data = aggregated_data[namespace]

                    # Update aggregations for this namespace
                    ns_data["total_stamps"] += 1
                    ns_data["total_size_bytes"] += metadata.file_size

                    # Track file types with bounded memory (O(1) with MAX_SAMPLES limit)
                    content_type = metadata.content_type
                    if content_type not in ns_data["_file_types_seen"]:
                        ns_data["unique_file_types_count"] += 1
                        ns_data["_file_types_seen"].add(content_type)

                        # Keep bounded samples for backward compatibility (max MAX_SAMPLES)
                        if len(ns_data["file_types"]) < MAX_SAMPLES:
                            ns_data["file_types"].add(content_type)

                        # Bound internal tracking set when it grows too large
                        # Keep most recent 2*MAX_SAMPLES for reasonable tracking accuracy
                        if len(ns_data["_file_types_seen"]) > MAX_SAMPLES * 2:
                            # Reset to current samples to bound memory
                            ns_data["_file_types_seen"] = ns_data["file_types"].copy()

                    # Track workflow IDs with bounded memory (O(1) with MAX_SAMPLES limit)
                    workflow_id_str = str(metadata.workflow_id)
                    if workflow_id_str not in ns_data["_workflow_ids_seen"]:
                        ns_data["unique_workflows_count"] += 1
                        ns_data["_workflow_ids_seen"].add(workflow_id_str)

                        # Keep bounded samples for backward compatibility (max MAX_SAMPLES)
                        if len(ns_data["workflow_ids"]) < MAX_SAMPLES:
                            ns_data["workflow_ids"].add(workflow_id_str)

                        # Bound internal tracking set when it grows too large
                        if len(ns_data["_workflow_ids_seen"]) > MAX_SAMPLES * 2:
                            # Reset to current samples to bound memory
                            ns_data["_workflow_ids_seen"] = ns_data[
                                "workflow_ids"
                            ].copy()

                    # Track FSM state using FSM manager (in-memory only)
                    workflow_id = metadata.workflow_id
                    workflow_state = metadata.workflow_state

                    # Initialize workflow in FSM if not already tracked
                    if self._fsm_manager.get_state(workflow_id) is None:
                        await self._fsm_manager.initialize_workflow(
                            workflow_id=workflow_id,
                            initial_state=workflow_state,
                            metadata={"namespace": namespace},
                        )
                    else:
                        # Workflow already exists, check if state changed
                        current_state = self._fsm_manager.get_state(workflow_id)
                        if (
                            current_state is not None
                            and current_state != workflow_state
                        ):
                            # Transition to new state
                            try:
                                await self._fsm_manager.transition_state(
                                    workflow_id=workflow_id,
                                    from_state=current_state,
                                    to_state=workflow_state,
                                    trigger="aggregation_update",
                                    metadata={"namespace": namespace},
                                )
                            except ValueError as e:
                                # Log invalid transition but continue processing
                                logger.warning(
                                    f"FSM transition failed for {workflow_id}: {e}"
                                )

                    # Store state in output dict
                    fsm_states[str(workflow_id)] = workflow_state

                # Publish BATCH_PROCESSED event
                batches_processed += 1
                await self._publish_event(
                    EnumReducerEvent.BATCH_PROCESSED,
                    {
                        "aggregation_id": aggregation_id,
                        "batch_number": batches_processed,
                        "batch_size": len(metadata_batch),
                        "total_items_processed": total_items,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

            # Convert bounded sets to lists for JSON serialization
            # Remove internal tracking sets (prefixed with _) before output
            for namespace_data in aggregated_data.values():
                # Convert bounded sample sets to lists (maintains backward compatibility)
                namespace_data["file_types"] = list(namespace_data["file_types"])
                namespace_data["workflow_ids"] = list(namespace_data["workflow_ids"])

                # Remove internal tracking sets to reduce output size and prevent leaking internal state
                namespace_data.pop("_file_types_seen", None)
                namespace_data.pop("_workflow_ids_seen", None)

            # Calculate total size across all namespaces
            total_size_bytes = sum(
                data["total_size_bytes"] for data in aggregated_data.values()
            )

            # Intent: Persist aggregated state (if StateManagement configured)
            if self._state_config or hasattr(contract, "state_management"):
                intents.append(
                    ModelIntent(
                        intent_type=EnumIntentType.PERSIST_STATE.value,
                        target="store_effect",
                        payload={
                            "aggregated_data": dict(aggregated_data),
                            "fsm_states": fsm_states,
                            "aggregation_id": aggregation_id,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                        priority=1,  # High priority for persistence
                    )
                )

                # Publish STATE_PERSISTED event
                await self._publish_event(
                    EnumReducerEvent.STATE_PERSISTED,
                    {
                        "aggregation_id": aggregation_id,
                        "namespaces_count": len(aggregated_data),
                        "workflows_count": len(fsm_states),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

            # Calculate performance metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            items_per_second = (
                total_items / (duration_ms / 1000) if duration_ms > 0 else 0.0
            )

            # Publish AGGREGATION_COMPLETED event
            await self._publish_event(
                EnumReducerEvent.AGGREGATION_COMPLETED,
                {
                    "aggregation_id": aggregation_id,
                    "total_items": total_items,
                    "batches_processed": batches_processed,
                    "total_size_bytes": total_size_bytes,
                    "duration_ms": duration_ms,
                    "items_per_second": items_per_second,
                    "namespaces_count": len(aggregated_data),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            # Collect intents from FSMStateManager (FSM transition persistence intents)
            fsm_intents = self._fsm_manager.get_pending_intents()
            intents.extend(fsm_intents)

            # Collect event publishing intents (ONEX v2.0 pure function pattern)
            event_intents = self.get_pending_event_intents()
            intents.extend(event_intents)

            # Record successful aggregation metrics
            self.metrics_collector.record_aggregation_items(
                aggregation_type_str, total_items
            )
            self.metrics_collector.record_operation_latency(
                "aggregation", duration_ms / 1000
            )

            # Return aggregation results with intents
            return ModelReducerOutputState(
                aggregation_type=aggregation_type,
                total_items=total_items,
                total_size_bytes=total_size_bytes,
                namespaces=list(aggregated_data.keys()),
                aggregations=dict(aggregated_data),
                fsm_states=fsm_states,
                intents=intents,
                aggregation_duration_ms=duration_ms,
                items_per_second=items_per_second,
            )

        except Exception as e:
            # Calculate failure metrics
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Record failed aggregation metrics
            self.metrics_collector.record_error(type(e).__name__, "execute_reduction")

            # Publish AGGREGATION_FAILED event
            await self._publish_event(
                EnumReducerEvent.AGGREGATION_FAILED,
                {
                    "aggregation_id": aggregation_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_ms": duration_ms,
                    "aggregation_type": aggregation_type.value,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            # Re-raise exception after logging
            raise

    async def _stream_metadata(
        self,
        contract: ModelContractReducer,
        batch_size: Optional[int] = None,
    ) -> AsyncIterator[list[ModelReducerInputState]]:
        """
        Stream metadata from input using streaming configuration.

        Implements windowed streaming with batching for efficient processing.
        Uses contract.input_stream if available, otherwise processes input_state.

        Args:
            contract: Reducer contract with streaming configuration
            batch_size: Number of items per batch

        Yields:
            Batches of stamp metadata items
        """
        # Use configured batch size if not provided
        if batch_size is None:
            batch_manager = get_batch_manager()
            batch_size = batch_manager.reducer_batch_size

        # Check if contract has streaming input
        if hasattr(contract, "input_stream") and contract.input_stream:
            # Stream from async iterator
            batch: list[ModelReducerInputState] = []

            async for item in contract.input_stream:
                # Convert dict to ModelReducerInputState if needed
                if isinstance(item, dict):
                    metadata_item = ModelReducerInputState(**item)
                else:
                    metadata_item = item

                batch.append(metadata_item)

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
                # Check if it's a container with items
                items = input_data.get("items", [input_data])
            elif isinstance(input_data, list):
                items = input_data
            else:
                items = [input_data]

            # Convert to ModelReducerInputState and yield in batches
            batch = []
            for item in items:
                if isinstance(item, dict):
                    metadata_item = ModelReducerInputState(**item)
                else:
                    metadata_item = item

                batch.append(metadata_item)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

    def _get_aggregation_type(
        self,
        contract: ModelContractReducer,
    ) -> EnumAggregationType:
        """
        Extract aggregation type from contract configuration.

        Args:
            contract: Reducer contract

        Returns:
            Aggregation type enum value
        """
        # Check for aggregation subcontract
        if hasattr(contract, "aggregation") and contract.aggregation:
            # Get strategy from aggregation config
            if hasattr(contract.aggregation, "aggregation_type"):
                return EnumAggregationType(contract.aggregation.aggregation_type)

        # Default to namespace grouping
        return EnumAggregationType.NAMESPACE_GROUPING

    def _get_window_size(self, contract: ModelContractReducer) -> int:
        """
        Extract window size from streaming configuration.

        Args:
            contract: Reducer contract

        Returns:
            Window size in milliseconds (default: 5000)
        """
        if hasattr(contract, "streaming") and contract.streaming:
            if hasattr(contract.streaming, "window_size"):
                return int(contract.streaming.window_size)

        # Default window size
        return 5000

    def _get_batch_size(self, contract: ModelContractReducer) -> int:
        """
        Extract batch size from streaming configuration.

        Args:
            contract: Reducer contract

        Returns:
            Batch size (default: 100)
        """
        if hasattr(contract, "streaming") and contract.streaming:
            if hasattr(contract.streaming, "batch_size"):
                return int(contract.streaming.batch_size)

        # Default batch size
        return 100

    async def _publish_event(
        self, event_type: EnumReducerEvent, data: dict[str, Any]
    ) -> None:
        """
        Generate PublishEvent intent and optionally publish to Kafka.

        ONEX v2.0 Pure Function Pattern:
        - Generates PublishEvent intent for orchestrator
        - Optionally publishes directly to Kafka if client is available
        - Intents allow replay and orchestrator tracking

        Args:
            event_type: Reducer event type identifier
            data: Event payload data
        """
        try:
            # Import ModelIntent and EnumIntentType here to avoid circular imports
            from .models.enum_intent_type import EnumIntentType
            from .models.model_intent import ModelIntent

            # Get Kafka topic name
            topic_name = event_type.get_topic_name(namespace=self.default_namespace)

            # Extract correlation ID from data
            correlation_id = data.get("correlation_id") or data.get("aggregation_id")

            # Generate PublishEvent intent (ONEX v2.0 pure function pattern)
            intent = ModelIntent(
                intent_type=EnumIntentType.PUBLISH_EVENT.value,
                target="event_bus",
                payload={
                    "event_type": event_type.value,
                    "topic": topic_name,
                    "data": data,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                priority=0,  # Normal priority for event publishing
            )
            self._pending_event_intents.append(intent)

            # Publish to Kafka if client is available (immediate feedback)
            if self.kafka_client and self.kafka_client.is_connected:
                # Extract correlation ID from data (aggregation_id)
                correlation_id = data.get("correlation_id") or data.get(
                    "aggregation_id"
                )

                # Add node metadata to payload
                payload = {
                    **data,
                    "node_id": getattr(self, "node_id", "reducer"),
                    "published_at": datetime.now(UTC).isoformat(),
                }

                # Publish with OnexEnvelopeV1 wrapping for standardized event format
                # Include Consul service_id for cross-service event correlation
                event_metadata = {
                    "event_category": "metadata_aggregation",
                    "node_type": "reducer",
                    "namespace": self.default_namespace,
                }

                # Add consul_service_id if available (enables cross-service correlation)
                if hasattr(self, "_consul_service_id"):
                    event_metadata["consul_service_id"] = self._consul_service_id

                success = await self.kafka_client.publish_with_envelope(
                    event_type=event_type.value,
                    source_node_id=str(getattr(self, "node_id", "reducer")),
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
                            "node_id": getattr(self, "node_id", "reducer"),
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
                            "node_id": getattr(self, "node_id", "reducer"),
                            "event_type": event_type.value,
                            "topic_name": topic_name,
                        },
                    )
            else:
                # Kafka client not available - log event only
                emit_log_event(
                    LogLevel.INFO,
                    f"Kafka event (no client): {event_type.value}",
                    {
                        "node_id": getattr(self, "node_id", "reducer"),
                        "event_type": event_type.value,
                        "data": data,
                    },
                )

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Error publishing Kafka event: {event_type.value}",
                {
                    "node_id": getattr(self, "node_id", "reducer"),
                    "event_type": event_type.value,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    # Lifecycle Methods

    async def startup(self) -> None:
        """
        Node startup lifecycle hook.

        Performs:
        1. Kafka client connection
        2. FSM state recovery intent generation (pure, no I/O)
        3. Introspection data publishing
        4. Background task startup

        Should be called when node is ready to serve requests.

        Note:
            FSM state recovery now uses Intent Publisher pattern.
            The startup method generates recovery intents but does not
            execute them directly. An orchestrator should:
            1. Call startup() to generate recovery intents
            2. Collect intents via _fsm_manager.get_pending_intents()
            3. Route intents to NodeBridgeStoreEffect for execution
            4. Process recovered state data and restore _state_cache
        """
        logger.info(
            f"NodeBridgeReducer starting up: {getattr(self, 'node_id', 'unknown')}"
        )

        # Connect to Kafka if client is available
        if self.kafka_client and not self.kafka_client.is_connected:
            try:
                await self.kafka_client.connect()
                emit_log_event(
                    LogLevel.INFO,
                    "Kafka client connected successfully in reducer",
                    {"node_id": getattr(self, "node_id", "reducer")},
                )
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Failed to connect Kafka client: {e}",
                    {
                        "node_id": getattr(self, "node_id", "reducer"),
                        "error": str(e),
                    },
                )

        # Generate FSM state recovery intent (pure, no I/O)
        recovery_stats = await self._fsm_manager.recover_states()

        # Note: Orchestrator collects and executes recovery intents from
        # _fsm_manager._pending_intents via NodeBridgeStoreEffect (EFFECT node)
        if recovery_stats.get("deferred", False):
            logger.info(
                "FSM state recovery intent generated - "
                "orchestrator should execute via NodeBridgeStoreEffect"
            )
        else:
            logger.info(
                f"FSM state recovery: {recovery_stats['recovered']} workflows recovered, "
                f"{recovery_stats['failed']} failed"
            )

        # Publish initial introspection broadcast
        await self.publish_introspection(reason="startup")

        # Start introspection background tasks (heartbeat, registry listener)
        await self.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=30,
            enable_registry_listener=True,
        )

        logger.info(
            f"NodeBridgeReducer startup complete: {getattr(self, 'node_id', 'unknown')}"
        )

    async def shutdown(self) -> None:
        """
        Node shutdown lifecycle hook.

        Stops background tasks and cleans up resources.
        Should be called when node is preparing to exit.
        """
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"NodeBridgeReducer shutting down: {getattr(self, 'node_id', 'unknown')}"
        )

        # Stop introspection background tasks
        await self.stop_introspection_tasks()

        # Flush any pending aggregations
        if self._aggregation_buffer:
            # In production, would persist final aggregations here
            pass

        # Disconnect Kafka client if connected
        if self.kafka_client and self.kafka_client.is_connected:
            try:
                await self.kafka_client.disconnect()
                emit_log_event(
                    LogLevel.INFO,
                    "Kafka client disconnected successfully",
                    {"node_id": getattr(self, "node_id", "reducer")},
                )
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Error disconnecting Kafka client: {e}",
                    {
                        "node_id": getattr(self, "node_id", "reducer"),
                        "error": str(e),
                    },
                )

        # Deregister from Consul for clean service discovery
        self._deregister_from_consul()

        logger.info(
            f"NodeBridgeReducer shutdown complete: {getattr(self, 'node_id', 'unknown')}"
        )

    # Health Check Methods

    def _register_component_checks(self) -> None:
        """
        Register component health checks for pure reducer.

        Checks:
        - node_runtime: Basic node operational status (critical)
        - aggregation_buffer: Internal aggregation buffer health (non-critical)

        Note: Pure reducer has no I/O dependencies (no DB, no Kafka).
        """
        # Parent class registers node_runtime check
        super()._register_component_checks()

        # Register aggregation buffer health check (non-critical)
        self.register_component_check(
            "aggregation_buffer",
            self._check_aggregation_buffer_health,
            critical=False,
            timeout_seconds=performance_config.AGGREGATION_BUFFER_HEALTH_CHECK_TIMEOUT_SECONDS,
        )

    async def _check_aggregation_buffer_health(
        self,
    ) -> tuple[HealthStatus, str, dict[str, Any]]:
        """
        Check aggregation buffer and FSM manager health.

        Returns:
            Tuple of (status, message, details)
        """
        try:
            # Check buffer size
            buffer_size = len(self._aggregation_buffer)

            # Check FSM manager state cache size
            fsm_cache_size = len(self._fsm_manager._state_cache)
            fsm_history_size = sum(
                len(history)
                for history in self._fsm_manager._transition_history.values()
            )

            # Warn if buffer is growing too large (potential memory issue)
            # Threshold configurable via environment variable
            if buffer_size > performance_config.AGGREGATION_BUFFER_WARNING_THRESHOLD:
                return (
                    HealthStatus.DEGRADED,
                    "Aggregation buffer large, may need flushing",
                    {
                        "buffer_size": buffer_size,
                        "fsm_cache_size": fsm_cache_size,
                        "fsm_history_size": fsm_history_size,
                        "threshold": performance_config.AGGREGATION_BUFFER_WARNING_THRESHOLD,
                    },
                )

            # Warn if FSM cache is growing too large
            # Threshold configurable via environment variable
            if fsm_cache_size > performance_config.FSM_CACHE_WARNING_THRESHOLD:
                return (
                    HealthStatus.DEGRADED,
                    "FSM state cache large, may need cleanup",
                    {
                        "buffer_size": buffer_size,
                        "fsm_cache_size": fsm_cache_size,
                        "fsm_history_size": fsm_history_size,
                        "threshold": performance_config.FSM_CACHE_WARNING_THRESHOLD,
                    },
                )

            return (
                HealthStatus.HEALTHY,
                "Aggregation buffer and FSM manager healthy",
                {
                    "buffer_size": buffer_size,
                    "fsm_cache_size": fsm_cache_size,
                    "fsm_history_size": fsm_history_size,
                },
            )

        except Exception as e:
            return (
                HealthStatus.DEGRADED,
                f"Aggregation buffer check failed: {e!s}",
                {"error": str(e)},
            )

    def _register_with_consul_sync(self) -> None:
        """
        Register reducer node with Consul for service discovery (synchronous).

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
            service_id = (
                f"omninode-bridge-reducer-{getattr(self, 'node_id', 'reducer')}"
            )

            # Get service port from config (default to 8061 for reducer)
            # ModelONEXContainer stores config in 'value' field
            config_dict = (
                self.container.value
                if hasattr(self.container, "value")
                else getattr(self.container, "config", {})
            )
            service_port = int(
                config_dict.get("service_port", 8061)
                if hasattr(config_dict, "get")
                else 8061
            )

            # Get service host from config (default to localhost)
            service_host = (
                config_dict.get("service_host", "localhost")
                if hasattr(config_dict, "get")
                else "localhost"
            )

            # Prepare service tags
            service_tags = [
                "onex",
                "bridge",
                "reducer",
                f"version:{getattr(self, 'version', '0.1.0')}",
                "omninode_bridge",
            ]

            # Prepare service metadata (note: python-consul v1.1.0 doesn't support meta parameter)
            # Metadata is encoded in tags for MVP compatibility
            service_tags.extend(
                [
                    "node_type:reducer",
                    f"namespace:{self.default_namespace}",
                    f"batch_size:{self._aggregation_batch_size}",
                    f"window_seconds:{self._aggregation_window_seconds}",
                    f"kafka_enabled:{self.kafka_client is not None}",
                ]
            )

            # Health check URL (assumes health endpoint is available)
            health_check_url = f"http://{service_host}:{service_port}/health"

            # Register service with Consul
            consul_client.agent.service.register(
                name="omninode-bridge-reducer",
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
                    "node_id": getattr(self, "node_id", "reducer"),
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
                {"node_id": getattr(self, "node_id", "reducer")},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                "Failed to register with Consul",
                {
                    "node_id": getattr(self, "node_id", "reducer"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def _deregister_from_consul(self) -> None:
        """
        Deregister reducer from Consul on shutdown (synchronous).

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
                    "node_id": getattr(self, "node_id", "reducer"),
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
                    "node_id": getattr(self, "node_id", "reducer"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )


def main() -> Any:
    """
    Entry point for node execution.

    Returns:
        NodeBase instance for ONEX runtime
    """
    try:
        from omnibase_core.infrastructure.node_base import NodeBase

        # Contract filename - standard ONEX pattern
        CONTRACT_FILENAME = "contract.yaml"

        return NodeBase(Path(__file__).parent / CONTRACT_FILENAME)
    except ImportError:
        # Fallback if omnibase runtime is not available
        print("Warning: omnibase runtime not available, running in standalone mode")
        return None


if __name__ == "__main__":
    main()
