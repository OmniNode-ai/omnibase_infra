# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dual Registration Reducer Node.

This module provides NodeDualRegistrationReducer for coordinating parallel
registration to Consul and PostgreSQL backends in the ONEX 2-way registration
pattern.

Architecture:
    NodeDualRegistrationReducer belongs to the REDUCER layer of ONEX 4-node
    architecture, aggregating registration state from multiple backends and
    implementing graceful degradation when one backend fails.

FSM Integration:
    The reducer loads its state machine from:
    contracts/fsm/dual_registration_reducer_fsm.yaml

    States:
    - idle: Waiting for introspection events
    - receiving_introspection: Parsing NODE_INTROSPECTION event
    - validating_payload: Validating event structure
    - registering_parallel: Parallel registration to both backends
    - aggregating_results: Combining registration outcomes
    - registration_complete: Both backends succeeded
    - partial_failure: One backend failed (graceful degradation)
    - registration_failed: Both backends failed

Performance Targets:
    - Dual registration time: <300ms
    - Aggregation overhead: <10ms

Related:
    - OMN-889: Infrastructure MVP - ModelNodeIntrospectionEvent
    - contracts/fsm/dual_registration_reducer_fsm.yaml: FSM contract
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import yaml

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers import ConsulHandler, DbAdapter
from omnibase_infra.models.registration import (
    ModelDualRegistrationResult,
    ModelNodeIntrospectionEvent,
    ModelNodeRegistration,
)
from omnibase_infra.nodes.reducers.enums import EnumFSMState, EnumFSMTrigger
from omnibase_infra.nodes.reducers.models import (
    ModelAggregationParams,
    ModelFSMContext,
    ModelFSMContract,
    ModelReducerMetrics,
)

# Valid ONEX node types for introspection event validation.
# This constant serves as defense-in-depth validation for mock objects in tests,
# since ModelNodeIntrospectionEvent.node_type already enforces this via Literal type.
_VALID_NODE_TYPES: frozenset[str] = frozenset(
    {"effect", "compute", "reducer", "orchestrator"}
)

logger = logging.getLogger(__name__)


def _find_contracts_dir() -> Path:
    """Find contracts directory by traversing up from current file.

    This is more robust than multiple .parent calls and handles
    different installation/development environments.

    Returns:
        Path to the contracts directory.

    Raises:
        RuntimeError: If contracts directory cannot be found.
    """
    # Start from current file's directory and walk up the tree until we find
    # a directory containing 'contracts/'. This handles both development
    # (source checkout) and installed package scenarios where the relative
    # path depth may vary.
    current = Path(__file__).resolve().parent
    while current != current.parent:
        contracts_dir = current / "contracts"
        if contracts_dir.is_dir():
            return contracts_dir
        current = current.parent
    raise RuntimeError(
        "Could not find contracts directory. "
        "Ensure the FSM contract exists at "
        "contracts/fsm/dual_registration_reducer_fsm.yaml"
    )


class NodeDualRegistrationReducer:
    """Dual registration reducer that aggregates registration state.

    Listens for NODE_INTROSPECTION events and coordinates dual registration
    with graceful degradation when one backend fails.

    This reducer implements the REDUCER pattern in ONEX 4-node architecture:
    - Aggregates state from multiple sources (Consul, PostgreSQL)
    - Implements graceful degradation for partial failures
    - Tracks registration metrics across all processed events
    - Uses FSM-driven workflow from YAML contract

    Status Semantics:
        - "success": Both Consul and PostgreSQL registrations succeeded
        - "partial": One registration succeeded, the other failed
        - "failed": Both registrations failed

    Attributes:
        consul_handler: Handler for Consul service discovery operations.
        db_adapter: Adapter for PostgreSQL database operations.
        metrics: Aggregation metrics tracking registration outcomes.
        fsm_contract: Loaded FSM contract from YAML.
        current_state: Current FSM state.
        fsm_context: Context variables maintained during FSM execution.

    Example:
        >>> from omnibase_infra.handlers import ConsulHandler, DbAdapter
        >>> from omnibase_infra.nodes.reducers import NodeDualRegistrationReducer
        >>>
        >>> consul = ConsulHandler()
        >>> db = DbAdapter()
        >>> reducer = NodeDualRegistrationReducer(consul, db)
        >>> await reducer.initialize()
        >>>
        >>> # Process introspection event
        >>> result = await reducer.execute(introspection_event, correlation_id)
        >>> print(result.status)  # "success", "partial", or "failed"
    """

    # Performance target constants
    _TARGET_DUAL_REGISTRATION_MS: float = 300.0
    _TARGET_AGGREGATION_OVERHEAD_MS: float = 10.0

    def __init__(
        self,
        consul_handler: ConsulHandler,
        db_adapter: DbAdapter,
        fsm_contract_path: Path | None = None,
    ) -> None:
        """Initialize dual registration reducer.

        Dependency Injection Pattern:
            This reducer uses constructor-based dependency injection for handlers.
            Handlers are injected rather than instantiated internally, enabling:

            1. **Testability**: Handlers can be replaced with mocks or stubs in tests
               without modifying the reducer code. Simply pass mock objects that
               implement the same interface (execute method returning status).

            2. **Loose Coupling**: The reducer depends on handler behavior (execute
               method signature), not concrete implementations. This allows handler
               implementations to evolve independently.

            3. **Lifecycle Management**: Handlers are managed externally, allowing
               shared connection pools, centralized initialization, and coordinated
               shutdown across multiple consumers.

            Note: While ONEX architecture prefers protocol-based resolution through
            ModelONEXContainer, the current DI pattern is acceptable for infrastructure
            nodes where handlers require external initialization (connections, auth).

        Args:
            consul_handler: Initialized ConsulHandler for service discovery.
                Must have an async execute() method that accepts an envelope dict
                and returns a response with a status attribute.
            db_adapter: Initialized DbAdapter for PostgreSQL operations.
                Must have an async execute() method that accepts an envelope dict
                and returns a response with a status attribute.
            fsm_contract_path: Optional path to FSM contract YAML. If not provided,
                defaults to contracts/fsm/dual_registration_reducer_fsm.yaml.
        """
        self._consul_handler = consul_handler
        self._db_adapter = db_adapter
        self._metrics = ModelReducerMetrics()
        self._initialized = False

        # FSM state management
        self._current_state = EnumFSMState.IDLE
        self._fsm_context = ModelFSMContext()
        self._fsm_contract: ModelFSMContract | None = None
        self._transition_map: dict[
            tuple[EnumFSMState, EnumFSMTrigger], EnumFSMState
        ] = {}

        # FSM contract path - use robust path resolution
        # _find_contracts_dir() traverses up from current file to find contracts/
        # directory, which is more reliable than counting .parent calls.
        # If contracts directory is not found, RuntimeError propagates to caller
        # since this is a genuine configuration error that should be surfaced.
        if fsm_contract_path is None:
            contracts_dir = _find_contracts_dir()
            self._fsm_contract_path = (
                contracts_dir / "fsm" / "dual_registration_reducer_fsm.yaml"
            )
        else:
            self._fsm_contract_path = fsm_contract_path

    @property
    def metrics(self) -> ModelReducerMetrics:
        """Return current aggregation metrics."""
        return self._metrics

    @property
    def current_state(self) -> EnumFSMState:
        """Return current FSM state."""
        return self._current_state

    @property
    def fsm_contract(self) -> ModelFSMContract | None:
        """Return loaded FSM contract."""
        return self._fsm_contract

    async def initialize(self) -> None:
        """Initialize reducer and load FSM contract.

        Loads the FSM contract from YAML and validates its structure.
        The consul_handler and db_adapter must be initialized separately
        before calling this method.

        Raises:
            RuntimeHostError: If FSM contract loading fails.
        """
        init_correlation_id = uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="initialize",
            target_name="dual_registration_reducer",
            correlation_id=init_correlation_id,
        )

        try:
            # Load FSM contract from YAML
            if not self._fsm_contract_path.exists():
                raise RuntimeHostError(
                    f"FSM contract not found: {self._fsm_contract_path}",
                    context=ctx,
                )

            with open(self._fsm_contract_path) as f:
                contract_data = yaml.safe_load(f)

            # Parse contract into structured model
            state_transitions = contract_data.get("state_transitions", {})
            self._fsm_contract = ModelFSMContract(
                contract_version=contract_data.get("contract_version", "1.0.0"),
                name=contract_data.get("name", "unknown"),
                description=contract_data.get("description", ""),
                initial_state=state_transitions.get("initial_state", "idle"),
                states=state_transitions.get("states", []),
                transitions=state_transitions.get("transitions", []),
                error_handling=contract_data.get("error_handling", {}),
            )

            # Validate initial state
            initial_state_str = self._fsm_contract.initial_state
            try:
                self._current_state = EnumFSMState(initial_state_str)
            except ValueError:
                raise RuntimeHostError(
                    f"Invalid initial state in FSM contract: {initial_state_str}",
                    context=ctx,
                )

            # Build transition map dynamically from contract
            self._transition_map = {}
            for transition in self._fsm_contract.transitions:
                try:
                    from_state = EnumFSMState(transition.from_state)
                    to_state = EnumFSMState(transition.to_state)
                    trigger = EnumFSMTrigger(transition.trigger)
                    self._transition_map[(from_state, trigger)] = to_state
                except ValueError as e:
                    raise RuntimeHostError(
                        f"Invalid transition in FSM contract: {transition.trigger}",
                        context=ctx,
                    ) from e

            self._initialized = True
            logger.info(
                "NodeDualRegistrationReducer initialized",
                extra={
                    "fsm_contract": self._fsm_contract.name,
                    "initial_state": self._current_state.value,
                    "correlation_id": str(init_correlation_id),
                },
            )

        except RuntimeHostError:
            raise
        except Exception as e:
            raise RuntimeHostError(
                f"Failed to initialize reducer: {type(e).__name__}",
                context=ctx,
            ) from e

    async def shutdown(self) -> None:
        """Shutdown reducer and reset state.

        Resets FSM to idle state and clears context.
        Does NOT shutdown consul_handler or db_adapter - those are managed
        externally.
        """
        self._current_state = EnumFSMState.IDLE
        self._fsm_context = ModelFSMContext()
        self._initialized = False
        logger.info("NodeDualRegistrationReducer shutdown complete")

    async def execute(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID | None = None,
    ) -> ModelDualRegistrationResult:
        """Execute dual registration workflow for introspection event.

        Processes a NODE_INTROSPECTION event through the FSM workflow,
        coordinating parallel registration to Consul and PostgreSQL
        with graceful degradation.

        Args:
            event: Node introspection event to process.
            correlation_id: Optional correlation ID for tracing. If not provided,
                uses event.correlation_id or generates a new one.

        Returns:
            ModelDualRegistrationResult with registration outcomes.

        Raises:
            RuntimeHostError: If reducer not initialized or workflow fails.
        """
        # Resolve correlation ID
        cid = correlation_id or event.correlation_id or uuid4()

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="execute",
            target_name="dual_registration_reducer",
            correlation_id=cid,
        )

        if not self._initialized:
            raise RuntimeHostError(
                "Reducer not initialized. Call initialize() first.",
                context=ctx,
            )

        # Initialize FSM context
        self._fsm_context = ModelFSMContext(
            correlation_id=cid,
            node_id=event.node_id,
            node_type=event.node_type,
            introspection_payload=event,
            registration_start_time=time.perf_counter(),
        )

        try:
            # FSM: idle -> receiving_introspection
            await self._transition(EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED)

            # FSM: receiving_introspection -> validating_payload
            await self._transition(EnumFSMTrigger.EVENT_PARSED)

            # Validate payload - returns (is_valid, error_message) tuple
            validation_passed, validation_error = self._validate_payload(event, cid)

            if not validation_passed:
                # FSM: validating_payload -> registration_failed
                await self._transition(EnumFSMTrigger.VALIDATION_FAILED)
                # FSM: registration_failed -> idle
                await self._transition(EnumFSMTrigger.FAILURE_RESULT_EMITTED)
                return self._build_failed_result(cid, validation_error)

            # FSM: validating_payload -> registering_parallel
            await self._transition(EnumFSMTrigger.VALIDATION_PASSED)

            # Execute parallel registration
            result = await self._register_parallel(event, cid)

            # FSM: registering_parallel -> aggregating_results
            await self._transition(EnumFSMTrigger.REGISTRATION_ATTEMPTS_COMPLETE)

            # Determine outcome and transition to terminal state
            if result.status == "success":
                # FSM: aggregating_results -> registration_complete
                await self._transition(EnumFSMTrigger.ALL_BACKENDS_SUCCEEDED)
                # FSM: registration_complete -> idle
                await self._transition(EnumFSMTrigger.RESULT_EMITTED)
            elif result.status == "partial":
                # FSM: aggregating_results -> partial_failure
                await self._transition(EnumFSMTrigger.PARTIAL_SUCCESS)
                # FSM: partial_failure -> idle
                await self._transition(EnumFSMTrigger.PARTIAL_RESULT_EMITTED)
            else:
                # FSM: aggregating_results -> registration_failed
                await self._transition(EnumFSMTrigger.ALL_BACKENDS_FAILED)
                # FSM: registration_failed -> idle
                await self._transition(EnumFSMTrigger.FAILURE_RESULT_EMITTED)

            return result

        except RuntimeHostError:
            raise
        except Exception as e:
            # =================================================================
            # ERROR RECOVERY PATTERN: FSM State Reset
            # =================================================================
            # When an unexpected exception occurs during the registration workflow,
            # we must ensure the FSM returns to the IDLE state to allow subsequent
            # events to be processed. Without this recovery, the reducer would be
            # stuck in an intermediate state and unable to handle new events.
            #
            # Recovery steps:
            # 1. Log the exception with full context for debugging
            # 2. Force FSM to REGISTRATION_FAILED state (error terminal)
            # 3. Transition from REGISTRATION_FAILED -> IDLE via FAILURE_RESULT_EMITTED
            #    (defined in FSM contract at lines 286-292)
            # 4. Re-raise as RuntimeHostError to signal failure to caller
            #
            # The try/except around the transition prevents transition errors from
            # masking the original exception that caused the failure.
            # =================================================================
            logger.exception(
                "Dual registration workflow failed",
                extra={
                    "node_id": self._fsm_context.node_id,
                    "current_state": self._current_state.value,
                    "correlation_id": str(cid),
                },
            )

            # Force transition to failed state (handles any intermediate state)
            self._current_state = EnumFSMState.REGISTRATION_FAILED

            # FSM: registration_failed -> idle (complete the FSM cycle)
            # This ensures the reducer can process subsequent events after exception.
            # The transition is defined in the FSM contract at lines 286-292.
            try:
                await self._transition(EnumFSMTrigger.FAILURE_RESULT_EMITTED)
            except Exception as transition_error:
                # Log but don't mask the original error - FSM will be in
                # REGISTRATION_FAILED state, which is recoverable on next init
                logger.exception(
                    "Failed to transition to idle after exception - "
                    "FSM may need reinitialization",
                    extra={
                        "original_error": type(e).__name__,
                        "transition_error": type(transition_error).__name__,
                        "current_state": self._current_state.value,
                        "correlation_id": str(cid),
                    },
                )

            raise RuntimeHostError(
                f"Dual registration failed: {type(e).__name__}",
                context=ctx,
            ) from e

    async def _transition(self, trigger: EnumFSMTrigger) -> None:
        """Execute FSM state transition.

        Validates the transition against the FSM contract and updates
        the current state. The transition map is built dynamically from
        the YAML contract during initialize().

        Args:
            trigger: Trigger to fire for state transition.

        Raises:
            RuntimeHostError: If transition is invalid or contract not loaded.
        """
        # Ensure transition map is populated (contract must be loaded)
        if not self._transition_map:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="fsm_transition",
                target_name="dual_registration_reducer",
                correlation_id=self._fsm_context.correlation_id,
            )
            raise RuntimeHostError(
                "FSM transition map not initialized. Call initialize() first.",
                context=ctx,
            )

        key = (self._current_state, trigger)
        if key not in self._transition_map:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="fsm_transition",
                target_name="dual_registration_reducer",
                correlation_id=self._fsm_context.correlation_id,
            )
            raise RuntimeHostError(
                f"Invalid FSM transition: "
                f"{self._current_state.value} + {trigger.value}",
                context=ctx,
                current_state=self._current_state.value,
                trigger=trigger.value,
            )

        old_state = self._current_state
        self._current_state = self._transition_map[key]

        logger.debug(
            "FSM transition",
            extra={
                "from_state": old_state.value,
                "to_state": self._current_state.value,
                "trigger": trigger.value,
                "node_id": self._fsm_context.node_id,
                "correlation_id": str(self._fsm_context.correlation_id),
            },
        )

    def _validate_payload(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> tuple[bool, str]:
        """Validate introspection event payload.

        Validates required fields (node_id, node_type) and logs validation
        errors with distinct error messages for each failure type.

        This method provides defense-in-depth validation. While Pydantic already
        validates ModelNodeIntrospectionEvent at construction time, this explicit
        validation serves two purposes:
        1. Catches issues with mock objects in tests that bypass Pydantic validation
        2. Documents business rules explicitly in code for maintainability

        Error Context:
            Creates a ModelInfraErrorContext for consistent distributed tracing.
            The context is used in logging to ensure correlation_id propagates
            through the validation operation, enabling end-to-end request tracing.

        Args:
            event: Introspection event to validate.
            correlation_id: Correlation ID for logging and error context.

        Returns:
            Tuple of (validation_passed, error_message).
            If validation passes, error_message is empty string.
        """
        # Create error context for consistent distributed tracing
        # This ensures all validation errors can be correlated with the
        # original request through the correlation_id
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="validate_payload",
            target_name="dual_registration_reducer",
            correlation_id=correlation_id,
        )

        # Defense-in-depth: Pydantic validates at model construction, but mocks
        # and edge cases may bypass this. Explicit checks provide safety net.
        if event.node_id is None:
            error_msg = "node_id is required but was None"
            logger.warning(
                "Payload validation failed: %s",
                error_msg,
                extra={
                    "correlation_id": str(correlation_id),
                    "operation": ctx.operation,
                    "target_name": ctx.target_name,
                },
            )
            return False, error_msg

        if event.node_type not in _VALID_NODE_TYPES:
            valid_types = ", ".join(sorted(_VALID_NODE_TYPES))
            error_msg = (
                f"node_type '{event.node_type}' is not valid. "
                f"Expected one of: {valid_types}"
            )
            logger.warning(
                "Payload validation failed: %s",
                error_msg,
                extra={
                    "node_type": event.node_type,
                    "valid_types": list(_VALID_NODE_TYPES),
                    "correlation_id": str(correlation_id),
                    "operation": ctx.operation,
                    "target_name": ctx.target_name,
                },
            )
            return False, error_msg

        return True, ""

    async def _register_parallel(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> ModelDualRegistrationResult:
        """Execute parallel registration to both backends.

        Uses asyncio.gather to execute Consul and PostgreSQL registrations
        concurrently, with return_exceptions=True for graceful error handling.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelDualRegistrationResult with aggregated outcomes.
        """
        start_time = time.perf_counter()

        # Run both registrations in parallel
        consul_task = self._register_consul(event, correlation_id)
        postgres_task = self._register_postgres(event, correlation_id)

        results = await asyncio.gather(
            consul_task,
            postgres_task,
            return_exceptions=True,
        )

        consul_result = results[0]
        postgres_result = results[1]

        # Calculate elapsed time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Aggregate results using parameter model
        params = ModelAggregationParams(
            consul_result=consul_result,
            postgres_result=postgres_result,
            node_id=event.node_id,
            correlation_id=correlation_id,
            registration_time_ms=elapsed_ms,
        )
        return self._aggregate_results(params)

    async def _register_consul(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> bool:
        """Register node with Consul service discovery.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            True if registration succeeded, False otherwise.

        Raises:
            Exception: If registration fails (caught by asyncio.gather).
        """
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="register",
            target_name="consul_handler",
            correlation_id=correlation_id,
        )

        try:
            # Build service registration payload
            service_id = f"node-{event.node_type}-{event.node_id}"
            health_endpoint = event.endpoints.get("health")

            # Build health check configuration
            check_config: dict[str, object] | None = None
            if health_endpoint:
                check_config = {
                    "http": health_endpoint,
                    "interval": "10s",
                    "timeout": "5s",
                }

            payload: dict[str, object] = {
                "name": f"onex-{event.node_type}",
                "service_id": service_id,
                "tags": [
                    f"node_type:{event.node_type}",
                    f"node_version:{event.node_version}",
                ],
            }

            if check_config is not None:
                payload["check"] = check_config

            # Execute registration
            envelope: dict[str, object] = {
                "operation": "consul.register",
                "payload": payload,
                "correlation_id": correlation_id,
            }

            response = await self._consul_handler.execute(envelope)

            # NOTE: Both ConsulHandler and DbAdapter now return typed Pydantic models
            # (ModelConsulHandlerResponse and ModelDbQueryResponse respectively).
            # This provides consistent attribute access patterns across both handlers.
            if response.status == "success":
                logger.info(
                    "Consul registration succeeded",
                    extra={
                        "service_id": service_id,
                        "node_id": str(event.node_id),
                        "correlation_id": str(correlation_id),
                    },
                )
                return True

            logger.warning(
                "Consul registration returned non-success status",
                extra={
                    "status": response.status,
                    "correlation_id": str(correlation_id),
                },
            )
            return False

        except Exception as e:
            logger.exception(
                "Consul registration failed",
                extra={
                    "node_id": str(event.node_id),
                    "correlation_id": str(correlation_id),
                },
            )
            raise InfraConnectionError(
                "Consul registration failed",
                context=ctx,
            ) from e

    async def _register_postgres(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> bool:
        """Register node in PostgreSQL node registry.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            True if registration succeeded, False otherwise.

        Raises:
            Exception: If registration fails (caught by asyncio.gather).
        """
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="register",
            target_name="db_adapter",
            correlation_id=correlation_id,
        )

        try:
            # Build node registration model
            now = datetime.now(UTC)
            registration = ModelNodeRegistration(
                node_id=event.node_id,
                node_type=event.node_type,
                node_version=event.node_version,
                capabilities=event.capabilities,
                endpoints=event.endpoints,
                metadata=event.metadata,
                health_endpoint=event.endpoints.get("health"),
                registered_at=now,
                updated_at=now,
            )

            # Convert to JSON for insertion
            registration_data = registration.model_dump(mode="json")

            # Upsert into node_registrations table
            sql = """
                INSERT INTO node_registrations (
                    node_id, node_type, node_version, capabilities,
                    endpoints, metadata, health_endpoint,
                    registered_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9
                )
                ON CONFLICT (node_id) DO UPDATE SET
                    node_type = EXCLUDED.node_type,
                    node_version = EXCLUDED.node_version,
                    capabilities = EXCLUDED.capabilities,
                    endpoints = EXCLUDED.endpoints,
                    metadata = EXCLUDED.metadata,
                    health_endpoint = EXCLUDED.health_endpoint,
                    updated_at = EXCLUDED.updated_at
            """

            envelope: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": sql,
                    "parameters": [
                        str(registration_data["node_id"]),
                        registration_data["node_type"],
                        registration_data["node_version"],
                        json.dumps(registration_data["capabilities"]),
                        json.dumps(registration_data["endpoints"]),
                        json.dumps(registration_data["metadata"]),
                        registration_data.get("health_endpoint"),
                        registration_data["registered_at"],
                        registration_data["updated_at"],
                    ],
                },
                "correlation_id": correlation_id,
            }

            response = await self._db_adapter.execute(envelope)

            # NOTE: Both DbAdapter and ConsulHandler now return typed Pydantic models
            # (ModelDbQueryResponse and ModelConsulHandlerResponse respectively).
            # This provides consistent attribute access patterns across both handlers.
            if response.status == "success":
                logger.info(
                    "PostgreSQL registration succeeded",
                    extra={
                        "node_id": str(event.node_id),
                        "correlation_id": str(correlation_id),
                    },
                )
                return True

            logger.warning(
                "PostgreSQL registration returned non-success status",
                extra={
                    "status": response.status,
                    "correlation_id": str(correlation_id),
                },
            )
            return False

        except Exception as e:
            logger.exception(
                "PostgreSQL registration failed",
                extra={
                    "node_id": str(event.node_id),
                    "correlation_id": str(correlation_id),
                },
            )
            raise InfraConnectionError(
                "PostgreSQL registration failed",
                context=ctx,
            ) from e

    def _aggregate_results(
        self,
        params: ModelAggregationParams,
    ) -> ModelDualRegistrationResult:
        """Aggregate registration results with graceful degradation.

        Determines overall status based on individual backend results:
        - Both succeed -> status="success"
        - One succeeds -> status="partial"
        - Both fail -> status="failed"

        Args:
            params: Aggregation parameters containing registration results,
                node_id, correlation_id, and timing information.

        Returns:
            ModelDualRegistrationResult with aggregated outcomes.
        """
        # Determine Consul success
        consul_registered = params.consul_result is True
        consul_error: str | None = None
        if isinstance(params.consul_result, BaseException):
            consul_error = (
                f"{type(params.consul_result).__name__}: {params.consul_result}"
            )

        # Determine PostgreSQL success
        postgres_registered = params.postgres_result is True
        postgres_error: str | None = None
        if isinstance(params.postgres_result, BaseException):
            postgres_error = (
                f"{type(params.postgres_result).__name__}: {params.postgres_result}"
            )

        # Update FSM context
        self._fsm_context.consul_registered = consul_registered
        self._fsm_context.postgres_registered = postgres_registered
        self._fsm_context.consul_error = consul_error
        self._fsm_context.postgres_error = postgres_error
        self._fsm_context.success_count = (1 if consul_registered else 0) + (
            1 if postgres_registered else 0
        )

        # Determine status
        status: Literal["success", "partial", "failed"]
        if consul_registered and postgres_registered:
            status = "success"
            self._metrics.success_count += 1
        elif consul_registered or postgres_registered:
            status = "partial"
            self._metrics.partial_count += 1
        else:
            status = "failed"
            self._metrics.failure_count += 1

        # Always increment total
        self._metrics.total_registrations += 1

        # Log performance metrics
        if params.registration_time_ms > self._TARGET_DUAL_REGISTRATION_MS:
            logger.warning(
                "Dual registration exceeded performance target",
                extra={
                    "registration_time_ms": params.registration_time_ms,
                    "target_ms": self._TARGET_DUAL_REGISTRATION_MS,
                    "node_id": str(params.node_id),
                    "correlation_id": str(params.correlation_id),
                },
            )

        # Build result - node_id is already UUID from params
        return ModelDualRegistrationResult(
            node_id=params.node_id,
            consul_registered=consul_registered,
            postgres_registered=postgres_registered,
            status=status,
            consul_error=consul_error,
            postgres_error=postgres_error,
            registration_time_ms=params.registration_time_ms,
            correlation_id=params.correlation_id,
        )

    def _build_failed_result(
        self,
        correlation_id: UUID,
        error_message: str,
    ) -> ModelDualRegistrationResult:
        """Build a failed result for validation or pre-registration errors.

        Args:
            correlation_id: Correlation ID for tracing.
            error_message: Error message describing the failure.

        Returns:
            ModelDualRegistrationResult with failed status.
        """
        self._metrics.failure_count += 1
        self._metrics.total_registrations += 1

        # Get node_id from FSM context, use a nil UUID as fallback
        # node_id is stored as UUID in ModelFSMContext
        node_id_value = self._fsm_context.node_id
        if node_id_value is None:
            node_id = UUID(int=0)  # Nil UUID as fallback
        elif isinstance(node_id_value, UUID):
            node_id = node_id_value
        else:
            # Handle string case for backwards compatibility
            try:
                node_id = UUID(str(node_id_value))
            except ValueError:
                node_id = UUID(int=0)  # Nil UUID as fallback

        # Use generic error message for both backends since registration
        # was not attempted (pre-registration failure)
        generic_error = f"Registration not attempted: {error_message}"

        return ModelDualRegistrationResult(
            node_id=node_id,
            consul_registered=False,
            postgres_registered=False,
            status="failed",
            consul_error=generic_error,
            postgres_error=generic_error,
            registration_time_ms=0.0,
            correlation_id=correlation_id,
        )


__all__ = ["NodeDualRegistrationReducer"]
