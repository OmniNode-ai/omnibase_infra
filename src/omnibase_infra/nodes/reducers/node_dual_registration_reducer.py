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
    current = Path(__file__).resolve().parent
    while current != current.parent:
        contracts_dir = current / "contracts"
        if contracts_dir.is_dir():
            return contracts_dir
        current = current.parent
    raise RuntimeError(
        "Could not find contracts directory. "
        "Ensure the FSM contract exists at contracts/fsm/dual_registration_reducer_fsm.yaml"
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

        Args:
            consul_handler: Initialized ConsulHandler for service discovery.
            db_adapter: Initialized DbAdapter for PostgreSQL operations.
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

        # FSM contract path - use robust path resolution
        if fsm_contract_path is None:
            try:
                contracts_dir = _find_contracts_dir()
                self._fsm_contract_path = (
                    contracts_dir / "fsm" / "dual_registration_reducer_fsm.yaml"
                )
            except RuntimeError:
                # Fallback to relative path construction for backwards compatibility
                self._fsm_contract_path = (
                    Path(__file__).parent.parent.parent.parent.parent
                    / "contracts"
                    / "fsm"
                    / "dual_registration_reducer_fsm.yaml"
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

            # Validate payload
            validation_passed = self._validate_payload(event, cid)

            if not validation_passed:
                # FSM: validating_payload -> registration_failed
                await self._transition(EnumFSMTrigger.VALIDATION_FAILED)
                # FSM: registration_failed -> idle
                await self._transition(EnumFSMTrigger.FAILURE_RESULT_EMITTED)
                return self._build_failed_result(cid, "Validation failed")

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
            # Log error and transition to failed state
            logger.exception(
                "Dual registration workflow failed",
                extra={
                    "node_id": self._fsm_context.node_id,
                    "correlation_id": str(cid),
                },
            )
            # Force transition to failed state
            self._current_state = EnumFSMState.REGISTRATION_FAILED
            raise RuntimeHostError(
                f"Dual registration failed: {type(e).__name__}",
                context=ctx,
            ) from e

    async def _transition(self, trigger: EnumFSMTrigger) -> None:
        """Execute FSM state transition.

        Validates the transition against the FSM contract and updates
        the current state.

        Args:
            trigger: Trigger to fire for state transition.

        Raises:
            RuntimeHostError: If transition is invalid.
        """
        # State transition map based on FSM contract
        transition_map: dict[tuple[EnumFSMState, EnumFSMTrigger], EnumFSMState] = {
            # idle -> receiving_introspection
            (
                EnumFSMState.IDLE,
                EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED,
            ): EnumFSMState.RECEIVING_INTROSPECTION,
            # receiving_introspection -> validating_payload
            (
                EnumFSMState.RECEIVING_INTROSPECTION,
                EnumFSMTrigger.EVENT_PARSED,
            ): EnumFSMState.VALIDATING_PAYLOAD,
            # validating_payload -> registering_parallel
            (
                EnumFSMState.VALIDATING_PAYLOAD,
                EnumFSMTrigger.VALIDATION_PASSED,
            ): EnumFSMState.REGISTERING_PARALLEL,
            # validating_payload -> registration_failed
            (
                EnumFSMState.VALIDATING_PAYLOAD,
                EnumFSMTrigger.VALIDATION_FAILED,
            ): EnumFSMState.REGISTRATION_FAILED,
            # registering_parallel -> aggregating_results
            (
                EnumFSMState.REGISTERING_PARALLEL,
                EnumFSMTrigger.REGISTRATION_ATTEMPTS_COMPLETE,
            ): EnumFSMState.AGGREGATING_RESULTS,
            # aggregating_results -> registration_complete
            (
                EnumFSMState.AGGREGATING_RESULTS,
                EnumFSMTrigger.ALL_BACKENDS_SUCCEEDED,
            ): EnumFSMState.REGISTRATION_COMPLETE,
            # aggregating_results -> partial_failure
            (
                EnumFSMState.AGGREGATING_RESULTS,
                EnumFSMTrigger.PARTIAL_SUCCESS,
            ): EnumFSMState.PARTIAL_FAILURE,
            # aggregating_results -> registration_failed
            (
                EnumFSMState.AGGREGATING_RESULTS,
                EnumFSMTrigger.ALL_BACKENDS_FAILED,
            ): EnumFSMState.REGISTRATION_FAILED,
            # registration_complete -> idle
            (
                EnumFSMState.REGISTRATION_COMPLETE,
                EnumFSMTrigger.RESULT_EMITTED,
            ): EnumFSMState.IDLE,
            # partial_failure -> idle
            (
                EnumFSMState.PARTIAL_FAILURE,
                EnumFSMTrigger.PARTIAL_RESULT_EMITTED,
            ): EnumFSMState.IDLE,
            # registration_failed -> idle
            (
                EnumFSMState.REGISTRATION_FAILED,
                EnumFSMTrigger.FAILURE_RESULT_EMITTED,
            ): EnumFSMState.IDLE,
        }

        key = (self._current_state, trigger)
        if key not in transition_map:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="fsm_transition",
                target_name="dual_registration_reducer",
                correlation_id=self._fsm_context.correlation_id,
            )
            raise RuntimeHostError(
                f"Invalid FSM transition: {self._current_state.value} + {trigger.value}",
                context=ctx,
                current_state=self._current_state.value,
                trigger=trigger.value,
            )

        old_state = self._current_state
        self._current_state = transition_map[key]

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
    ) -> bool:
        """Validate introspection event payload.

        Validates required fields (node_id, node_type) and logs validation
        errors.

        Args:
            event: Introspection event to validate.
            correlation_id: Correlation ID for logging.

        Returns:
            True if validation passes, False otherwise.
        """
        # ModelNodeIntrospectionEvent already validates via Pydantic
        # Additional business validation can be added here
        if event.node_id is None:
            logger.warning(
                "Validation failed: missing node_id",
                extra={"correlation_id": str(correlation_id)},
            )
            return False

        if event.node_type not in _VALID_NODE_TYPES:
            logger.warning(
                "Validation failed: invalid node_type",
                extra={
                    "node_type": event.node_type,
                    "correlation_id": str(correlation_id),
                },
            )
            return False

        return True

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

            # ConsulHandler returns dict response - use dict access pattern
            if response.get("status") == "success":
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
                    "status": response.get("status"),
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

            # DbAdapter returns ModelDbQueryResponse object - use attribute access
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
        """Build a failed result for validation errors.

        Args:
            correlation_id: Correlation ID for tracing.
            error_message: Error message to include.

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

        return ModelDualRegistrationResult(
            node_id=node_id,
            consul_registered=False,
            postgres_registered=False,
            status="failed",
            consul_error=error_message,
            postgres_error=error_message,
            registration_time_ms=0.0,
            correlation_id=correlation_id,
        )


__all__ = ["NodeDualRegistrationReducer"]
