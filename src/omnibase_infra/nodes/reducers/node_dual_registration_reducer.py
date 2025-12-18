# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dual Registration Reducer Node.

This module provides NodeDualRegistrationReducer for coordinating parallel
registration to Consul and PostgreSQL backends in the ONEX 2-way registration
pattern.

Architecture:
    NodeDualRegistrationReducer belongs to the REDUCER layer of ONEX 4-node
    architecture. As a PURE reducer, it performs NO I/O operations. Instead,
    it emits typed intents (ModelConsulRegisterIntent, ModelPostgresUpsertRegistrationIntent)
    that describe the desired side effects. An Effect node is responsible for
    executing these intents.

    This design ensures:
    - Reducer purity: same inputs always produce same outputs
    - Testability: no mocking required for I/O
    - Replay capability: given same inputs, replay produces identical outputs
    - Separation of concerns: business logic (reducer) vs infrastructure (effect)

FSM Integration:
    The reducer loads its state machine from:
    contracts/fsm/dual_registration_reducer_fsm.yaml

    States:
    - idle: Waiting for introspection events
    - receiving_introspection: Parsing NODE_INTROSPECTION event
    - validating_payload: Validating event structure
    - building_intents: Building typed registration intents (was: registering_parallel)
    - aggregating_results: Combining registration outcomes
    - registration_complete: Both intents emitted successfully
    - partial_failure: One intent could not be built (validation failure)
    - registration_failed: No intents could be emitted

Intent Emission:
    The reducer emits typed intents from omnibase_core.models.intents:
    - ModelConsulRegisterIntent: Declares Consul service registration
    - ModelPostgresUpsertRegistrationIntent: Declares PostgreSQL record upsert

    Effect nodes receive these intents and execute the actual I/O operations.

Related:
    - OMN-889: Infrastructure MVP - ModelNodeIntrospectionEvent
    - OMN-912: ModelIntent typed payloads
    - contracts/fsm/dual_registration_reducer_fsm.yaml: FSM contract
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import yaml
from omnibase_core.models.intents import (
    ModelConsulRegisterIntent,
    ModelCoreRegistrationIntent,
    ModelPostgresUpsertRegistrationIntent,
)

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.models.registration import (
    ModelNodeIntrospectionEvent,
    ModelNodeRegistrationRecord,
)
from omnibase_infra.nodes.reducers.enums import EnumFSMState, EnumFSMTrigger
from omnibase_infra.nodes.reducers.models import (
    ModelDualRegistrationReducerOutput,
    ModelFSMContext,
    ModelFSMContract,
    ModelReducerMetrics,
)

logger = logging.getLogger(__name__)


# Maximum directory traversal depth to prevent infinite loops
_MAX_CONTRACTS_DIR_SEARCH_DEPTH: int = 15

# Expected marker file within contracts directory for validation
_EXPECTED_FSM_CONTRACT_SUBPATH: str = "fsm/dual_registration_reducer_fsm.yaml"


def _find_contracts_dir() -> Path:
    """Find contracts directory by traversing up from current file.

    This is more robust than multiple .parent calls and handles
    different installation/development environments.

    Validation Strategy:
        To avoid finding the wrong contracts/ directory (e.g., from another
        package or a mypy cache), this function performs two validations:

        1. **Marker file validation**: The contracts/ directory must contain
           the expected FSM contract file (fsm/dual_registration_reducer_fsm.yaml).

        2. **Parent path validation**: The contracts/ directory must be within
           the omnibase_infra package (i.e., "omnibase_infra" must appear in
           the resolved path).

        Both validations must pass before returning the contracts directory.
        If a contracts/ directory fails either validation, the search continues
        up the directory tree.

    Returns:
        Path to the contracts directory containing the expected FSM contract.

    Raises:
        RuntimeError: If contracts directory with expected FSM file cannot be found
            within the traversal depth limit, or if no valid omnibase_infra
            contracts directory exists.
    """
    # Start from current file's directory and walk up the tree until we find
    # a directory containing 'contracts/' with the expected FSM contract.
    # This handles both development (source checkout) and installed package
    # scenarios where the relative path depth may vary.
    current = Path(__file__).resolve().parent
    search_depth = 0

    while current != current.parent and search_depth < _MAX_CONTRACTS_DIR_SEARCH_DEPTH:
        contracts_dir = current / "contracts"

        # Validation 1: Check if contracts directory exists
        if contracts_dir.is_dir():
            # Validation 2: Verify expected FSM contract file exists
            expected_fsm = contracts_dir / _EXPECTED_FSM_CONTRACT_SUBPATH
            if expected_fsm.exists():
                # Validation 3: Ensure this is the omnibase_infra contracts directory
                # This prevents accidentally finding contracts/ from another package
                # or from cache directories (e.g., .mypy_cache)
                resolved_path = str(contracts_dir.resolve())
                if "omnibase_infra" in resolved_path:
                    return contracts_dir
                # Found contracts/ with FSM file but wrong package - log and continue
                logger.debug(
                    "Found contracts directory with FSM file but outside omnibase_infra: %s",
                    contracts_dir,
                )

        current = current.parent
        search_depth += 1

    raise RuntimeError(
        "Could not find omnibase_infra contracts directory. "
        f"Searched {search_depth} levels up from {Path(__file__).resolve().parent}. "
        f"Ensure the FSM contract exists at contracts/{_EXPECTED_FSM_CONTRACT_SUBPATH} "
        "within the omnibase_infra package."
    )


class NodeDualRegistrationReducer:
    """Pure dual registration reducer that emits typed intents.

    Listens for NODE_INTROSPECTION events and emits typed registration intents
    for Consul and PostgreSQL backends. As a PURE reducer, this class performs
    NO I/O operations - it only computes and emits intents.

    Pure Reducer Architecture:
        This reducer follows the ONEX Pure Reducer Architecture pattern:
        - NO I/O operations (no database, network, file system access)
        - NO side effects (logging is acceptable for debugging)
        - Deterministic: same inputs always produce same outputs
        - Emits typed ModelIntent objects for Effect layer execution

        The separation of concerns ensures:
        1. **Replay capability**: Given same inputs, replay produces identical outputs
        2. **Testability**: No mocking required for I/O
        3. **Predictability**: FSM transitions are deterministic

    Status Semantics (Intent Emission, not Registration):
        - "success": Both Consul and PostgreSQL intents were emitted
        - "partial": Only one intent could be emitted (validation failure)
        - "failed": No intents could be emitted (event validation failed)

    Attributes:
        metrics: Aggregation metrics tracking intent emission outcomes.
        fsm_contract: Loaded FSM contract from YAML.
        current_state: Current FSM state.
        fsm_context: Context variables maintained during FSM execution.

    Example:
        >>> from omnibase_infra.nodes.reducers import NodeDualRegistrationReducer
        >>>
        >>> reducer = NodeDualRegistrationReducer()
        >>> await reducer.initialize()
        >>>
        >>> # Process introspection event - returns intents, not registration result
        >>> output = await reducer.execute(introspection_event, correlation_id)
        >>> print(output.status)  # "success", "partial", or "failed"
        >>> print(len(output.intents))  # Number of intents emitted
        >>>
        >>> # Effect layer executes the intents
        >>> for intent in output.intents:
        ...     await effect_node.execute(intent)
    """

    # Default performance target constants
    _DEFAULT_TARGET_INTENT_BUILD_MS: float = 50.0
    _DEFAULT_TARGET_AGGREGATION_OVERHEAD_MS: float = 10.0

    def __init__(
        self,
        fsm_contract_path: Path | None = None,
        target_intent_build_ms: float | None = None,
        target_aggregation_overhead_ms: float | None = None,
    ) -> None:
        """Initialize pure dual registration reducer.

        Pure Reducer Design:
            This reducer is PURE - it performs no I/O operations. Unlike the
            previous design that injected ConsulHandler and DbAdapter, this
            reducer only builds typed intents that describe the desired
            side effects.

            The Effect layer (not this reducer) is responsible for:
            1. Receiving the emitted intents
            2. Executing actual I/O operations (Consul registration, DB upsert)
            3. Reporting success/failure back to the orchestrator

            This separation ensures:
            - Reducer is testable without I/O mocking
            - Replay produces identical outputs
            - FSM transitions are deterministic

        Args:
            fsm_contract_path: Optional path to FSM contract YAML. If not provided,
                defaults to contracts/fsm/dual_registration_reducer_fsm.yaml.
            target_intent_build_ms: Optional performance target for intent building
                in milliseconds. Defaults to 50.0ms. Exceeded thresholds trigger
                warning logs.
            target_aggregation_overhead_ms: Optional performance target for
                aggregation overhead in milliseconds. Defaults to 10.0ms.
        """
        self._metrics = ModelReducerMetrics()
        self._initialized = False

        # Performance thresholds (configurable for different environments)
        self._target_intent_build_ms = (
            target_intent_build_ms
            if target_intent_build_ms is not None
            else self._DEFAULT_TARGET_INTENT_BUILD_MS
        )
        self._target_aggregation_overhead_ms = (
            target_aggregation_overhead_ms
            if target_aggregation_overhead_ms is not None
            else self._DEFAULT_TARGET_AGGREGATION_OVERHEAD_MS
        )

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
        As a pure reducer, no I/O handlers need to be initialized.

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
        As a pure reducer, there are no I/O resources to release.
        """
        self._current_state = EnumFSMState.IDLE
        self._fsm_context = ModelFSMContext()
        self._initialized = False
        logger.info("NodeDualRegistrationReducer shutdown complete")

    async def execute(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID | None = None,
    ) -> ModelDualRegistrationReducerOutput:
        """Execute pure dual registration workflow for introspection event.

        Processes a NODE_INTROSPECTION event through the FSM workflow,
        building typed registration intents for Consul and PostgreSQL.
        This method performs NO I/O - it only builds and emits intents.

        Pure Reducer Semantics:
            This method returns intents describing the desired registrations,
            NOT the registration results. The Effect layer is responsible for:
            1. Receiving the emitted intents
            2. Executing actual I/O operations
            3. Reporting success/failure

        Args:
            event: Node introspection event to process.
            correlation_id: Optional correlation ID for tracing. If not provided,
                uses event.correlation_id or generates a new one.

        Returns:
            ModelDualRegistrationReducerOutput containing typed intents.

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
                return self._build_failed_output(cid, validation_error)

            # FSM: validating_payload -> registering_parallel
            # (Note: state name unchanged for FSM compatibility, but we're building intents)
            await self._transition(EnumFSMTrigger.VALIDATION_PASSED)

            # Build registration intents (pure computation, no I/O)
            output = self._build_registration_intents(event, cid)

            # FSM: registering_parallel -> aggregating_results
            await self._transition(EnumFSMTrigger.REGISTRATION_ATTEMPTS_COMPLETE)

            # Determine outcome and transition to terminal state
            if output.status == "success":
                # FSM: aggregating_results -> registration_complete
                await self._transition(EnumFSMTrigger.ALL_BACKENDS_SUCCEEDED)
                # FSM: registration_complete -> idle
                await self._transition(EnumFSMTrigger.RESULT_EMITTED)
            elif output.status == "partial":
                # FSM: aggregating_results -> partial_failure
                await self._transition(EnumFSMTrigger.PARTIAL_SUCCESS)
                # FSM: partial_failure -> idle
                await self._transition(EnumFSMTrigger.PARTIAL_RESULT_EMITTED)
            else:
                # FSM: aggregating_results -> registration_failed
                await self._transition(EnumFSMTrigger.ALL_BACKENDS_FAILED)
                # FSM: registration_failed -> idle
                await self._transition(EnumFSMTrigger.FAILURE_RESULT_EMITTED)

            return output

        except RuntimeHostError:
            raise
        except Exception as e:
            # =================================================================
            # ERROR RECOVERY PATTERN: FSM State Reset
            # =================================================================
            # When an unexpected exception occurs during the intent building workflow,
            # we must ensure the FSM returns to the IDLE state to allow subsequent
            # events to be processed. Without this recovery, the reducer would be
            # stuck in an intermediate state and unable to handle new events.
            #
            # Recovery steps:
            # 1. Log the exception with full context for debugging
            # 2. Force FSM to REGISTRATION_FAILED state (error terminal)
            # 3. Transition from REGISTRATION_FAILED -> IDLE via FAILURE_RESULT_EMITTED
            # 4. Re-raise as RuntimeHostError to signal failure to caller
            # =================================================================
            logger.exception(
                "Dual registration intent building failed",
                extra={
                    "node_id": self._fsm_context.node_id,
                    "current_state": self._current_state.value,
                    "correlation_id": str(cid),
                },
            )

            # Force transition to failed state (handles any intermediate state)
            self._current_state = EnumFSMState.REGISTRATION_FAILED

            # FSM: registration_failed -> idle (complete the FSM cycle)
            try:
                await self._transition(EnumFSMTrigger.FAILURE_RESULT_EMITTED)
            except Exception as transition_error:
                # Log but don't mask the original error
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
                f"Intent building failed: {type(e).__name__}",
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

        Validates required fields and logs validation errors with distinct
        error messages for each failure type.

        Note:
            node_type validation is handled by Pydantic at model construction
            via Literal["effect", "compute", "reducer", "orchestrator"]. This
            method only validates fields that may be None despite being required.

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

        # Validate node_id is present (Pydantic marks it required but mocks may bypass)
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

        # node_type validation is enforced by Pydantic Literal type at model
        # construction - no duplicate validation needed here

        return True, ""

    def _build_registration_intents(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> ModelDualRegistrationReducerOutput:
        """Build typed registration intents for both backends.

        This is a PURE method - it performs no I/O operations.
        It builds typed intent objects that describe the desired
        registration operations for the Effect layer to execute.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelDualRegistrationReducerOutput with typed intents.
        """
        start_time = time.perf_counter()
        intents: list[ModelCoreRegistrationIntent] = []

        # Build Consul registration intent
        consul_intent = self._build_consul_intent(event, correlation_id)
        consul_emitted = consul_intent is not None
        if consul_intent is not None:
            intents.append(consul_intent)

        # Build PostgreSQL registration intent
        postgres_intent = self._build_postgres_intent(event, correlation_id)
        postgres_emitted = postgres_intent is not None
        if postgres_intent is not None:
            intents.append(postgres_intent)

        # Calculate elapsed time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Determine status based on intent emission
        status: Literal["success", "partial", "failed"]
        if consul_emitted and postgres_emitted:
            status = "success"
            self._metrics.success_count += 1
        elif consul_emitted or postgres_emitted:
            status = "partial"
            self._metrics.partial_count += 1
        else:
            status = "failed"
            self._metrics.failure_count += 1

        # Always increment total
        self._metrics.total_registrations += 1

        # Update FSM context for intent emission (not registration result)
        self._fsm_context.consul_registered = consul_emitted
        self._fsm_context.postgres_registered = postgres_emitted
        self._fsm_context.success_count = (1 if consul_emitted else 0) + (
            1 if postgres_emitted else 0
        )

        # Log performance metrics
        if elapsed_ms > self._target_intent_build_ms:
            logger.warning(
                "Intent building exceeded performance target",
                extra={
                    "intent_build_time_ms": elapsed_ms,
                    "target_ms": self._target_intent_build_ms,
                    "node_id": str(event.node_id),
                    "correlation_id": str(correlation_id),
                },
            )

        return ModelDualRegistrationReducerOutput(
            node_id=event.node_id,
            intents=tuple(intents),
            status=status,
            consul_intent_emitted=consul_emitted,
            postgres_intent_emitted=postgres_emitted,
            validation_error=None,
            processing_time_ms=elapsed_ms,
            correlation_id=correlation_id,
        )

    def _build_consul_intent(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> ModelConsulRegisterIntent | None:
        """Build Consul registration intent.

        This is a PURE method - it performs no I/O operations.
        It builds a typed ModelConsulRegisterIntent that declares
        the desired Consul service registration.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelConsulRegisterIntent if intent can be built, None otherwise.
        """
        # Build service ID
        service_id = f"node-{event.node_type}-{event.node_id}"
        service_name = f"onex-{event.node_type}"

        # Build tags
        tags = [
            f"node_type:{event.node_type}",
            f"node_version:{event.node_version}",
        ]

        # Build health check configuration
        health_endpoint = event.endpoints.get("health")
        health_check: dict[str, str] | None = None
        if health_endpoint:
            health_check = {
                "HTTP": health_endpoint,
                "Interval": "10s",
                "Timeout": "5s",
            }

        logger.debug(
            "Building Consul registration intent",
            extra={
                "service_id": service_id,
                "service_name": service_name,
                "node_id": str(event.node_id),
                "correlation_id": str(correlation_id),
            },
        )

        return ModelConsulRegisterIntent(
            correlation_id=correlation_id,
            service_id=service_id,
            service_name=service_name,
            tags=tags,
            health_check=health_check,
        )

    def _build_postgres_intent(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> ModelPostgresUpsertRegistrationIntent | None:
        """Build PostgreSQL upsert registration intent.

        This is a PURE method - it performs no I/O operations.
        It builds a typed ModelPostgresUpsertRegistrationIntent that
        declares the desired PostgreSQL record upsert.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelPostgresUpsertRegistrationIntent if intent can be built, None otherwise.
        """
        # Build registration record
        now = datetime.now(UTC)

        # Convert capabilities to dict if it's a model
        if hasattr(event.capabilities, "model_dump"):
            capabilities_dict = event.capabilities.model_dump(mode="json")
        else:
            capabilities_dict = dict(event.capabilities) if event.capabilities else {}

        # Convert metadata to dict if it's a model
        if hasattr(event.metadata, "model_dump"):
            metadata_dict = event.metadata.model_dump(mode="json")
        else:
            metadata_dict = dict(event.metadata) if event.metadata else {}

        record = ModelNodeRegistrationRecord(
            node_id=event.node_id,
            node_type=event.node_type,
            node_version=event.node_version,
            capabilities=capabilities_dict,
            endpoints=dict(event.endpoints) if event.endpoints else {},
            metadata=metadata_dict,
            health_endpoint=event.endpoints.get("health") if event.endpoints else None,
            registered_at=now,
            updated_at=now,
        )

        logger.debug(
            "Building PostgreSQL upsert registration intent",
            extra={
                "node_id": str(event.node_id),
                "node_type": event.node_type,
                "correlation_id": str(correlation_id),
            },
        )

        return ModelPostgresUpsertRegistrationIntent(
            correlation_id=correlation_id,
            record=record,
        )

    def _build_failed_output(
        self,
        correlation_id: UUID,
        error_message: str,
    ) -> ModelDualRegistrationReducerOutput:
        """Build a failed output for validation or pre-intent-building errors.

        Args:
            correlation_id: Correlation ID for tracing.
            error_message: Error message describing the failure.

        Returns:
            ModelDualRegistrationReducerOutput with failed status and no intents.
        """
        self._metrics.failure_count += 1
        self._metrics.total_registrations += 1

        # Get node_id from FSM context, use a nil UUID as fallback
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

        return ModelDualRegistrationReducerOutput(
            node_id=node_id,
            intents=(),
            status="failed",
            consul_intent_emitted=False,
            postgres_intent_emitted=False,
            validation_error=error_message,
            processing_time_ms=0.0,
            correlation_id=correlation_id,
        )


__all__ = ["NodeDualRegistrationReducer"]
