# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Canonical Registration Reducer following ProtocolReducer pattern.

This reducer replaces the legacy NodeDualRegistrationReducer (887 lines)
with a pure function implementation (~80 lines) that follows the canonical
ONEX reducer pattern defined in DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md.

Architecture:
    - Pure function: reduce(state, event) -> ModelReducerOutput
    - No internal state - state passed in and returned
    - No I/O - emits intents for Effect layer
    - Deterministic - same inputs produce same outputs

Key Differences from Legacy:
    - No FSM machinery (FSM was for tracking async I/O, which pure reducers don't do)
    - No initialize()/shutdown() lifecycle methods
    - No internal metrics tracking (metrics are output-level concerns)
    - State is external and immutable (ModelRegistrationState)
    - Uses omnibase_core's ModelReducerOutput[T] for standard output format

Intent Emission:
    The reducer emits ModelIntent objects (reducer-layer intents) that wrap
    the typed infrastructure intents:
    - consul.register: Consul service registration
    - postgres.upsert_registration: PostgreSQL record upsert

    The payload contains the serialized typed intent for Effect layer execution.

Related:
    - NodeDualRegistrationReducer: Legacy 887-line implementation (deprecated)
    - DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md: Architecture design
    - OMN-889: Infrastructure MVP - ModelNodeIntrospectionEvent
    - OMN-912: ModelIntent typed payloads
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from omnibase_core.enums import EnumReductionType, EnumStreamingMode
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.nodes import ModelReducerOutput

from omnibase_infra.models.registration import (
    ModelNodeIntrospectionEvent,
    ModelNodeRegistrationRecord,
)
from omnibase_infra.nodes.reducers.models.model_registration_state import (
    ModelRegistrationState,
)


class RegistrationReducer:
    """Pure reducer for node registration workflow.

    Follows ProtocolReducer pattern:
    - reduce(state, event) -> ModelReducerOutput
    - Pure function, no side effects
    - Emits intents for Consul and PostgreSQL registration

    This is a stateless class - all state is passed in and returned via
    ModelRegistrationState. The class exists to group related pure functions.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        >>> from omnibase_infra.nodes.reducers import RegistrationReducer
        >>> from omnibase_infra.nodes.reducers.models import ModelRegistrationState
        >>>
        >>> reducer = RegistrationReducer()
        >>> state = ModelRegistrationState()  # Initial idle state
        >>> event = ModelNodeIntrospectionEvent(
        ...     node_id=uuid4(),
        ...     node_type="effect",
        ...     node_version="1.0.0",
        ...     endpoints={"health": "http://localhost:8080/health"},
        ... )
        >>> output = reducer.reduce(state, event)
        >>> print(output.result.status)  # "pending"
        >>> print(len(output.intents))   # 2 (Consul + PostgreSQL)
    """

    def reduce(
        self,
        state: ModelRegistrationState,
        event: ModelNodeIntrospectionEvent,
    ) -> ModelReducerOutput[ModelRegistrationState]:
        """Pure reduce function: state + event -> new_state + intents.

        Processes a node introspection event and emits registration intents
        for both Consul and PostgreSQL backends. The returned output contains
        the new state and any intents to be executed by the Effect layer.

        Idempotency:
            If the event has already been processed (based on event_id), the
            reducer returns immediately with the current state and no intents.

        Validation:
            If the event fails validation (e.g., missing node_id), the reducer
            transitions to failed state with no intents.

        Args:
            state: Current registration state (immutable).
            event: Node introspection event to process.

        Returns:
            ModelReducerOutput containing new_state and intents tuple.
            The result field contains the new ModelRegistrationState.
            The intents field contains registration intents for Effect layer.
        """
        import time

        start_time = time.perf_counter()

        # Resolve event ID for idempotency
        event_id = event.correlation_id or uuid4()

        # Idempotency guard - skip if we've already processed this event
        if state.is_duplicate_event(event_id):
            return self._build_output(
                state=state,
                intents=(),
                processing_time_ms=0.0,
                items_processed=0,
            )

        # Validate event
        if not self._is_valid(event):
            new_state = state.with_failure("validation_failed", event_id)
            return self._build_output(
                state=new_state,
                intents=(),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                items_processed=0,
            )

        # Build registration intents
        correlation_id = event.correlation_id or event_id
        consul_intent = self._build_consul_intent(event, correlation_id)
        postgres_intent = self._build_postgres_intent(event, correlation_id)

        # Collect non-None intents
        intents: tuple[ModelIntent, ...] = tuple(
            intent for intent in [consul_intent, postgres_intent] if intent is not None
        )

        # Transition to pending state
        new_state = state.with_pending_registration(event.node_id, event_id)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return self._build_output(
            state=new_state,
            intents=intents,
            processing_time_ms=processing_time_ms,
            items_processed=1,
        )

    def _is_valid(self, event: ModelNodeIntrospectionEvent) -> bool:
        """Validate introspection event.

        Validates that required fields are present. Pydantic handles type
        validation at model construction; this method checks semantic validity.

        Args:
            event: Introspection event to validate.

        Returns:
            True if the event is valid for processing.
        """
        # node_id is required for registration
        return event.node_id is not None

    def _build_consul_intent(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> ModelIntent | None:
        """Build Consul registration intent (pure, no I/O).

        Creates a ModelIntent that describes the desired Consul service
        registration. The Effect layer is responsible for executing this intent.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelIntent with intent_type="consul.register" and Consul payload.
        """
        service_id = f"node-{event.node_type}-{event.node_id}"
        service_name = f"onex-{event.node_type}"
        tags = [
            f"node_type:{event.node_type}",
            f"node_version:{event.node_version}",
        ]

        # Build health check configuration if health endpoint is provided
        health_endpoint = event.endpoints.get("health") if event.endpoints else None
        health_check: dict[str, Any] | None = None
        if health_endpoint:
            health_check = {
                "HTTP": health_endpoint,
                "Interval": "10s",
                "Timeout": "5s",
            }

        # Build payload for Consul registration
        payload: dict[str, Any] = {
            "correlation_id": str(correlation_id),
            "service_id": service_id,
            "service_name": service_name,
            "tags": tags,
        }
        if health_check:
            payload["health_check"] = health_check

        return ModelIntent(
            intent_type="consul.register",
            target=f"consul://service/{service_name}",
            payload=payload,
        )

    def _build_postgres_intent(
        self,
        event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> ModelIntent | None:
        """Build PostgreSQL upsert intent (pure, no I/O).

        Creates a ModelIntent that describes the desired PostgreSQL record
        upsert. The Effect layer is responsible for executing this intent.

        Args:
            event: Introspection event containing node data.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelIntent with intent_type="postgres.upsert_registration" and record payload.
        """
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

        # Build the registration record as a Pydantic model for validation,
        # then serialize to dict for the intent payload
        record = ModelNodeRegistrationRecord(
            node_id=event.node_id,
            node_type=event.node_type,
            node_version=event.node_version,
            capabilities=capabilities_dict,
            endpoints=dict(event.endpoints) if event.endpoints else {},
            metadata=metadata_dict,
            health_endpoint=(
                event.endpoints.get("health") if event.endpoints else None
            ),
            registered_at=now,
            updated_at=now,
        )

        # Build payload with correlation_id and serialized record
        payload: dict[str, Any] = {
            "correlation_id": str(correlation_id),
            "record": record.model_dump(mode="json"),
        }

        return ModelIntent(
            intent_type="postgres.upsert_registration",
            target=f"postgres://node_registrations/{event.node_id}",
            payload=payload,
        )

    def _build_output(
        self,
        state: ModelRegistrationState,
        intents: tuple[ModelIntent, ...],
        processing_time_ms: float,
        items_processed: int,
    ) -> ModelReducerOutput[ModelRegistrationState]:
        """Build standardized ModelReducerOutput.

        Creates the output model with all required fields for the
        omnibase_core reducer output contract.

        Args:
            state: New registration state to return.
            intents: Tuple of ModelIntent objects to emit.
            processing_time_ms: Time taken to process the event.
            items_processed: Number of events processed (0 or 1).

        Returns:
            ModelReducerOutput containing the state and intents.
        """
        return ModelReducerOutput(
            result=state,
            operation_id=uuid4(),
            reduction_type=EnumReductionType.MERGE,
            processing_time_ms=processing_time_ms,
            items_processed=items_processed,
            conflicts_resolved=0,
            streaming_mode=EnumStreamingMode.BATCH,
            batches_processed=1,
            intents=intents,
        )


__all__ = ["RegistrationReducer"]
