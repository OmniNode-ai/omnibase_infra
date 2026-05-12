# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatcher adapter for delegation routing request commands.

Routes ModelDelegationRequest payloads to HandlerDelegationRouting.delta(),
which computes the deterministic routing decision (model, endpoint, tier).

Related:
    - OMN-7040: Node-based delegation pipeline
    - OMN-10868: Wire delegation routing dispatcher in DI kernel
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumDispatchStatus,
    EnumInfraTransportType,
    EnumMessageCategory,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing import (
    delta,
)
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers._util_envelope_extract import (
    extract_envelope_fields,
)
from omnibase_infra.utils import sanitize_error_message

__all__ = ["DispatcherDelegationRoutingRequest"]

logger = logging.getLogger(__name__)

TOPIC_ID_DELEGATION_ROUTING_REQUEST = "delegation.routing-request"


class DispatcherDelegationRoutingRequest(MixinAsyncCircuitBreaker):
    """Dispatcher for incoming delegation routing request commands."""

    def __init__(self) -> None:
        self._init_circuit_breaker(
            threshold=3,
            reset_timeout=20.0,
            service_name="dispatcher.delegation.routing-request",
            transport_type=EnumInfraTransportType.KAFKA,
        )

    @property
    def dispatcher_id(self) -> str:
        return "dispatcher.delegation.routing-request"

    @property
    def category(self) -> EnumMessageCategory:
        return EnumMessageCategory.COMMAND

    @property
    def message_types(self) -> set[str]:
        return {"ModelDelegationRequest", "omnibase-infra.delegation-routing-request"}

    @property
    def node_kind(self) -> EnumNodeKind:
        return EnumNodeKind.REDUCER

    async def handle(
        self,
        envelope: ModelEventEnvelope[object] | dict[str, object],
    ) -> ModelDispatchResult:
        started_at = datetime.now(UTC)
        correlation_id, raw_payload = extract_envelope_fields(envelope)

        try:
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker("handle", correlation_id)

            payload = raw_payload
            if not isinstance(payload, ModelDelegationRequest):
                if isinstance(payload, dict):
                    payload = ModelDelegationRequest.model_validate(payload)
                else:
                    return ModelDispatchResult(
                        dispatch_id=uuid4(),
                        status=EnumDispatchStatus.INVALID_MESSAGE,
                        topic=TOPIC_ID_DELEGATION_ROUTING_REQUEST,
                        dispatcher_id=self.dispatcher_id,
                        started_at=started_at,
                        completed_at=started_at,
                        duration_ms=0.0,
                        error_message=f"Expected ModelDelegationRequest, got {type(payload).__name__}",
                        correlation_id=correlation_id,
                        output_events=[],
                    )

            assert isinstance(payload, ModelDelegationRequest)

            routing_decision = delta(payload)

            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.info(
                "DispatcherDelegationRoutingRequest processed request",
                extra={
                    "correlation_id": str(correlation_id),
                    "task_type": payload.task_type,
                    "selected_model": routing_decision.selected_model,
                    "cost_tier": routing_decision.cost_tier,
                    "duration_ms": duration_ms,
                },
            )

            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.SUCCESS,
                topic=TOPIC_ID_DELEGATION_ROUTING_REQUEST,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                output_events=[routing_decision],
            )

        except InfraUnavailableError as e:
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000
            logger.error(  # noqa: TRY400
                "DispatcherDelegationRoutingRequest circuit open: %s",
                sanitize_error_message(e),
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.HANDLER_ERROR,
                topic=TOPIC_ID_DELEGATION_ROUTING_REQUEST,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitize_error_message(e),
                correlation_id=correlation_id,
                output_events=[],
            )

        except (ValidationError, ValueError, KeyError) as e:
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000
            logger.warning(
                "DispatcherDelegationRoutingRequest validation error: %s",
                sanitize_error_message(e),
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.INVALID_MESSAGE,
                topic=TOPIC_ID_DELEGATION_ROUTING_REQUEST,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitize_error_message(e),
                correlation_id=correlation_id,
                output_events=[],
            )

        except Exception as e:  # noqa: BLE001
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("handle")
            logger.error(  # noqa: TRY400
                "DispatcherDelegationRoutingRequest failed: %s",
                sanitize_error_message(e),
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.HANDLER_ERROR,
                topic=TOPIC_ID_DELEGATION_ROUTING_REQUEST,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitize_error_message(e),
                correlation_id=correlation_id,
                output_events=[],
            )
