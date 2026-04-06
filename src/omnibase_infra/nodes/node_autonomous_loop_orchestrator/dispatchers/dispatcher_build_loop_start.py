# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatcher adapter for incoming build loop start commands.

Routes ModelLoopStartCommand payloads to HandlerLoopOrchestrator.handle(),
which runs the 6-phase autonomous build loop cycle.

Related:
    - OMN-7319: node_autonomous_loop_orchestrator
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
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
from omnibase_infra.nodes.node_autonomous_loop_orchestrator.models.model_loop_start_command import (
    ModelLoopStartCommand,
)
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers._util_envelope_extract import (
    extract_envelope_fields,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_autonomous_loop_orchestrator.handlers.handler_loop_orchestrator import (
        HandlerLoopOrchestrator,
    )

__all__ = ["DispatcherBuildLoopStart"]

logger = logging.getLogger(__name__)

TOPIC_ID_BUILD_LOOP_START = "build-loop.start"


class DispatcherBuildLoopStart(MixinAsyncCircuitBreaker):
    """Dispatcher for incoming build loop start commands."""

    def __init__(
        self,
        handler: HandlerLoopOrchestrator,
    ) -> None:
        self._handler = handler
        self._init_circuit_breaker(
            threshold=3,
            reset_timeout=20.0,
            service_name="dispatcher.build-loop.start",
            transport_type=EnumInfraTransportType.KAFKA,
        )

    @property
    def dispatcher_id(self) -> str:
        return "dispatcher.build-loop.start"

    @property
    def category(self) -> EnumMessageCategory:
        return EnumMessageCategory.COMMAND

    @property
    def message_types(self) -> set[str]:
        return {"ModelLoopStartCommand", "omnibase-infra.build-loop-start"}

    @property
    def node_kind(self) -> EnumNodeKind:
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self,
        envelope: ModelEventEnvelope[object] | dict[str, object],
    ) -> ModelDispatchResult:
        started_at = datetime.now(UTC)
        logger.info(
            "[BUILD-LOOP] === DISPATCHER ENTRY === DispatcherBuildLoopStart.handle() "
            "called (envelope_type=%s)",
            type(envelope).__name__,
        )
        correlation_id, raw_payload = extract_envelope_fields(envelope)
        logger.info(
            "[BUILD-LOOP] Dispatcher extracted fields: correlation_id=%s, "
            "payload_type=%s, payload_keys=%s",
            correlation_id,
            type(raw_payload).__name__,
            list(raw_payload.keys()) if isinstance(raw_payload, dict) else "N/A",
        )

        try:
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker("handle", correlation_id)

            payload = raw_payload
            if not isinstance(payload, ModelLoopStartCommand):
                if isinstance(payload, dict):
                    logger.info(
                        "[BUILD-LOOP] Dispatcher deserializing dict to "
                        "ModelLoopStartCommand (correlation_id=%s)",
                        correlation_id,
                    )
                    payload = ModelLoopStartCommand.model_validate(payload)
                    logger.info(
                        "[BUILD-LOOP] Dispatcher deserialization success: "
                        "max_cycles=%d, dry_run=%s, skip_closeout=%s "
                        "(correlation_id=%s)",
                        payload.max_cycles,
                        payload.dry_run,
                        payload.skip_closeout,
                        correlation_id,
                    )
                else:
                    logger.warning(
                        "[BUILD-LOOP] Dispatcher received unexpected payload type: %s "
                        "(correlation_id=%s)",
                        type(payload).__name__,
                        correlation_id,
                    )
                    return ModelDispatchResult(
                        dispatch_id=uuid4(),
                        status=EnumDispatchStatus.INVALID_MESSAGE,
                        topic=TOPIC_ID_BUILD_LOOP_START,
                        dispatcher_id=self.dispatcher_id,
                        started_at=started_at,
                        completed_at=started_at,
                        duration_ms=0.0,
                        error_message=f"Expected ModelLoopStartCommand, got {type(payload).__name__}",
                        correlation_id=correlation_id,
                        output_events=[],
                    )

            assert isinstance(payload, ModelLoopStartCommand)

            logger.info(
                "[BUILD-LOOP] Dispatcher invoking handler.handle() (correlation_id=%s)",
                correlation_id,
            )
            result = await self._handler.handle(payload)

            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.info(
                "DispatcherBuildLoopStart processed command",
                extra={
                    "correlation_id": str(correlation_id),
                    "cycles_completed": result.cycles_completed,
                    "cycles_failed": result.cycles_failed,
                    "duration_ms": duration_ms,
                },
            )

            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.SUCCESS,
                topic=TOPIC_ID_BUILD_LOOP_START,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                output_events=[],
            )

        except InfraUnavailableError as e:
            completed_at = datetime.now(UTC)
            duration_ms = (completed_at - started_at).total_seconds() * 1000
            logger.error(  # noqa: TRY400
                "DispatcherBuildLoopStart circuit open: %s",
                sanitize_error_message(e),
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.HANDLER_ERROR,
                topic=TOPIC_ID_BUILD_LOOP_START,
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
                "DispatcherBuildLoopStart validation error: %s",
                sanitize_error_message(e),
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.INVALID_MESSAGE,
                topic=TOPIC_ID_BUILD_LOOP_START,
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
                "DispatcherBuildLoopStart failed: %s",
                sanitize_error_message(e),
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.HANDLER_ERROR,
                topic=TOPIC_ID_BUILD_LOOP_START,
                dispatcher_id=self.dispatcher_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=sanitize_error_message(e),
                correlation_id=correlation_id,
                output_events=[],
            )
