# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dispatch result applier for processing ModelDispatchResult outputs.

This module provides the DispatchResultApplier, a runtime-level service
that processes the output of MessageDispatchEngine dispatch operations. It
handles publishing output events to the event bus and delegating intents to
the IntentExecutor.

Architecture:
    The applier sits between the dispatch engine and the event bus:

    EventBusSubcontractWiring -> MessageDispatchEngine -> DispatchResultApplier
                                                          |-> publish output events
                                                          |-> delegate intents (Phase C)

    This separation keeps the dispatch engine pure (routing only) while the
    applier handles side effects (publishing, intent execution).

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - EventBusSubcontractWiring: Creates subscriptions that feed the engine
    - MessageDispatchEngine: Routes messages to dispatchers
    - IntentExecutor: Executes intents from dispatch results (Phase C)

.. versionadded:: 0.7.0
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus, EnumInfraTransportType
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
    from omnibase_infra.protocols import ProtocolEventBusLike
    from omnibase_infra.runtime.service_intent_executor import IntentExecutor

logger = logging.getLogger(__name__)


class DispatchResultApplier:
    """Processes ModelDispatchResult: publishes output events and delegates intents.

    This service is injected into the dispatch callback chain by
    EventBusSubcontractWiring. After the dispatch engine routes a message
    to a dispatcher and receives a ModelDispatchResult, this applier:

    1. Publishes output events to the configured output topic
    2. Delegates intents to IntentExecutor (Phase C)
    3. Records dispatch metrics for observability

    Thread Safety:
        This class is designed for single-threaded async use. The underlying
        event bus implementations handle their own thread safety.

    Attributes:
        _event_bus: Event bus for publishing output events.
        _output_topic: Topic to publish output events to.
        _intent_executor: Optional intent executor for delegating intents
            to effect layer handlers (Phase C).

    Example:
        ```python
        applier = DispatchResultApplier(
            event_bus=event_bus,
            output_topic="onex.evt.platform.node-registration-result.v1",
        )
        await applier.apply(dispatch_result)
        ```

    .. versionadded:: 0.7.0
    """

    def __init__(
        self,
        event_bus: ProtocolEventBusLike,
        output_topic: str,
        intent_executor: IntentExecutor | None = None,
    ) -> None:
        """Initialize the dispatch result applier.

        Args:
            event_bus: Event bus for publishing output events.
            output_topic: Topic to publish output events to.
            intent_executor: Optional intent executor for delegating intents
                to effect layer handlers. When provided, intents from dispatch
                results are forwarded to the executor for effect layer processing.
        """
        self._event_bus = event_bus
        self._output_topic = output_topic
        self._intent_executor = intent_executor

    async def apply(
        self,
        result: ModelDispatchResult,
        correlation_id: UUID | None = None,
    ) -> None:
        """Process a dispatch result: publish output events and delegate intents.

        Args:
            result: The dispatch result from the dispatch engine.
            correlation_id: Optional correlation ID for tracing.
        """
        effective_correlation_id = correlation_id or result.correlation_id or uuid4()

        if result.status != EnumDispatchStatus.SUCCESS:
            logger.debug(
                "Skipping result apply for non-success status=%s "
                "dispatcher_id=%s correlation_id=%s",
                result.status.value if result.status else "unknown",
                result.dispatcher_id,
                str(effective_correlation_id),
            )
            return

        # Publish output events
        if result.output_events:
            for output_event in result.output_events:
                try:
                    output_envelope: ModelEventEnvelope[BaseModel] = ModelEventEnvelope(
                        payload=output_event,
                        correlation_id=effective_correlation_id,
                        envelope_timestamp=datetime.now(UTC),
                    )

                    await self._event_bus.publish_envelope(
                        envelope=output_envelope,
                        topic=self._output_topic,
                    )

                    logger.info(
                        "Published output event to %s (correlation_id=%s)",
                        self._output_topic,
                        str(effective_correlation_id),
                        extra={
                            "output_event_type": type(output_event).__name__,
                            "envelope_id": str(output_envelope.envelope_id),
                            "dispatcher_id": result.dispatcher_id,
                        },
                    )
                except Exception as pub_err:
                    logger.warning(
                        "Failed to publish output event: %s (correlation_id=%s)",
                        sanitize_error_message(pub_err),
                        str(effective_correlation_id),
                        extra={
                            "error_type": type(pub_err).__name__,
                            "dispatcher_id": result.dispatcher_id,
                        },
                    )
                    # Re-raise so the caller can classify the error and
                    # apply retry/DLQ logic. Swallowing publish failures
                    # causes offset commit despite lost output events.
                    raise

            logger.debug(
                "Applied %d output events from dispatcher=%s (correlation_id=%s)",
                len(result.output_events),
                result.dispatcher_id,
                str(effective_correlation_id),
            )

        # Delegate intents to effect layer via IntentExecutor
        output_intents = result.output_intents
        if output_intents and self._intent_executor is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="dispatch_result_applier.apply_intents",
            )
            raise RuntimeHostError(
                f"Dispatch result contains {len(output_intents)} intent(s) but no "
                f"IntentExecutor is configured — intents would be lost "
                f"(dispatcher_id={result.dispatcher_id})",
                context=context,
            )
        if self._intent_executor is not None and output_intents:
            try:
                await self._intent_executor.execute_all(
                    output_intents,
                    correlation_id=effective_correlation_id,
                )
                logger.debug(
                    "Delegated %d intents from dispatcher=%s (correlation_id=%s)",
                    len(output_intents),
                    result.dispatcher_id,
                    str(effective_correlation_id),
                )
            except Exception as intent_err:
                logger.warning(
                    "Failed to execute intents: %s (correlation_id=%s)",
                    sanitize_error_message(intent_err),
                    str(effective_correlation_id),
                    extra={
                        "error_type": type(intent_err).__name__,
                        "dispatcher_id": result.dispatcher_id,
                        "intent_count": len(output_intents),
                    },
                )
                # Re-raise so the caller (EventBusSubcontractWiring) can
                # classify the error and apply retry/DLQ logic. Swallowing
                # intent errors here would cause Kafka offset commit despite
                # failed PostgreSQL upserts, leading to data loss.
                raise


__all__: list[str] = ["DispatchResultApplier"]
