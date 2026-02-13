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
                                                          |-> delegate intents (writes first)
                                                          |-> publish output events

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
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4, uuid5

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

    1. Delegates intents to IntentExecutor (writes first)
    2. Publishes output events to the configured output topic
    3. Records dispatch metrics for observability

    Partition Key Extraction:
        When publishing output events, the applier extracts a partition key
        from the event payload to ensure per-entity ordering in Kafka. The
        key is resolved from the first available field in precedence order:
        ``entity_id > node_id > session_id > correlation_id``. If no key
        field is found, the event is published without a key (round-robin).

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
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the dispatch result applier.

        Args:
            event_bus: Event bus for publishing output events.
            output_topic: Topic to publish output events to.
            intent_executor: Optional intent executor for delegating intents
                to effect layer handlers. When provided, intents from dispatch
                results are forwarded to the executor for effect layer processing.
            clock: Optional callable returning current UTC datetime. Defaults to
                ``datetime.now(UTC)``. Inject for deterministic replay/testing.
        """
        self._event_bus = event_bus
        self._output_topic = output_topic
        self._intent_executor = intent_executor
        self._clock = clock or (lambda: datetime.now(UTC))

    def _resolve_partition_key(self, event: BaseModel) -> bytes | None:
        """Extract partition key from event model for per-entity ordering.

        Scans the event model for well-known identity fields and returns the
        first non-None value encoded as UTF-8 bytes. This key is intended for
        Kafka partition assignment so that all events for the same entity land
        on the same partition, preserving per-entity ordering.

        Precedence: ``entity_id > node_id > session_id > correlation_id``.

        Returns ``None`` (round-robin) if no key field is found on the event.

        Args:
            event: The output event payload (a Pydantic BaseModel).

        Returns:
            UTF-8 encoded partition key bytes, or ``None`` if no identity
            field is present on the event model.
        """
        for attr in ("entity_id", "node_id", "session_id", "correlation_id"):
            value = getattr(event, attr, None)
            if value is not None:
                return str(value).encode("utf-8")
        return None

    async def apply(
        self,
        result: ModelDispatchResult,
        correlation_id: UUID | None = None,
    ) -> None:
        """Process a dispatch result: execute intents then publish output events.

        Ordering Contract:
            Intents (writes) execute BEFORE output events are published. This
            ensures read models are consistent before downstream consumers can
            observe the events. Matches the original handler ordering where
            projection persistence preceded event emission.

        At-Least-Once Semantics:
            Output events are published sequentially. If event N fails, events
            1..N-1 are already published with no compensation. The exception
            propagates to the caller, preventing Kafka offset commit. On
            redelivery, events 1..N-1 will be published again as duplicates.
            Downstream consumers must be idempotent.

        Args:
            result: The dispatch result from the dispatch engine.
            correlation_id: Optional correlation ID for tracing.
        """
        effective_correlation_id = correlation_id or result.correlation_id
        if effective_correlation_id is None:
            effective_correlation_id = uuid4()
            logger.warning(
                "No correlation_id available — generated uuid4() fallback. "
                "Deterministic envelope_id deduplication will not work for "
                "this dispatch result (dispatcher_id=%s).",
                result.dispatcher_id,
            )

        if result.status != EnumDispatchStatus.SUCCESS:
            logger.debug(
                "Skipping result apply for non-success status=%s "
                "dispatcher_id=%s correlation_id=%s",
                result.status.value if result.status else "unknown",
                result.dispatcher_id,
                str(effective_correlation_id),
            )
            return

        # Phase 1: Execute intents (writes) BEFORE publishing output events.
        # This ensures read models (PostgreSQL projections, Consul registrations)
        # are consistent before downstream consumers can observe the events.
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
                logger.info(
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

        # Phase 2: Publish output events AFTER intents have committed.
        if result.output_events:
            for idx, output_event in enumerate(result.output_events):
                try:
                    # Deterministic envelope_id: uuid5(correlation_id, "type:index")
                    # ensures redeliveries produce identical IDs, enabling
                    # downstream consumers to deduplicate at-least-once events.
                    deterministic_id = uuid5(
                        effective_correlation_id,
                        f"{type(output_event).__name__}:{idx}",
                    )
                    output_envelope: ModelEventEnvelope[BaseModel] = ModelEventEnvelope(
                        envelope_id=deterministic_id,
                        payload=output_event,
                        correlation_id=effective_correlation_id,
                        envelope_timestamp=self._clock(),
                    )

                    # Extract partition key for per-entity ordering.
                    partition_key = self._resolve_partition_key(output_event)
                    if partition_key is not None:
                        logger.debug(
                            "Resolved partition key for output event "
                            "(type=%s, key=%s, correlation_id=%s)",
                            type(output_event).__name__,
                            partition_key.decode("utf-8"),
                            str(effective_correlation_id),
                        )

                    await self._event_bus.publish_envelope(
                        envelope=output_envelope,
                        topic=self._output_topic,
                        key=partition_key,
                    )

                    logger.info(
                        "Published output event to %s (correlation_id=%s)",
                        self._output_topic,
                        str(effective_correlation_id),
                        extra={
                            "output_event_type": type(output_event).__name__,
                            "envelope_id": str(output_envelope.envelope_id),
                            "dispatcher_id": result.dispatcher_id,
                            "partition_key": (
                                partition_key.decode("utf-8")
                                if partition_key is not None
                                else None
                            ),
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


__all__: list[str] = ["DispatchResultApplier"]
