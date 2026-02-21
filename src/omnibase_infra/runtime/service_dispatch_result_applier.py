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
                                                          |-> execute projection (OMN-2510)
                                                          |-> delegate intents (writes first)
                                                          |-> publish output events

    This separation keeps the dispatch engine pure (routing only) while the
    applier handles side effects (publishing, intent execution, projection).

Ordering guarantee (OMN-2363 / OMN-2510):
    reduce() -> NodeProjectionEffect.execute() -> intent execution -> Kafka publish

    Projection failure BLOCKS Kafka publish entirely.  If the projection write
    raises, the applier re-raises without touching the event bus, preventing
    offset commit.  The message is redelivered on the next consumer poll.

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - OMN-2363: Projection ordering guarantee epic
    - OMN-2510: Runtime wires projection before Kafka publish (this ticket)
    - EventBusSubcontractWiring: Creates subscriptions that feed the engine
    - MessageDispatchEngine: Routes messages to dispatchers
    - IntentExecutor: Executes intents from dispatch results

.. versionadded:: 0.7.0
.. versionchanged:: 0.9.0
    Added projection phase (OMN-2510): NodeProjectionEffect executes
    synchronously before Kafka publish to eliminate the projection/publish
    race condition.
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
from omnibase_infra.errors.error_projection import ProjectionError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
    from omnibase_infra.models.projection.model_projection_intent import (
        ModelProjectionIntent,
    )
    from omnibase_infra.protocols import ProtocolEventBusLike
    from omnibase_infra.runtime.protocol_projection_effect import (
        ProtocolProjectionEffect,
    )
    from omnibase_infra.runtime.service_intent_executor import IntentExecutor

logger = logging.getLogger(__name__)


class DispatchResultApplier:
    """Processes ModelDispatchResult: runs projection, publishes output events, delegates intents.

    This service is injected into the dispatch callback chain by
    EventBusSubcontractWiring. After the dispatch engine routes a message
    to a dispatcher and receives a ModelDispatchResult, this applier:

    1. Executes NodeProjectionEffect synchronously (OMN-2510 — writes first)
    2. Delegates remaining intents to IntentExecutor
    3. Publishes output events to the configured output topic

    Ordering guarantee (OMN-2363):
        Kafka publish is GATED on successful projection execution.  If
        NodeProjectionEffect.execute() raises, the applier re-raises without
        calling the event bus, preventing offset commit.

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
            to effect layer handlers.
        _projection_effect: Optional synchronous projection effect.  When
            provided, its ``execute()`` is called with each ModelProjectionIntent
            in the dispatch result before any Kafka publish occurs.

    Example:
        ```python
        applier = DispatchResultApplier(
            event_bus=event_bus,
            output_topic="onex.evt.platform.node-registration-result.v1",
            projection_effect=node_projection_effect,
        )
        await applier.apply(dispatch_result)
        ```

    .. versionadded:: 0.7.0
    .. versionchanged:: 0.9.0
        Added ``projection_effect`` parameter (OMN-2510).
    """

    def __init__(
        self,
        event_bus: ProtocolEventBusLike,
        output_topic: str,
        intent_executor: IntentExecutor | None = None,
        clock: Callable[[], datetime] | None = None,
        projection_effect: ProtocolProjectionEffect | None = None,
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
            projection_effect: Optional synchronous projection effect
                (OMN-2510).  When provided, its ``execute()`` is called with
                each ``ModelProjectionIntent`` in the dispatch result.
                Kafka publish is skipped if ``execute()`` raises.
        """
        self._event_bus = event_bus
        self._output_topic = output_topic
        self._intent_executor = intent_executor
        self._clock = clock or (lambda: datetime.now(UTC))
        self._projection_effect = projection_effect

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

    def _execute_projection(
        self,
        intent: ModelProjectionIntent,
        correlation_id: UUID,
        dispatcher_id: str | None,
    ) -> None:
        """Execute a single projection intent synchronously.

        Calls NodeProjectionEffect.execute() and blocks until the projection
        is persisted.  Raises ProjectionError (or re-raises the underlying
        exception) on failure, which prevents the caller from publishing to
        Kafka.

        Args:
            intent: The projection intent to execute.
            correlation_id: Correlation ID for log context and error tracking.
            dispatcher_id: Dispatcher identifier for logging context.

        Raises:
            ProjectionError: When the projection effect raises any exception.
                Wraps the original exception for structured logging.
        """
        assert self._projection_effect is not None  # caller guards this
        try:
            result = self._projection_effect.execute(intent)
            if not result.success:
                # Explicit failure result — treat like a raised exception.
                error_msg = (
                    result.error
                    or "projection returned success=False without error detail"
                )
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="dispatch_result_applier.execute_projection",
                )
                raise ProjectionError(
                    f"Projection write failed for {intent.projection_type}: {error_msg}",
                    context=context,
                    originating_event_id=intent.causation_event_id,
                    projection_type=intent.projection_type,
                )
            logger.info(
                "Projection persisted: type=%s aggregate_id=%s artifact_ref=%s "
                "dispatcher_id=%s correlation_id=%s",
                intent.projection_type,
                str(intent.aggregate_id),
                result.artifact_ref,
                dispatcher_id,
                str(correlation_id),
            )
        except ProjectionError:
            raise
        except Exception as exc:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="dispatch_result_applier.execute_projection",
            )
            raise ProjectionError(
                f"Projection write raised exception for {intent.projection_type}: "
                f"{sanitize_error_message(exc)}",
                context=context,
                originating_event_id=intent.causation_event_id,
                projection_type=intent.projection_type,
            ) from exc

    async def apply(
        self,
        result: ModelDispatchResult,
        correlation_id: UUID | None = None,
    ) -> None:
        """Process a dispatch result: project, execute intents, then publish output events.

        Ordering Contract (OMN-2363 / OMN-2510):
            1. NodeProjectionEffect.execute() — synchronous, blocks Kafka on failure
            2. IntentExecutor.execute_all() — remaining effects (writes first)
            3. EventBus.publish_envelope() — Kafka publish, only if 1 and 2 succeed

        At-Least-Once Semantics:
            Output events are published sequentially. If event N fails, events
            1..N-1 are already published with no compensation. The exception
            propagates to the caller, preventing Kafka offset commit. On
            redelivery, events 1..N-1 will be published again as duplicates.
            Downstream consumers must be idempotent.

        Projection Failure Semantics (OMN-2510):
            If the projection effect raises, the applier re-raises immediately
            without executing intents or publishing any Kafka messages.  Zero
            intents are published on projection failure — no partial state
            emission.

        Args:
            result: The dispatch result from the dispatch engine.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            ProjectionError: If the projection effect raises.
            RuntimeHostError: If intent execution misconfiguration is detected.
            Exception: Re-raised from intent execution or Kafka publish failures.
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

        # Phase 0: Execute projection synchronously (OMN-2510).
        # Projection MUST complete before any Kafka publish.  If the projection
        # effect raises, we re-raise immediately — no intents published, no
        # Kafka messages emitted.
        projection_intents: list[ModelProjectionIntent] = list(
            getattr(result, "projection_intents", None) or []
        )
        if projection_intents and self._projection_effect is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="dispatch_result_applier.execute_projection",
            )
            raise RuntimeHostError(
                f"Dispatch result contains {len(projection_intents)} projection intent(s) "
                f"but no ProtocolProjectionEffect is configured — projection would be "
                f"skipped and Kafka publish would race (dispatcher_id={result.dispatcher_id})",
                context=context,
            )
        if self._projection_effect is not None and projection_intents:
            for proj_intent in projection_intents:
                # _execute_projection raises ProjectionError on any failure.
                # We do NOT catch it here — the caller must see it to skip
                # Kafka offset commit.
                try:
                    self._execute_projection(
                        proj_intent,
                        correlation_id=effective_correlation_id,
                        dispatcher_id=result.dispatcher_id,
                    )
                except ProjectionError as proj_err:
                    logger.exception(
                        "Projection failed — Kafka publish blocked: "
                        "projection_type=%s aggregate_id=%s originating_event_id=%s "
                        "dispatcher_id=%s correlation_id=%s",
                        proj_intent.projection_type,
                        str(proj_intent.aggregate_id),
                        str(proj_intent.causation_event_id),
                        result.dispatcher_id,
                        str(effective_correlation_id),
                        extra={
                            "error_type": type(proj_err).__name__,
                            "projection_type": proj_intent.projection_type,
                            "aggregate_id": str(proj_intent.aggregate_id),
                            "dispatcher_id": result.dispatcher_id,
                        },
                    )
                    raise

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

        # Phase 2: Publish output events AFTER projection and intents have committed.
        if result.output_events:
            for idx, output_event in enumerate(result.output_events):
                try:
                    # Deterministic envelope_id: uuid5(correlation_id, "type:index")
                    # ensures redeliveries produce identical IDs, enabling
                    # downstream consumers to deduplicate at-least-once events.
                    # NOTE: If correlation_id is a uuid4 fallback (see above),
                    # each retry generates a new namespace, defeating deduplication.
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
