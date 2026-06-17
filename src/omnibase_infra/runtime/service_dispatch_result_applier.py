# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""# ai-slop-ok: pre-existingDispatch result applier for processing ModelDispatchResult outputs.

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
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus, EnumInfraTransportType
from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
    EnumOmnibaseInfraTopic,
)
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.errors.error_projection import ProjectionError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_core.models.projectors.model_projection_intent import (
        ModelProjectionIntent,
    )
    from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
    from omnibase_infra.protocols import ProtocolEventBusLike
    from omnibase_infra.runtime.protocol_projection_effect import (
        ProtocolProjectionEffect,
    )
    from omnibase_infra.runtime.service_intent_executor import IntentExecutor

logger = logging.getLogger(__name__)

# Delegation intent topics are resolved from the contract-sourced topic registry
# (ServiceTopicRegistry, OMN-5839) rather than importing the legacy TOPIC_*
# string constants from event_bus/topic_constants.py. Each resolved string equals
# the topic declared in the owning node contract.yaml (OMN-12803 / OMN-13191).
_TOPIC_REGISTRY = ServiceTopicRegistry.from_defaults()

_DELEGATION_INTENT_TOPIC_BY_CLASS: dict[str, str] = {
    "ModelBaselineIntent": _TOPIC_REGISTRY.resolve(
        topic_keys.DELEGATION_BASELINE_COMPARISON
    ),
    "ModelInferenceIntent": _TOPIC_REGISTRY.resolve(
        topic_keys.DELEGATION_INFERENCE_REQUEST
    ),
    "ModelInvocationCommand": EnumOmnibaseInfraTopic.CMD_REMOTE_AGENT_INVOKE_V1.value,
    "ModelQualityGateIntent": _TOPIC_REGISTRY.resolve(
        topic_keys.DELEGATION_QUALITY_GATE_REQUEST
    ),
    "ModelRoutingIntent": _TOPIC_REGISTRY.resolve(
        topic_keys.DELEGATION_ROUTING_REQUEST
    ),
}


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
            output_topic="onex.evt.platform.node-registration-result.v1",  # onex-topic-allow: pending contract auto-wiring
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
        topic_router: dict[str, str] | None = None,
        output_topic_map: dict[str, str] | None = None,
        output_event_handler: Callable[[BaseModel], Awaitable[BaseModel | None]]
        | None = None,
        allowed_output_topics: Iterable[str] | None = None,
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
            topic_router: Optional mapping of Python event class names to their
                declared Kafka topics (e.g. ``{"ModelNodeRegistrationAccepted":
                "onex.evt.platform.node-registration-accepted.v1"}``).  When  # onex-topic-allow: pending contract auto-wiring
                provided, each output event is published to its per-type topic
                instead of the single ``output_topic`` fallback.  Events whose
                class name is not in the map fall back to ``output_topic``.
                Build this map with ``build_topic_router_from_contract()``
                (OMN-4881).
            output_topic_map: Optional mapping of event_type names (from contract
                ``published_events``) to their declared Kafka topics. Uses short
                names (``Model`` prefix stripped) with full class name fallback.
                Build with ``load_published_events_map()`` (OMN-5132).
            output_event_handler: Optional async hook for output events that are
                executed in-process instead of being published. If it returns a
                non-None model, the original output event is considered handled.
            allowed_output_topics: Optional contract-derived allowlist of publish
                topics. This should include ``event_bus.publish_topics`` so
                per-instance embedded topics can select any declared terminal
                topic, even when multiple terminal outcomes share one event model.
        """
        self._event_bus = event_bus
        self._output_topic = output_topic
        self._intent_executor = intent_executor
        self._clock = clock or (lambda: datetime.now(UTC))
        self._projection_effect = projection_effect
        self._topic_router: dict[str, str] = topic_router or {}
        self._output_topic_map: dict[str, str] = output_topic_map or {}
        self._output_event_handler = output_event_handler
        self._allowed_output_topic_set: set[str] = {
            topic.strip()
            for topic in (allowed_output_topics or ())
            if isinstance(topic, str) and topic.strip()
        }

    @property
    def published_events_map(self) -> dict[str, str]:
        """Return the published_events topic routing map.

        Exposes the output_topic_map for health check introspection (OMN-5164).
        Returns the mapping of event_type names to their declared Kafka topics.
        """
        return self._output_topic_map

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

    def _publish_payload_for_output_event(self, event: BaseModel) -> BaseModel:
        """Return the payload that should be wrapped in the bus envelope.

        Some domain handlers return a topic-bearing domain envelope whose
        ``payload`` is the actual event payload. The runtime uses the outer model
        to resolve the topic, but Kafka consumers and projections expect the
        inner payload as ``ModelEventEnvelope.payload``.
        """
        if type(event).__name__ == "ModelDelegationEventEnvelope":
            inner_payload = getattr(event, "payload", None)
            if isinstance(inner_payload, BaseModel):
                return inner_payload
        return event

    def _resolve_output_topic(self, event: BaseModel) -> str:
        """Resolve the output topic for an event using the output_topic_map.

        Typed event models may carry an explicit ``topic`` field. When present,
        that is the most specific contract boundary. Otherwise this tries the
        short name (class name with ``Model`` prefix removed), then the full
        class name, and falls back to ``_output_topic``.

        Args:
            event: The output event payload (a Pydantic BaseModel).

        Returns:
            The resolved topic string.
        """
        embedded_topic = self._resolve_embedded_output_topic(event)
        if embedded_topic is not None:
            return embedded_topic
        return self._resolve_mapped_output_topic(event)

    def _resolve_embedded_output_topic(self, event: BaseModel) -> str | None:
        """Return a declared topic carried by the event payload, if present."""
        embedded_topic = getattr(event, "topic", None)
        if isinstance(embedded_topic, str) and embedded_topic.strip():
            candidate_topic = embedded_topic.strip()
            if candidate_topic in self._allowed_output_topics():
                return candidate_topic
            logger.warning(
                "Ignoring undeclared embedded output topic",
                extra={
                    "topic": candidate_topic,
                    "event_type": type(event).__name__,
                },
            )
        return None

    def _resolve_mapped_output_topic(self, event: BaseModel) -> str:
        """Resolve output topic from configured class maps or fallback topic."""
        class_name = type(event).__name__
        delegation_intent_topic = _DELEGATION_INTENT_TOPIC_BY_CLASS.get(class_name)
        if not self._output_topic_map:
            return delegation_intent_topic or self._output_topic
        short_name = class_name.removeprefix("Model")
        return self._output_topic_map.get(
            short_name,
            self._output_topic_map.get(
                class_name,
                delegation_intent_topic or self._output_topic,
            ),
        )

    def _allowed_output_topics(self) -> set[str]:
        """Return contract-derived topics the applier may publish to."""
        return {
            self._output_topic,
            *_DELEGATION_INTENT_TOPIC_BY_CLASS.values(),
            *self._output_topic_map.values(),
            *self._topic_router.values(),
            *self._allowed_output_topic_set,
        }

    @staticmethod
    def _derive_event_type_from_topic(topic: str) -> str | None:
        """Derive the event_type routing key from an ONEX topic name.

        ONEX topics follow the convention::

            onex.{kind}.{producer}.{event-name}.v{n}

        This method extracts ``{producer}.{event-name}`` as a dot-path routing
        key suitable for ``ModelEventEnvelope.event_type``, which is the alias
        format used by dispatcher registration (e.g.
        ``omnimarket.swarm-endpoint-health-completed``).

        Args:
            topic: Full topic name following ONEX naming convention
                (e.g., ``'onex.evt.omnimarket.swarm-endpoint-health-completed.v1'``).

        Returns:
            Derived event_type as ``'{producer}.{event-name}'``
            (e.g., ``'omnimarket.swarm-endpoint-health-completed'``), or ``None``
            if the topic does not follow the expected ONEX format.

        .. versionadded:: 0.41.0 (OMN-12116)
        """
        parts = topic.split(".")
        if len(parts) >= 5 and parts[0] == "onex":
            # onex.{kind}.{producer}.{event-name}.v{n}
            producer = parts[2]
            event_name = parts[3]
            return f"{producer}.{event_name}"
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
                    f"Projection write failed for {intent.projector_key}: {error_msg}",
                    context=context,
                    projection_type=intent.projector_key,
                )
            logger.info(
                "Projection persisted: projector_key=%s event_type=%s artifact_ref=%s "
                "dispatcher_id=%s correlation_id=%s",
                intent.projector_key,
                intent.event_type,
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
                f"Projection write raised exception for {intent.projector_key}: "
                f"{sanitize_error_message(exc)}",
                context=context,
                projection_type=intent.projector_key,
            ) from exc

    async def apply(
        self,
        result: ModelDispatchResult | None,
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

        None-result opt-out (OMN-12151):
            When ``result`` is ``None``, the applier exits immediately without
            publishing any terminal event or executing any intents.  This is the
            canonical opt-out for multi-step FSM orchestrators that publish
            sub-commands mid-flight and must not emit a terminal event until the
            FSM reaches its final state.  Handlers that drive their own event
            publication (e.g. ``handle_async`` flushes sub-commands directly)
            should return ``None`` for non-terminal FSM states.

        Args:
            result: The dispatch result from the dispatch engine, or ``None``
                to suppress terminal-event publication entirely.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            ProjectionError: If the projection effect raises.
            RuntimeHostError: If intent execution misconfiguration is detected.
            Exception: Re-raised from intent execution or Kafka publish failures.
        """
        if result is None:
            logger.debug(
                "DispatchResultApplier.apply received None result — "
                "suppressing terminal event publication (OMN-12151)"
            )
            return

        effective_correlation_id = correlation_id or result.correlation_id
        if effective_correlation_id is None:
            effective_correlation_id = uuid4()
            logger.warning(
                "No correlation_id available — generated uuid4() fallback. "
                "Deterministic envelope_id deduplication will not work for "
                "this dispatch result (dispatcher_id=%s).",
                result.dispatcher_id,
            )

        # Per-handler result application (OMN-12416): a multi-handler contract
        # aggregates every matched handler into one ModelDispatchResult, so a
        # single sibling handler's failure marks the aggregate status as
        # HANDLER_ERROR even when another handler on the same contract succeeded
        # and produced output. Gating purely on ``status == SUCCESS`` would then
        # drop the successful handler's output — letting one handler's outcome
        # suppress another's. Only HANDLER_ERROR represents that partial-success
        # shape; cancellation/timeouts and other non-success statuses must not
        # publish side effects even if they happen to carry output fields.
        has_applicable_output = bool(
            result.output_events or result.output_intents or result.projection_intents
        )
        is_partial_handler_failure = result.status == EnumDispatchStatus.HANDLER_ERROR
        if result.status != EnumDispatchStatus.SUCCESS and (
            not is_partial_handler_failure or not has_applicable_output
        ):
            logger.debug(
                "Skipping result apply for non-success status=%s "
                "dispatcher_id=%s correlation_id=%s",
                result.status.value if result.status else "unknown",
                result.dispatcher_id,
                str(effective_correlation_id),
            )
            return
        if is_partial_handler_failure and has_applicable_output:
            logger.info(
                "Applying partial-success dispatch output despite status=%s — "
                "a sibling handler failed but %d event(s)/%d intent(s)/%d "
                "projection(s) from succeeding handler(s) are published "
                "(dispatcher_id=%s correlation_id=%s)",
                result.status.value if result.status else "unknown",
                len(result.output_events),
                len(result.output_intents),
                len(result.projection_intents),
                result.dispatcher_id,
                str(effective_correlation_id),
            )

        # Phase 0: Execute projection synchronously (OMN-2510).
        # Projection MUST complete before any Kafka publish.  If the projection
        # effect raises, we re-raise immediately — no intents published, no
        # Kafka messages emitted.
        projection_intents: list[ModelProjectionIntent] = list(
            result.projection_intents
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
                        "projector_key=%s event_type=%s "
                        "dispatcher_id=%s correlation_id=%s",
                        proj_intent.projector_key,
                        proj_intent.event_type,
                        result.dispatcher_id,
                        str(effective_correlation_id),
                        extra={
                            "error_type": type(proj_err).__name__,
                            "projector_key": proj_intent.projector_key,
                            "event_type": proj_intent.event_type,
                            "dispatcher_id": result.dispatcher_id,
                        },
                    )
                    raise

        # Phase 1: Execute intents (writes) BEFORE publishing output events.
        # This ensures read models (PostgreSQL projections) are consistent
        # before downstream consumers can observe the events.
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
                    publish_payload = self._publish_payload_for_output_event(
                        output_event
                    )
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
                        payload=publish_payload,
                        correlation_id=effective_correlation_id,
                        envelope_timestamp=self._clock(),
                    )

                    # Extract partition key for per-entity ordering.
                    partition_key = self._resolve_partition_key(publish_payload)
                    if partition_key is not None:
                        logger.debug(
                            "Resolved partition key for output event "
                            "(type=%s, key=%s, correlation_id=%s)",
                            type(publish_payload).__name__,
                            partition_key.decode("utf-8"),
                            str(effective_correlation_id),
                        )

                    embedded_topic = self._resolve_embedded_output_topic(output_event)
                    resolved_topic = (
                        embedded_topic
                        if embedded_topic is not None
                        else self._topic_router.get(
                            type(output_event).__name__,
                            self._resolve_mapped_output_topic(output_event),
                        )
                    )

                    # OMN-12116: Derive event_type from the resolved topic so
                    # multi-step FSM orchestrators can match the dispatcher
                    # registration alias (e.g. 'omnimarket.swarm-endpoint-health-completed').
                    # Without this, the envelope arrives with event_type=None
                    # and the orchestrator's dispatcher — registered under the
                    # topic-derived alias — cannot find a matching handler,
                    # causing the response event to be routed to DLQ.
                    derived_event_type = self._derive_event_type_from_topic(
                        resolved_topic
                    )
                    if derived_event_type is not None:
                        output_envelope = output_envelope.model_copy(
                            update={"event_type": derived_event_type}
                        )

                    await self._event_bus.publish_envelope(
                        envelope=output_envelope,
                        topic=resolved_topic,
                        key=partition_key,
                    )

                    logger.info(
                        "Published output event to %s (correlation_id=%s)",
                        resolved_topic,
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
