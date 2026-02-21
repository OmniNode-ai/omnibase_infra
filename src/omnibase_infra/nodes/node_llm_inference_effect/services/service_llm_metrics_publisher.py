# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Service that publishes LLM call metrics to the event bus after each inference.

This service wraps a handler (HandlerLlmOpenaiCompatible or HandlerLlmOllama)
and publishes ``ContractLlmCallMetrics`` to
``onex.evt.omniintelligence.llm-call-completed.v1`` after every successful
inference call.

Architecture:
    - Wraps the inner handler: delegates ``handle()`` calls to it
    - Reads ``last_call_metrics`` after each call (fire-and-forget, never
      breaks inference on publish failure)
    - Publishes via ``ProtocolEventPublisher``; publish errors are logged
      at WARNING level and never propagated to callers
    - Handlers MUST NOT publish directly (ARCH-002); this service lives at
      the service layer where event publishing is permitted

Wiring:
    ``RegistryInfraLlmInferenceEffect.register_openai_compatible_with_metrics``
    and ``register_ollama_with_metrics`` create and wire this service.
    When no publisher is supplied, the plain handlers are used as-is (no
    metrics emission) so that local / test environments work without Kafka.

Related:
    - OMN-2443: Wire NodeLlmInferenceEffect to emit llm-call-completed events
    - OMN-2238: Token usage normalization
    - OMN-2235: LLM cost tracking contracts (SPI layer)
    - TOPIC_LLM_CALL_COMPLETED: ``onex.evt.omniintelligence.llm-call-completed.v1``
    - ARCH-002: No handler publishing (validator_no_handler_publishing.py)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.event_bus.topic_constants import TOPIC_LLM_CALL_COMPLETED

if TYPE_CHECKING:
    from omnibase_infra.nodes.effects.models.model_llm_inference_response import (
        ModelLlmInferenceResponse,
    )
    from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
        HandlerLlmOpenaiCompatible,
    )
    from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )

logger = logging.getLogger(__name__)


class ServiceLlmMetricsPublisher:
    """Wraps an LLM handler and publishes call metrics after each inference.

    This service delegates inference calls to the inner handler and then
    publishes ``ContractLlmCallMetrics`` to the canonical LLM call
    completed topic.  Publish failures never break inference; they are
    logged at WARNING and silently dropped.

    This class is intentionally NOT named ``Handler*`` to avoid triggering
    the ARCH-002 AST validator, which prohibits event-bus access in handlers.
    The service layer (named ``Service*``) is explicitly allowed to publish.

    Attributes:
        _handler: The inner HandlerLlmOpenaiCompatible instance.
        _publisher: Callable that accepts ``(event_type, payload, correlation_id)``
            and returns an awaitable bool.  Typically an
            ``AdapterProtocolEventPublisherKafka.publish`` method.

    Example:
        >>> from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        >>> from omnibase_infra.event_bus.adapters import AdapterProtocolEventPublisherKafka
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> bus = EventBusKafka.default()
        >>> await bus.start()
        >>> adapter = AdapterProtocolEventPublisherKafka(
        ...     container=ModelONEXContainer(), bus=bus
        ... )
        >>> inner_handler = HandlerLlmOpenaiCompatible(transport)
        >>> service = ServiceLlmMetricsPublisher(
        ...     handler=inner_handler,
        ...     publisher=adapter.publish,
        ... )
        >>> response = await service.handle(request)
    """

    def __init__(
        self,
        handler: HandlerLlmOpenaiCompatible,
        publisher: object,
    ) -> None:
        """Initialise with inner handler and publisher callable.

        Args:
            handler: The wrapped LLM inference handler.  Must expose a
                ``last_call_metrics`` attribute (``ContractLlmCallMetrics | None``)
                that is populated after each ``handle()`` call.
            publisher: An async callable with the signature::

                    async def publish(
                        event_type: str,
                        payload: JsonType,
                        correlation_id: str | None = None,
                    ) -> bool: ...

                Typically ``AdapterProtocolEventPublisherKafka.publish`` or an
                equivalent in-memory stub.
        """
        self._handler = handler
        self._publisher = publisher

    async def handle(
        self,
        request: ModelLlmInferenceRequest,
        correlation_id: UUID | None = None,
    ) -> ModelLlmInferenceResponse:
        """Delegate inference to inner handler and emit metrics on completion.

        Always returns the response from the inner handler.  Metrics
        emission is fire-and-forget: if publishing fails for any reason
        the exception is caught, logged, and the response is still returned.

        Args:
            request: LLM inference request.
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            ``ModelLlmInferenceResponse`` from the inner handler.

        Raises:
            Any exception raised by the inner handler is propagated unchanged.
        """
        if correlation_id is None:
            correlation_id = uuid4()

        response = await self._handler.handle(request, correlation_id=correlation_id)

        # Emit metrics -- fire-and-forget: must never break the inference result.
        await self._emit_metrics(correlation_id)

        return response

    async def _emit_metrics(self, correlation_id: UUID) -> None:
        """Read last_call_metrics from the handler and publish to Kafka.

        Safe wrapper around the publish path.  All exceptions are caught
        and logged so that a Kafka outage never impacts inference callers.

        Args:
            correlation_id: Correlation ID for the publish call.
        """
        metrics = self._handler.last_call_metrics
        if metrics is None:
            logger.debug(
                "No LLM call metrics to publish (last_call_metrics is None). "
                "correlation_id=%s",
                correlation_id,
            )
            return

        try:
            payload = json.loads(metrics.model_dump_json())
            await self._publisher(  # type: ignore[operator]
                TOPIC_LLM_CALL_COMPLETED,
                payload,
                str(correlation_id),
            )
            logger.debug(
                "Published LLM call metrics. topic=%s model=%s "
                "prompt_tokens=%d completion_tokens=%d correlation_id=%s",
                TOPIC_LLM_CALL_COMPLETED,
                metrics.model_id,
                metrics.prompt_tokens,
                metrics.completion_tokens,
                correlation_id,
            )
        except Exception:
            logger.warning(
                "Failed to publish LLM call metrics to topic=%s; ignoring. "
                "model=%s correlation_id=%s",
                TOPIC_LLM_CALL_COMPLETED,
                getattr(metrics, "model_id", "<unknown>"),
                correlation_id,
                exc_info=True,
            )


__all__: list[str] = ["ServiceLlmMetricsPublisher"]
