# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent effect adapter for Consul service deregistration.

This module provides the IntentEffectConsulDeregister adapter, which bridges
ModelPayloadConsulDeregister intent payloads to actual Consul deregistration
operations via the HandlerConsul mixin's _deregister_service method.

Architecture:
    Reducer
        -> ModelPayloadConsulDeregister (intent payload)
        -> IntentExecutor
        -> IntentEffectConsulDeregister.execute()
        -> HandlerConsul._deregister_service() (Consul API)

Related:
    - OMN-2115: Bus audit layer 1 - generic bus health diagnostics
    - ModelPayloadConsulDeregister: Intent payload model
    - IntentEffectConsulRegister: Register counterpart
    - HandlerConsul: Consul handler with _deregister_service()

.. versionadded:: 0.8.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.nodes.reducers.models.model_payload_consul_deregister import (
    ModelPayloadConsulDeregister,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.handlers import HandlerConsul

logger = logging.getLogger(__name__)


class IntentEffectConsulDeregister:
    """Intent effect adapter for Consul service deregistration.

    Bridges ModelPayloadConsulDeregister intent payloads to HandlerConsul
    operations. The adapter extracts service_id from the payload and delegates
    to the consul handler for deregistration.

    Thread Safety:
        This class is designed for single-threaded async use. The underlying
        HandlerConsul manages its own thread pool for synchronous Consul
        API calls.

    Attributes:
        _consul_handler: HandlerConsul for Consul service deregistration.

    Example:
        ```python
        effect = IntentEffectConsulDeregister(consul_handler=consul_handler)
        await effect.execute(payload, correlation_id=correlation_id)
        ```

    .. versionadded:: 0.8.0
    """

    def __init__(self, consul_handler: HandlerConsul) -> None:
        """Initialize the Consul deregister intent effect.

        Args:
            consul_handler: HandlerConsul for Consul service deregistration.
                Must be fully initialized with a valid Consul client.
        """
        self._consul_handler = consul_handler

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        """Execute a Consul service deregistration from an intent payload.

        Extracts service_id from the payload and delegates to
        HandlerConsul via the handler's execute() method with
        operation="consul.deregister".

        Args:
            payload: The ModelPayloadConsulDeregister intent payload.
                Validated via isinstance at entry.
            correlation_id: Optional correlation ID for tracing.
                Falls back to payload.correlation_id if not provided.

        Raises:
            RuntimeHostError: If the Consul deregistration fails.
        """
        effective_correlation_id = (
            correlation_id or getattr(payload, "correlation_id", None) or uuid4()
        )

        if not isinstance(payload, ModelPayloadConsulDeregister):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="intent_effect_consul_deregister",
            )
            raise RuntimeHostError(
                f"Expected ModelPayloadConsulDeregister, got {type(payload).__name__}",
                context=context,
            )

        try:
            envelope: dict[str, object] = {
                "operation": "consul.deregister",
                "payload": {"service_id": payload.service_id},
                "correlation_id": effective_correlation_id,
                "envelope_id": str(uuid4()),
            }
            await self._consul_handler.execute(envelope)

            logger.info(
                "Consul deregistration executed: service_id=%s correlation_id=%s",
                payload.service_id,
                str(effective_correlation_id),
            )

        except RuntimeHostError:
            raise
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="intent_effect_consul_deregister",
            )
            logger.warning(
                "Consul deregistration intent failed: error=%s correlation_id=%s",
                sanitize_error_message(e),
                str(effective_correlation_id),
                extra={
                    "error_type": type(e).__name__,
                    "service_id": payload.service_id,
                },
            )
            raise RuntimeHostError(
                "Failed to execute Consul deregistration intent",
                context=context,
            ) from e


__all__: list[str] = ["IntentEffectConsulDeregister"]
