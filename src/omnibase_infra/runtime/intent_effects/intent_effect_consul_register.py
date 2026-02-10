# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent effect adapter for Consul service registration.

This module provides the IntentEffectConsulRegister adapter, which bridges
ModelPayloadConsulRegister intent payloads to actual Consul registration
operations via the HandlerConsul mixin's _register_service method.

Architecture:
    HandlerNodeIntrospected
        -> ModelPayloadConsulRegister (intent payload)
        -> IntentExecutor
        -> IntentEffectConsulRegister.execute()
        -> HandlerConsul._register_service() (Consul API)

    The adapter extracts service_id, service_name, tags, and health_check
    from the intent payload and delegates to the HandlerConsul for actual
    Consul agent registration.

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - ModelPayloadConsulRegister: Intent payload model
    - HandlerConsul: Consul handler with _register_service()
    - MixinConsulService: Mixin providing _register_service implementation

.. versionadded:: 0.7.0
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
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.handlers import HandlerConsul

logger = logging.getLogger(__name__)


class IntentEffectConsulRegister:
    """Intent effect adapter for Consul service registration.

    Bridges ModelPayloadConsulRegister intent payloads to HandlerConsul
    operations. The adapter extracts service registration fields from the
    payload and delegates to the consul handler.

    Thread Safety:
        This class is designed for single-threaded async use. The underlying
        HandlerConsul manages its own thread pool for synchronous Consul
        API calls.

    Attributes:
        _consul_handler: HandlerConsul for Consul service registration.

    Example:
        ```python
        effect = IntentEffectConsulRegister(consul_handler=consul_handler)
        await effect.execute(payload, correlation_id=correlation_id)
        ```

    .. versionadded:: 0.7.0
    """

    def __init__(self, consul_handler: HandlerConsul) -> None:
        """Initialize the Consul register intent effect.

        Args:
            consul_handler: HandlerConsul for Consul service registration.
                Must be fully initialized with a valid Consul client.
        """
        self._consul_handler = consul_handler

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        """Execute a Consul service registration from an intent payload.

        Extracts service registration fields from the payload and delegates
        to HandlerConsul._register_service() via the mixin's handle() method.

        The payload fields are converted to the dict format expected by
        the HandlerConsul mixin's _register_service method.

        Args:
            payload: The ModelPayloadConsulRegister intent payload.
                Validated via isinstance at entry.
            correlation_id: Optional correlation ID for tracing.
                Falls back to payload.correlation_id if not provided.

        Raises:
            RuntimeHostError: If the Consul registration fails.
        """
        # Compute effective correlation_id before type checks so error contexts
        # always carry a non-None ID, preserving any ID from the payload when
        # available and falling back to uuid4() only as a last resort.
        effective_correlation_id = (
            correlation_id or getattr(payload, "correlation_id", None) or uuid4()
        )

        if not isinstance(payload, ModelPayloadConsulRegister):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="intent_effect_consul_register",
            )
            raise RuntimeHostError(
                f"Expected ModelPayloadConsulRegister, got {type(payload).__name__}",
                context=context,
            )

        try:
            # Build the payload dict expected by HandlerConsul._register_service
            register_payload: dict[str, object] = {
                "name": payload.service_name,
                "service_id": payload.service_id,
                "tags": payload.tags,
            }

            if payload.address is not None:
                register_payload["address"] = payload.address
            if payload.port is not None:
                register_payload["port"] = payload.port

            if payload.health_check is not None:
                register_payload["check"] = payload.health_check

            # Store event_bus_config if present (for topic routing lookups)
            if payload.event_bus_config is not None:
                register_payload["event_bus_config"] = (
                    payload.event_bus_config.model_dump()
                )

            # Delegate to HandlerConsul via its execute() method
            # The handler routes internally based on the "operation" key
            envelope: dict[str, object] = {
                "operation": "consul.register",
                "payload": register_payload,
                "correlation_id": effective_correlation_id,
                "envelope_id": str(uuid4()),
            }
            await self._consul_handler.execute(envelope)

            logger.info(
                "Consul registration executed: service_id=%s service_name=%s "
                "correlation_id=%s",
                payload.service_id,
                payload.service_name,
                str(effective_correlation_id),
            )

        except RuntimeHostError:
            raise
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="intent_effect_consul_register",
            )
            logger.warning(
                "Consul registration intent failed: error=%s correlation_id=%s",
                sanitize_error_message(e),
                str(effective_correlation_id),
                extra={
                    "error_type": type(e).__name__,
                    "service_id": payload.service_id,
                },
            )
            raise RuntimeHostError(
                "Failed to execute Consul registration intent",
                context=context,
            ) from e


__all__: list[str] = ["IntentEffectConsulRegister"]
