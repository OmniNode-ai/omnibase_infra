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

            if payload.node_id is not None:
                register_payload["node_id"] = payload.node_id

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
            handler_output = await self._consul_handler.execute(envelope)

            # Guard against None output before accessing result.
            if handler_output is None:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=effective_correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="intent_effect_consul_register",
                )
                raise RuntimeHostError(
                    "Consul handler returned None output",
                    context=context,
                )

            # Detect silent failures: handler may return an error-status result
            # instead of raising, so check the response status explicitly.
            #
            # Note: In practice, HandlerConsul._build_response always calls
            # ModelHandlerOutput.for_compute with a non-None ModelConsulHandlerResponse
            # result, so `consul_response is None` should never occur on the normal
            # success path. The None guard below is a defensive check against
            # hypothetical broken or future handler implementations that might
            # return None — it is NOT the standard EFFECT success path. Only a
            # non-None result with `is_error=True` indicates an error that was
            # returned in-band rather than raised.
            consul_response = handler_output.result
            # The two blocks below are mutually exclusive paths:
            #
            # (a) Missing attribute: the first block fires when `consul_response`
            #     is a non-None object that does not have an `is_error` attribute
            #     at all. This indicates a broken or unexpected handler
            #     implementation. The block raises immediately, so the second
            #     block is never reached in this case.
            #
            # (b) Attribute present but True: the second block fires when
            #     `consul_response` has `is_error` (confirmed by the first block
            #     not raising) and the value is True, meaning the handler
            #     returned an error status in-band rather than raising an
            #     exception. Because the first block raises on a missing
            #     attribute, the `hasattr` check in the second condition is
            #     implicitly guaranteed by reaching that line.
            if consul_response is None:
                # Abnormal: HandlerConsul._build_response always returns a
                # non-None result via for_compute, so None here signals a
                # broken or non-conformant handler implementation.
                logger.warning(
                    "Consul handler returned None result for service_id=%s "
                    "(expected ModelConsulHandlerResponse); registration may "
                    "be incomplete. correlation_id=%s",
                    payload.service_id,
                    str(effective_correlation_id),
                )
            if consul_response is not None:
                # Defensive guard: consul_response is typed as `object` at the
                # handler output level. If the handler returns an unexpected type
                # that lacks `is_error`, accessing the attribute would raise an
                # AttributeError. Guard with hasattr first; if the attribute is
                # missing, raise a RuntimeHostError rather than letting an
                # AttributeError propagate with a misleading stack trace.
                if not hasattr(consul_response, "is_error"):
                    context = ModelInfraErrorContext.with_correlation(
                        correlation_id=effective_correlation_id,
                        transport_type=EnumInfraTransportType.CONSUL,
                        operation="intent_effect_consul_register",
                    )
                    raise RuntimeHostError(
                        "Consul handler result missing is_error attribute",
                        context=context,
                    )
            if consul_response is not None and consul_response.is_error:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=effective_correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="intent_effect_consul_register",
                )
                raise RuntimeHostError(
                    f"Consul registration returned error status for "
                    f"service_id={payload.service_id}",
                    context=context,
                )

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
