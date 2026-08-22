# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerGatewayLinkHealthProjection - heartbeat event -> ModelIntent transformer.

Mirrors the canonical HandlerPrStateProjection / HandlerBuildLoopProjection
shape. Receives a Kafka ModelEventMessage carrying
``onex.evt.omnibase-infra.gateway-heartbeat.v1`` (``ModelGatewayHeartbeat``,
published by ``node_bus_forwarder_effect``), extracts the tenant edge
identity and freshness stamp, and emits a ModelIntent with a
ModelPayloadGatewayLinkHealthUpsert payload for
NodeGatewayLinkHealthWriteEffect to persist.

Ticket: OMN-15570 (G3, gateway lift Phase 0)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.types import JsonType
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_gateway_link_health_projection_compute.models.model_payload_gateway_link_health_upsert import (
    ModelPayloadGatewayLinkHealthUpsert,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)

HANDLER_ID_GATEWAY_LINK_HEALTH_PROJECTION: str = (
    "gateway-link-health-projection-handler"
)


def _require_str(
    body: dict[str, JsonType],
    key: str,
    correlation_id: UUID | None,
) -> str:
    """Return a required non-empty string field, or raise RuntimeHostError."""
    value = body.get(key)
    if isinstance(value, str) and value:
        return value
    context = ModelInfraErrorContext.with_correlation(
        correlation_id=correlation_id,
        transport_type=EnumInfraTransportType.KAFKA,
        operation="gateway_link_health_projection.extract_payload",
    )
    raise RuntimeHostError(
        f"Gateway heartbeat event missing required field '{key}'",
        error_code=EnumCoreErrorCode.INVALID_INPUT,
        context=context,
    )


def _require_non_negative_int(
    body: dict[str, JsonType],
    key: str,
    correlation_id: UUID | None,
) -> int:
    """Return a required non-negative int field, or raise RuntimeHostError.

    ``bool`` is rejected explicitly: it is an ``int`` subclass in Python, so
    a producer sending ``true`` would otherwise be silently recorded as 1.
    """
    value = body.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    context = ModelInfraErrorContext.with_correlation(
        correlation_id=correlation_id,
        transport_type=EnumInfraTransportType.KAFKA,
        operation="gateway_link_health_projection.extract_payload",
    )
    raise RuntimeHostError(
        f"Gateway heartbeat event missing required non-negative int field '{key}'",
        error_code=EnumCoreErrorCode.INVALID_INPUT,
        context=context,
    )


class HandlerGatewayLinkHealthProjection:
    """COMPUTE handler that projects gateway heartbeat events into write intents."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def initialize(self, config: dict[str, object]) -> None:
        self._initialized = True
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        self._initialized = False
        logger.info("HandlerGatewayLinkHealthProjection shutdown complete")

    def project(self, message: ModelEventMessage) -> ModelIntent:
        """Transform a gateway heartbeat event message into an upsert intent."""
        payload = self._extract_payload(message)
        return ModelIntent(
            intent_type=payload.intent_type,
            target=f"postgres://gateway_link_health/{payload.tenant_id}",
            payload=payload,
        )

    async def handle(
        self,
        message: ModelEventMessage,
    ) -> ModelHandlerOutput[ModelIntent]:
        """Canonical def-B dispatch entrypoint.

        The shared runtime adapter validates the wire payload into the
        contract-declared ``ModelEventMessage`` input model and hands it here
        -- the envelope boundary lives in the runtime adapter, not in this
        core (definition B, OMN-14355). This projects the gateway heartbeat
        event into a ``gateway_link_health.upsert`` intent for
        ``NodeGatewayLinkHealthWriteEffect`` to persist.
        """
        raw_message = self._coerce_event_message(message)
        intent = self.project(raw_message)
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=raw_message.headers.correlation_id,
            handler_id=HANDLER_ID_GATEWAY_LINK_HEALTH_PROJECTION,
            result=intent,
        )

    def _extract_payload(
        self, message: ModelEventMessage
    ) -> ModelPayloadGatewayLinkHealthUpsert:
        """Extract heartbeat fields into a ModelPayloadGatewayLinkHealthUpsert.

        Required: tenant_id, principal_id, local_transport_flavor, a
        parseable emitted_at, status, and consecutive_failures -- a heartbeat
        without an identifiable tenant edge is a producer bug, not a
        partial-data case.

        status and consecutive_failures are required rather than defaulted
        (OMN-15742/G2 put both on every ModelGatewayHeartbeat). Defaulting a
        missing status to "active" would let a malformed or non-canonical
        producer be scored HEALTHY, which is the one failure mode this
        projection must not have: silence is already visible as staleness,
        but a wrongly-healthy verdict is invisible. Raising instead surfaces
        the producer bug.

        lag_messages / lag_seconds are still always absent from
        ModelGatewayHeartbeat (see the payload module docstring) and default
        to None.
        """
        headers = message.headers
        header_correlation_id = headers.correlation_id if headers else None

        try:
            decoded = json.loads(message.value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=header_correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="gateway_link_health_projection.extract_payload",
            )
            raise RuntimeHostError(
                f"Cannot decode gateway heartbeat event body as JSON: {type(e).__name__}",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            ) from e

        if not isinstance(decoded, dict):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=header_correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="gateway_link_health_projection.extract_payload",
            )
            raise RuntimeHostError(
                "Gateway heartbeat event body must decode to a JSON object, "
                f"got {type(decoded).__name__}.",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            )

        body: dict[str, JsonType] = (
            decoded["payload"] if isinstance(decoded.get("payload"), dict) else decoded
        )

        tenant_id = _require_str(body, "tenant_id", header_correlation_id)
        principal_id = _require_str(body, "principal_id", header_correlation_id)
        local_transport_flavor = _require_str(
            body, "local_transport_flavor", header_correlation_id
        )
        last_seen_at = self._extract_timestamp(body)
        reported_status = _require_str(body, "status", header_correlation_id)
        consecutive_failures = _require_non_negative_int(
            body, "consecutive_failures", header_correlation_id
        )
        lag_messages_raw = body.get("lag_messages")
        lag_messages = (
            lag_messages_raw
            if isinstance(lag_messages_raw, int)
            and not isinstance(lag_messages_raw, bool)
            else None
        )
        lag_seconds_raw = body.get("lag_seconds")
        lag_seconds = (
            float(lag_seconds_raw)
            if isinstance(lag_seconds_raw, (int, float))
            and not isinstance(lag_seconds_raw, bool)
            else None
        )

        return ModelPayloadGatewayLinkHealthUpsert(
            tenant_id=tenant_id,
            principal_id=principal_id,
            local_transport_flavor=local_transport_flavor,
            last_seen_at=last_seen_at,
            reported_status=reported_status,
            consecutive_failures=consecutive_failures,
            lag_messages=lag_messages,
            lag_seconds=lag_seconds,
        )

    @staticmethod
    def _coerce_event_message(raw: object) -> ModelEventMessage:
        """Accept direct ModelEventMessage or an auto-wired envelope wrapper."""
        if isinstance(raw, ModelEventMessage):
            return raw
        payload = getattr(raw, "payload", raw)
        if isinstance(raw, dict):
            payload = raw.get("payload", raw)
        return ModelEventMessage.model_validate(payload)

    @staticmethod
    def _extract_timestamp(body: dict[str, JsonType]) -> datetime:
        for k in ("emitted_at", "as_of", "timestamp", "occurred_at"):
            raw = body.get(k)
            if isinstance(raw, str) and raw:
                try:
                    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    continue
                return (
                    parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
                )
        return datetime.now(UTC)


__all__ = [
    "HandlerGatewayLinkHealthProjection",
    "HANDLER_ID_GATEWAY_LINK_HEALTH_PROJECTION",
]
