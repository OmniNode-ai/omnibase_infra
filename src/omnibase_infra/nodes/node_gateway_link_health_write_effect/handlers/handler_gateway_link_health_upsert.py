# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerGatewayLinkHealthUpsert - UPSERT one link-health intent into gateway_link_health.

gateway_link_health is a latest-known-state projection: ON CONFLICT
(tenant_id) DO UPDATE keeps exactly one row per tenant edge, refreshed on
every heartbeat. The row is never deleted, so "absence of progress" (a
tenant edge that stopped heartbeating) shows up as a stale `last_seen_at` on
an existing row, evaluated live by the `gateway_link_health_status` view --
never as a missing row. RETURNING (xmax = 0) detects insert-vs-update,
matching HandlerPrStateUpsert's idiom.

Composes with HandlerDb (PostgreSQL transport) for circuit-breaker
protection, error classification, and connection-pool management, mirroring
HandlerPrStateUpsert / HandlerLedgerAppend (OMN-14140's internally-composed
HandlerDb pattern).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.nodes.node_gateway_link_health_projection_compute.models import (
    ModelPayloadGatewayLinkHealthUpsert,
)
from omnibase_infra.nodes.node_gateway_link_health_write_effect.models import (
    ModelGatewayLinkHealthUpsertResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)

HANDLER_ID_GATEWAY_LINK_HEALTH_UPSERT: str = "gateway-link-health-upsert-handler"

# UPSERT keyed on tenant_id. RETURNING (xmax = 0) distinguishes an INSERT
# (was_insert=True) from an UPDATE via the ON CONFLICT branch -- same idiom
# as HandlerPrStateUpsert's _SQL_UPSERT.
_SQL_UPSERT = """
INSERT INTO public.gateway_link_health (
    tenant_id,
    principal_id,
    local_transport_flavor,
    last_seen_at,
    lag_messages,
    lag_seconds,
    updated_at
) VALUES (
    $1, $2, $3, $4, $5, $6, NOW()
)
ON CONFLICT (tenant_id) DO UPDATE SET
    principal_id            = EXCLUDED.principal_id,
    local_transport_flavor  = EXCLUDED.local_transport_flavor,
    last_seen_at             = EXCLUDED.last_seen_at,
    lag_messages             = EXCLUDED.lag_messages,
    lag_seconds              = EXCLUDED.lag_seconds,
    updated_at                = NOW()
RETURNING (xmax = 0) AS was_insert
"""


class HandlerGatewayLinkHealthUpsert:
    """EFFECT handler that UPSERTs one row into public.gateway_link_health."""

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        db_dsn: str | None = None,
    ) -> None:
        """Initialize the gateway_link_health upsert handler.

        Args:
            container: ONEX dependency injection container. HandlerDb is
                composed internally from this container (matching
                OMN-14140's pattern). Defaults to ``None`` so the generic
                auto-wiring resolver's zero-required-param fast path can
                construct this handler directly.
            db_dsn: Optional PostgreSQL DSN supplied by the runtime
                auto-wiring boundary. Handlers do not read environment
                directly; runtime composition owns that IO boundary.
        """
        self._container = container
        self._db_handler = HandlerDb(container) if container is not None else None
        self._db_dsn = db_dsn.strip() if db_dsn else ""
        self._initialized: bool = False
        self._db_init_lock = asyncio.Lock()

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        config_dsn = config.get("dsn")
        if isinstance(config_dsn, str) and config_dsn.strip():
            self._db_dsn = config_dsn.strip()
        await self._ensure_db_ready()
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        if self._initialized and self._db_handler is not None:
            await self._db_handler.shutdown()
        self._initialized = False
        logger.info("HandlerGatewayLinkHealthUpsert shutdown complete")

    async def _ensure_db_ready(self) -> None:
        """Lazily connect the composed HandlerDb on first real use.

        The auto-wiring resolver constructs contract-routed handlers from
        `container` alone and never calls their `initialize()` method
        (OMN-14140), so this handler owns its HandlerDb connection lifecycle
        instead of relying on an external initialize() call.
        """
        if self._initialized:
            return
        async with self._db_init_lock:
            if self._initialized:
                return
            dsn = self._db_dsn
            if self._container is None or self._db_handler is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="gateway_link_health.upsert.connect",
                )
                raise RuntimeHostError(
                    "Missing ONEX container for gateway_link_health persistence "
                    "-- provide container at construction",
                    context=ctx,
                )
            if not dsn:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="gateway_link_health.upsert.connect",
                )
                raise RuntimeHostError(
                    "Missing PostgreSQL DSN for gateway_link_health persistence "
                    "-- provide db_dsn at construction or initialize({'dsn': ...})",
                    context=ctx,
                )
            await self._db_handler.initialize({"dsn": dsn})
            self._initialized = True

    async def upsert(
        self,
        payload: ModelPayloadGatewayLinkHealthUpsert,
        *,
        correlation_id: UUID | None = None,
    ) -> ModelGatewayLinkHealthUpsertResult:
        """UPSERT one row into gateway_link_health."""
        correlation_id = correlation_id or uuid4()

        await self._ensure_db_ready()

        parameters: list[object] = [
            payload.tenant_id,  # $1
            payload.principal_id,  # $2
            payload.local_transport_flavor,  # $3
            payload.last_seen_at,  # $4
            payload.lag_messages,  # $5
            payload.lag_seconds,  # $6
        ]

        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {
                "sql": _SQL_UPSERT,
                "parameters": parameters,
            },
            "correlation_id": str(correlation_id),
        }

        logger.debug(
            "Upserting gateway_link_health row",
            extra={
                "tenant_id": payload.tenant_id,
                "correlation_id": str(correlation_id),
            },
        )

        if self._db_handler is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="gateway_link_health.upsert",
            )
            raise RuntimeHostError("Database handler is not available", context=ctx)

        db_result = await self._db_handler.execute(envelope)

        if db_result.result is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="gateway_link_health.upsert",
            )
            raise RuntimeHostError("Database operation returned no result", context=ctx)

        rows = db_result.result.payload.rows
        if not rows:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="gateway_link_health.upsert",
            )
            raise RuntimeHostError(
                "UPSERT into gateway_link_health returned no row", context=ctx
            )

        was_insert = bool(rows[0]["was_insert"])
        return ModelGatewayLinkHealthUpsertResult(
            success=True,
            tenant_id=payload.tenant_id,
            was_insert=was_insert,
        )

    async def handle(
        self,
        payload: object,
    ) -> ModelHandlerOutput[ModelGatewayLinkHealthUpsertResult]:
        """Contract-routed operation_match entry point.

        Named ``payload`` (an ONEX canonical-shape magic parameter name, per
        ``canonical_handler_shape.py``'s ``MAGIC_PARAM_NAMES``) rather than
        typed directly as ``ModelPayloadGatewayLinkHealthUpsert``: this
        handler is invoked two ways -- the contract-declared
        ``operation_match`` dispatch path (a raw dict/object envelope) and
        ``IntentEffectDispatchBridge`` (OMN-14516), which always calls
        ``handle({"payload": ..., "correlation_id": ...})`` regardless of
        this parameter's static type, so an ``object`` annotation reflects
        the real invocation contract honestly rather than asserting a type
        the runtime does not actually guarantee.
        """
        payload_raw = self._extract_envelope_field(payload, "payload")
        if payload_raw is None:
            payload_raw = payload
        typed_payload = (
            payload_raw
            if isinstance(payload_raw, ModelPayloadGatewayLinkHealthUpsert)
            else ModelPayloadGatewayLinkHealthUpsert.model_validate(payload_raw)
        )

        envelope_correlation_id = self._extract_envelope_field(
            payload, "correlation_id"
        )
        correlation_id = self._safe_correlation_id(envelope_correlation_id)
        result = await self.upsert(typed_payload, correlation_id=correlation_id)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_GATEWAY_LINK_HEALTH_UPSERT,
            result=result,
        )

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelGatewayLinkHealthUpsertResult]:
        """ProtocolHandler entry point."""
        correlation_id = self._safe_correlation_id(envelope.get("correlation_id"))
        input_envelope_id = uuid4()

        payload_raw = envelope.get("payload")
        if not isinstance(payload_raw, dict):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="gateway_link_health.upsert",
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        payload = ModelPayloadGatewayLinkHealthUpsert.model_validate(payload_raw)
        result = await self.upsert(payload, correlation_id=correlation_id)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_GATEWAY_LINK_HEALTH_UPSERT,
            result=result,
        )

    @staticmethod
    def _extract_envelope_field(envelope: object, key: str) -> object:
        if isinstance(envelope, dict):
            return envelope.get(key)
        return getattr(envelope, key, None)

    @staticmethod
    def _safe_correlation_id(raw: object) -> UUID:
        """Parse a correlation ID from envelope-supplied raw input.

        Returns a fresh UUID if `raw` is missing, empty, or unparseable --
        gateway_link_health projection is best-effort read-model refresh and
        must never drop a refresh over a malformed correlation_id.
        """
        if not raw:
            return uuid4()
        try:
            return UUID(str(raw))
        except (ValueError, TypeError):
            return uuid4()


__all__ = [
    "HandlerGatewayLinkHealthUpsert",
    "HANDLER_ID_GATEWAY_LINK_HEALTH_UPSERT",
]
