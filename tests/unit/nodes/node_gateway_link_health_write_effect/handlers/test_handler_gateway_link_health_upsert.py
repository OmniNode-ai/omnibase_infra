# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerGatewayLinkHealthUpsert (OMN-15570, G3).

Tests validate:
- Constructor composes HandlerDb internally from `container` alone
  (OMN-14140 pattern).
- Lazy DB connection guard (missing DSN raises RuntimeHostError).
- Insert-vs-update detection via `RETURNING (xmax = 0) AS was_insert`.
- handle() drives the real dispatch-shaped entry point (dict-and-attribute
  envelope shapes).

Mirrors test_handler_pr_state_upsert.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.handlers.models import ModelDbQueryPayload, ModelDbQueryResponse
from omnibase_infra.nodes.node_gateway_link_health_projection_compute.models import (
    ModelPayloadGatewayLinkHealthUpsert,
)
from omnibase_infra.nodes.node_gateway_link_health_write_effect.handlers.handler_gateway_link_health_upsert import (
    HandlerGatewayLinkHealthUpsert,
)

pytestmark = [pytest.mark.unit]


def make_mock_container() -> MagicMock:
    return MagicMock(spec=ModelONEXContainer)


def make_handler_with_mock_db(
    initialized: bool = True,
) -> tuple[HandlerGatewayLinkHealthUpsert, AsyncMock]:
    container = make_mock_container()
    handler = HandlerGatewayLinkHealthUpsert(container)
    db_handler = AsyncMock()
    handler._db_handler = db_handler
    handler._initialized = initialized
    return handler, db_handler


def make_db_result(rows: list[dict[str, object]]) -> MagicMock:
    correlation_id = uuid4()
    payload = ModelDbQueryPayload(rows=rows, row_count=len(rows))
    response = ModelDbQueryResponse(
        status=EnumResponseStatus.SUCCESS,
        payload=payload,
        correlation_id=correlation_id,
    )
    result_wrapper = MagicMock()
    result_wrapper.result = response
    return result_wrapper


def make_minimal_payload(
    **overrides: object,
) -> ModelPayloadGatewayLinkHealthUpsert:
    defaults: dict[str, object] = {
        "tenant_id": "beta-gateway-canary-79afa7263852",
        "principal_id": "t-abc123",
        "local_transport_flavor": "containerized",
        "last_seen_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return ModelPayloadGatewayLinkHealthUpsert(**defaults)


class TestHandlerGatewayLinkHealthUpsertComposition:
    """HandlerGatewayLinkHealthUpsert composes HandlerDb from `container` alone."""

    def test_constructor_takes_container_and_optional_dsn(self) -> None:
        container = make_mock_container()

        handler = HandlerGatewayLinkHealthUpsert(container)

        assert isinstance(handler._db_handler, HandlerDb)
        assert handler._initialized is False


class TestHandlerGatewayLinkHealthUpsertInitialization:
    """Lazy HandlerDb connection lifecycle."""

    @pytest.mark.asyncio
    async def test_upsert_raises_when_dsn_not_configured(self) -> None:
        container = make_mock_container()
        handler = HandlerGatewayLinkHealthUpsert(container)

        payload = make_minimal_payload()
        with pytest.raises(RuntimeHostError, match="Missing PostgreSQL DSN"):
            await handler.upsert(payload)

    @pytest.mark.asyncio
    async def test_ensure_db_ready_is_idempotent(self) -> None:
        container = make_mock_container()
        handler = HandlerGatewayLinkHealthUpsert(
            container, db_dsn="postgresql://test-dsn"
        )
        handler._db_handler.initialize = AsyncMock()  # type: ignore[method-assign]

        await handler._ensure_db_ready()
        await handler._ensure_db_ready()

        handler._db_handler.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_sets_initialized_false(self) -> None:
        handler, db_handler = make_handler_with_mock_db(initialized=True)
        assert handler._initialized is True

        await handler.shutdown()

        assert handler._initialized is False
        db_handler.shutdown.assert_awaited_once()


class TestHandlerGatewayLinkHealthUpsertInsertVsUpdate:
    """Insert-vs-update detection via RETURNING (xmax = 0) AS was_insert."""

    @pytest.mark.asyncio
    async def test_first_seen_tenant_is_insert(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": True}])
        )

        payload = make_minimal_payload()
        result = await handler.upsert(payload)

        assert result.success is True
        assert result.was_insert is True
        assert result.tenant_id == payload.tenant_id

    @pytest.mark.asyncio
    async def test_refreshed_tenant_is_update(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": False}])
        )

        payload = make_minimal_payload()
        result = await handler.upsert(payload)

        assert result.success is True
        assert result.was_insert is False

    @pytest.mark.asyncio
    async def test_upsert_raises_when_no_row_returned(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        payload = make_minimal_payload()
        with pytest.raises(RuntimeHostError, match="returned no row"):
            await handler.upsert(payload)


class TestHandlerGatewayLinkHealthUpsertLagFields:
    """lag_messages/lag_seconds reach the UPSERT SQL parameters even though
    no real producer populates them today (forward-compat)."""

    @pytest.mark.asyncio
    async def test_upsert_sends_lag_fields_as_sql_parameters(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": True}])
        )
        payload = make_minimal_payload(lag_messages=12, lag_seconds=3.5)

        await handler.upsert(payload)

        db_handler.execute.assert_awaited_once()
        (envelope,), _ = db_handler.execute.call_args
        parameters = envelope["payload"]["parameters"]
        assert parameters[4] == 12
        assert parameters[5] == 3.5

    @pytest.mark.asyncio
    async def test_upsert_defaults_lag_fields_to_none(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": True}])
        )
        payload = make_minimal_payload()
        assert payload.lag_messages is None
        assert payload.lag_seconds is None

        await handler.upsert(payload)

        (envelope,), _ = db_handler.execute.call_args
        parameters = envelope["payload"]["parameters"]
        assert parameters[4] is None
        assert parameters[5] is None


class TestHandlerGatewayLinkHealthUpsertHandle:
    """handle() is the auto-wiring entry point the real dispatch path calls."""

    @pytest.mark.asyncio
    async def test_handle_accepts_dict_shaped_envelope(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": True}])
        )
        payload = make_minimal_payload()
        envelope = {
            "payload": payload.model_dump(mode="json"),
            "correlation_id": str(uuid4()),
        }

        output = await handler.handle(envelope)

        assert output.result is not None
        assert output.result.success is True
        assert output.result.tenant_id == payload.tenant_id

    @pytest.mark.asyncio
    async def test_handle_accepts_typed_payload_directly(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": False}])
        )
        payload = make_minimal_payload()

        output = await handler.handle({"payload": payload})

        assert output.result is not None
        assert output.result.was_insert is False

    @pytest.mark.asyncio
    async def test_handle_falls_back_to_fresh_correlation_id(self) -> None:
        """A missing/malformed correlation_id degrades to a fresh UUID rather
        than raising -- link-health refresh is best-effort and must never
        drop a row over a bad correlation_id."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": True}])
        )
        payload = make_minimal_payload()

        output = await handler.handle(
            {"payload": payload, "correlation_id": "not-a-uuid"}
        )

        assert output.correlation_id is not None
