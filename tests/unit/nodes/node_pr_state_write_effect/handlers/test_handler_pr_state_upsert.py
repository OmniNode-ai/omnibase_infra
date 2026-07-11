# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerPrStateUpsert (OMN-14375).

Tests validate:
- Constructor composes HandlerDb internally from `container` alone
  (OMN-14140 pattern — the auto-wiring resolver's known-injectable set).
- Lazy DB connection guard (missing DSN raises RuntimeHostError).
- Insert-vs-update detection via `RETURNING (xmax = 0) AS was_insert`.
- handle() drives the real dispatch-shaped entry point (dict-and-attribute
  envelope shapes, mirroring HandlerLedgerAppend._extract_envelope_field —
  OMN-14134/OMN-14139's dict-vs-attribute lesson applied from day one here).

Related Tickets:
    - OMN-14375: GitHub-state projection — WS-L fan-in producer.
    - OMN-14140: HandlerDb composed internally from `container`.
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
from omnibase_infra.nodes.node_pr_state_projection_compute.models import (
    ModelPayloadPrStateUpsert,
)
from omnibase_infra.nodes.node_pr_state_write_effect.handlers.handler_pr_state_upsert import (
    HandlerPrStateUpsert,
)

pytestmark = [pytest.mark.unit]


def make_mock_container() -> MagicMock:
    return MagicMock(spec=ModelONEXContainer)


def make_handler_with_mock_db(
    initialized: bool = True,
) -> tuple[HandlerPrStateUpsert, AsyncMock]:
    """Create a HandlerPrStateUpsert with its composed HandlerDb replaced by a mock."""
    container = make_mock_container()
    handler = HandlerPrStateUpsert(container)
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


def make_minimal_payload(**overrides: object) -> ModelPayloadPrStateUpsert:
    defaults: dict[str, object] = {
        "repo": "OmniNode-ai/omnibase_infra",
        "pr_number": 2260,
        "triage_state": "needs_review",
        "as_of": datetime.now(UTC),
    }
    defaults.update(overrides)
    return ModelPayloadPrStateUpsert(**defaults)


class TestHandlerPrStateUpsertComposition:
    """HandlerPrStateUpsert composes HandlerDb from `container` alone."""

    def test_constructor_takes_container_and_optional_dsn(self) -> None:
        container = make_mock_container()

        handler = HandlerPrStateUpsert(container)

        assert isinstance(handler._db_handler, HandlerDb)
        assert handler._initialized is False


class TestHandlerPrStateUpsertInitialization:
    """Lazy HandlerDb connection lifecycle."""

    @pytest.mark.asyncio
    async def test_upsert_raises_when_dsn_not_configured(self) -> None:
        container = make_mock_container()
        handler = HandlerPrStateUpsert(container)

        payload = make_minimal_payload()
        with pytest.raises(RuntimeHostError, match="Missing PostgreSQL DSN"):
            await handler.upsert(payload)

    @pytest.mark.asyncio
    async def test_ensure_db_ready_is_idempotent(self) -> None:
        container = make_mock_container()
        handler = HandlerPrStateUpsert(container, db_dsn="postgresql://test-dsn")
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


class TestHandlerPrStateUpsertInsertVsUpdate:
    """Insert-vs-update detection via RETURNING (xmax = 0) AS was_insert."""

    @pytest.mark.asyncio
    async def test_first_seen_pr_is_insert(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": True}])
        )

        payload = make_minimal_payload()
        result = await handler.upsert(payload)

        assert result.success is True
        assert result.was_insert is True
        assert result.repo == payload.repo
        assert result.pr_number == payload.pr_number

    @pytest.mark.asyncio
    async def test_refreshed_pr_is_update(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": False}])
        )

        payload = make_minimal_payload(triage_state="ready_to_merge")
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


class TestHandlerPrStateUpsertIsDraft:
    """OMN-14394: is_draft must reach the UPSERT SQL parameters -- the seam
    gap this ticket closes was is_draft silently dropped between the
    producer payload and the persisted row."""

    @pytest.mark.asyncio
    async def test_upsert_sends_is_draft_as_final_sql_parameter(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": True}])
        )
        payload = make_minimal_payload(is_draft=True)

        await handler.upsert(payload)

        db_handler.execute.assert_awaited_once()
        (envelope,), _ = db_handler.execute.call_args
        parameters = envelope["payload"]["parameters"]
        assert parameters[-1] is True

    @pytest.mark.asyncio
    async def test_upsert_defaults_is_draft_false(self) -> None:
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": True}])
        )
        payload = make_minimal_payload()
        assert payload.is_draft is False

        await handler.upsert(payload)

        (envelope,), _ = db_handler.execute.call_args
        parameters = envelope["payload"]["parameters"]
        assert parameters[-1] is False


class TestHandlerPrStateUpsertHandle:
    """handle() is the auto-wiring entry point the real dispatch path calls."""

    @pytest.mark.asyncio
    async def test_handle_accepts_dict_shaped_envelope(self) -> None:
        """MessageDispatchEngine delivers a DICT envelope on the live dispatch
        path (OMN-14139's lesson — extraction must not silently no-op on
        dict-shaped envelopes)."""
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
        assert output.result.repo == payload.repo

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
        than raising -- pr_state refresh is best-effort and must never drop a
        row over a bad correlation_id."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"was_insert": True}])
        )
        payload = make_minimal_payload()

        output = await handler.handle(
            {"payload": payload, "correlation_id": "not-a-uuid"}
        )

        assert output.correlation_id is not None
