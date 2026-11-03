# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""IntentEffectDispatchBridge reaches a typed-request handler (OMN-21700).

``HandlerValidationLedgerAppend.handle(request: ModelPayloadValidationLedgerAppend)``
takes one typed model.  The kernel registers every audit/projection write effect
behind the one generic ``IntentEffectDispatchBridge``, which used to call
``handle({"payload": ..., "correlation_id": ...})`` for every target.  A typed
handler then raised ``AttributeError`` on ``payload.correlation_id`` and the
validation ledger stopped receiving rows.  The bridge owns the conversion: a
target whose ``handle`` takes a pydantic model gets that model, with the intent's
correlation id on the dispatch-envelope context; a target that still takes the
dict envelope gets the dict it got before.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.handlers.models import ModelDbQueryPayload, ModelDbQueryResponse
from omnibase_infra.models.validation_ledger import ModelPayloadValidationLedgerAppend
from omnibase_infra.nodes.node_validation_ledger_write_effect.handlers.handler_validation_ledger_append import (
    HandlerValidationLedgerAppend,
)
from omnibase_infra.runtime.dispatch_envelope_context import current_dispatch_envelope
from omnibase_infra.runtime.intent_effects import IntentEffectDispatchBridge


class _ProbeRequest(BaseModel):
    value: int


class _TypedProbe:
    def __init__(self) -> None:
        self.seen: list[UUID | None] = []

    async def handle(self, request: _ProbeRequest) -> None:
        envelope = current_dispatch_envelope()
        self.seen.append(envelope.correlation_id if envelope else None)


class _LegacyProbe:
    def __init__(self) -> None:
        self.received: list[object] = []

    async def handle(self, payload: object) -> None:
        self.received.append(payload)


def _payload(**overrides: object) -> ModelPayloadValidationLedgerAppend:
    defaults: dict[str, object] = {
        "run_id": uuid4(),
        "repo_id": "omnibase_core",
        "event_type": "onex.evt.validation.cross-repo-run-started.v1",
        "event_version": "v1",
        "occurred_at": datetime.now(UTC),
        "kafka_topic": "onex.evt.validation.cross-repo-run-started.v1",
        "kafka_partition": 0,
        "kafka_offset": 42,
        "envelope_bytes": "SGVsbG8gV29ybGQ=",
        "envelope_hash": "a" * 64,
    }
    defaults.update(overrides)
    return ModelPayloadValidationLedgerAppend.model_validate(defaults)


def _handler_with_mock_db(
    ledger_entry_id: UUID,
) -> tuple[HandlerValidationLedgerAppend, AsyncMock]:
    handler = HandlerValidationLedgerAppend(MagicMock(spec=ModelONEXContainer))
    db = AsyncMock()
    response = ModelDbQueryResponse(
        status=EnumResponseStatus.SUCCESS,
        payload=ModelDbQueryPayload(rows=[{"id": str(ledger_entry_id)}], row_count=1),
        correlation_id=uuid4(),
    )
    wrapper = MagicMock()
    wrapper.result = response
    db.execute = AsyncMock(return_value=wrapper)
    handler._db_handler = db
    handler._initialized = True
    return handler, db


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bridge_writes_validation_ledger_row_through_typed_handle() -> None:
    """The bridge-routed intent reaches append() and the DB write is issued."""
    handler, db = _handler_with_mock_db(uuid4())
    correlation_id = uuid4()

    await IntentEffectDispatchBridge(handler).execute(
        _payload(), correlation_id=correlation_id
    )

    db.execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bridge_carries_correlation_id_to_the_database_call() -> None:
    """The intent's correlation id rides to the DB envelope; payload id wins."""
    handler, db = _handler_with_mock_db(uuid4())
    correlation_id = uuid4()

    await IntentEffectDispatchBridge(handler).execute(
        _payload(correlation_id=None), correlation_id=correlation_id
    )

    sent = db.execute.await_args.args[0]
    assert sent["correlation_id"] == str(correlation_id)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bridge_coerces_a_wire_dict_payload_to_the_typed_request() -> None:
    """A dict intent payload is validated into the handler's request model."""
    handler, db = _handler_with_mock_db(uuid4())

    await IntentEffectDispatchBridge(handler).execute(
        _payload().model_dump(mode="json"), correlation_id=uuid4()
    )

    db.execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bridge_binds_dispatch_envelope_only_for_the_call() -> None:
    """The synthetic envelope context does not leak past the handler call."""
    typed = _TypedProbe()
    correlation_id = uuid4()
    await IntentEffectDispatchBridge(typed).execute(
        {"value": 1}, correlation_id=correlation_id
    )

    assert typed.seen == [correlation_id]
    assert current_dispatch_envelope() is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bridge_still_sends_dict_envelope_to_dict_envelope_handlers() -> None:
    """Handlers that take the raw envelope receive what they always received."""
    legacy = _LegacyProbe()
    correlation_id = uuid4()
    await IntentEffectDispatchBridge(legacy).execute(
        {"k": "v"}, correlation_id=correlation_id
    )

    assert legacy.received == [
        {"payload": {"k": "v"}, "correlation_id": correlation_id}
    ]
