# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536: validation ledger handles one typed payload."""

import inspect
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.validation_ledger import ModelPayloadValidationLedgerAppend
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from tests.unit.nodes.node_validation_ledger_write_effect.handlers.test_handler_validation_ledger_append import (
    make_db_result,
    make_handler_with_mock_db,
    make_minimal_payload,
)


async def test_handle_typed_payload_preserves_append_result_and_correlation() -> None:
    handler, db = make_handler_with_mock_db()
    ledger_entry_id = uuid4()
    correlation_id = uuid4()
    request = make_minimal_payload(correlation_id=correlation_id)
    db.execute.return_value = make_db_result([{"id": str(ledger_entry_id)}])

    output = await handler.handle(request)

    assert output.correlation_id == correlation_id
    assert output.handler_id == "validation-ledger-append-handler"
    assert output.result is not None
    assert output.result.success is True
    assert output.result.duplicate is False
    assert output.result.ledger_entry_id == ledger_entry_id
    envelope = db.execute.call_args.args[0]
    assert envelope["correlation_id"] == str(correlation_id)
    assert envelope["payload"]["parameters"][8] == b"Hello World"


async def test_handle_duplicate_with_default_correlation() -> None:
    handler, db = make_handler_with_mock_db()
    db.execute.return_value = make_db_result([])

    output = await handler.handle(make_minimal_payload())

    assert isinstance(output.correlation_id, UUID)
    assert output.result is not None
    assert output.result.duplicate is True
    assert output.result.ledger_entry_id is None


async def test_runtime_unwraps_and_validates_payload_before_handle() -> None:
    handler, db = make_handler_with_mock_db()
    request = make_minimal_payload(correlation_id=uuid4())
    db.execute.return_value = make_db_result([])
    handler.append = AsyncMock(wraps=handler.append)
    callback = _make_dispatch_callback(handler)

    result = await callback(
        ModelEventEnvelope[object](
            payload=request.model_dump(mode="json"),
            correlation_id=request.correlation_id,
        )
    )

    assert result is not None
    handler.append.assert_awaited_once_with(request)


def test_payload_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelPayloadValidationLedgerAppend.model_validate(
            dict(make_minimal_payload()) | {"unknown": True}
        )


def test_handle_signature() -> None:
    handler, _ = make_handler_with_mock_db()
    assert tuple(inspect.signature(handler.handle).parameters) == ("request",)
