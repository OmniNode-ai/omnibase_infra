# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536: derived SQL writes accept one typed static request."""

import inspect
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.handler_projection_derived_write import (
    HandlerProjectionDerivedWrite,
)
from omnibase_infra.models.projection.model_projection_derived_write_request import (
    ModelProjectionDerivedWriteRequest,
)
from omnibase_infra.protocols.protocol_projection_write_cursor import (
    ProtocolProjectionWriteCursor,
)


def _request() -> ModelProjectionDerivedWriteRequest:
    cursor = MagicMock(spec=ProtocolProjectionWriteCursor)
    cursor.run_guarded_derived_update.return_value = [
        {"correlation_id": "run", "payload": "input", "summary": "derived:input"}
    ]
    return ModelProjectionDerivedWriteRequest(
        cursor=cursor,
        physical_schema="omninode_internal",
        table="generation_events",
        stored_rows=({"correlation_id": "run", "payload": "input"},),
        conflict_columns=("correlation_id",),
        input_columns=("payload",),
        derived_columns=("summary",),
        returning=("correlation_id", "payload", "summary"),
        derive=lambda row: {"summary": f"derived:{row['payload']}"},
        adapt=lambda row: row,
    )


def test_handle_typed_request_returns_guarded_derived_rows() -> None:
    request = _request()

    rows = HandlerProjectionDerivedWrite.handle(request)

    assert rows == [
        {"correlation_id": "run", "payload": "input", "summary": "derived:input"}
    ]
    request.cursor.run_guarded_derived_update.assert_called_once()
    call = request.cursor.run_guarded_derived_update.call_args
    assert call.args[1] == {
        "derived_0": "derived:input",
        "input_0": "run",
        "input_1": "input",
    }
    assert call.kwargs == {"returning": request.returning}
    assert "omninode_internal" in repr(call.args[0])
    assert "generation_events" in repr(call.args[0])


def test_handle_without_returning_discards_internal_readback() -> None:
    request = _request().model_copy(update={"returning": ()})
    assert HandlerProjectionDerivedWrite.handle(request) == []
    assert request.cursor.run_guarded_derived_update.call_args.kwargs == {
        "returning": ("correlation_id",)
    }


def test_request_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelProjectionDerivedWriteRequest.model_validate(
            dict(_request()) | {"unknown": True}
        )


def test_request_is_frozen() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="frozen_instance"):
        request.table = "other"


def test_handle_signature() -> None:
    assert isinstance(
        inspect.getattr_static(HandlerProjectionDerivedWrite, "handle"), staticmethod
    )
    assert tuple(
        inspect.signature(HandlerProjectionDerivedWrite.handle).parameters
    ) == ("request",)
