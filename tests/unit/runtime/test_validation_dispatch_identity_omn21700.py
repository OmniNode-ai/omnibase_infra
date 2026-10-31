# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validation handlers retain identity across the typed dispatch boundary."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.validation import ModelValidationExecutionRequested
from omnibase_infra.models.validation_ledger import ModelPayloadValidationLedgerAppend
from omnibase_infra.nodes.node_validation_executor.handlers.handler_run_checks import (
    HandlerRunChecks,
)
from omnibase_infra.nodes.node_validation_ledger_write_effect.handlers.handler_validation_ledger_append import (
    HandlerValidationLedgerAppend,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from omnibase_infra.runtime.dispatch_envelope_context import (
    bind_dispatch_envelope,
    current_dispatch_envelope,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.unit


def _ledger_request(
    correlation_id: UUID | None = None,
) -> ModelPayloadValidationLedgerAppend:
    return ModelPayloadValidationLedgerAppend(
        run_id=uuid4(),
        repo_id="validation-test",
        event_type="ValidationChecksCompleted",
        event_version="v1",
        occurred_at=datetime.now(UTC),
        kafka_topic="onex.evt.platform.validation-checks-completed.v1",
        kafka_partition=0,
        kafka_offset=42,
        envelope_bytes="SGVsbG8=",
        envelope_hash="a" * 64,
        correlation_id=correlation_id,
    )


@pytest.mark.parametrize("payload_has_correlation", [False, True])
async def test_ledger_preserves_dispatch_identity(
    payload_has_correlation: bool,
) -> None:
    handler = HandlerValidationLedgerAppend(MagicMock(spec=ModelONEXContainer))
    handler._initialized = True
    db = AsyncMock()
    db.execute.return_value = MagicMock(result=MagicMock(payload=MagicMock(rows=[])))
    handler._db_handler = db
    payload_correlation = uuid4() if payload_has_correlation else None
    request = _ledger_request(payload_correlation)
    envelope = ModelEventEnvelope[object](payload=request, correlation_id=uuid4())

    with bind_dispatch_envelope(envelope):
        output = await handler.handle(request)

    expected_correlation = payload_correlation or envelope.correlation_id
    assert output.correlation_id == expected_correlation
    assert output.input_envelope_id == envelope.envelope_id
    assert db.execute.call_args.args[0]["correlation_id"] == str(expected_correlation)
    assert request.correlation_id == payload_correlation
    assert current_dispatch_envelope() is None


async def test_executor_dispatches_keep_per_message_correlation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = HandlerRunChecks(correlation_id=uuid4())
    engine = MessageDispatchEngine()
    topic = "onex.evt.platform.validation-execution-requested.v1"
    engine.register_dispatcher(
        dispatcher_id="validation-executor",
        dispatcher=_make_dispatch_callback(handler),
        category=EnumMessageCategory.EVENT,
        message_types={"ValidationExecutionRequested"},
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="validation-executor-route",
            topic_pattern=topic,
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="validation-executor",
        )
    )
    engine.freeze()
    envelopes = [
        ModelEventEnvelope[object](
            payload=ModelValidationExecutionRequested(candidate_id=uuid4(), checks=()),
            correlation_id=uuid4(),
            event_type="ValidationExecutionRequested",
        )
        for _ in range(2)
    ]

    with caplog.at_level(logging.INFO, logger=HandlerRunChecks.__module__):
        results = await asyncio.gather(
            *(engine.dispatch(topic, envelope) for envelope in envelopes)
        )

    assert all(result.status is EnumDispatchStatus.SUCCESS for result in results)
    observed = {
        record.args[-1]
        for record in caplog.records
        if record.msg.startswith("Executing validation plan")
    }
    assert observed == {envelope.correlation_id for envelope in envelopes}
    assert current_dispatch_envelope() is None
