# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A successful apply and a successful intent execution write no INFO line.

OMN-26152: in a 30 minute read of the dev runtime log (36,161 lines) the result applier's
"Published output event" (5,842) and "Delegated N intents" (2,026) and the intent executor's
"Intent executed" (2,026) were 9,894 lines, 27.4%, one set per runtime tick. They sit at DEBUG;
a failed intent still logs a WARNING.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)
from omnibase_infra.runtime.service_intent_executor import (
    IntentExecutor as ServiceIntentExecutor,
)

_SAMPLE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures/omn26152/dev_runtime_log_sample_20261126T0607Z.json"
    ).read_text(encoding="utf-8")
)
_EXECUTOR_LOGGER = "omnibase_infra.runtime.service_intent_executor"
_LOGGER = "omnibase_infra.runtime.service_dispatch_result_applier"


class _Out(BaseModel):
    value: str = "x"


class _Intent(BaseModel):
    intent_type: str = "ledger.append"


def _result(*, intents: bool) -> ModelDispatchResult:
    return ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic="onex.evt.platform.runtime-tick.v1",
        started_at=datetime.now(UTC),
        correlation_id=uuid4(),
        dispatcher_id="omn26152-tick",
        output_events=[_Out()],
        output_intents=[
            ModelIntent(
                intent_type="ledger.append",
                target="postgres://ledger/x",
                payload=_Intent(),
            )
        ]
        if intents
        else [],
    )


@pytest.mark.unit
def test_fixture_records_the_sampled_share() -> None:
    lines = (
        _SAMPLE["applier_published_output_event_info_lines"]
        + _SAMPLE["applier_delegated_intents_info_lines"]
        + _SAMPLE["intent_executor_intent_executed_info_lines"]
    )
    assert lines == 9894
    assert round(100 * lines / _SAMPLE["total_lines"], 1) == 27.4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_successful_apply_writes_no_info_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = AsyncMock()
    bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = DispatchResultApplier(
        event_bus=bus,
        output_topic="test.out.topic",
        intent_executor=executor,
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        await applier.apply(_result(intents=True))
    executor.execute_all.assert_awaited_once()
    bus.publish_envelope.assert_awaited_once()
    info = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert not [
        m for m in info if m.startswith(("Published output event", "Delegated"))
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_debug_still_writes_the_success_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    applier = DispatchResultApplier(
        event_bus=AsyncMock(spec=ProtocolEventBusLike),
        output_topic="test.out.topic",
        intent_executor=AsyncMock(),
    )
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await applier.apply(_result(intents=True))
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(m.startswith("Published output event") for m in debug)
    assert any(m.startswith("Delegated 1 intents") for m in debug)


class _Handler:
    async def execute(self, payload: object, correlation_id: object = None) -> None:
        return None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_successful_intent_execution_writes_no_info_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = ServiceIntentExecutor(
        container=MagicMock(), effect_handlers={"ledger.append": _Handler()}
    )
    intent = _result(intents=True).output_intents[0]
    with caplog.at_level(logging.INFO, logger=_EXECUTOR_LOGGER):
        await executor.execute(intent)
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=_EXECUTOR_LOGGER):
        await executor.execute(intent)
    assert any(r.getMessage().startswith("Intent executed") for r in caplog.records)
