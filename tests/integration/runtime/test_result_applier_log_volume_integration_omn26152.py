# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof (OMN-26152): the real applier and executor at INFO write no success line.

A ``DispatchResultApplier`` driving a real ``IntentExecutor`` through a real INFO stream
handler must render no "Published output event", "Delegated" or "Intent executed" line.
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime
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
from omnibase_infra.runtime.service_intent_executor import IntentExecutor


class _Out(BaseModel):
    value: str = "x"


class _Intent(BaseModel):
    intent_type: str = "ledger.append"


class _Handler:
    async def execute(self, payload: object, correlation_id: object = None) -> None:
        return None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rendered_stream_holds_no_applier_or_executor_success_line() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    names = (
        "omnibase_infra.runtime.service_dispatch_result_applier",
        "omnibase_infra.runtime.service_intent_executor",
    )
    loggers = [logging.getLogger(n) for n in names]
    saved = [(lg.level, lg.propagate) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
    try:
        executor = IntentExecutor(
            container=MagicMock(), effect_handlers={"ledger.append": _Handler()}
        )
        applier = DispatchResultApplier(
            event_bus=AsyncMock(spec=ProtocolEventBusLike),
            output_topic="test.out.topic",
            intent_executor=executor,
        )
        await applier.apply(
            ModelDispatchResult(
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
                ],
            )
        )
    finally:
        for lg, (level, _) in zip(loggers, saved, strict=True):
            lg.removeHandler(handler)
            lg.setLevel(level)
    rendered = stream.getvalue()
    for text in ("Published output event", "Delegated", "Intent executed"):
        assert text not in rendered
