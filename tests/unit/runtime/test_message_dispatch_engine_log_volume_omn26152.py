# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A successful dispatch must not write two INFO lines and build two log contexts.

OMN-26152: the dev runtime log of 2026-11-24T06:15Z to 07:07Z held 404,208 lines, and the
engine's own "Dispatch started" and "Dispatch completed successfully" were 319,692 of them
(79.1% of the lines, 39,297,357 of 72,275,128 bytes) at 52.0 dispatches per second. Each
pair also validated a pydantic ``ModelDispatchLogContext`` twice before the line was written.
The success pair now sits at DEBUG and builds its context only when DEBUG is enabled; a failed
dispatch still logs ``Dispatch completed with errors`` at ERROR.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_SAMPLE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures/omn26152/dev_runtime_log_sample_20261124T0707Z.json"
    ).read_text(encoding="utf-8")
)
_SUCCESS_LINES = ("Dispatch started", "Dispatch completed successfully")


class _Payload:
    def __init__(self, value: str) -> None:
        self.value = value


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _engine(
    level: int, *, fail: bool = False
) -> tuple[MessageDispatchEngine, _Collect, logging.Logger]:
    logger = logging.getLogger(f"omn26152.engine.{uuid4().hex}")
    logger.setLevel(level)
    logger.propagate = False
    collected = _Collect()
    logger.addHandler(collected)

    async def dispatcher(envelope: ModelEventEnvelope[object]) -> str:
        if fail:
            raise RuntimeError("handler failed")
        return "output.topic.v1"

    engine = MessageDispatchEngine(logger=logger)
    engine.register_dispatcher(
        dispatcher_id="tick-dispatcher",
        dispatcher=dispatcher,
        category=EnumMessageCategory.EVENT,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="tick-route",
            topic_pattern="*.platform.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="tick-dispatcher",
        )
    )
    engine.freeze()
    return engine, collected, logger


def _fan_out_per_tick() -> int:
    """Dispatches one runtime tick fans out to, read from the recorded sample."""
    started = int(_SAMPLE["dispatch_started_lines"])
    return round(started / int(_SAMPLE["runtime_tick_dispatch_failures"]))


async def _dispatch_one_tick(engine: MessageDispatchEngine) -> None:
    for _ in range(_fan_out_per_tick()):
        result = await engine.dispatch(
            "dev.platform.events.v1",
            ModelEventEnvelope(payload=_Payload("tick"), correlation_id=uuid4()),
        )
        assert result.status == EnumDispatchStatus.SUCCESS


def _success_lines(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    return [r for r in records if r.getMessage() in _SUCCESS_LINES]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recorded_tick_fan_out_writes_no_info_line_per_successful_dispatch() -> (
    None
):
    engine, collected, _ = _engine(logging.INFO)

    await _dispatch_one_tick(engine)

    info_lines = [
        r for r in _success_lines(collected.records) if r.levelno >= logging.INFO
    ]
    assert _fan_out_per_tick() == 65
    assert info_lines == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_successful_dispatch_builds_no_log_context_below_debug() -> None:
    engine, _, _ = _engine(logging.INFO)

    with patch.object(
        engine, "_build_log_context", wraps=engine._build_log_context
    ) as build:
        await _dispatch_one_tick(engine)

    assert build.call_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_success_pair_is_still_written_when_debug_is_enabled() -> None:
    engine, collected, _ = _engine(logging.DEBUG)

    await _dispatch_one_tick(engine)

    lines = _success_lines(collected.records)
    assert len(lines) == 2 * _fan_out_per_tick()
    assert {r.levelno for r in lines} == {logging.DEBUG}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_dispatch_still_logs_error_at_info_level() -> None:
    engine, collected, _ = _engine(logging.INFO, fail=True)

    result = await engine.dispatch(
        "dev.platform.events.v1",
        ModelEventEnvelope(payload=_Payload("tick"), correlation_id=uuid4()),
    )

    assert result.status != EnumDispatchStatus.SUCCESS
    errors = [
        r
        for r in collected.records
        if r.getMessage() == "Dispatch completed with errors"
    ]
    assert len(errors) == 1
    assert errors[0].levelno == logging.ERROR
