# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof (OMN-26152): a real engine and a real stream handler at INFO.

A successful dispatch through a frozen ``MessageDispatchEngine`` must write no
"Dispatch started" or "Dispatch completed successfully" line to the rendered log
stream at the runtime's INFO level, and a failed dispatch must still reach it at ERROR.
"""

from __future__ import annotations

import io
import logging
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine


class _Payload:
    value = "tick"


def _engine(stream: io.StringIO, *, fail: bool) -> MessageDispatchEngine:
    logger = logging.getLogger(f"omn26152.integration.{uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)

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
    return engine


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rendered_info_stream_has_no_success_dispatch_lines() -> None:
    stream = io.StringIO()
    engine = _engine(stream, fail=False)

    for _ in range(65):
        result = await engine.dispatch(
            "dev.platform.events.v1",
            ModelEventEnvelope(payload=_Payload(), correlation_id=uuid4()),
        )
        assert result.status == EnumDispatchStatus.SUCCESS

    rendered = stream.getvalue()
    assert "Dispatch started" not in rendered
    assert "Dispatch completed successfully" not in rendered


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rendered_info_stream_keeps_failed_dispatch_error() -> None:
    stream = io.StringIO()
    engine = _engine(stream, fail=True)

    await engine.dispatch(
        "dev.platform.events.v1",
        ModelEventEnvelope(payload=_Payload(), correlation_id=uuid4()),
    )

    assert "[ERROR]" in stream.getvalue()
    assert "Dispatch completed with errors" in stream.getvalue()
