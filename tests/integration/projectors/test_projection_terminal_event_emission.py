# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for terminal event emission from projection callbacks (OMN-11187).

Verifies that _make_projection_dispatch_callback emits a terminal event envelope
to the declared terminal_event topic after each successful DB projection. This is
the integration-layer proof that the runtime wiring correctly bridges the projection
handler's DB write to the event bus observable by Pattern-B consumers.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_projection_dispatch_callback,
)

_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_sync_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"

TERMINAL_TOPIC = "onex.evt.omnimarket.projection-delegation-applied.v1"
DB_TABLES = [{"name": "delegation_events", "database": "omnidash_analytics"}]
SUBSCRIBE_TOPICS = ("onex.evt.omniclaude.task-delegated.v1",)


@pytest.mark.integration
def test_terminal_event_emitted_after_successful_projection() -> None:
    """After a successful DB projection, a terminal envelope is published to event_bus.

    This is the integration-level proof for OMN-11187: the projection callback
    wires handler.handle() → event_bus.publish(terminal_event, ...) when both
    event_bus and terminal_event are configured at the call site.
    """
    published: list[tuple[str, object, bytes]] = []
    correlation_id = uuid.uuid4()

    class FakeDelegationHandler:
        def handle(self, input_data: dict) -> dict:
            return {"rows_upserted": 1}

    class FakeEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    callback = _make_projection_dispatch_callback(
        FakeDelegationHandler(),
        DB_TABLES,
        SUBSCRIBE_TOPICS,
        event_bus=FakeEventBus(),
        terminal_event=TERMINAL_TOPIC,
    )

    envelope = MagicMock()
    envelope.topic = SUBSCRIBE_TOPICS[0]
    envelope.payload = {"task_type": "code-review", "delegated_to": "smoke-agent"}
    envelope.correlation_id = correlation_id

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://postgres:test@localhost:5436/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            asyncio.run(callback(envelope))

    assert len(published) == 1, "Exactly one terminal event must be published"
    topic, _, raw = published[0]
    assert topic == TERMINAL_TOPIC

    parsed = json.loads(raw.decode("utf-8"))
    assert parsed["event_type"] == TERMINAL_TOPIC
    assert parsed["correlation_id"] == str(correlation_id), (
        "correlation_id must propagate from source envelope to terminal event"
    )
    assert parsed["payload"] == {"projected": True}


@pytest.mark.integration
def test_terminal_event_not_emitted_when_projection_fails() -> None:
    """No terminal event is published when the projection handler raises an exception."""
    published: list[tuple] = []

    class FailingHandler:
        def handle(self, input_data: dict) -> dict:
            raise RuntimeError("DB write failed")

    class FakeEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    callback = _make_projection_dispatch_callback(
        FailingHandler(),
        DB_TABLES,
        SUBSCRIBE_TOPICS,
        event_bus=FakeEventBus(),
        terminal_event=TERMINAL_TOPIC,
    )

    envelope = MagicMock()
    envelope.topic = SUBSCRIBE_TOPICS[0]
    envelope.payload = {}
    envelope.correlation_id = uuid.uuid4()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://postgres:test@localhost:5436/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            asyncio.run(callback(envelope))

    assert len(published) == 0, (
        "No terminal event must be published when the handler raises"
    )


@pytest.mark.integration
def test_no_terminal_event_without_event_bus() -> None:
    """Projection callbacks without an event_bus configured emit no terminal event.

    Existing projection handlers that were wired before OMN-11187 pass event_bus=None
    (the default). This test ensures backward compatibility — those callbacks must
    not crash and must still successfully project to the DB.
    """
    call_count = [0]

    class FakeDelegationHandler:
        def handle(self, input_data: dict) -> dict:
            call_count[0] += 1
            return {"rows_upserted": 1}

    callback = _make_projection_dispatch_callback(
        FakeDelegationHandler(),
        DB_TABLES,
        SUBSCRIBE_TOPICS,
        event_bus=None,
        terminal_event=TERMINAL_TOPIC,
    )

    envelope = MagicMock()
    envelope.topic = SUBSCRIBE_TOPICS[0]
    envelope.payload = {"task_type": "code-review"}
    envelope.correlation_id = uuid.uuid4()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://postgres:test@localhost:5436/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            result = asyncio.run(callback(envelope))

    assert result is None
    assert call_count[0] == 1, "Handler must still be called when bus is None"
