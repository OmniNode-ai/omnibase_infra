# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime projection DB-injection dispatch [OMN-12245]."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_projection_dispatch_callback,
)

_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_sync_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"


@pytest.mark.integration
def test_runtime_projection_dispatch_skips_standalone_runner_classes() -> None:
    """Kafka projection runners must not be invoked by DB-injection auto-wiring."""

    class DelegationProjectionRunner:
        topics = ["onex.evt.omniclaude.task-delegated.v1"]

        def __init__(self) -> None:
            self.db = object()
            self.called = False

        async def project_event(self) -> None:
            self.called = True

        def handle(self, input_data: dict[str, object]) -> dict[str, bool]:
            self.called = True
            return {"projected": True}

    handler = DelegationProjectionRunner()
    callback = _make_projection_dispatch_callback(
        handler,
        [{"name": "delegation_events", "database": "omnidash_analytics"}],
        ("onex.evt.omniclaude.task-delegated.v1",),
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.omniclaude.task-delegated.v1"
    envelope.payload = {"correlation_id": "corr-1", "task_type": "release-proof"}

    result = asyncio.run(callback(envelope))

    assert result is None
    assert handler.called is False


@pytest.mark.integration
def test_runtime_projection_dispatch_runs_sync_handler_off_event_loop() -> None:
    """Regular sync projection handlers run in a worker thread under runtime dispatch."""

    loop_thread_id = threading.get_ident()
    handler_thread_ids: list[int] = []
    received: list[dict[str, object]] = []

    class HandlerProjectionDelegation:
        def handle(self, input_data: dict[str, object]) -> dict[str, int]:
            handler_thread_ids.append(threading.get_ident())
            received.append(dict(input_data))
            return {"rows_upserted": 1}

    callback = _make_projection_dispatch_callback(
        HandlerProjectionDelegation(),
        [{"name": "delegation_events", "database": "omnidash_analytics"}],
        ("onex.evt.omniclaude.task-delegated.v1",),
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.omniclaude.task-delegated.v1"
    envelope.payload = {
        "correlation_id": "corr-2",
        "task_type": "release-proof",
        "quality_gates_checked": 1,
    }
    fake_adapter = MagicMock()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
            result = asyncio.run(callback(envelope))

    assert result is None
    assert len(received) == 1
    assert received[0]["_db"] is fake_adapter
    assert received[0]["_event_type"] == "task-delegated"
    assert received[0]["task_type"] == "release-proof"
    assert handler_thread_ids == [handler_thread_ids[0]]
    assert handler_thread_ids[0] != loop_thread_id
