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
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)


@pytest.fixture(autouse=True)
def _configured_projection_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure every binding DSN the shipped topology declares.

    OMN-17142/OMN-17152: this used to set ``ONEX_TENANT_DB_URL`` alone, then
    OMN-15425 moved tenant-schema targets onto the ``tenant_projection``
    binding and this file only followed the one binding it happened to use.
    These tests are proving dispatch mechanics, not binding selection, so they
    take the whole topology-declared set — the next binding addition is
    covered without editing this fixture again.
    """
    configure_projection_dsns(monkeypatch)


_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"


@pytest.mark.integration
def test_runtime_projection_dispatch_skips_standalone_runner_classes() -> None:
    """Kafka projection runners must not be invoked by DB-injection auto-wiring."""

    class _SelfServedAdapter:
        """The runner's own async pool adapter, opened in its constructor."""

        async def connect(self) -> None: ...

        async def close(self) -> None: ...

    class DelegationProjectionRunner:
        topics = ["onex.evt.omniclaude.task-delegated.v1"]

        def __init__(self) -> None:
            # OMN-16874: the skip is decided by SHAPE + declared capability, not
            # by the class name. A standalone runner owns its consume loop
            # (`run`), its projection entrypoint, its topics and its own DB
            # adapter, and declares no in-process dispatch capability.
            self.db = _SelfServedAdapter()
            self.called = False

        async def run(self) -> None:
            raise AssertionError("the runtime must not drive run()")

        async def project_event(self) -> None:
            self.called = True

        def handle(self, input_data: dict[str, object]) -> dict[str, bool]:
            self.called = True
            return {"projected": True}

    handler = DelegationProjectionRunner()
    callback = _make_projection_dispatch_callback(
        handler,
        projection_database_target("delegation_events"),
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
        projection_database_target("delegation_events"),
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
