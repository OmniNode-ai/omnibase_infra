# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for auto-wiring DB injection into projection handlers [OMN-8684].

Verifies that the wiring bridge (_make_projection_dispatch_callback) correctly
converts a ModelEventEnvelope into the dict-with-_db protocol expected by
projection handlers, using a mock adapter to avoid requiring a live database.
"""

from __future__ import annotations

import asyncio
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
def test_projection_dispatch_bridge_injects_db_and_event_type() -> None:
    """Projection handler receives dict with _db and _event_type from envelope.

    This verifies the protocol bridge introduced for OMN-8684:
    before: handler.handle(envelope) → TypeError (projection handlers don't accept envelopes)
    after: handler.handle({"_db": adapter, "_event_type": "heartbeat", ...payload...})
    """
    received: list[dict] = []

    class FakeProjectionHandler:
        def handle(self, input_data: dict) -> dict:
            received.append(dict(input_data))
            return {"rows_upserted": 1}

    db_tables = [{"name": "node_service_registry", "database": "omnidash_analytics"}]
    handler = FakeProjectionHandler()
    callback = _make_projection_dispatch_callback(
        handler, db_tables, ("onex.evt.platform.node-heartbeat.v1",)
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {"service_name": "test-svc", "health_status": "healthy"}

    fake_adapter = MagicMock()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://postgres:test@localhost:5436/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
            asyncio.run(callback(envelope))

    assert len(received) == 1, "Handler should have been called exactly once"
    assert received[0]["_db"] is fake_adapter, "_db must be the injected adapter"
    assert received[0]["_event_type"] == "heartbeat", (
        "heartbeat topic → event_type=heartbeat"
    )
    assert received[0].get("service_name") == "test-svc", (
        "payload fields must be preserved"
    )


@pytest.mark.integration
def test_projection_dispatch_bridge_no_call_when_db_url_missing() -> None:
    """Projection handler is NOT called when OMNIDASH_ANALYTICS_DB_URL is unset.

    Verifies no silent error occurs — handler simply skipped, no exception raised.
    """
    call_count = [0]

    class FakeProjectionHandler:
        def handle(self, input_data: dict) -> dict:
            call_count[0] += 1
            return {}

    db_tables = [{"name": "node_service_registry", "database": "omnidash_analytics"}]
    handler = FakeProjectionHandler()
    callback = _make_projection_dispatch_callback(handler, db_tables, ())

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {}

    with patch(_PATCH_ENVIRON_GET, return_value=""):
        result = asyncio.run(callback(envelope))

    assert result is None
    assert call_count[0] == 0, "Handler must not be called when DB URL is absent"
