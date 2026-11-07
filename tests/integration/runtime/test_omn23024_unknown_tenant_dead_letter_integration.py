# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23024: an unknown-tenant event reaches the dead-letter topic once, over a real bus.

The unit file beside ``tests/unit/runtime/auto_wiring`` drives the dispatch
callback against recording doubles. This one gives the callback a real
``EventBusInmemory`` as its dead-letter sink and reads the record back off a
subscription on the declared dead-letter topic, so what is asserted is a message
that crossed the bus, not a call that was recorded.

Replaying the same unresolvable event a second time is a second dead letter, not
a retry loop inside one delivery: the callback acknowledges each delivery after
exactly one dead letter and never withholds the offset.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDispatchSinks,
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"
_TEST_DSN = "postgresql://user:***REDACTED***@host:5432/omnidash_analytics"

_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_DLQ_TOPIC = "onex.dlq.omnimarket.delegation-projection.v1"
_UNKNOWN_TENANT = "99ffb290-5fdc-4b8a-9c6b-e70239214928"
_RESOLUTION_MODULE = "omnimarket.projection.tenant_registry_resolution"

TenantRegistryResolutionError = type(
    "TenantRegistryResolutionError", (ValueError,), {"__module__": _RESOLUTION_MODULE}
)


class _UnknownTenantHandler:
    def __init__(self) -> None:
        self.calls = 0

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        raise TenantRegistryResolutionError(
            f"tenant_registry_mirror holds no row for '{_UNKNOWN_TENANT}'"
        )


def _recorded_envelope() -> MagicMock:
    envelope = MagicMock()
    envelope.topic = _TOPIC
    envelope.payload = {"tenant_id": _UNKNOWN_TENANT, "status": "completed"}
    envelope.correlation_id = "d2fddb35-a014-4068-bc40-4c57330c6b58"
    return envelope


@pytest.fixture(autouse=True)
def _configured_projection_dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch, url=_TEST_DSN)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unknown_tenant_event_is_dead_lettered_once_over_a_real_bus() -> None:
    bus = EventBusInmemory(environment="test", group="omn23024")
    await bus.start()
    try:
        dead_letters: asyncio.Queue[ModelEventMessage] = asyncio.Queue()

        async def collect(message: ModelEventMessage) -> None:
            await dead_letters.put(message)

        await bus.subscribe(_DLQ_TOPIC, group_id="dlq-collector", on_message=collect)

        handler = _UnknownTenantHandler()
        callback = _make_projection_dispatch_callback(
            handler,
            projection_database_target(
                "session_replay_snapshots", schema="omninode_internal"
            ),
            (_TOPIC,),
            sinks=ProjectionDispatchSinks(event_bus=bus, dlq_topics=[_DLQ_TOPIC]),
        )

        with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
            with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
                result = await callback(_recorded_envelope())

        first = await asyncio.wait_for(dead_letters.get(), timeout=5)
        body = first.value.decode("utf-8", "replace")

        assert result is None, "the delivery is acknowledged; the offset advances"
        assert handler.calls == 1
        assert "TenantRegistryResolutionError" in body
        assert _UNKNOWN_TENANT in body
        assert dead_letters.empty(), "exactly one dead letter per delivery"
    finally:
        await bus.close()
